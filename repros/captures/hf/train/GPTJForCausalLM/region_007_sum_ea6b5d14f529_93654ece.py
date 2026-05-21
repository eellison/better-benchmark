"""
Standalone repro captured via capture_hook.
Label: hf_GPTJForCausalLM_train
Pattern hash: ea6b5d14f529
Shape hash: 93654ece
"""
_shapes_config = "(T([128, 50400], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([1, 128, 4096], f32), T([1, 128, 4096], f32), T([128, 4096], f32), T([128, 16384], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([128, 4096], f32), T([4096], f32), T([1, 128, 4096], f32), T([1, 128, 1], f32), T([1, 128, 1], f32), T([1, 128, 4096], f32), T([1, 128], i64), T([], f32), S([50400]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([4096]), S([16384]), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]), S([1, 128, 4096]))"
import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

class Repro(torch.nn.Module):
    def forward(self, view_790: "f32[128, 50400]", view_792: "f32[1, 128, 4096]", mul_280: "f32[1, 128, 4096]", view_794: "f32[128, 4096]", view_797: "f32[128, 16384]", view_814: "f32[128, 4096]", view_816: "f32[128, 4096]", view_818: "f32[128, 4096]", add_268: "f32[1, 128, 4096]", mul_270: "f32[1, 128, 4096]", view_820: "f32[128, 4096]", view_823: "f32[128, 16384]", view_840: "f32[128, 4096]", view_842: "f32[128, 4096]", view_844: "f32[128, 4096]", add_280: "f32[1, 128, 4096]", mul_260: "f32[1, 128, 4096]", view_846: "f32[128, 4096]", view_849: "f32[128, 16384]", view_866: "f32[128, 4096]", view_868: "f32[128, 4096]", view_870: "f32[128, 4096]", add_292: "f32[1, 128, 4096]", mul_250: "f32[1, 128, 4096]", view_872: "f32[128, 4096]", view_875: "f32[128, 16384]", view_892: "f32[128, 4096]", view_894: "f32[128, 4096]", view_896: "f32[128, 4096]", add_304: "f32[1, 128, 4096]", mul_240: "f32[1, 128, 4096]", view_898: "f32[128, 4096]", view_901: "f32[128, 16384]", view_918: "f32[128, 4096]", view_920: "f32[128, 4096]", view_922: "f32[128, 4096]", add_316: "f32[1, 128, 4096]", mul_230: "f32[1, 128, 4096]", view_924: "f32[128, 4096]", view_927: "f32[128, 16384]", view_944: "f32[128, 4096]", view_946: "f32[128, 4096]", view_948: "f32[128, 4096]", add_328: "f32[1, 128, 4096]", mul_220: "f32[1, 128, 4096]", view_950: "f32[128, 4096]", view_953: "f32[128, 16384]", view_970: "f32[128, 4096]", view_972: "f32[128, 4096]", view_974: "f32[128, 4096]", add_340: "f32[1, 128, 4096]", mul_210: "f32[1, 128, 4096]", view_976: "f32[128, 4096]", view_979: "f32[128, 16384]", view_996: "f32[128, 4096]", view_998: "f32[128, 4096]", view_1000: "f32[128, 4096]", add_352: "f32[1, 128, 4096]", mul_200: "f32[1, 128, 4096]", view_1002: "f32[128, 4096]", view_1005: "f32[128, 16384]", view_1022: "f32[128, 4096]", view_1024: "f32[128, 4096]", view_1026: "f32[128, 4096]", add_364: "f32[1, 128, 4096]", mul_190: "f32[1, 128, 4096]", view_1028: "f32[128, 4096]", view_1031: "f32[128, 16384]", view_1048: "f32[128, 4096]", view_1050: "f32[128, 4096]", view_1052: "f32[128, 4096]", add_376: "f32[1, 128, 4096]", mul_180: "f32[1, 128, 4096]", view_1054: "f32[128, 4096]", view_1057: "f32[128, 16384]", view_1074: "f32[128, 4096]", view_1076: "f32[128, 4096]", view_1078: "f32[128, 4096]", add_388: "f32[1, 128, 4096]", mul_170: "f32[1, 128, 4096]", view_1080: "f32[128, 4096]", view_1083: "f32[128, 16384]", view_1100: "f32[128, 4096]", view_1102: "f32[128, 4096]", view_1104: "f32[128, 4096]", add_400: "f32[1, 128, 4096]", mul_160: "f32[1, 128, 4096]", view_1106: "f32[128, 4096]", view_1109: "f32[128, 16384]", view_1126: "f32[128, 4096]", view_1128: "f32[128, 4096]", view_1130: "f32[128, 4096]", add_412: "f32[1, 128, 4096]", mul_150: "f32[1, 128, 4096]", view_1132: "f32[128, 4096]", view_1135: "f32[128, 16384]", view_1152: "f32[128, 4096]", view_1154: "f32[128, 4096]", view_1156: "f32[128, 4096]", add_424: "f32[1, 128, 4096]", mul_140: "f32[1, 128, 4096]", view_1158: "f32[128, 4096]", view_1161: "f32[128, 16384]", view_1178: "f32[128, 4096]", view_1180: "f32[128, 4096]", view_1182: "f32[128, 4096]", add_436: "f32[1, 128, 4096]", mul_130: "f32[1, 128, 4096]", view_1184: "f32[128, 4096]", view_1187: "f32[128, 16384]", view_1204: "f32[128, 4096]", view_1206: "f32[128, 4096]", view_1208: "f32[128, 4096]", add_448: "f32[1, 128, 4096]", mul_120: "f32[1, 128, 4096]", view_1210: "f32[128, 4096]", view_1213: "f32[128, 16384]", view_1230: "f32[128, 4096]", view_1232: "f32[128, 4096]", view_1234: "f32[128, 4096]", add_460: "f32[1, 128, 4096]", mul_110: "f32[1, 128, 4096]", view_1236: "f32[128, 4096]", view_1239: "f32[128, 16384]", view_1256: "f32[128, 4096]", view_1258: "f32[128, 4096]", view_1260: "f32[128, 4096]", add_472: "f32[1, 128, 4096]", mul_100: "f32[1, 128, 4096]", view_1262: "f32[128, 4096]", view_1265: "f32[128, 16384]", view_1282: "f32[128, 4096]", view_1284: "f32[128, 4096]", view_1286: "f32[128, 4096]", add_484: "f32[1, 128, 4096]", mul_90: "f32[1, 128, 4096]", view_1288: "f32[128, 4096]", view_1291: "f32[128, 16384]", view_1308: "f32[128, 4096]", view_1310: "f32[128, 4096]", view_1312: "f32[128, 4096]", add_496: "f32[1, 128, 4096]", mul_80: "f32[1, 128, 4096]", view_1314: "f32[128, 4096]", view_1317: "f32[128, 16384]", view_1334: "f32[128, 4096]", view_1336: "f32[128, 4096]", view_1338: "f32[128, 4096]", add_508: "f32[1, 128, 4096]", mul_70: "f32[1, 128, 4096]", view_1340: "f32[128, 4096]", view_1343: "f32[128, 16384]", view_1360: "f32[128, 4096]", view_1362: "f32[128, 4096]", view_1364: "f32[128, 4096]", add_520: "f32[1, 128, 4096]", mul_60: "f32[1, 128, 4096]", view_1366: "f32[128, 4096]", view_1369: "f32[128, 16384]", view_1386: "f32[128, 4096]", view_1388: "f32[128, 4096]", view_1390: "f32[128, 4096]", add_532: "f32[1, 128, 4096]", mul_50: "f32[1, 128, 4096]", view_1392: "f32[128, 4096]", view_1395: "f32[128, 16384]", view_1412: "f32[128, 4096]", view_1414: "f32[128, 4096]", view_1416: "f32[128, 4096]", add_544: "f32[1, 128, 4096]", mul_40: "f32[1, 128, 4096]", view_1418: "f32[128, 4096]", view_1421: "f32[128, 16384]", view_1438: "f32[128, 4096]", view_1440: "f32[128, 4096]", view_1442: "f32[128, 4096]", add_556: "f32[1, 128, 4096]", mul_30: "f32[1, 128, 4096]", view_1444: "f32[128, 4096]", view_1447: "f32[128, 16384]", view_1464: "f32[128, 4096]", view_1466: "f32[128, 4096]", view_1468: "f32[128, 4096]", add_568: "f32[1, 128, 4096]", mul_20: "f32[1, 128, 4096]", view_1470: "f32[128, 4096]", view_1473: "f32[128, 16384]", view_1490: "f32[128, 4096]", view_1492: "f32[128, 4096]", view_1494: "f32[128, 4096]", add_580: "f32[1, 128, 4096]", mul_10: "f32[1, 128, 4096]", view_1496: "f32[128, 4096]", view_1499: "f32[128, 16384]", mm_440: "f32[128, 4096]", view_1516: "f32[128, 4096]", mm_445: "f32[128, 4096]", view_1518: "f32[128, 4096]", mm_447: "f32[128, 4096]", view_1520: "f32[128, 4096]", mm_449: "f32[128, 4096]", primals_3: "f32[4096]", embedding: "f32[1, 128, 4096]", getitem_1: "f32[1, 128, 1]", rsqrt: "f32[1, 128, 1]", add_581: "f32[1, 128, 4096]", primals_1: "i64[1, 128]", full_default_1: "f32[]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:623 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        permute_default: "f32[50400, 128]" = torch.ops.aten.permute.default(view_790, [1, 0])
        sum_dim_int_list: "f32[1, 50400]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
        reshape_default: "f32[50400]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:542 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_tensor: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_792, mul_280);  mul_280 = None
        sum_dim_int_list_1: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[4096]" = torch.ops.aten.sum.dim_IntList(view_792, [0, 1]);  view_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_1: "f32[4096, 128]" = torch.ops.aten.permute.default(view_794, [1, 0])
        sum_dim_int_list_3: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_794, [0], True);  view_794 = None
        reshape_default_1: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_2: "f32[16384, 128]" = torch.ops.aten.permute.default(view_797, [1, 0])
        sum_dim_int_list_4: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_797, [0], True);  view_797 = None
        reshape_default_2: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_3: "f32[4096, 128]" = torch.ops.aten.permute.default(view_814, [1, 0]);  view_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_4: "f32[4096, 128]" = torch.ops.aten.permute.default(view_816, [1, 0]);  view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_5: "f32[4096, 128]" = torch.ops.aten.permute.default(view_818, [1, 0]);  view_818 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_268, mul_270);  mul_270 = None
        sum_dim_int_list_5: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_6: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_268, [0, 1]);  add_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_6: "f32[4096, 128]" = torch.ops.aten.permute.default(view_820, [1, 0])
        sum_dim_int_list_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_820, [0], True);  view_820 = None
        reshape_default_3: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_7: "f32[16384, 128]" = torch.ops.aten.permute.default(view_823, [1, 0])
        sum_dim_int_list_8: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_823, [0], True);  view_823 = None
        reshape_default_4: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_4);  sum_dim_int_list_8 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_8: "f32[4096, 128]" = torch.ops.aten.permute.default(view_840, [1, 0]);  view_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_9: "f32[4096, 128]" = torch.ops.aten.permute.default(view_842, [1, 0]);  view_842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_10: "f32[4096, 128]" = torch.ops.aten.permute.default(view_844, [1, 0]);  view_844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_280, mul_260);  mul_260 = None
        sum_dim_int_list_9: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_10: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_280, [0, 1]);  add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_11: "f32[4096, 128]" = torch.ops.aten.permute.default(view_846, [1, 0])
        sum_dim_int_list_11: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
        reshape_default_5: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_12: "f32[16384, 128]" = torch.ops.aten.permute.default(view_849, [1, 0])
        sum_dim_int_list_12: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
        reshape_default_6: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_13: "f32[4096, 128]" = torch.ops.aten.permute.default(view_866, [1, 0]);  view_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_14: "f32[4096, 128]" = torch.ops.aten.permute.default(view_868, [1, 0]);  view_868 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_15: "f32[4096, 128]" = torch.ops.aten.permute.default(view_870, [1, 0]);  view_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_292, mul_250);  mul_250 = None
        sum_dim_int_list_13: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_14: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_292, [0, 1]);  add_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_16: "f32[4096, 128]" = torch.ops.aten.permute.default(view_872, [1, 0])
        sum_dim_int_list_15: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_872, [0], True);  view_872 = None
        reshape_default_7: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_7);  sum_dim_int_list_15 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_17: "f32[16384, 128]" = torch.ops.aten.permute.default(view_875, [1, 0])
        sum_dim_int_list_16: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_875, [0], True);  view_875 = None
        reshape_default_8: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_18: "f32[4096, 128]" = torch.ops.aten.permute.default(view_892, [1, 0]);  view_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_19: "f32[4096, 128]" = torch.ops.aten.permute.default(view_894, [1, 0]);  view_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_20: "f32[4096, 128]" = torch.ops.aten.permute.default(view_896, [1, 0]);  view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_4: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_304, mul_240);  mul_240 = None
        sum_dim_int_list_17: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_18: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_304, [0, 1]);  add_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_21: "f32[4096, 128]" = torch.ops.aten.permute.default(view_898, [1, 0])
        sum_dim_int_list_19: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
        reshape_default_9: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_9);  sum_dim_int_list_19 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_22: "f32[16384, 128]" = torch.ops.aten.permute.default(view_901, [1, 0])
        sum_dim_int_list_20: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_901, [0], True);  view_901 = None
        reshape_default_10: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_10);  sum_dim_int_list_20 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_23: "f32[4096, 128]" = torch.ops.aten.permute.default(view_918, [1, 0]);  view_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_24: "f32[4096, 128]" = torch.ops.aten.permute.default(view_920, [1, 0]);  view_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_25: "f32[4096, 128]" = torch.ops.aten.permute.default(view_922, [1, 0]);  view_922 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_5: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_316, mul_230);  mul_230 = None
        sum_dim_int_list_21: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_22: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_316, [0, 1]);  add_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_26: "f32[4096, 128]" = torch.ops.aten.permute.default(view_924, [1, 0])
        sum_dim_int_list_23: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_924, [0], True);  view_924 = None
        reshape_default_11: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_11);  sum_dim_int_list_23 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_27: "f32[16384, 128]" = torch.ops.aten.permute.default(view_927, [1, 0])
        sum_dim_int_list_24: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_927, [0], True);  view_927 = None
        reshape_default_12: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_12);  sum_dim_int_list_24 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_28: "f32[4096, 128]" = torch.ops.aten.permute.default(view_944, [1, 0]);  view_944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_29: "f32[4096, 128]" = torch.ops.aten.permute.default(view_946, [1, 0]);  view_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_30: "f32[4096, 128]" = torch.ops.aten.permute.default(view_948, [1, 0]);  view_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_6: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_328, mul_220);  mul_220 = None
        sum_dim_int_list_25: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_26: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_328, [0, 1]);  add_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_31: "f32[4096, 128]" = torch.ops.aten.permute.default(view_950, [1, 0])
        sum_dim_int_list_27: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_950, [0], True);  view_950 = None
        reshape_default_13: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_13);  sum_dim_int_list_27 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_32: "f32[16384, 128]" = torch.ops.aten.permute.default(view_953, [1, 0])
        sum_dim_int_list_28: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_953, [0], True);  view_953 = None
        reshape_default_14: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_14);  sum_dim_int_list_28 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_33: "f32[4096, 128]" = torch.ops.aten.permute.default(view_970, [1, 0]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_34: "f32[4096, 128]" = torch.ops.aten.permute.default(view_972, [1, 0]);  view_972 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_35: "f32[4096, 128]" = torch.ops.aten.permute.default(view_974, [1, 0]);  view_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_7: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_340, mul_210);  mul_210 = None
        sum_dim_int_list_29: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_30: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_340, [0, 1]);  add_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_36: "f32[4096, 128]" = torch.ops.aten.permute.default(view_976, [1, 0])
        sum_dim_int_list_31: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_976, [0], True);  view_976 = None
        reshape_default_15: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_15);  sum_dim_int_list_31 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_37: "f32[16384, 128]" = torch.ops.aten.permute.default(view_979, [1, 0])
        sum_dim_int_list_32: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_979, [0], True);  view_979 = None
        reshape_default_16: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_16);  sum_dim_int_list_32 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_38: "f32[4096, 128]" = torch.ops.aten.permute.default(view_996, [1, 0]);  view_996 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_39: "f32[4096, 128]" = torch.ops.aten.permute.default(view_998, [1, 0]);  view_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_40: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1000, [1, 0]);  view_1000 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_8: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_352, mul_200);  mul_200 = None
        sum_dim_int_list_33: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_34: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_352, [0, 1]);  add_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_41: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1002, [1, 0])
        sum_dim_int_list_35: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1002, [0], True);  view_1002 = None
        reshape_default_17: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_17);  sum_dim_int_list_35 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_42: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1005, [1, 0])
        sum_dim_int_list_36: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1005, [0], True);  view_1005 = None
        reshape_default_18: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_18);  sum_dim_int_list_36 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_43: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1022, [1, 0]);  view_1022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_44: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1024, [1, 0]);  view_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_45: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1026, [1, 0]);  view_1026 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_9: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_364, mul_190);  mul_190 = None
        sum_dim_int_list_37: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_38: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_364, [0, 1]);  add_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_46: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1028, [1, 0])
        sum_dim_int_list_39: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1028, [0], True);  view_1028 = None
        reshape_default_19: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_19);  sum_dim_int_list_39 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_47: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1031, [1, 0])
        sum_dim_int_list_40: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1031, [0], True);  view_1031 = None
        reshape_default_20: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_20);  sum_dim_int_list_40 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_48: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1048, [1, 0]);  view_1048 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_49: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1050, [1, 0]);  view_1050 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_50: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1052, [1, 0]);  view_1052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_10: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_376, mul_180);  mul_180 = None
        sum_dim_int_list_41: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_42: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_376, [0, 1]);  add_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_51: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1054, [1, 0])
        sum_dim_int_list_43: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1054, [0], True);  view_1054 = None
        reshape_default_21: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_21);  sum_dim_int_list_43 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_52: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1057, [1, 0])
        sum_dim_int_list_44: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1057, [0], True);  view_1057 = None
        reshape_default_22: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_22);  sum_dim_int_list_44 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_53: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1074, [1, 0]);  view_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_54: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1076, [1, 0]);  view_1076 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_55: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1078, [1, 0]);  view_1078 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_11: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_388, mul_170);  mul_170 = None
        sum_dim_int_list_45: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_46: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_388, [0, 1]);  add_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_56: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1080, [1, 0])
        sum_dim_int_list_47: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1080, [0], True);  view_1080 = None
        reshape_default_23: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_23);  sum_dim_int_list_47 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_57: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1083, [1, 0])
        sum_dim_int_list_48: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1083, [0], True);  view_1083 = None
        reshape_default_24: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_24);  sum_dim_int_list_48 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_58: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1100, [1, 0]);  view_1100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_59: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1102, [1, 0]);  view_1102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_60: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1104, [1, 0]);  view_1104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_12: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_400, mul_160);  mul_160 = None
        sum_dim_int_list_49: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_50: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_400, [0, 1]);  add_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_61: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1106, [1, 0])
        sum_dim_int_list_51: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1106, [0], True);  view_1106 = None
        reshape_default_25: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_25);  sum_dim_int_list_51 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_62: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1109, [1, 0])
        sum_dim_int_list_52: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1109, [0], True);  view_1109 = None
        reshape_default_26: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_26);  sum_dim_int_list_52 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_63: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1126, [1, 0]);  view_1126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_64: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1128, [1, 0]);  view_1128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_65: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1130, [1, 0]);  view_1130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_13: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_412, mul_150);  mul_150 = None
        sum_dim_int_list_53: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_54: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_412, [0, 1]);  add_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_66: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1132, [1, 0])
        sum_dim_int_list_55: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True);  view_1132 = None
        reshape_default_27: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_27);  sum_dim_int_list_55 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_67: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1135, [1, 0])
        sum_dim_int_list_56: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True);  view_1135 = None
        reshape_default_28: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_28);  sum_dim_int_list_56 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_68: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1152, [1, 0]);  view_1152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_69: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1154, [1, 0]);  view_1154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_70: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1156, [1, 0]);  view_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_14: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_424, mul_140);  mul_140 = None
        sum_dim_int_list_57: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_58: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_424, [0, 1]);  add_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_71: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1158, [1, 0])
        sum_dim_int_list_59: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1158, [0], True);  view_1158 = None
        reshape_default_29: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_29);  sum_dim_int_list_59 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_72: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1161, [1, 0])
        sum_dim_int_list_60: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1161, [0], True);  view_1161 = None
        reshape_default_30: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_30);  sum_dim_int_list_60 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_73: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1178, [1, 0]);  view_1178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_74: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1180, [1, 0]);  view_1180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_75: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1182, [1, 0]);  view_1182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_15: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_436, mul_130);  mul_130 = None
        sum_dim_int_list_61: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_62: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_436, [0, 1]);  add_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_76: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1184, [1, 0])
        sum_dim_int_list_63: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1184, [0], True);  view_1184 = None
        reshape_default_31: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_31);  sum_dim_int_list_63 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_77: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1187, [1, 0])
        sum_dim_int_list_64: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1187, [0], True);  view_1187 = None
        reshape_default_32: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_32);  sum_dim_int_list_64 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_78: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1204, [1, 0]);  view_1204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_79: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1206, [1, 0]);  view_1206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_80: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1208, [1, 0]);  view_1208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_16: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_448, mul_120);  mul_120 = None
        sum_dim_int_list_65: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_66: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_448, [0, 1]);  add_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_81: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1210, [1, 0])
        sum_dim_int_list_67: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
        reshape_default_33: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_33);  sum_dim_int_list_67 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_82: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1213, [1, 0])
        sum_dim_int_list_68: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1213, [0], True);  view_1213 = None
        reshape_default_34: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_34);  sum_dim_int_list_68 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_83: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1230, [1, 0]);  view_1230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_84: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1232, [1, 0]);  view_1232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_85: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1234, [1, 0]);  view_1234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_17: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_460, mul_110);  mul_110 = None
        sum_dim_int_list_69: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_70: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_460, [0, 1]);  add_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_86: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1236, [1, 0])
        sum_dim_int_list_71: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1236, [0], True);  view_1236 = None
        reshape_default_35: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_35);  sum_dim_int_list_71 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_87: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1239, [1, 0])
        sum_dim_int_list_72: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1239, [0], True);  view_1239 = None
        reshape_default_36: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_36);  sum_dim_int_list_72 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_88: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1256, [1, 0]);  view_1256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_89: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1258, [1, 0]);  view_1258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_90: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1260, [1, 0]);  view_1260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_18: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_472, mul_100);  mul_100 = None
        sum_dim_int_list_73: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_74: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_472, [0, 1]);  add_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_91: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1262, [1, 0])
        sum_dim_int_list_75: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1262, [0], True);  view_1262 = None
        reshape_default_37: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_37);  sum_dim_int_list_75 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_92: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1265, [1, 0])
        sum_dim_int_list_76: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1265, [0], True);  view_1265 = None
        reshape_default_38: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_38);  sum_dim_int_list_76 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_93: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1282, [1, 0]);  view_1282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_94: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1284, [1, 0]);  view_1284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_95: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1286, [1, 0]);  view_1286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_19: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_484, mul_90);  mul_90 = None
        sum_dim_int_list_77: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_78: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_484, [0, 1]);  add_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_96: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1288, [1, 0])
        sum_dim_int_list_79: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1288, [0], True);  view_1288 = None
        reshape_default_39: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_39);  sum_dim_int_list_79 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_97: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1291, [1, 0])
        sum_dim_int_list_80: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1291, [0], True);  view_1291 = None
        reshape_default_40: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_40);  sum_dim_int_list_80 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_98: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1308, [1, 0]);  view_1308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_99: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1310, [1, 0]);  view_1310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_100: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1312, [1, 0]);  view_1312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_20: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_496, mul_80);  mul_80 = None
        sum_dim_int_list_81: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_82: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_496, [0, 1]);  add_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_101: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1314, [1, 0])
        sum_dim_int_list_83: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1314, [0], True);  view_1314 = None
        reshape_default_41: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_41);  sum_dim_int_list_83 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_102: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1317, [1, 0])
        sum_dim_int_list_84: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1317, [0], True);  view_1317 = None
        reshape_default_42: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, _shape_param_42);  sum_dim_int_list_84 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_103: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1334, [1, 0]);  view_1334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_104: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1336, [1, 0]);  view_1336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_105: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1338, [1, 0]);  view_1338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_21: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_508, mul_70);  mul_70 = None
        sum_dim_int_list_85: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_86: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_508, [0, 1]);  add_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_106: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1340, [1, 0])
        sum_dim_int_list_87: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1340, [0], True);  view_1340 = None
        reshape_default_43: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_43);  sum_dim_int_list_87 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_107: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1343, [1, 0])
        sum_dim_int_list_88: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1343, [0], True);  view_1343 = None
        reshape_default_44: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_44);  sum_dim_int_list_88 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_108: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1360, [1, 0]);  view_1360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_109: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1362, [1, 0]);  view_1362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_110: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1364, [1, 0]);  view_1364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_22: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_520, mul_60);  mul_60 = None
        sum_dim_int_list_89: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_90: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_520, [0, 1]);  add_520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_111: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1366, [1, 0])
        sum_dim_int_list_91: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1366, [0], True);  view_1366 = None
        reshape_default_45: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_45);  sum_dim_int_list_91 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_112: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1369, [1, 0])
        sum_dim_int_list_92: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1369, [0], True);  view_1369 = None
        reshape_default_46: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_46);  sum_dim_int_list_92 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_113: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1386, [1, 0]);  view_1386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_114: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1388, [1, 0]);  view_1388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_115: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1390, [1, 0]);  view_1390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_23: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_532, mul_50);  mul_50 = None
        sum_dim_int_list_93: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_94: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_532, [0, 1]);  add_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_116: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1392, [1, 0])
        sum_dim_int_list_95: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1392, [0], True);  view_1392 = None
        reshape_default_47: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_47);  sum_dim_int_list_95 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_117: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1395, [1, 0])
        sum_dim_int_list_96: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1395, [0], True);  view_1395 = None
        reshape_default_48: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_48);  sum_dim_int_list_96 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_118: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1412, [1, 0]);  view_1412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_119: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1414, [1, 0]);  view_1414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_120: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1416, [1, 0]);  view_1416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_24: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_544, mul_40);  mul_40 = None
        sum_dim_int_list_97: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_98: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_544, [0, 1]);  add_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_121: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1418, [1, 0])
        sum_dim_int_list_99: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1418, [0], True);  view_1418 = None
        reshape_default_49: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_49);  sum_dim_int_list_99 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_122: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1421, [1, 0])
        sum_dim_int_list_100: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1421, [0], True);  view_1421 = None
        reshape_default_50: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_50);  sum_dim_int_list_100 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_123: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1438, [1, 0]);  view_1438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_124: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1440, [1, 0]);  view_1440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_125: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1442, [1, 0]);  view_1442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_25: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_556, mul_30);  mul_30 = None
        sum_dim_int_list_101: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1]);  mul_tensor_25 = None
        sum_dim_int_list_102: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_556, [0, 1]);  add_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_126: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1444, [1, 0])
        sum_dim_int_list_103: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1444, [0], True);  view_1444 = None
        reshape_default_51: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_51);  sum_dim_int_list_103 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_127: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1447, [1, 0])
        sum_dim_int_list_104: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1447, [0], True);  view_1447 = None
        reshape_default_52: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_52);  sum_dim_int_list_104 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_128: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1464, [1, 0]);  view_1464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_129: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1466, [1, 0]);  view_1466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_130: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1468, [1, 0]);  view_1468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_26: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_568, mul_20);  mul_20 = None
        sum_dim_int_list_105: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        sum_dim_int_list_106: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_568, [0, 1]);  add_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_131: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1470, [1, 0])
        sum_dim_int_list_107: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1470, [0], True);  view_1470 = None
        reshape_default_53: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_53);  sum_dim_int_list_107 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_132: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1473, [1, 0])
        sum_dim_int_list_108: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1473, [0], True);  view_1473 = None
        reshape_default_54: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_54);  sum_dim_int_list_108 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_133: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1490, [1, 0]);  view_1490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_134: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1492, [1, 0]);  view_1492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_135: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1494, [1, 0]);  view_1494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:401 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_tensor_27: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_580, mul_10);  mul_10 = None
        sum_dim_int_list_109: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1]);  mul_tensor_27 = None
        sum_dim_int_list_110: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_580, [0, 1]);  add_580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:377 in forward, code: hidden_states = self.fc_out(hidden_states)
        permute_default_136: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1496, [1, 0])
        sum_dim_int_list_111: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1496, [0], True);  view_1496 = None
        reshape_default_55: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_55);  sum_dim_int_list_111 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:375 in forward, code: hidden_states = self.fc_in(hidden_states)
        permute_default_137: "f32[16384, 128]" = torch.ops.aten.permute.default(view_1499, [1, 0])
        sum_dim_int_list_112: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_1499, [0], True);  view_1499 = None
        reshape_default_56: "f32[16384]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_56);  sum_dim_int_list_112 = _shape_param_56 = None
        reshape_default_57: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_440, _shape_param_57);  mm_440 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:184 in forward, code: value = self.v_proj(hidden_states)
        permute_default_138: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1516, [1, 0]);  view_1516 = None
        reshape_default_58: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_445, _shape_param_58);  mm_445 = _shape_param_58 = None
        add_tensor: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(reshape_default_57, reshape_default_58);  reshape_default_57 = reshape_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:183 in forward, code: key = self.k_proj(hidden_states)
        permute_default_139: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1518, [1, 0]);  view_1518 = None
        reshape_default_59: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_447, _shape_param_59);  mm_447 = _shape_param_59 = None
        add_tensor_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_59);  add_tensor = reshape_default_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:182 in forward, code: query = self.q_proj(hidden_states)
        permute_default_140: "f32[4096, 128]" = torch.ops.aten.permute.default(view_1520, [1, 0]);  view_1520 = None
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
        add_tensor_3: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_581, mul_tensor_33);  add_581 = mul_tensor_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gptj/modeling_gptj.py:494 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_scalar: "b8[1, 128]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 128, 4096]" = torch.ops.aten.where.self(unsqueeze_default, full_default_1, add_tensor_3);  unsqueeze_default = full_default_1 = add_tensor_3 = None
        full_default: "f32[50400, 4096]" = torch.ops.aten.full.default([50400, 4096], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[50400, 4096]" = torch.ops.aten.index_put.default(full_default, [primals_1], where_self, True);  full_default = primals_1 = where_self = None
        return (permute_default, reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default_1, reshape_default_1, permute_default_2, reshape_default_2, permute_default_3, permute_default_4, permute_default_5, sum_dim_int_list_5, sum_dim_int_list_6, permute_default_6, reshape_default_3, permute_default_7, reshape_default_4, permute_default_8, permute_default_9, permute_default_10, sum_dim_int_list_9, sum_dim_int_list_10, permute_default_11, reshape_default_5, permute_default_12, reshape_default_6, permute_default_13, permute_default_14, permute_default_15, sum_dim_int_list_13, sum_dim_int_list_14, permute_default_16, reshape_default_7, permute_default_17, reshape_default_8, permute_default_18, permute_default_19, permute_default_20, sum_dim_int_list_17, sum_dim_int_list_18, permute_default_21, reshape_default_9, permute_default_22, reshape_default_10, permute_default_23, permute_default_24, permute_default_25, sum_dim_int_list_21, sum_dim_int_list_22, permute_default_26, reshape_default_11, permute_default_27, reshape_default_12, permute_default_28, permute_default_29, permute_default_30, sum_dim_int_list_25, sum_dim_int_list_26, permute_default_31, reshape_default_13, permute_default_32, reshape_default_14, permute_default_33, permute_default_34, permute_default_35, sum_dim_int_list_29, sum_dim_int_list_30, permute_default_36, reshape_default_15, permute_default_37, reshape_default_16, permute_default_38, permute_default_39, permute_default_40, sum_dim_int_list_33, sum_dim_int_list_34, permute_default_41, reshape_default_17, permute_default_42, reshape_default_18, permute_default_43, permute_default_44, permute_default_45, sum_dim_int_list_37, sum_dim_int_list_38, permute_default_46, reshape_default_19, permute_default_47, reshape_default_20, permute_default_48, permute_default_49, permute_default_50, sum_dim_int_list_41, sum_dim_int_list_42, permute_default_51, reshape_default_21, permute_default_52, reshape_default_22, permute_default_53, permute_default_54, permute_default_55, sum_dim_int_list_45, sum_dim_int_list_46, permute_default_56, reshape_default_23, permute_default_57, reshape_default_24, permute_default_58, permute_default_59, permute_default_60, sum_dim_int_list_49, sum_dim_int_list_50, permute_default_61, reshape_default_25, permute_default_62, reshape_default_26, permute_default_63, permute_default_64, permute_default_65, sum_dim_int_list_53, sum_dim_int_list_54, permute_default_66, reshape_default_27, permute_default_67, reshape_default_28, permute_default_68, permute_default_69, permute_default_70, sum_dim_int_list_57, sum_dim_int_list_58, permute_default_71, reshape_default_29, permute_default_72, reshape_default_30, permute_default_73, permute_default_74, permute_default_75, sum_dim_int_list_61, sum_dim_int_list_62, permute_default_76, reshape_default_31, permute_default_77, reshape_default_32, permute_default_78, permute_default_79, permute_default_80, sum_dim_int_list_65, sum_dim_int_list_66, permute_default_81, reshape_default_33, permute_default_82, reshape_default_34, permute_default_83, permute_default_84, permute_default_85, sum_dim_int_list_69, sum_dim_int_list_70, permute_default_86, reshape_default_35, permute_default_87, reshape_default_36, permute_default_88, permute_default_89, permute_default_90, sum_dim_int_list_73, sum_dim_int_list_74, permute_default_91, reshape_default_37, permute_default_92, reshape_default_38, permute_default_93, permute_default_94, permute_default_95, sum_dim_int_list_77, sum_dim_int_list_78, permute_default_96, reshape_default_39, permute_default_97, reshape_default_40, permute_default_98, permute_default_99, permute_default_100, sum_dim_int_list_81, sum_dim_int_list_82, permute_default_101, reshape_default_41, permute_default_102, reshape_default_42, permute_default_103, permute_default_104, permute_default_105, sum_dim_int_list_85, sum_dim_int_list_86, permute_default_106, reshape_default_43, permute_default_107, reshape_default_44, permute_default_108, permute_default_109, permute_default_110, sum_dim_int_list_89, sum_dim_int_list_90, permute_default_111, reshape_default_45, permute_default_112, reshape_default_46, permute_default_113, permute_default_114, permute_default_115, sum_dim_int_list_93, sum_dim_int_list_94, permute_default_116, reshape_default_47, permute_default_117, reshape_default_48, permute_default_118, permute_default_119, permute_default_120, sum_dim_int_list_97, sum_dim_int_list_98, permute_default_121, reshape_default_49, permute_default_122, reshape_default_50, permute_default_123, permute_default_124, permute_default_125, sum_dim_int_list_101, sum_dim_int_list_102, permute_default_126, reshape_default_51, permute_default_127, reshape_default_52, permute_default_128, permute_default_129, permute_default_130, sum_dim_int_list_105, sum_dim_int_list_106, permute_default_131, reshape_default_53, permute_default_132, reshape_default_54, permute_default_133, permute_default_134, permute_default_135, sum_dim_int_list_109, sum_dim_int_list_110, permute_default_136, reshape_default_55, permute_default_137, reshape_default_56, permute_default_138, permute_default_139, permute_default_140, sum_dim_int_list_115, sum_dim_int_list_116, index_put_default)



def make_inputs():
    return [
    torch.randn([128, 50400], dtype=torch.float32, device='cuda'),
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
    [50400],  # _shape_param_0
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
