"""
Standalone repro captured via capture_hook.
Label: torchbench_BERT_pytorch_train
Pattern hash: 4096af0624c9
Shape hash: ef1bf327
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 2
_shapes_config = "(T([16384, 20005], f32), T([128, 2], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 3072], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([128, 128, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([16384, 768], f32), T([768], f32), T([128, 128, 768], f32), T([128, 128, 1], f32), T([128, 128, 768], f32), T([], f32), T([128, 128, 768], b8), T([128, 128], i64, gen=Index(3)), T([128, 128], i64, gen=Index(20005)), S([20005]), S([2]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([768]), S([3072]), S([768]), S([768]), S([768]), S([768]), S([128, 128, 768]), S([768]), S([128, 128, 768]), S([768]), S([128, 128, 768]), S([768]), S([768]), S([128, 128, 768]))"

class Repro(torch.nn.Module):
    def forward(self, view_266: "f32[16384, 20005]", sub_41: "f32[128, 2]", view_270: "f32[16384, 768]", view_273: "f32[16384, 3072]", view_275: "f32[128, 128, 768]", div_50: "f32[128, 128, 768]", sub_35: "f32[128, 128, 768]", view_278: "f32[16384, 768]", view_289: "f32[16384, 768]", view_293: "f32[16384, 768]", view_297: "f32[16384, 768]", add_93: "f32[128, 128, 768]", div_56: "f32[128, 128, 768]", sub_33: "f32[128, 128, 768]", view_302: "f32[16384, 768]", view_305: "f32[16384, 3072]", view_307: "f32[128, 128, 768]", div_61: "f32[128, 128, 768]", sub_32: "f32[128, 128, 768]", view_310: "f32[16384, 768]", view_321: "f32[16384, 768]", view_325: "f32[16384, 768]", view_329: "f32[16384, 768]", add_103: "f32[128, 128, 768]", div_67: "f32[128, 128, 768]", sub_30: "f32[128, 128, 768]", view_334: "f32[16384, 768]", view_337: "f32[16384, 3072]", view_339: "f32[128, 128, 768]", div_72: "f32[128, 128, 768]", sub_29: "f32[128, 128, 768]", view_342: "f32[16384, 768]", view_353: "f32[16384, 768]", view_357: "f32[16384, 768]", view_361: "f32[16384, 768]", add_113: "f32[128, 128, 768]", div_78: "f32[128, 128, 768]", sub_27: "f32[128, 128, 768]", view_366: "f32[16384, 768]", view_369: "f32[16384, 3072]", view_371: "f32[128, 128, 768]", div_83: "f32[128, 128, 768]", sub_26: "f32[128, 128, 768]", view_374: "f32[16384, 768]", view_385: "f32[16384, 768]", view_389: "f32[16384, 768]", view_393: "f32[16384, 768]", add_123: "f32[128, 128, 768]", div_89: "f32[128, 128, 768]", sub_24: "f32[128, 128, 768]", view_398: "f32[16384, 768]", view_401: "f32[16384, 3072]", view_403: "f32[128, 128, 768]", div_94: "f32[128, 128, 768]", sub_23: "f32[128, 128, 768]", view_406: "f32[16384, 768]", view_417: "f32[16384, 768]", view_421: "f32[16384, 768]", view_425: "f32[16384, 768]", add_133: "f32[128, 128, 768]", div_100: "f32[128, 128, 768]", sub_21: "f32[128, 128, 768]", view_430: "f32[16384, 768]", view_433: "f32[16384, 3072]", view_435: "f32[128, 128, 768]", div_105: "f32[128, 128, 768]", sub_20: "f32[128, 128, 768]", view_438: "f32[16384, 768]", view_449: "f32[16384, 768]", view_453: "f32[16384, 768]", view_457: "f32[16384, 768]", add_143: "f32[128, 128, 768]", div_111: "f32[128, 128, 768]", sub_18: "f32[128, 128, 768]", view_462: "f32[16384, 768]", view_465: "f32[16384, 3072]", view_467: "f32[128, 128, 768]", div_116: "f32[128, 128, 768]", sub_17: "f32[128, 128, 768]", view_470: "f32[16384, 768]", view_481: "f32[16384, 768]", view_485: "f32[16384, 768]", view_489: "f32[16384, 768]", add_153: "f32[128, 128, 768]", div_122: "f32[128, 128, 768]", sub_15: "f32[128, 128, 768]", view_494: "f32[16384, 768]", view_497: "f32[16384, 3072]", view_499: "f32[128, 128, 768]", div_127: "f32[128, 128, 768]", sub_14: "f32[128, 128, 768]", view_502: "f32[16384, 768]", view_513: "f32[16384, 768]", view_517: "f32[16384, 768]", view_521: "f32[16384, 768]", add_163: "f32[128, 128, 768]", div_133: "f32[128, 128, 768]", sub_12: "f32[128, 128, 768]", view_526: "f32[16384, 768]", view_529: "f32[16384, 3072]", view_531: "f32[128, 128, 768]", div_138: "f32[128, 128, 768]", sub_11: "f32[128, 128, 768]", view_534: "f32[16384, 768]", view_545: "f32[16384, 768]", view_549: "f32[16384, 768]", view_553: "f32[16384, 768]", add_173: "f32[128, 128, 768]", div_144: "f32[128, 128, 768]", sub_9: "f32[128, 128, 768]", view_558: "f32[16384, 768]", view_561: "f32[16384, 3072]", view_563: "f32[128, 128, 768]", div_149: "f32[128, 128, 768]", sub_8: "f32[128, 128, 768]", view_566: "f32[16384, 768]", view_577: "f32[16384, 768]", view_581: "f32[16384, 768]", view_585: "f32[16384, 768]", add_183: "f32[128, 128, 768]", div_155: "f32[128, 128, 768]", sub_6: "f32[128, 128, 768]", view_590: "f32[16384, 768]", view_593: "f32[16384, 3072]", view_595: "f32[128, 128, 768]", div_160: "f32[128, 128, 768]", sub_5: "f32[128, 128, 768]", view_598: "f32[16384, 768]", view_609: "f32[16384, 768]", view_613: "f32[16384, 768]", view_617: "f32[16384, 768]", add_193: "f32[128, 128, 768]", div_166: "f32[128, 128, 768]", sub_3: "f32[128, 128, 768]", view_622: "f32[16384, 768]", view_625: "f32[16384, 3072]", view_627: "f32[128, 128, 768]", div_171: "f32[128, 128, 768]", sub_2: "f32[128, 128, 768]", view_630: "f32[16384, 768]", view_641: "f32[16384, 768]", mm_142: "f32[16384, 768]", view_645: "f32[16384, 768]", mm_144: "f32[16384, 768]", view_649: "f32[16384, 768]", mm_146: "f32[16384, 768]", primals_6: "f32[768]", sub: "f32[128, 128, 768]", sqrt: "f32[128, 128, 1]", add_201: "f32[128, 128, 768]", full_default_13: "f32[]", gt_1: "b8[128, 128, 768]", primals_5: "i64[128, 128]", primals_1: "i64[128, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125):
        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:61 in forward, code: return self.softmax(self.linear(x))
        permute_default: "f32[20005, 16384]" = torch.ops.aten.permute.default(view_266, [1, 0])
        sum_dim_int_list: "f32[1, 20005]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
        reshape_default: "f32[20005]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/language_model.py:42 in forward, code: return self.softmax(self.linear(x[:, 0]))
        permute_default_1: "f32[2, 128]" = torch.ops.aten.permute.default(sub_41, [1, 0])
        sum_dim_int_list_1: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(sub_41, [0], True);  sub_41 = None
        reshape_default_1: "f32[2]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_2: "f32[768, 16384]" = torch.ops.aten.permute.default(view_270, [1, 0])
        sum_dim_int_list_2: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
        reshape_default_2: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_2);  sum_dim_int_list_2 = _shape_param_2 = None
        permute_default_3: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_273, [1, 0])
        sum_dim_int_list_3: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
        reshape_default_3: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_3);  sum_dim_int_list_3 = _shape_param_3 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_4: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_275, [0, 1], True);  view_275 = None
        reshape_default_4: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_4);  sum_dim_int_list_4 = _shape_param_4 = None
        mul_tensor: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_50, sub_35);  div_50 = sub_35 = None
        sum_dim_int_list_5: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True);  mul_tensor = None
        reshape_default_5: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_5);  sum_dim_int_list_5 = _shape_param_5 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_4: "f32[768, 16384]" = torch.ops.aten.permute.default(view_278, [1, 0])
        sum_dim_int_list_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
        reshape_default_6: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_6);  sum_dim_int_list_6 = _shape_param_6 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_5: "f32[768, 16384]" = torch.ops.aten.permute.default(view_289, [1, 0])
        sum_dim_int_list_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
        reshape_default_7: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_7);  sum_dim_int_list_7 = _shape_param_7 = None
        permute_default_6: "f32[768, 16384]" = torch.ops.aten.permute.default(view_293, [1, 0])
        sum_dim_int_list_8: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
        reshape_default_8: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_8);  sum_dim_int_list_8 = _shape_param_8 = None
        permute_default_7: "f32[768, 16384]" = torch.ops.aten.permute.default(view_297, [1, 0])
        sum_dim_int_list_9: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_297, [0], True);  view_297 = None
        reshape_default_9: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_9);  sum_dim_int_list_9 = _shape_param_9 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_10: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_93, [0, 1], True);  add_93 = None
        reshape_default_10: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_10);  sum_dim_int_list_10 = _shape_param_10 = None
        mul_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_33);  div_56 = sub_33 = None
        sum_dim_int_list_11: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default_11: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_11);  sum_dim_int_list_11 = _shape_param_11 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_8: "f32[768, 16384]" = torch.ops.aten.permute.default(view_302, [1, 0])
        sum_dim_int_list_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_302, [0], True);  view_302 = None
        reshape_default_12: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_12);  sum_dim_int_list_12 = _shape_param_12 = None
        permute_default_9: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_305, [1, 0])
        sum_dim_int_list_13: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_305, [0], True);  view_305 = None
        reshape_default_13: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_13);  sum_dim_int_list_13 = _shape_param_13 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_14: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1], True);  view_307 = None
        reshape_default_14: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_14);  sum_dim_int_list_14 = _shape_param_14 = None
        mul_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_32);  div_61 = sub_32 = None
        sum_dim_int_list_15: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True);  mul_tensor_2 = None
        reshape_default_15: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_15);  sum_dim_int_list_15 = _shape_param_15 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_10: "f32[768, 16384]" = torch.ops.aten.permute.default(view_310, [1, 0])
        sum_dim_int_list_16: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_310, [0], True);  view_310 = None
        reshape_default_16: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_16);  sum_dim_int_list_16 = _shape_param_16 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_11: "f32[768, 16384]" = torch.ops.aten.permute.default(view_321, [1, 0])
        sum_dim_int_list_17: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True);  view_321 = None
        reshape_default_17: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_17);  sum_dim_int_list_17 = _shape_param_17 = None
        permute_default_12: "f32[768, 16384]" = torch.ops.aten.permute.default(view_325, [1, 0])
        sum_dim_int_list_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True);  view_325 = None
        reshape_default_18: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_18);  sum_dim_int_list_18 = _shape_param_18 = None
        permute_default_13: "f32[768, 16384]" = torch.ops.aten.permute.default(view_329, [1, 0])
        sum_dim_int_list_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_329, [0], True);  view_329 = None
        reshape_default_19: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_19);  sum_dim_int_list_19 = _shape_param_19 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_20: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_103, [0, 1], True);  add_103 = None
        reshape_default_20: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_20);  sum_dim_int_list_20 = _shape_param_20 = None
        mul_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_67, sub_30);  div_67 = sub_30 = None
        sum_dim_int_list_21: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_21: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_21);  sum_dim_int_list_21 = _shape_param_21 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_14: "f32[768, 16384]" = torch.ops.aten.permute.default(view_334, [1, 0])
        sum_dim_int_list_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_334, [0], True);  view_334 = None
        reshape_default_22: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_22);  sum_dim_int_list_22 = _shape_param_22 = None
        permute_default_15: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_337, [1, 0])
        sum_dim_int_list_23: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_337, [0], True);  view_337 = None
        reshape_default_23: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_23);  sum_dim_int_list_23 = _shape_param_23 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_24: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0, 1], True);  view_339 = None
        reshape_default_24: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_24);  sum_dim_int_list_24 = _shape_param_24 = None
        mul_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_72, sub_29);  div_72 = sub_29 = None
        sum_dim_int_list_25: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True);  mul_tensor_4 = None
        reshape_default_25: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, _shape_param_25);  sum_dim_int_list_25 = _shape_param_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_16: "f32[768, 16384]" = torch.ops.aten.permute.default(view_342, [1, 0])
        sum_dim_int_list_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_342, [0], True);  view_342 = None
        reshape_default_26: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_26);  sum_dim_int_list_26 = _shape_param_26 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_17: "f32[768, 16384]" = torch.ops.aten.permute.default(view_353, [1, 0])
        sum_dim_int_list_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
        reshape_default_27: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_27);  sum_dim_int_list_27 = _shape_param_27 = None
        permute_default_18: "f32[768, 16384]" = torch.ops.aten.permute.default(view_357, [1, 0])
        sum_dim_int_list_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_357, [0], True);  view_357 = None
        reshape_default_28: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_28);  sum_dim_int_list_28 = _shape_param_28 = None
        permute_default_19: "f32[768, 16384]" = torch.ops.aten.permute.default(view_361, [1, 0])
        sum_dim_int_list_29: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
        reshape_default_29: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_29);  sum_dim_int_list_29 = _shape_param_29 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_30: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_113, [0, 1], True);  add_113 = None
        reshape_default_30: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_30);  sum_dim_int_list_30 = _shape_param_30 = None
        mul_tensor_5: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_78, sub_27);  div_78 = sub_27 = None
        sum_dim_int_list_31: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_31: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_31);  sum_dim_int_list_31 = _shape_param_31 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_20: "f32[768, 16384]" = torch.ops.aten.permute.default(view_366, [1, 0])
        sum_dim_int_list_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
        reshape_default_32: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_32);  sum_dim_int_list_32 = _shape_param_32 = None
        permute_default_21: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_369, [1, 0])
        sum_dim_int_list_33: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_369, [0], True);  view_369 = None
        reshape_default_33: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_33);  sum_dim_int_list_33 = _shape_param_33 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_34: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_371, [0, 1], True);  view_371 = None
        reshape_default_34: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_34);  sum_dim_int_list_34 = _shape_param_34 = None
        mul_tensor_6: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_83, sub_26);  div_83 = sub_26 = None
        sum_dim_int_list_35: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], True);  mul_tensor_6 = None
        reshape_default_35: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_35);  sum_dim_int_list_35 = _shape_param_35 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_22: "f32[768, 16384]" = torch.ops.aten.permute.default(view_374, [1, 0])
        sum_dim_int_list_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_374, [0], True);  view_374 = None
        reshape_default_36: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_36);  sum_dim_int_list_36 = _shape_param_36 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_23: "f32[768, 16384]" = torch.ops.aten.permute.default(view_385, [1, 0])
        sum_dim_int_list_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_385, [0], True);  view_385 = None
        reshape_default_37: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_37);  sum_dim_int_list_37 = _shape_param_37 = None
        permute_default_24: "f32[768, 16384]" = torch.ops.aten.permute.default(view_389, [1, 0])
        sum_dim_int_list_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_389, [0], True);  view_389 = None
        reshape_default_38: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_38);  sum_dim_int_list_38 = _shape_param_38 = None
        permute_default_25: "f32[768, 16384]" = torch.ops.aten.permute.default(view_393, [1, 0])
        sum_dim_int_list_39: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_393, [0], True);  view_393 = None
        reshape_default_39: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_39);  sum_dim_int_list_39 = _shape_param_39 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_40: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_123, [0, 1], True);  add_123 = None
        reshape_default_40: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_40);  sum_dim_int_list_40 = _shape_param_40 = None
        mul_tensor_7: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_89, sub_24);  div_89 = sub_24 = None
        sum_dim_int_list_41: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_41: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_41);  sum_dim_int_list_41 = _shape_param_41 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_26: "f32[768, 16384]" = torch.ops.aten.permute.default(view_398, [1, 0])
        sum_dim_int_list_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_398, [0], True);  view_398 = None
        reshape_default_42: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_42);  sum_dim_int_list_42 = _shape_param_42 = None
        permute_default_27: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_401, [1, 0])
        sum_dim_int_list_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_401, [0], True);  view_401 = None
        reshape_default_43: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_43);  sum_dim_int_list_43 = _shape_param_43 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_44: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_403, [0, 1], True);  view_403 = None
        reshape_default_44: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_44);  sum_dim_int_list_44 = _shape_param_44 = None
        mul_tensor_8: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_94, sub_23);  div_94 = sub_23 = None
        sum_dim_int_list_45: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1], True);  mul_tensor_8 = None
        reshape_default_45: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_45, _shape_param_45);  sum_dim_int_list_45 = _shape_param_45 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_28: "f32[768, 16384]" = torch.ops.aten.permute.default(view_406, [1, 0])
        sum_dim_int_list_46: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_406, [0], True);  view_406 = None
        reshape_default_46: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_46);  sum_dim_int_list_46 = _shape_param_46 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_29: "f32[768, 16384]" = torch.ops.aten.permute.default(view_417, [1, 0])
        sum_dim_int_list_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_417, [0], True);  view_417 = None
        reshape_default_47: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_47);  sum_dim_int_list_47 = _shape_param_47 = None
        permute_default_30: "f32[768, 16384]" = torch.ops.aten.permute.default(view_421, [1, 0])
        sum_dim_int_list_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_421, [0], True);  view_421 = None
        reshape_default_48: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_48);  sum_dim_int_list_48 = _shape_param_48 = None
        permute_default_31: "f32[768, 16384]" = torch.ops.aten.permute.default(view_425, [1, 0])
        sum_dim_int_list_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True);  view_425 = None
        reshape_default_49: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, _shape_param_49);  sum_dim_int_list_49 = _shape_param_49 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_50: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_133, [0, 1], True);  add_133 = None
        reshape_default_50: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_50);  sum_dim_int_list_50 = _shape_param_50 = None
        mul_tensor_9: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_100, sub_21);  div_100 = sub_21 = None
        sum_dim_int_list_51: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_51: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_51);  sum_dim_int_list_51 = _shape_param_51 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_32: "f32[768, 16384]" = torch.ops.aten.permute.default(view_430, [1, 0])
        sum_dim_int_list_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
        reshape_default_52: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_52);  sum_dim_int_list_52 = _shape_param_52 = None
        permute_default_33: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_433, [1, 0])
        sum_dim_int_list_53: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
        reshape_default_53: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_53);  sum_dim_int_list_53 = _shape_param_53 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_54: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_435, [0, 1], True);  view_435 = None
        reshape_default_54: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_54);  sum_dim_int_list_54 = _shape_param_54 = None
        mul_tensor_10: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_105, sub_20);  div_105 = sub_20 = None
        sum_dim_int_list_55: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1], True);  mul_tensor_10 = None
        reshape_default_55: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_55);  sum_dim_int_list_55 = _shape_param_55 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_34: "f32[768, 16384]" = torch.ops.aten.permute.default(view_438, [1, 0])
        sum_dim_int_list_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
        reshape_default_56: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_56);  sum_dim_int_list_56 = _shape_param_56 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_35: "f32[768, 16384]" = torch.ops.aten.permute.default(view_449, [1, 0])
        sum_dim_int_list_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_449, [0], True);  view_449 = None
        reshape_default_57: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_57);  sum_dim_int_list_57 = _shape_param_57 = None
        permute_default_36: "f32[768, 16384]" = torch.ops.aten.permute.default(view_453, [1, 0])
        sum_dim_int_list_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
        reshape_default_58: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_58);  sum_dim_int_list_58 = _shape_param_58 = None
        permute_default_37: "f32[768, 16384]" = torch.ops.aten.permute.default(view_457, [1, 0])
        sum_dim_int_list_59: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_457, [0], True);  view_457 = None
        reshape_default_59: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_59);  sum_dim_int_list_59 = _shape_param_59 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_60: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1], True);  add_143 = None
        reshape_default_60: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_60);  sum_dim_int_list_60 = _shape_param_60 = None
        mul_tensor_11: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_111, sub_18);  div_111 = sub_18 = None
        sum_dim_int_list_61: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_61: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, _shape_param_61);  sum_dim_int_list_61 = _shape_param_61 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_38: "f32[768, 16384]" = torch.ops.aten.permute.default(view_462, [1, 0])
        sum_dim_int_list_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
        reshape_default_62: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_62);  sum_dim_int_list_62 = _shape_param_62 = None
        permute_default_39: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_465, [1, 0])
        sum_dim_int_list_63: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_465, [0], True);  view_465 = None
        reshape_default_63: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_63);  sum_dim_int_list_63 = _shape_param_63 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_64: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_467, [0, 1], True);  view_467 = None
        reshape_default_64: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_64);  sum_dim_int_list_64 = _shape_param_64 = None
        mul_tensor_12: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_116, sub_17);  div_116 = sub_17 = None
        sum_dim_int_list_65: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1], True);  mul_tensor_12 = None
        reshape_default_65: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_65, _shape_param_65);  sum_dim_int_list_65 = _shape_param_65 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_40: "f32[768, 16384]" = torch.ops.aten.permute.default(view_470, [1, 0])
        sum_dim_int_list_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True);  view_470 = None
        reshape_default_66: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_66);  sum_dim_int_list_66 = _shape_param_66 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_41: "f32[768, 16384]" = torch.ops.aten.permute.default(view_481, [1, 0])
        sum_dim_int_list_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
        reshape_default_67: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_67);  sum_dim_int_list_67 = _shape_param_67 = None
        permute_default_42: "f32[768, 16384]" = torch.ops.aten.permute.default(view_485, [1, 0])
        sum_dim_int_list_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_485, [0], True);  view_485 = None
        reshape_default_68: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_68);  sum_dim_int_list_68 = _shape_param_68 = None
        permute_default_43: "f32[768, 16384]" = torch.ops.aten.permute.default(view_489, [1, 0])
        sum_dim_int_list_69: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_489, [0], True);  view_489 = None
        reshape_default_69: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, _shape_param_69);  sum_dim_int_list_69 = _shape_param_69 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_70: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1], True);  add_153 = None
        reshape_default_70: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_70);  sum_dim_int_list_70 = _shape_param_70 = None
        mul_tensor_13: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_122, sub_15);  div_122 = sub_15 = None
        sum_dim_int_list_71: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_71: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_71);  sum_dim_int_list_71 = _shape_param_71 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_44: "f32[768, 16384]" = torch.ops.aten.permute.default(view_494, [1, 0])
        sum_dim_int_list_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_494, [0], True);  view_494 = None
        reshape_default_72: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_72);  sum_dim_int_list_72 = _shape_param_72 = None
        permute_default_45: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_497, [1, 0])
        sum_dim_int_list_73: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_497, [0], True);  view_497 = None
        reshape_default_73: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_73);  sum_dim_int_list_73 = _shape_param_73 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_74: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_499, [0, 1], True);  view_499 = None
        reshape_default_74: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, _shape_param_74);  sum_dim_int_list_74 = _shape_param_74 = None
        mul_tensor_14: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_127, sub_14);  div_127 = sub_14 = None
        sum_dim_int_list_75: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1], True);  mul_tensor_14 = None
        reshape_default_75: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_75);  sum_dim_int_list_75 = _shape_param_75 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_46: "f32[768, 16384]" = torch.ops.aten.permute.default(view_502, [1, 0])
        sum_dim_int_list_76: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_502, [0], True);  view_502 = None
        reshape_default_76: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_76);  sum_dim_int_list_76 = _shape_param_76 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_47: "f32[768, 16384]" = torch.ops.aten.permute.default(view_513, [1, 0])
        sum_dim_int_list_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_513, [0], True);  view_513 = None
        reshape_default_77: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_77);  sum_dim_int_list_77 = _shape_param_77 = None
        permute_default_48: "f32[768, 16384]" = torch.ops.aten.permute.default(view_517, [1, 0])
        sum_dim_int_list_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_517, [0], True);  view_517 = None
        reshape_default_78: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, _shape_param_78);  sum_dim_int_list_78 = _shape_param_78 = None
        permute_default_49: "f32[768, 16384]" = torch.ops.aten.permute.default(view_521, [1, 0])
        sum_dim_int_list_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_521, [0], True);  view_521 = None
        reshape_default_79: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_79);  sum_dim_int_list_79 = _shape_param_79 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_80: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_163, [0, 1], True);  add_163 = None
        reshape_default_80: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_80);  sum_dim_int_list_80 = _shape_param_80 = None
        mul_tensor_15: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_133, sub_12);  div_133 = sub_12 = None
        sum_dim_int_list_81: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_81: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_81);  sum_dim_int_list_81 = _shape_param_81 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_50: "f32[768, 16384]" = torch.ops.aten.permute.default(view_526, [1, 0])
        sum_dim_int_list_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True);  view_526 = None
        reshape_default_82: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_82);  sum_dim_int_list_82 = _shape_param_82 = None
        permute_default_51: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_529, [1, 0])
        sum_dim_int_list_83: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
        reshape_default_83: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_83);  sum_dim_int_list_83 = _shape_param_83 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_84: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_531, [0, 1], True);  view_531 = None
        reshape_default_84: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, _shape_param_84);  sum_dim_int_list_84 = _shape_param_84 = None
        mul_tensor_16: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_138, sub_11);  div_138 = sub_11 = None
        sum_dim_int_list_85: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1], True);  mul_tensor_16 = None
        reshape_default_85: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_85, _shape_param_85);  sum_dim_int_list_85 = _shape_param_85 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_52: "f32[768, 16384]" = torch.ops.aten.permute.default(view_534, [1, 0])
        sum_dim_int_list_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
        reshape_default_86: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_86);  sum_dim_int_list_86 = _shape_param_86 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_53: "f32[768, 16384]" = torch.ops.aten.permute.default(view_545, [1, 0])
        sum_dim_int_list_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_545, [0], True);  view_545 = None
        reshape_default_87: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_87);  sum_dim_int_list_87 = _shape_param_87 = None
        permute_default_54: "f32[768, 16384]" = torch.ops.aten.permute.default(view_549, [1, 0])
        sum_dim_int_list_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
        reshape_default_88: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_88);  sum_dim_int_list_88 = _shape_param_88 = None
        permute_default_55: "f32[768, 16384]" = torch.ops.aten.permute.default(view_553, [1, 0])
        sum_dim_int_list_89: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_553, [0], True);  view_553 = None
        reshape_default_89: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_89, _shape_param_89);  sum_dim_int_list_89 = _shape_param_89 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_90: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_173, [0, 1], True);  add_173 = None
        reshape_default_90: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_90);  sum_dim_int_list_90 = _shape_param_90 = None
        mul_tensor_17: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_144, sub_9);  div_144 = sub_9 = None
        sum_dim_int_list_91: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_91: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_91);  sum_dim_int_list_91 = _shape_param_91 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_56: "f32[768, 16384]" = torch.ops.aten.permute.default(view_558, [1, 0])
        sum_dim_int_list_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
        reshape_default_92: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_92);  sum_dim_int_list_92 = _shape_param_92 = None
        permute_default_57: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_561, [1, 0])
        sum_dim_int_list_93: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
        reshape_default_93: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_93);  sum_dim_int_list_93 = _shape_param_93 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_94: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1], True);  view_563 = None
        reshape_default_94: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, _shape_param_94);  sum_dim_int_list_94 = _shape_param_94 = None
        mul_tensor_18: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_149, sub_8);  div_149 = sub_8 = None
        sum_dim_int_list_95: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1], True);  mul_tensor_18 = None
        reshape_default_95: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_95);  sum_dim_int_list_95 = _shape_param_95 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_58: "f32[768, 16384]" = torch.ops.aten.permute.default(view_566, [1, 0])
        sum_dim_int_list_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
        reshape_default_96: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_96);  sum_dim_int_list_96 = _shape_param_96 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_59: "f32[768, 16384]" = torch.ops.aten.permute.default(view_577, [1, 0])
        sum_dim_int_list_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_577, [0], True);  view_577 = None
        reshape_default_97: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_97);  sum_dim_int_list_97 = _shape_param_97 = None
        permute_default_60: "f32[768, 16384]" = torch.ops.aten.permute.default(view_581, [1, 0])
        sum_dim_int_list_98: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_581, [0], True);  view_581 = None
        reshape_default_98: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_98, _shape_param_98);  sum_dim_int_list_98 = _shape_param_98 = None
        permute_default_61: "f32[768, 16384]" = torch.ops.aten.permute.default(view_585, [1, 0])
        sum_dim_int_list_99: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
        reshape_default_99: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_99);  sum_dim_int_list_99 = _shape_param_99 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_100: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_183, [0, 1], True);  add_183 = None
        reshape_default_100: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_100);  sum_dim_int_list_100 = _shape_param_100 = None
        mul_tensor_19: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_155, sub_6);  div_155 = sub_6 = None
        sum_dim_int_list_101: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_101: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, _shape_param_101);  sum_dim_int_list_101 = _shape_param_101 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_62: "f32[768, 16384]" = torch.ops.aten.permute.default(view_590, [1, 0])
        sum_dim_int_list_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        reshape_default_102: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_102);  sum_dim_int_list_102 = _shape_param_102 = None
        permute_default_63: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_593, [1, 0])
        sum_dim_int_list_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
        reshape_default_103: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_103);  sum_dim_int_list_103 = _shape_param_103 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_104: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_595, [0, 1], True);  view_595 = None
        reshape_default_104: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_104);  sum_dim_int_list_104 = _shape_param_104 = None
        mul_tensor_20: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_160, sub_5);  div_160 = sub_5 = None
        sum_dim_int_list_105: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1], True);  mul_tensor_20 = None
        reshape_default_105: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_105, _shape_param_105);  sum_dim_int_list_105 = _shape_param_105 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_64: "f32[768, 16384]" = torch.ops.aten.permute.default(view_598, [1, 0])
        sum_dim_int_list_106: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_598, [0], True);  view_598 = None
        reshape_default_106: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, _shape_param_106);  sum_dim_int_list_106 = _shape_param_106 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_65: "f32[768, 16384]" = torch.ops.aten.permute.default(view_609, [1, 0])
        sum_dim_int_list_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_609, [0], True);  view_609 = None
        reshape_default_107: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_107);  sum_dim_int_list_107 = _shape_param_107 = None
        permute_default_66: "f32[768, 16384]" = torch.ops.aten.permute.default(view_613, [1, 0])
        sum_dim_int_list_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_613, [0], True);  view_613 = None
        reshape_default_108: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_108);  sum_dim_int_list_108 = _shape_param_108 = None
        permute_default_67: "f32[768, 16384]" = torch.ops.aten.permute.default(view_617, [1, 0])
        sum_dim_int_list_109: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True);  view_617 = None
        reshape_default_109: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_109, _shape_param_109);  sum_dim_int_list_109 = _shape_param_109 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_110: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_193, [0, 1], True);  add_193 = None
        reshape_default_110: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_110);  sum_dim_int_list_110 = _shape_param_110 = None
        mul_tensor_21: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_166, sub_3);  div_166 = sub_3 = None
        sum_dim_int_list_111: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_111: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_111);  sum_dim_int_list_111 = _shape_param_111 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:15 in forward, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
        permute_default_68: "f32[768, 16384]" = torch.ops.aten.permute.default(view_622, [1, 0])
        sum_dim_int_list_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
        reshape_default_112: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_112);  sum_dim_int_list_112 = _shape_param_112 = None
        permute_default_69: "f32[3072, 16384]" = torch.ops.aten.permute.default(view_625, [1, 0])
        sum_dim_int_list_113: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
        reshape_default_113: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_113);  sum_dim_int_list_113 = _shape_param_113 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_114: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_627, [0, 1], True);  view_627 = None
        reshape_default_114: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_114, _shape_param_114);  sum_dim_int_list_114 = _shape_param_114 = None
        mul_tensor_22: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_171, sub_2);  div_171 = sub_2 = None
        sum_dim_int_list_115: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1], True);  mul_tensor_22 = None
        reshape_default_115: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_115, _shape_param_115);  sum_dim_int_list_115 = _shape_param_115 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:53 in forward, code: return self.output_linear(x)
        permute_default_70: "f32[768, 16384]" = torch.ops.aten.permute.default(view_630, [1, 0])
        sum_dim_int_list_116: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_630, [0], True);  view_630 = None
        reshape_default_116: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_116);  sum_dim_int_list_116 = _shape_param_116 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:43 in forward, code: l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        permute_default_71: "f32[768, 16384]" = torch.ops.aten.permute.default(view_641, [1, 0])
        sum_dim_int_list_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_641, [0], True);  view_641 = None
        reshape_default_117: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_117);  sum_dim_int_list_117 = _shape_param_117 = None
        reshape_default_118: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_142, _shape_param_118);  mm_142 = _shape_param_118 = None
        permute_default_72: "f32[768, 16384]" = torch.ops.aten.permute.default(view_645, [1, 0])
        sum_dim_int_list_118: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
        reshape_default_119: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_118, _shape_param_119);  sum_dim_int_list_118 = _shape_param_119 = None
        reshape_default_120: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_144, _shape_param_120);  mm_144 = _shape_param_120 = None
        add_tensor: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(reshape_default_118, reshape_default_120);  reshape_default_118 = reshape_default_120 = None
        permute_default_73: "f32[768, 16384]" = torch.ops.aten.permute.default(view_649, [1, 0])
        sum_dim_int_list_119: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_649, [0], True);  view_649 = None
        reshape_default_121: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, _shape_param_121);  sum_dim_int_list_119 = _shape_param_121 = None
        reshape_default_122: "f32[128, 128, 768]" = torch.ops.aten.reshape.default(mm_146, _shape_param_122);  mm_146 = _shape_param_122 = None
        add_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_122);  add_tensor = reshape_default_122 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17 in forward, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        sum_dim_int_list_120: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1], True)
        reshape_default_123: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, _shape_param_123);  sum_dim_int_list_120 = _shape_param_123 = None
        mul_tensor_23: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(primals_6, sub)
        add_tensor_2: "f32[128, 128, 1]" = torch.ops.aten.add.Tensor(sqrt, 1e-06)
        div_tensor: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(mul_tensor_23, add_tensor_2);  mul_tensor_23 = None
        div_tensor_1: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(div_tensor, add_tensor_2);  div_tensor = None
        neg_default: "f32[128, 128, 768]" = torch.ops.aten.neg.default(add_tensor_1)
        mul_tensor_24: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(neg_default, div_tensor_1);  neg_default = div_tensor_1 = None
        div_tensor_2: "f32[128, 128, 768]" = torch.ops.aten.div.Tensor(add_tensor_1, add_tensor_2);  add_tensor_1 = add_tensor_2 = None
        sum_dim_int_list_121: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [2], True);  mul_tensor_24 = None
        mul_tensor_25: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor_2, primals_6);  primals_6 = None
        mul_tensor_26: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(div_tensor_2, sub);  div_tensor_2 = None
        sum_dim_int_list_122: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1], True);  mul_tensor_26 = None
        reshape_default_124: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_122, _shape_param_124);  sum_dim_int_list_122 = _shape_param_124 = None
        neg_default_1: "f32[128, 128, 768]" = torch.ops.aten.neg.default(mul_tensor_25)
        sum_dim_int_list_123: "f32[128, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_default_1, [2], True);  neg_default_1 = None
        add_tensor_3: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_201, mul_tensor_25);  add_201 = mul_tensor_25 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16 in forward, code: std = x.std(-1, keepdim=True)
        mul_scalar: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(sqrt, 2)
        div_tensor_3: "f32[128, 128, 1]" = torch.ops.aten.div.Tensor(sum_dim_int_list_121, mul_scalar);  sum_dim_int_list_121 = mul_scalar = None
        eq_scalar: "b8[128, 128, 1]" = torch.ops.aten.eq.Scalar(sqrt, 0);  sqrt = None
        where_self: "f32[128, 128, 1]" = torch.ops.aten.where.self(eq_scalar, full_default_13, div_tensor_3);  eq_scalar = div_tensor_3 = None
        mul_scalar_1: "f32[128, 128, 1]" = torch.ops.aten.mul.Scalar(where_self, 0.002607561929595828);  where_self = None
        mul_tensor_27: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(mul_scalar_1, sub);  mul_scalar_1 = sub = None
        add_tensor_4: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_27);  add_tensor_3 = mul_tensor_27 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15 in forward, code: mean = x.mean(-1, keepdim=True)
        expand_default: "f32[128, 128, 768]" = torch.ops.aten.expand.default(sum_dim_int_list_123, _shape_param_125);  sum_dim_int_list_123 = _shape_param_125 = None
        div_scalar: "f32[128, 128, 768]" = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        add_tensor_5: "f32[128, 128, 768]" = torch.ops.aten.add.Tensor(add_tensor_4, div_scalar);  add_tensor_4 = div_scalar = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:33 in forward, code: return self.dropout(x)
        convert_element_type_default: "f32[128, 128, 768]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_tensor_28: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_29: "f32[128, 128, 768]" = torch.ops.aten.mul.Tensor(add_tensor_5, mul_tensor_28);  add_tensor_5 = mul_tensor_28 = None

        # File: /tmp/pytorch-work/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32 in forward, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        eq_scalar_1: "b8[128, 128]" = torch.ops.aten.eq.Scalar(primals_5, 0)
        unsqueeze_default: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[128, 128, 768]" = torch.ops.aten.where.self(unsqueeze_default, full_default_13, mul_tensor_29);  unsqueeze_default = None
        full_default: "f32[3, 768]" = torch.ops.aten.full.default([3, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[3, 768]" = torch.ops.aten.index_put.default(full_default, [primals_5], where_self_1, True);  full_default = primals_5 = where_self_1 = None
        eq_scalar_2: "b8[128, 128]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_1: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_2, -1);  eq_scalar_2 = None
        where_self_2: "f32[128, 128, 768]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_13, mul_tensor_29);  unsqueeze_default_1 = full_default_13 = mul_tensor_29 = None
        full_default_14: "f32[20005, 768]" = torch.ops.aten.full.default([20005, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[20005, 768]" = torch.ops.aten.index_put.default(full_default_14, [primals_1], where_self_2, True);  full_default_14 = primals_1 = where_self_2 = None
        return (permute_default, reshape_default, permute_default_1, reshape_default_1, permute_default_2, reshape_default_2, permute_default_3, reshape_default_3, reshape_default_4, reshape_default_5, permute_default_4, reshape_default_6, permute_default_5, reshape_default_7, permute_default_6, reshape_default_8, permute_default_7, reshape_default_9, reshape_default_10, reshape_default_11, permute_default_8, reshape_default_12, permute_default_9, reshape_default_13, reshape_default_14, reshape_default_15, permute_default_10, reshape_default_16, permute_default_11, reshape_default_17, permute_default_12, reshape_default_18, permute_default_13, reshape_default_19, reshape_default_20, reshape_default_21, permute_default_14, reshape_default_22, permute_default_15, reshape_default_23, reshape_default_24, reshape_default_25, permute_default_16, reshape_default_26, permute_default_17, reshape_default_27, permute_default_18, reshape_default_28, permute_default_19, reshape_default_29, reshape_default_30, reshape_default_31, permute_default_20, reshape_default_32, permute_default_21, reshape_default_33, reshape_default_34, reshape_default_35, permute_default_22, reshape_default_36, permute_default_23, reshape_default_37, permute_default_24, reshape_default_38, permute_default_25, reshape_default_39, reshape_default_40, reshape_default_41, permute_default_26, reshape_default_42, permute_default_27, reshape_default_43, reshape_default_44, reshape_default_45, permute_default_28, reshape_default_46, permute_default_29, reshape_default_47, permute_default_30, reshape_default_48, permute_default_31, reshape_default_49, reshape_default_50, reshape_default_51, permute_default_32, reshape_default_52, permute_default_33, reshape_default_53, reshape_default_54, reshape_default_55, permute_default_34, reshape_default_56, permute_default_35, reshape_default_57, permute_default_36, reshape_default_58, permute_default_37, reshape_default_59, reshape_default_60, reshape_default_61, permute_default_38, reshape_default_62, permute_default_39, reshape_default_63, reshape_default_64, reshape_default_65, permute_default_40, reshape_default_66, permute_default_41, reshape_default_67, permute_default_42, reshape_default_68, permute_default_43, reshape_default_69, reshape_default_70, reshape_default_71, permute_default_44, reshape_default_72, permute_default_45, reshape_default_73, reshape_default_74, reshape_default_75, permute_default_46, reshape_default_76, permute_default_47, reshape_default_77, permute_default_48, reshape_default_78, permute_default_49, reshape_default_79, reshape_default_80, reshape_default_81, permute_default_50, reshape_default_82, permute_default_51, reshape_default_83, reshape_default_84, reshape_default_85, permute_default_52, reshape_default_86, permute_default_53, reshape_default_87, permute_default_54, reshape_default_88, permute_default_55, reshape_default_89, reshape_default_90, reshape_default_91, permute_default_56, reshape_default_92, permute_default_57, reshape_default_93, reshape_default_94, reshape_default_95, permute_default_58, reshape_default_96, permute_default_59, reshape_default_97, permute_default_60, reshape_default_98, permute_default_61, reshape_default_99, reshape_default_100, reshape_default_101, permute_default_62, reshape_default_102, permute_default_63, reshape_default_103, reshape_default_104, reshape_default_105, permute_default_64, reshape_default_106, permute_default_65, reshape_default_107, permute_default_66, reshape_default_108, permute_default_67, reshape_default_109, reshape_default_110, reshape_default_111, permute_default_68, reshape_default_112, permute_default_69, reshape_default_113, reshape_default_114, reshape_default_115, permute_default_70, reshape_default_116, permute_default_71, reshape_default_117, permute_default_72, reshape_default_119, permute_default_73, reshape_default_121, reshape_default_123, reshape_default_124, index_put_default, index_put_default_1)



def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
