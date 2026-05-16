"""
Standalone repro captured via capture_hook.
Label: hf_qwen2_0.5b_train
Pattern hash: 2491eaca4cc8
Shape hash: 4c800e4a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, convert_element_type_555: "f32[4, 512, 896]", rsqrt_48: "f32[4, 512, 1]", view_490: "bf16[4, 512, 896]", _shape_param_0, view_492: "bf16[2048, 896]", view_494: "bf16[2048, 4864]", view_496: "bf16[2048, 4864]", convert_element_type_545: "f32[4, 512, 896]", rsqrt_47: "f32[4, 512, 1]", add_176: "bf16[4, 512, 896]", _shape_param_1, view_499: "bf16[2048, 896]", getitem_207: "bf16[4, 14, 512, 64]", _shape_param_2, _shape_param_3, view_505: "bf16[2048, 128]", _shape_param_4, view_509: "bf16[2048, 128]", _shape_param_5, view_513: "bf16[2048, 896]", _shape_param_6, convert_element_type_532: "f32[4, 512, 896]", rsqrt_46: "f32[4, 512, 1]", add_186: "bf16[4, 512, 896]", _shape_param_7, view_517: "bf16[2048, 896]", view_519: "bf16[2048, 4864]", view_521: "bf16[2048, 4864]", convert_element_type_522: "f32[4, 512, 896]", rsqrt_45: "f32[4, 512, 1]", add_191: "bf16[4, 512, 896]", _shape_param_8, view_524: "bf16[2048, 896]", getitem_198: "bf16[4, 14, 512, 64]", _shape_param_9, _shape_param_10, view_530: "bf16[2048, 128]", _shape_param_11, view_534: "bf16[2048, 128]", _shape_param_12, view_538: "bf16[2048, 896]", _shape_param_13, convert_element_type_509: "f32[4, 512, 896]", rsqrt_44: "f32[4, 512, 1]", add_201: "bf16[4, 512, 896]", _shape_param_14, view_542: "bf16[2048, 896]", view_544: "bf16[2048, 4864]", view_546: "bf16[2048, 4864]", convert_element_type_499: "f32[4, 512, 896]", rsqrt_43: "f32[4, 512, 1]", add_206: "bf16[4, 512, 896]", _shape_param_15, view_549: "bf16[2048, 896]", getitem_189: "bf16[4, 14, 512, 64]", _shape_param_16, _shape_param_17, view_555: "bf16[2048, 128]", _shape_param_18, view_559: "bf16[2048, 128]", _shape_param_19, view_563: "bf16[2048, 896]", _shape_param_20, convert_element_type_486: "f32[4, 512, 896]", rsqrt_42: "f32[4, 512, 1]", add_216: "bf16[4, 512, 896]", _shape_param_21, view_567: "bf16[2048, 896]", view_569: "bf16[2048, 4864]", view_571: "bf16[2048, 4864]", convert_element_type_476: "f32[4, 512, 896]", rsqrt_41: "f32[4, 512, 1]", add_221: "bf16[4, 512, 896]", _shape_param_22, view_574: "bf16[2048, 896]", getitem_180: "bf16[4, 14, 512, 64]", _shape_param_23, _shape_param_24, view_580: "bf16[2048, 128]", _shape_param_25, view_584: "bf16[2048, 128]", _shape_param_26, view_588: "bf16[2048, 896]", _shape_param_27, convert_element_type_463: "f32[4, 512, 896]", rsqrt_40: "f32[4, 512, 1]", add_231: "bf16[4, 512, 896]", _shape_param_28, view_592: "bf16[2048, 896]", view_594: "bf16[2048, 4864]", view_596: "bf16[2048, 4864]", convert_element_type_453: "f32[4, 512, 896]", rsqrt_39: "f32[4, 512, 1]", add_236: "bf16[4, 512, 896]", _shape_param_29, view_599: "bf16[2048, 896]", getitem_171: "bf16[4, 14, 512, 64]", _shape_param_30, _shape_param_31, view_605: "bf16[2048, 128]", _shape_param_32, view_609: "bf16[2048, 128]", _shape_param_33, view_613: "bf16[2048, 896]", _shape_param_34, convert_element_type_440: "f32[4, 512, 896]", rsqrt_38: "f32[4, 512, 1]", add_246: "bf16[4, 512, 896]", _shape_param_35, view_617: "bf16[2048, 896]", view_619: "bf16[2048, 4864]", view_621: "bf16[2048, 4864]", convert_element_type_430: "f32[4, 512, 896]", rsqrt_37: "f32[4, 512, 1]", add_251: "bf16[4, 512, 896]", _shape_param_36, view_624: "bf16[2048, 896]", getitem_162: "bf16[4, 14, 512, 64]", _shape_param_37, _shape_param_38, view_630: "bf16[2048, 128]", _shape_param_39, view_634: "bf16[2048, 128]", _shape_param_40, view_638: "bf16[2048, 896]", _shape_param_41, convert_element_type_417: "f32[4, 512, 896]", rsqrt_36: "f32[4, 512, 1]", add_261: "bf16[4, 512, 896]", _shape_param_42, view_642: "bf16[2048, 896]", view_644: "bf16[2048, 4864]", view_646: "bf16[2048, 4864]", convert_element_type_407: "f32[4, 512, 896]", rsqrt_35: "f32[4, 512, 1]", add_266: "bf16[4, 512, 896]", _shape_param_43, view_649: "bf16[2048, 896]", getitem_153: "bf16[4, 14, 512, 64]", _shape_param_44, _shape_param_45, view_655: "bf16[2048, 128]", _shape_param_46, view_659: "bf16[2048, 128]", _shape_param_47, view_663: "bf16[2048, 896]", _shape_param_48, convert_element_type_394: "f32[4, 512, 896]", rsqrt_34: "f32[4, 512, 1]", add_276: "bf16[4, 512, 896]", _shape_param_49, view_667: "bf16[2048, 896]", view_669: "bf16[2048, 4864]", view_671: "bf16[2048, 4864]", convert_element_type_384: "f32[4, 512, 896]", rsqrt_33: "f32[4, 512, 1]", add_281: "bf16[4, 512, 896]", _shape_param_50, view_674: "bf16[2048, 896]", getitem_144: "bf16[4, 14, 512, 64]", _shape_param_51, _shape_param_52, view_680: "bf16[2048, 128]", _shape_param_53, view_684: "bf16[2048, 128]", _shape_param_54, view_688: "bf16[2048, 896]", _shape_param_55, convert_element_type_371: "f32[4, 512, 896]", rsqrt_32: "f32[4, 512, 1]", add_291: "bf16[4, 512, 896]", _shape_param_56, view_692: "bf16[2048, 896]", view_694: "bf16[2048, 4864]", view_696: "bf16[2048, 4864]", convert_element_type_361: "f32[4, 512, 896]", rsqrt_31: "f32[4, 512, 1]", add_296: "bf16[4, 512, 896]", _shape_param_57, view_699: "bf16[2048, 896]", getitem_135: "bf16[4, 14, 512, 64]", _shape_param_58, _shape_param_59, view_705: "bf16[2048, 128]", _shape_param_60, view_709: "bf16[2048, 128]", _shape_param_61, view_713: "bf16[2048, 896]", _shape_param_62, convert_element_type_348: "f32[4, 512, 896]", rsqrt_30: "f32[4, 512, 1]", add_306: "bf16[4, 512, 896]", _shape_param_63, view_717: "bf16[2048, 896]", view_719: "bf16[2048, 4864]", view_721: "bf16[2048, 4864]", convert_element_type_338: "f32[4, 512, 896]", rsqrt_29: "f32[4, 512, 1]", add_311: "bf16[4, 512, 896]", _shape_param_64, view_724: "bf16[2048, 896]", getitem_126: "bf16[4, 14, 512, 64]", _shape_param_65, _shape_param_66, view_730: "bf16[2048, 128]", _shape_param_67, view_734: "bf16[2048, 128]", _shape_param_68, view_738: "bf16[2048, 896]", _shape_param_69, convert_element_type_325: "f32[4, 512, 896]", rsqrt_28: "f32[4, 512, 1]", add_321: "bf16[4, 512, 896]", _shape_param_70, view_742: "bf16[2048, 896]", view_744: "bf16[2048, 4864]", view_746: "bf16[2048, 4864]", convert_element_type_315: "f32[4, 512, 896]", rsqrt_27: "f32[4, 512, 1]", add_326: "bf16[4, 512, 896]", _shape_param_71, view_749: "bf16[2048, 896]", getitem_117: "bf16[4, 14, 512, 64]", _shape_param_72, _shape_param_73, view_755: "bf16[2048, 128]", _shape_param_74, view_759: "bf16[2048, 128]", _shape_param_75, view_763: "bf16[2048, 896]", _shape_param_76, convert_element_type_302: "f32[4, 512, 896]", rsqrt_26: "f32[4, 512, 1]", add_336: "bf16[4, 512, 896]", _shape_param_77, view_767: "bf16[2048, 896]", view_769: "bf16[2048, 4864]", view_771: "bf16[2048, 4864]", convert_element_type_292: "f32[4, 512, 896]", rsqrt_25: "f32[4, 512, 1]", add_341: "bf16[4, 512, 896]", _shape_param_78, view_774: "bf16[2048, 896]", getitem_108: "bf16[4, 14, 512, 64]", _shape_param_79, _shape_param_80, view_780: "bf16[2048, 128]", _shape_param_81, view_784: "bf16[2048, 128]", _shape_param_82, view_788: "bf16[2048, 896]", _shape_param_83, convert_element_type_279: "f32[4, 512, 896]", rsqrt_24: "f32[4, 512, 1]", add_351: "bf16[4, 512, 896]", _shape_param_84, view_792: "bf16[2048, 896]", view_794: "bf16[2048, 4864]", view_796: "bf16[2048, 4864]", convert_element_type_269: "f32[4, 512, 896]", rsqrt_23: "f32[4, 512, 1]", add_356: "bf16[4, 512, 896]", _shape_param_85, view_799: "bf16[2048, 896]", getitem_99: "bf16[4, 14, 512, 64]", _shape_param_86, _shape_param_87, view_805: "bf16[2048, 128]", _shape_param_88, view_809: "bf16[2048, 128]", _shape_param_89, view_813: "bf16[2048, 896]", _shape_param_90, convert_element_type_256: "f32[4, 512, 896]", rsqrt_22: "f32[4, 512, 1]", add_366: "bf16[4, 512, 896]", _shape_param_91, view_817: "bf16[2048, 896]", view_819: "bf16[2048, 4864]", view_821: "bf16[2048, 4864]", convert_element_type_246: "f32[4, 512, 896]", rsqrt_21: "f32[4, 512, 1]", add_371: "bf16[4, 512, 896]", _shape_param_92, view_824: "bf16[2048, 896]", getitem_90: "bf16[4, 14, 512, 64]", _shape_param_93, _shape_param_94, view_830: "bf16[2048, 128]", _shape_param_95, view_834: "bf16[2048, 128]", _shape_param_96, view_838: "bf16[2048, 896]", _shape_param_97, convert_element_type_233: "f32[4, 512, 896]", rsqrt_20: "f32[4, 512, 1]", add_381: "bf16[4, 512, 896]", _shape_param_98, view_842: "bf16[2048, 896]", view_844: "bf16[2048, 4864]", view_846: "bf16[2048, 4864]", convert_element_type_223: "f32[4, 512, 896]", rsqrt_19: "f32[4, 512, 1]", add_386: "bf16[4, 512, 896]", _shape_param_99, view_849: "bf16[2048, 896]", getitem_81: "bf16[4, 14, 512, 64]", _shape_param_100, _shape_param_101, view_855: "bf16[2048, 128]", _shape_param_102, view_859: "bf16[2048, 128]", _shape_param_103, view_863: "bf16[2048, 896]", _shape_param_104, convert_element_type_210: "f32[4, 512, 896]", rsqrt_18: "f32[4, 512, 1]", add_396: "bf16[4, 512, 896]", _shape_param_105, view_867: "bf16[2048, 896]", view_869: "bf16[2048, 4864]", view_871: "bf16[2048, 4864]", convert_element_type_200: "f32[4, 512, 896]", rsqrt_17: "f32[4, 512, 1]", add_401: "bf16[4, 512, 896]", _shape_param_106, view_874: "bf16[2048, 896]", getitem_72: "bf16[4, 14, 512, 64]", _shape_param_107, _shape_param_108, view_880: "bf16[2048, 128]", _shape_param_109, view_884: "bf16[2048, 128]", _shape_param_110, view_888: "bf16[2048, 896]", _shape_param_111, convert_element_type_187: "f32[4, 512, 896]", rsqrt_16: "f32[4, 512, 1]", add_411: "bf16[4, 512, 896]", _shape_param_112, view_892: "bf16[2048, 896]", view_894: "bf16[2048, 4864]", view_896: "bf16[2048, 4864]", convert_element_type_177: "f32[4, 512, 896]", rsqrt_15: "f32[4, 512, 1]", add_416: "bf16[4, 512, 896]", _shape_param_113, view_899: "bf16[2048, 896]", getitem_63: "bf16[4, 14, 512, 64]", _shape_param_114, _shape_param_115, view_905: "bf16[2048, 128]", _shape_param_116, view_909: "bf16[2048, 128]", _shape_param_117, view_913: "bf16[2048, 896]", _shape_param_118, convert_element_type_164: "f32[4, 512, 896]", rsqrt_14: "f32[4, 512, 1]", add_426: "bf16[4, 512, 896]", _shape_param_119, view_917: "bf16[2048, 896]", view_919: "bf16[2048, 4864]", view_921: "bf16[2048, 4864]", convert_element_type_154: "f32[4, 512, 896]", rsqrt_13: "f32[4, 512, 1]", add_431: "bf16[4, 512, 896]", _shape_param_120, view_924: "bf16[2048, 896]", getitem_54: "bf16[4, 14, 512, 64]", _shape_param_121, _shape_param_122, view_930: "bf16[2048, 128]", _shape_param_123, view_934: "bf16[2048, 128]", _shape_param_124, view_938: "bf16[2048, 896]", _shape_param_125, convert_element_type_141: "f32[4, 512, 896]", rsqrt_12: "f32[4, 512, 1]", add_441: "bf16[4, 512, 896]", _shape_param_126, view_942: "bf16[2048, 896]", view_944: "bf16[2048, 4864]", view_946: "bf16[2048, 4864]", convert_element_type_131: "f32[4, 512, 896]", rsqrt_11: "f32[4, 512, 1]", add_446: "bf16[4, 512, 896]", _shape_param_127, view_949: "bf16[2048, 896]", getitem_45: "bf16[4, 14, 512, 64]", _shape_param_128, _shape_param_129, view_955: "bf16[2048, 128]", _shape_param_130, view_959: "bf16[2048, 128]", _shape_param_131, view_963: "bf16[2048, 896]", _shape_param_132, convert_element_type_118: "f32[4, 512, 896]", rsqrt_10: "f32[4, 512, 1]", add_456: "bf16[4, 512, 896]", _shape_param_133, view_967: "bf16[2048, 896]", view_969: "bf16[2048, 4864]", view_971: "bf16[2048, 4864]", convert_element_type_108: "f32[4, 512, 896]", rsqrt_9: "f32[4, 512, 1]", add_461: "bf16[4, 512, 896]", _shape_param_134, view_974: "bf16[2048, 896]", getitem_36: "bf16[4, 14, 512, 64]", _shape_param_135, _shape_param_136, view_980: "bf16[2048, 128]", _shape_param_137, view_984: "bf16[2048, 128]", _shape_param_138, view_988: "bf16[2048, 896]", _shape_param_139, convert_element_type_95: "f32[4, 512, 896]", rsqrt_8: "f32[4, 512, 1]", add_471: "bf16[4, 512, 896]", _shape_param_140, view_992: "bf16[2048, 896]", view_994: "bf16[2048, 4864]", view_996: "bf16[2048, 4864]", convert_element_type_85: "f32[4, 512, 896]", rsqrt_7: "f32[4, 512, 1]", add_476: "bf16[4, 512, 896]", _shape_param_141, view_999: "bf16[2048, 896]", getitem_27: "bf16[4, 14, 512, 64]", _shape_param_142, _shape_param_143, view_1005: "bf16[2048, 128]", _shape_param_144, view_1009: "bf16[2048, 128]", _shape_param_145, view_1013: "bf16[2048, 896]", _shape_param_146, convert_element_type_72: "f32[4, 512, 896]", rsqrt_6: "f32[4, 512, 1]", add_486: "bf16[4, 512, 896]", _shape_param_147, view_1017: "bf16[2048, 896]", view_1019: "bf16[2048, 4864]", view_1021: "bf16[2048, 4864]", convert_element_type_62: "f32[4, 512, 896]", rsqrt_5: "f32[4, 512, 1]", add_491: "bf16[4, 512, 896]", _shape_param_148, view_1024: "bf16[2048, 896]", getitem_18: "bf16[4, 14, 512, 64]", _shape_param_149, _shape_param_150, view_1030: "bf16[2048, 128]", _shape_param_151, view_1034: "bf16[2048, 128]", _shape_param_152, view_1038: "bf16[2048, 896]", _shape_param_153, convert_element_type_49: "f32[4, 512, 896]", rsqrt_4: "f32[4, 512, 1]", add_501: "bf16[4, 512, 896]", _shape_param_154, view_1042: "bf16[2048, 896]", view_1044: "bf16[2048, 4864]", view_1046: "bf16[2048, 4864]", convert_element_type_39: "f32[4, 512, 896]", rsqrt_3: "f32[4, 512, 1]", add_506: "bf16[4, 512, 896]", _shape_param_155, view_1049: "bf16[2048, 896]", getitem_9: "bf16[4, 14, 512, 64]", _shape_param_156, _shape_param_157, view_1055: "bf16[2048, 128]", _shape_param_158, view_1059: "bf16[2048, 128]", _shape_param_159, view_1063: "bf16[2048, 896]", _shape_param_160, convert_element_type_26: "f32[4, 512, 896]", rsqrt_2: "f32[4, 512, 1]", add_516: "bf16[4, 512, 896]", _shape_param_161, view_1067: "bf16[2048, 896]", view_1069: "bf16[2048, 4864]", view_1071: "bf16[2048, 4864]", convert_element_type_16: "f32[4, 512, 896]", rsqrt_1: "f32[4, 512, 1]", add_521: "bf16[4, 512, 896]", _shape_param_162, view_1074: "bf16[2048, 896]", getitem: "bf16[4, 14, 512, 64]", _shape_param_163, _shape_param_164, view_1080: "bf16[2048, 128]", _shape_param_165, mm_429: "bf16[2048, 896]", _shape_param_166, view_1084: "bf16[2048, 128]", _shape_param_167, mm_431: "bf16[2048, 896]", _shape_param_168, view_1088: "bf16[2048, 896]", _shape_param_169, mm_433: "bf16[2048, 896]", _shape_param_170, primals_4: "bf16[896]", embedding: "bf16[4, 512, 896]", rsqrt: "f32[4, 512, 1]", _shape_param_171, _shape_param_172, add_523: "bf16[4, 512, 896]", primals_1: "i64[4, 512]", full_default_49: "f32[]", mm_97: "bf16[151936, 896]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_555, rsqrt_48);  convert_element_type_555 = rsqrt_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(view_490, convert_element_type_default);  view_490 = convert_element_type_default = None
        sum_dim_int_list: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True);  mul_tensor_1 = None
        reshape_default: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_492, [1, 0]);  view_492 = None
        permute_default_1: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_494, [1, 0]);  view_494 = None
        permute_default_2: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_496, [1, 0]);  view_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_2: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_545, rsqrt_47);  convert_element_type_545 = rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_1: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_176, convert_element_type_default_1);  add_176 = convert_element_type_default_1 = None
        sum_dim_int_list_1: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True);  mul_tensor_3 = None
        reshape_default_1: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_1, _shape_param_1);  sum_dim_int_list_1 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_3: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_499, [1, 0]);  view_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_4: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_207, [0, 2, 1, 3]);  getitem_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_2: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_2);  permute_default_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_3: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_2, _shape_param_3);  reshape_default_2 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_5: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_505, [1, 0])
        sum_dim_int_list_2: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_505, [0], True);  view_505 = None
        reshape_default_4: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_2, _shape_param_4);  sum_dim_int_list_2 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_6: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_509, [1, 0])
        sum_dim_int_list_3: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
        reshape_default_5: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_5);  sum_dim_int_list_3 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_7: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_513, [1, 0])
        sum_dim_int_list_4: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_513, [0], True);  view_513 = None
        reshape_default_6: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_6);  sum_dim_int_list_4 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_4: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_532, rsqrt_46);  convert_element_type_532 = rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_2: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        mul_tensor_5: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_186, convert_element_type_default_2);  add_186 = convert_element_type_default_2 = None
        sum_dim_int_list_5: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True);  mul_tensor_5 = None
        reshape_default_7: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_5, _shape_param_7);  sum_dim_int_list_5 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_8: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_517, [1, 0]);  view_517 = None
        permute_default_9: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_519, [1, 0]);  view_519 = None
        permute_default_10: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_521, [1, 0]);  view_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_6: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_522, rsqrt_45);  convert_element_type_522 = rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_3: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        mul_tensor_7: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_191, convert_element_type_default_3);  add_191 = convert_element_type_default_3 = None
        sum_dim_int_list_6: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], True);  mul_tensor_7 = None
        reshape_default_8: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_8);  sum_dim_int_list_6 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_11: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_524, [1, 0]);  view_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_12: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_198, [0, 2, 1, 3]);  getitem_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_9: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_12, _shape_param_9);  permute_default_12 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_10: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_9, _shape_param_10);  reshape_default_9 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_13: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_530, [1, 0])
        sum_dim_int_list_7: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_530, [0], True);  view_530 = None
        reshape_default_11: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_11);  sum_dim_int_list_7 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_14: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_534, [1, 0])
        sum_dim_int_list_8: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_534, [0], True);  view_534 = None
        reshape_default_12: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_8, _shape_param_12);  sum_dim_int_list_8 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_15: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_538, [1, 0])
        sum_dim_int_list_9: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
        reshape_default_13: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_13);  sum_dim_int_list_9 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_8: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_509, rsqrt_44);  convert_element_type_509 = rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_4: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        mul_tensor_9: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_201, convert_element_type_default_4);  add_201 = convert_element_type_default_4 = None
        sum_dim_int_list_10: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], True);  mul_tensor_9 = None
        reshape_default_14: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_14);  sum_dim_int_list_10 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_16: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_542, [1, 0]);  view_542 = None
        permute_default_17: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_544, [1, 0]);  view_544 = None
        permute_default_18: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_546, [1, 0]);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_10: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_499, rsqrt_43);  convert_element_type_499 = rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_5: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        mul_tensor_11: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_206, convert_element_type_default_5);  add_206 = convert_element_type_default_5 = None
        sum_dim_int_list_11: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], True);  mul_tensor_11 = None
        reshape_default_15: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_15);  sum_dim_int_list_11 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_19: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_549, [1, 0]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_20: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_16: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_20, _shape_param_16);  permute_default_20 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_17: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_16, _shape_param_17);  reshape_default_16 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_21: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_555, [1, 0])
        sum_dim_int_list_12: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True);  view_555 = None
        reshape_default_18: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_18);  sum_dim_int_list_12 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_22: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_559, [1, 0])
        sum_dim_int_list_13: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True);  view_559 = None
        reshape_default_19: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_19);  sum_dim_int_list_13 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_23: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_563, [1, 0])
        sum_dim_int_list_14: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_563, [0], True);  view_563 = None
        reshape_default_20: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_14, _shape_param_20);  sum_dim_int_list_14 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_12: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_486, rsqrt_42);  convert_element_type_486 = rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_6: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        mul_tensor_13: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_216, convert_element_type_default_6);  add_216 = convert_element_type_default_6 = None
        sum_dim_int_list_15: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1], True);  mul_tensor_13 = None
        reshape_default_21: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_15, _shape_param_21);  sum_dim_int_list_15 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_24: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_567, [1, 0]);  view_567 = None
        permute_default_25: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_569, [1, 0]);  view_569 = None
        permute_default_26: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_571, [1, 0]);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_14: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_476, rsqrt_41);  convert_element_type_476 = rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_7: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_14, torch.bfloat16);  mul_tensor_14 = None
        mul_tensor_15: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_221, convert_element_type_default_7);  add_221 = convert_element_type_default_7 = None
        sum_dim_int_list_16: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True);  mul_tensor_15 = None
        reshape_default_22: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_22);  sum_dim_int_list_16 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_27: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_574, [1, 0]);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_28: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_180, [0, 2, 1, 3]);  getitem_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_23: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_28, _shape_param_23);  permute_default_28 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_24: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_23, _shape_param_24);  reshape_default_23 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_29: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_580, [1, 0])
        sum_dim_int_list_17: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
        reshape_default_25: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_25);  sum_dim_int_list_17 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_30: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_584, [1, 0])
        sum_dim_int_list_18: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
        reshape_default_26: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_26);  sum_dim_int_list_18 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_31: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_588, [1, 0])
        sum_dim_int_list_19: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_588, [0], True);  view_588 = None
        reshape_default_27: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_19, _shape_param_27);  sum_dim_int_list_19 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_16: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_463, rsqrt_40);  convert_element_type_463 = rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_8: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16);  mul_tensor_16 = None
        mul_tensor_17: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_231, convert_element_type_default_8);  add_231 = convert_element_type_default_8 = None
        sum_dim_int_list_20: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], True);  mul_tensor_17 = None
        reshape_default_28: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_28);  sum_dim_int_list_20 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_32: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_592, [1, 0]);  view_592 = None
        permute_default_33: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_594, [1, 0]);  view_594 = None
        permute_default_34: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_596, [1, 0]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_18: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_453, rsqrt_39);  convert_element_type_453 = rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_9: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_18, torch.bfloat16);  mul_tensor_18 = None
        mul_tensor_19: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_236, convert_element_type_default_9);  add_236 = convert_element_type_default_9 = None
        sum_dim_int_list_21: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], True);  mul_tensor_19 = None
        reshape_default_29: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_29);  sum_dim_int_list_21 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_35: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_599, [1, 0]);  view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_36: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_171, [0, 2, 1, 3]);  getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_30: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_36, _shape_param_30);  permute_default_36 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_31: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_30, _shape_param_31);  reshape_default_30 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_37: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_605, [1, 0])
        sum_dim_int_list_22: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_605, [0], True);  view_605 = None
        reshape_default_32: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_32);  sum_dim_int_list_22 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_38: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_609, [1, 0])
        sum_dim_int_list_23: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_609, [0], True);  view_609 = None
        reshape_default_33: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_33);  sum_dim_int_list_23 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_39: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_613, [1, 0])
        sum_dim_int_list_24: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_613, [0], True);  view_613 = None
        reshape_default_34: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_34);  sum_dim_int_list_24 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_20: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_440, rsqrt_38);  convert_element_type_440 = rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_10: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.bfloat16);  mul_tensor_20 = None
        mul_tensor_21: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_246, convert_element_type_default_10);  add_246 = convert_element_type_default_10 = None
        sum_dim_int_list_25: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], True);  mul_tensor_21 = None
        reshape_default_35: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_25, _shape_param_35);  sum_dim_int_list_25 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_40: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_617, [1, 0]);  view_617 = None
        permute_default_41: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_619, [1, 0]);  view_619 = None
        permute_default_42: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_621, [1, 0]);  view_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_22: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_430, rsqrt_37);  convert_element_type_430 = rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_11: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_22, torch.bfloat16);  mul_tensor_22 = None
        mul_tensor_23: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_251, convert_element_type_default_11);  add_251 = convert_element_type_default_11 = None
        sum_dim_int_list_26: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1], True);  mul_tensor_23 = None
        reshape_default_36: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_36);  sum_dim_int_list_26 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_43: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_624, [1, 0]);  view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_44: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_37: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_44, _shape_param_37);  permute_default_44 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_38: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_37, _shape_param_38);  reshape_default_37 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_45: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_630, [1, 0])
        sum_dim_int_list_27: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_630, [0], True);  view_630 = None
        reshape_default_39: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_39);  sum_dim_int_list_27 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_46: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_634, [1, 0])
        sum_dim_int_list_28: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_634, [0], True);  view_634 = None
        reshape_default_40: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_28, _shape_param_40);  sum_dim_int_list_28 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_47: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_638, [1, 0])
        sum_dim_int_list_29: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_638, [0], True);  view_638 = None
        reshape_default_41: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_41);  sum_dim_int_list_29 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_24: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_417, rsqrt_36);  convert_element_type_417 = rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_12: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.bfloat16);  mul_tensor_24 = None
        mul_tensor_25: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_261, convert_element_type_default_12);  add_261 = convert_element_type_default_12 = None
        sum_dim_int_list_30: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], True);  mul_tensor_25 = None
        reshape_default_42: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_42);  sum_dim_int_list_30 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_48: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_642, [1, 0]);  view_642 = None
        permute_default_49: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_644, [1, 0]);  view_644 = None
        permute_default_50: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_646, [1, 0]);  view_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_26: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_407, rsqrt_35);  convert_element_type_407 = rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_13: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_26, torch.bfloat16);  mul_tensor_26 = None
        mul_tensor_27: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_266, convert_element_type_default_13);  add_266 = convert_element_type_default_13 = None
        sum_dim_int_list_31: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], True);  mul_tensor_27 = None
        reshape_default_43: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_43);  sum_dim_int_list_31 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_51: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_649, [1, 0]);  view_649 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_52: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_44: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_52, _shape_param_44);  permute_default_52 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_45: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_44, _shape_param_45);  reshape_default_44 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_53: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_655, [1, 0])
        sum_dim_int_list_32: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_655, [0], True);  view_655 = None
        reshape_default_46: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_46);  sum_dim_int_list_32 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_54: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_659, [1, 0])
        sum_dim_int_list_33: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
        reshape_default_47: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_47);  sum_dim_int_list_33 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_55: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_663, [1, 0])
        sum_dim_int_list_34: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True);  view_663 = None
        reshape_default_48: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_34, _shape_param_48);  sum_dim_int_list_34 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_28: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_394, rsqrt_34);  convert_element_type_394 = rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_14: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.bfloat16);  mul_tensor_28 = None
        mul_tensor_29: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_276, convert_element_type_default_14);  add_276 = convert_element_type_default_14 = None
        sum_dim_int_list_35: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1], True);  mul_tensor_29 = None
        reshape_default_49: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_35, _shape_param_49);  sum_dim_int_list_35 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_56: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_667, [1, 0]);  view_667 = None
        permute_default_57: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_669, [1, 0]);  view_669 = None
        permute_default_58: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_671, [1, 0]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_30: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_384, rsqrt_33);  convert_element_type_384 = rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_15: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_30, torch.bfloat16);  mul_tensor_30 = None
        mul_tensor_31: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_281, convert_element_type_default_15);  add_281 = convert_element_type_default_15 = None
        sum_dim_int_list_36: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1], True);  mul_tensor_31 = None
        reshape_default_50: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_50);  sum_dim_int_list_36 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_59: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_674, [1, 0]);  view_674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_60: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3]);  getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_51: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_60, _shape_param_51);  permute_default_60 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_52: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_51, _shape_param_52);  reshape_default_51 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_61: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_680, [1, 0])
        sum_dim_int_list_37: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_680, [0], True);  view_680 = None
        reshape_default_53: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_53);  sum_dim_int_list_37 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_62: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_684, [1, 0])
        sum_dim_int_list_38: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_684, [0], True);  view_684 = None
        reshape_default_54: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_54);  sum_dim_int_list_38 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_63: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_688, [1, 0])
        sum_dim_int_list_39: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_688, [0], True);  view_688 = None
        reshape_default_55: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_39, _shape_param_55);  sum_dim_int_list_39 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_32: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_371, rsqrt_32);  convert_element_type_371 = rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_16: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_32, torch.bfloat16);  mul_tensor_32 = None
        mul_tensor_33: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_291, convert_element_type_default_16);  add_291 = convert_element_type_default_16 = None
        sum_dim_int_list_40: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1], True);  mul_tensor_33 = None
        reshape_default_56: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_56);  sum_dim_int_list_40 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_64: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_692, [1, 0]);  view_692 = None
        permute_default_65: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_694, [1, 0]);  view_694 = None
        permute_default_66: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_696, [1, 0]);  view_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_34: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_361, rsqrt_31);  convert_element_type_361 = rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_17: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_34, torch.bfloat16);  mul_tensor_34 = None
        mul_tensor_35: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_296, convert_element_type_default_17);  add_296 = convert_element_type_default_17 = None
        sum_dim_int_list_41: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1], True);  mul_tensor_35 = None
        reshape_default_57: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_57);  sum_dim_int_list_41 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_67: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_699, [1, 0]);  view_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_68: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_135, [0, 2, 1, 3]);  getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_58: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_68, _shape_param_58);  permute_default_68 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_59: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_58, _shape_param_59);  reshape_default_58 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_69: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_705, [1, 0])
        sum_dim_int_list_42: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_705, [0], True);  view_705 = None
        reshape_default_60: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_60);  sum_dim_int_list_42 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_70: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_709, [1, 0])
        sum_dim_int_list_43: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
        reshape_default_61: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_61);  sum_dim_int_list_43 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_71: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_713, [1, 0])
        sum_dim_int_list_44: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_713, [0], True);  view_713 = None
        reshape_default_62: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_44, _shape_param_62);  sum_dim_int_list_44 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_36: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_348, rsqrt_30);  convert_element_type_348 = rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_18: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_36, torch.bfloat16);  mul_tensor_36 = None
        mul_tensor_37: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_306, convert_element_type_default_18);  add_306 = convert_element_type_default_18 = None
        sum_dim_int_list_45: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1], True);  mul_tensor_37 = None
        reshape_default_63: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_45, _shape_param_63);  sum_dim_int_list_45 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_72: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_717, [1, 0]);  view_717 = None
        permute_default_73: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_719, [1, 0]);  view_719 = None
        permute_default_74: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_721, [1, 0]);  view_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_38: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_338, rsqrt_29);  convert_element_type_338 = rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_19: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_38, torch.bfloat16);  mul_tensor_38 = None
        mul_tensor_39: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_311, convert_element_type_default_19);  add_311 = convert_element_type_default_19 = None
        sum_dim_int_list_46: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1], True);  mul_tensor_39 = None
        reshape_default_64: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_64);  sum_dim_int_list_46 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_75: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_724, [1, 0]);  view_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_76: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_65: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_76, _shape_param_65);  permute_default_76 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_66: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_65, _shape_param_66);  reshape_default_65 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_77: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_730, [1, 0])
        sum_dim_int_list_47: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_730, [0], True);  view_730 = None
        reshape_default_67: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_67);  sum_dim_int_list_47 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_78: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_734, [1, 0])
        sum_dim_int_list_48: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
        reshape_default_68: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_48, _shape_param_68);  sum_dim_int_list_48 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_79: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_738, [1, 0])
        sum_dim_int_list_49: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_738, [0], True);  view_738 = None
        reshape_default_69: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_49, _shape_param_69);  sum_dim_int_list_49 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_40: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_325, rsqrt_28);  convert_element_type_325 = rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_20: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_40, torch.bfloat16);  mul_tensor_40 = None
        mul_tensor_41: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_321, convert_element_type_default_20);  add_321 = convert_element_type_default_20 = None
        sum_dim_int_list_50: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1], True);  mul_tensor_41 = None
        reshape_default_70: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_70);  sum_dim_int_list_50 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_80: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_742, [1, 0]);  view_742 = None
        permute_default_81: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_744, [1, 0]);  view_744 = None
        permute_default_82: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_746, [1, 0]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_42: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_315, rsqrt_27);  convert_element_type_315 = rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_21: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_42, torch.bfloat16);  mul_tensor_42 = None
        mul_tensor_43: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_326, convert_element_type_default_21);  add_326 = convert_element_type_default_21 = None
        sum_dim_int_list_51: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1], True);  mul_tensor_43 = None
        reshape_default_71: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_71);  sum_dim_int_list_51 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_83: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_749, [1, 0]);  view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_84: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_72: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_84, _shape_param_72);  permute_default_84 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_73: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_72, _shape_param_73);  reshape_default_72 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_85: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_755, [1, 0])
        sum_dim_int_list_52: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_755, [0], True);  view_755 = None
        reshape_default_74: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_74);  sum_dim_int_list_52 = _shape_param_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_86: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_759, [1, 0])
        sum_dim_int_list_53: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_759, [0], True);  view_759 = None
        reshape_default_75: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_75);  sum_dim_int_list_53 = _shape_param_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_87: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_763, [1, 0])
        sum_dim_int_list_54: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_763, [0], True);  view_763 = None
        reshape_default_76: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_76);  sum_dim_int_list_54 = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_44: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_302, rsqrt_26);  convert_element_type_302 = rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_22: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_44, torch.bfloat16);  mul_tensor_44 = None
        mul_tensor_45: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_336, convert_element_type_default_22);  add_336 = convert_element_type_default_22 = None
        sum_dim_int_list_55: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1], True);  mul_tensor_45 = None
        reshape_default_77: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_55, _shape_param_77);  sum_dim_int_list_55 = _shape_param_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_88: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_767, [1, 0]);  view_767 = None
        permute_default_89: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_769, [1, 0]);  view_769 = None
        permute_default_90: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_771, [1, 0]);  view_771 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_46: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_292, rsqrt_25);  convert_element_type_292 = rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_23: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_46, torch.bfloat16);  mul_tensor_46 = None
        mul_tensor_47: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_341, convert_element_type_default_23);  add_341 = convert_element_type_default_23 = None
        sum_dim_int_list_56: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1], True);  mul_tensor_47 = None
        reshape_default_78: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_78);  sum_dim_int_list_56 = _shape_param_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_91: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_774, [1, 0]);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_92: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3]);  getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_79: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_92, _shape_param_79);  permute_default_92 = _shape_param_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_80: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_79, _shape_param_80);  reshape_default_79 = _shape_param_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_93: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_780, [1, 0])
        sum_dim_int_list_57: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_780, [0], True);  view_780 = None
        reshape_default_81: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_81);  sum_dim_int_list_57 = _shape_param_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_94: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_784, [1, 0])
        sum_dim_int_list_58: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_784, [0], True);  view_784 = None
        reshape_default_82: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_58, _shape_param_82);  sum_dim_int_list_58 = _shape_param_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_95: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_788, [1, 0])
        sum_dim_int_list_59: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_788, [0], True);  view_788 = None
        reshape_default_83: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_83);  sum_dim_int_list_59 = _shape_param_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_48: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_279, rsqrt_24);  convert_element_type_279 = rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_24: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_48, torch.bfloat16);  mul_tensor_48 = None
        mul_tensor_49: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_351, convert_element_type_default_24);  add_351 = convert_element_type_default_24 = None
        sum_dim_int_list_60: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_49, [0, 1], True);  mul_tensor_49 = None
        reshape_default_84: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_84);  sum_dim_int_list_60 = _shape_param_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_96: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_792, [1, 0]);  view_792 = None
        permute_default_97: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_794, [1, 0]);  view_794 = None
        permute_default_98: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_796, [1, 0]);  view_796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_50: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_269, rsqrt_23);  convert_element_type_269 = rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_25: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_50, torch.bfloat16);  mul_tensor_50 = None
        mul_tensor_51: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_356, convert_element_type_default_25);  add_356 = convert_element_type_default_25 = None
        sum_dim_int_list_61: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [0, 1], True);  mul_tensor_51 = None
        reshape_default_85: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, _shape_param_85);  sum_dim_int_list_61 = _shape_param_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_99: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_799, [1, 0]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_100: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_99, [0, 2, 1, 3]);  getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_86: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_100, _shape_param_86);  permute_default_100 = _shape_param_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_87: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_86, _shape_param_87);  reshape_default_86 = _shape_param_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_101: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_805, [1, 0])
        sum_dim_int_list_62: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_805, [0], True);  view_805 = None
        reshape_default_88: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_88);  sum_dim_int_list_62 = _shape_param_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_102: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_809, [1, 0])
        sum_dim_int_list_63: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_809, [0], True);  view_809 = None
        reshape_default_89: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_89);  sum_dim_int_list_63 = _shape_param_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_103: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_813, [1, 0])
        sum_dim_int_list_64: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_813, [0], True);  view_813 = None
        reshape_default_90: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_64, _shape_param_90);  sum_dim_int_list_64 = _shape_param_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_52: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_256, rsqrt_22);  convert_element_type_256 = rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_26: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_52, torch.bfloat16);  mul_tensor_52 = None
        mul_tensor_53: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_366, convert_element_type_default_26);  add_366 = convert_element_type_default_26 = None
        sum_dim_int_list_65: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_53, [0, 1], True);  mul_tensor_53 = None
        reshape_default_91: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_65, _shape_param_91);  sum_dim_int_list_65 = _shape_param_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_104: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_817, [1, 0]);  view_817 = None
        permute_default_105: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_819, [1, 0]);  view_819 = None
        permute_default_106: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_821, [1, 0]);  view_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_54: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_246, rsqrt_21);  convert_element_type_246 = rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_27: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_54, torch.bfloat16);  mul_tensor_54 = None
        mul_tensor_55: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_371, convert_element_type_default_27);  add_371 = convert_element_type_default_27 = None
        sum_dim_int_list_66: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_55, [0, 1], True);  mul_tensor_55 = None
        reshape_default_92: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_92);  sum_dim_int_list_66 = _shape_param_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_107: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_824, [1, 0]);  view_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_108: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_90, [0, 2, 1, 3]);  getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_93: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_108, _shape_param_93);  permute_default_108 = _shape_param_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_94: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_93, _shape_param_94);  reshape_default_93 = _shape_param_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_109: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_830, [1, 0])
        sum_dim_int_list_67: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_830, [0], True);  view_830 = None
        reshape_default_95: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_95);  sum_dim_int_list_67 = _shape_param_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_110: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_834, [1, 0])
        sum_dim_int_list_68: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_834, [0], True);  view_834 = None
        reshape_default_96: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_96);  sum_dim_int_list_68 = _shape_param_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_111: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_838, [1, 0])
        sum_dim_int_list_69: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_838, [0], True);  view_838 = None
        reshape_default_97: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, _shape_param_97);  sum_dim_int_list_69 = _shape_param_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_56: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_233, rsqrt_20);  convert_element_type_233 = rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_28: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_56, torch.bfloat16);  mul_tensor_56 = None
        mul_tensor_57: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_381, convert_element_type_default_28);  add_381 = convert_element_type_default_28 = None
        sum_dim_int_list_70: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_57, [0, 1], True);  mul_tensor_57 = None
        reshape_default_98: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_98);  sum_dim_int_list_70 = _shape_param_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_112: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_842, [1, 0]);  view_842 = None
        permute_default_113: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_844, [1, 0]);  view_844 = None
        permute_default_114: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_846, [1, 0]);  view_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_58: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_223, rsqrt_19);  convert_element_type_223 = rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_29: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_58, torch.bfloat16);  mul_tensor_58 = None
        mul_tensor_59: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_386, convert_element_type_default_29);  add_386 = convert_element_type_default_29 = None
        sum_dim_int_list_71: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_59, [0, 1], True);  mul_tensor_59 = None
        reshape_default_99: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_99);  sum_dim_int_list_71 = _shape_param_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_115: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_849, [1, 0]);  view_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_116: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3]);  getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_100: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_116, _shape_param_100);  permute_default_116 = _shape_param_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_101: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_100, _shape_param_101);  reshape_default_100 = _shape_param_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_117: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_855, [1, 0])
        sum_dim_int_list_72: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_855, [0], True);  view_855 = None
        reshape_default_102: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_102);  sum_dim_int_list_72 = _shape_param_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_118: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_859, [1, 0])
        sum_dim_int_list_73: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_859, [0], True);  view_859 = None
        reshape_default_103: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_103);  sum_dim_int_list_73 = _shape_param_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_119: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_863, [1, 0])
        sum_dim_int_list_74: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_863, [0], True);  view_863 = None
        reshape_default_104: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, _shape_param_104);  sum_dim_int_list_74 = _shape_param_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_60: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_210, rsqrt_18);  convert_element_type_210 = rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_30: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_60, torch.bfloat16);  mul_tensor_60 = None
        mul_tensor_61: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_396, convert_element_type_default_30);  add_396 = convert_element_type_default_30 = None
        sum_dim_int_list_75: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_61, [0, 1], True);  mul_tensor_61 = None
        reshape_default_105: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_75, _shape_param_105);  sum_dim_int_list_75 = _shape_param_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_120: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_867, [1, 0]);  view_867 = None
        permute_default_121: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_869, [1, 0]);  view_869 = None
        permute_default_122: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_871, [1, 0]);  view_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_62: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_200, rsqrt_17);  convert_element_type_200 = rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_31: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_62, torch.bfloat16);  mul_tensor_62 = None
        mul_tensor_63: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_401, convert_element_type_default_31);  add_401 = convert_element_type_default_31 = None
        sum_dim_int_list_76: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_63, [0, 1], True);  mul_tensor_63 = None
        reshape_default_106: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_106);  sum_dim_int_list_76 = _shape_param_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_123: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_874, [1, 0]);  view_874 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_124: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_107: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_124, _shape_param_107);  permute_default_124 = _shape_param_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_108: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_107, _shape_param_108);  reshape_default_107 = _shape_param_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_125: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_880, [1, 0])
        sum_dim_int_list_77: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
        reshape_default_109: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_109);  sum_dim_int_list_77 = _shape_param_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_126: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_884, [1, 0])
        sum_dim_int_list_78: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_884, [0], True);  view_884 = None
        reshape_default_110: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, _shape_param_110);  sum_dim_int_list_78 = _shape_param_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_127: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_888, [1, 0])
        sum_dim_int_list_79: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_888, [0], True);  view_888 = None
        reshape_default_111: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_79, _shape_param_111);  sum_dim_int_list_79 = _shape_param_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_64: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_187, rsqrt_16);  convert_element_type_187 = rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_32: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_64, torch.bfloat16);  mul_tensor_64 = None
        mul_tensor_65: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_411, convert_element_type_default_32);  add_411 = convert_element_type_default_32 = None
        sum_dim_int_list_80: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_65, [0, 1], True);  mul_tensor_65 = None
        reshape_default_112: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_112);  sum_dim_int_list_80 = _shape_param_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_128: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_892, [1, 0]);  view_892 = None
        permute_default_129: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_894, [1, 0]);  view_894 = None
        permute_default_130: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_896, [1, 0]);  view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_66: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_177, rsqrt_15);  convert_element_type_177 = rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_33: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_66, torch.bfloat16);  mul_tensor_66 = None
        mul_tensor_67: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_416, convert_element_type_default_33);  add_416 = convert_element_type_default_33 = None
        sum_dim_int_list_81: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_67, [0, 1], True);  mul_tensor_67 = None
        reshape_default_113: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_113);  sum_dim_int_list_81 = _shape_param_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_131: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_899, [1, 0]);  view_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_132: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_63, [0, 2, 1, 3]);  getitem_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_114: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_132, _shape_param_114);  permute_default_132 = _shape_param_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_115: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_114, _shape_param_115);  reshape_default_114 = _shape_param_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_133: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_905, [1, 0])
        sum_dim_int_list_82: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
        reshape_default_116: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_116);  sum_dim_int_list_82 = _shape_param_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_134: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_909, [1, 0])
        sum_dim_int_list_83: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_909, [0], True);  view_909 = None
        reshape_default_117: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_117);  sum_dim_int_list_83 = _shape_param_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_135: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_913, [1, 0])
        sum_dim_int_list_84: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_913, [0], True);  view_913 = None
        reshape_default_118: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_84, _shape_param_118);  sum_dim_int_list_84 = _shape_param_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_68: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_164, rsqrt_14);  convert_element_type_164 = rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_34: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_68, torch.bfloat16);  mul_tensor_68 = None
        mul_tensor_69: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_426, convert_element_type_default_34);  add_426 = convert_element_type_default_34 = None
        sum_dim_int_list_85: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_69, [0, 1], True);  mul_tensor_69 = None
        reshape_default_119: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_85, _shape_param_119);  sum_dim_int_list_85 = _shape_param_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_136: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_917, [1, 0]);  view_917 = None
        permute_default_137: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_919, [1, 0]);  view_919 = None
        permute_default_138: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_921, [1, 0]);  view_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_70: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_154, rsqrt_13);  convert_element_type_154 = rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_35: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_70, torch.bfloat16);  mul_tensor_70 = None
        mul_tensor_71: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_431, convert_element_type_default_35);  add_431 = convert_element_type_default_35 = None
        sum_dim_int_list_86: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_71, [0, 1], True);  mul_tensor_71 = None
        reshape_default_120: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_120);  sum_dim_int_list_86 = _shape_param_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_139: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_924, [1, 0]);  view_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_140: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_54, [0, 2, 1, 3]);  getitem_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_121: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_140, _shape_param_121);  permute_default_140 = _shape_param_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_122: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_121, _shape_param_122);  reshape_default_121 = _shape_param_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_141: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_930, [1, 0])
        sum_dim_int_list_87: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_930, [0], True);  view_930 = None
        reshape_default_123: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_123);  sum_dim_int_list_87 = _shape_param_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_142: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_934, [1, 0])
        sum_dim_int_list_88: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_934, [0], True);  view_934 = None
        reshape_default_124: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_88, _shape_param_124);  sum_dim_int_list_88 = _shape_param_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_143: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_938, [1, 0])
        sum_dim_int_list_89: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_938, [0], True);  view_938 = None
        reshape_default_125: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_89, _shape_param_125);  sum_dim_int_list_89 = _shape_param_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_72: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_141, rsqrt_12);  convert_element_type_141 = rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_36: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_72, torch.bfloat16);  mul_tensor_72 = None
        mul_tensor_73: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_441, convert_element_type_default_36);  add_441 = convert_element_type_default_36 = None
        sum_dim_int_list_90: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_73, [0, 1], True);  mul_tensor_73 = None
        reshape_default_126: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_126);  sum_dim_int_list_90 = _shape_param_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_144: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_942, [1, 0]);  view_942 = None
        permute_default_145: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_944, [1, 0]);  view_944 = None
        permute_default_146: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_946, [1, 0]);  view_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_74: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_131, rsqrt_11);  convert_element_type_131 = rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_37: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_74, torch.bfloat16);  mul_tensor_74 = None
        mul_tensor_75: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_446, convert_element_type_default_37);  add_446 = convert_element_type_default_37 = None
        sum_dim_int_list_91: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_75, [0, 1], True);  mul_tensor_75 = None
        reshape_default_127: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_127);  sum_dim_int_list_91 = _shape_param_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_147: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_949, [1, 0]);  view_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_148: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_45, [0, 2, 1, 3]);  getitem_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_128: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_148, _shape_param_128);  permute_default_148 = _shape_param_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_129: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_128, _shape_param_129);  reshape_default_128 = _shape_param_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_149: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_955, [1, 0])
        sum_dim_int_list_92: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_955, [0], True);  view_955 = None
        reshape_default_130: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_130);  sum_dim_int_list_92 = _shape_param_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_150: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_959, [1, 0])
        sum_dim_int_list_93: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_959, [0], True);  view_959 = None
        reshape_default_131: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_131);  sum_dim_int_list_93 = _shape_param_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_151: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_963, [1, 0])
        sum_dim_int_list_94: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_963, [0], True);  view_963 = None
        reshape_default_132: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_94, _shape_param_132);  sum_dim_int_list_94 = _shape_param_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_76: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_118, rsqrt_10);  convert_element_type_118 = rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_38: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_76, torch.bfloat16);  mul_tensor_76 = None
        mul_tensor_77: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_456, convert_element_type_default_38);  add_456 = convert_element_type_default_38 = None
        sum_dim_int_list_95: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_77, [0, 1], True);  mul_tensor_77 = None
        reshape_default_133: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_133);  sum_dim_int_list_95 = _shape_param_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_152: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_967, [1, 0]);  view_967 = None
        permute_default_153: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_969, [1, 0]);  view_969 = None
        permute_default_154: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_971, [1, 0]);  view_971 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_78: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_108, rsqrt_9);  convert_element_type_108 = rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_39: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_78, torch.bfloat16);  mul_tensor_78 = None
        mul_tensor_79: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_461, convert_element_type_default_39);  add_461 = convert_element_type_default_39 = None
        sum_dim_int_list_96: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_79, [0, 1], True);  mul_tensor_79 = None
        reshape_default_134: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_134);  sum_dim_int_list_96 = _shape_param_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_155: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_974, [1, 0]);  view_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_156: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_36, [0, 2, 1, 3]);  getitem_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_135: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_156, _shape_param_135);  permute_default_156 = _shape_param_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_136: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_135, _shape_param_136);  reshape_default_135 = _shape_param_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_157: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_980, [1, 0])
        sum_dim_int_list_97: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_980, [0], True);  view_980 = None
        reshape_default_137: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_137);  sum_dim_int_list_97 = _shape_param_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_158: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_984, [1, 0])
        sum_dim_int_list_98: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_984, [0], True);  view_984 = None
        reshape_default_138: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_98, _shape_param_138);  sum_dim_int_list_98 = _shape_param_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_159: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_988, [1, 0])
        sum_dim_int_list_99: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_988, [0], True);  view_988 = None
        reshape_default_139: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_139);  sum_dim_int_list_99 = _shape_param_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_80: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_95, rsqrt_8);  convert_element_type_95 = rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_40: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_80, torch.bfloat16);  mul_tensor_80 = None
        mul_tensor_81: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_471, convert_element_type_default_40);  add_471 = convert_element_type_default_40 = None
        sum_dim_int_list_100: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_81, [0, 1], True);  mul_tensor_81 = None
        reshape_default_140: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_140);  sum_dim_int_list_100 = _shape_param_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_160: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_992, [1, 0]);  view_992 = None
        permute_default_161: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_994, [1, 0]);  view_994 = None
        permute_default_162: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_996, [1, 0]);  view_996 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_82: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_85, rsqrt_7);  convert_element_type_85 = rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_41: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_82, torch.bfloat16);  mul_tensor_82 = None
        mul_tensor_83: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_476, convert_element_type_default_41);  add_476 = convert_element_type_default_41 = None
        sum_dim_int_list_101: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_83, [0, 1], True);  mul_tensor_83 = None
        reshape_default_141: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, _shape_param_141);  sum_dim_int_list_101 = _shape_param_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_163: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_999, [1, 0]);  view_999 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_164: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_142: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_164, _shape_param_142);  permute_default_164 = _shape_param_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_143: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_142, _shape_param_143);  reshape_default_142 = _shape_param_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_165: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_1005, [1, 0])
        sum_dim_int_list_102: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1005, [0], True);  view_1005 = None
        reshape_default_144: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_144);  sum_dim_int_list_102 = _shape_param_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_166: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_1009, [1, 0])
        sum_dim_int_list_103: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1009, [0], True);  view_1009 = None
        reshape_default_145: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_145);  sum_dim_int_list_103 = _shape_param_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_167: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_1013, [1, 0])
        sum_dim_int_list_104: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_1013, [0], True);  view_1013 = None
        reshape_default_146: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_146);  sum_dim_int_list_104 = _shape_param_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_84: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_72, rsqrt_6);  convert_element_type_72 = rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_42: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_84, torch.bfloat16);  mul_tensor_84 = None
        mul_tensor_85: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_486, convert_element_type_default_42);  add_486 = convert_element_type_default_42 = None
        sum_dim_int_list_105: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_85, [0, 1], True);  mul_tensor_85 = None
        reshape_default_147: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_105, _shape_param_147);  sum_dim_int_list_105 = _shape_param_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_168: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_1017, [1, 0]);  view_1017 = None
        permute_default_169: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_1019, [1, 0]);  view_1019 = None
        permute_default_170: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_1021, [1, 0]);  view_1021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_86: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_62, rsqrt_5);  convert_element_type_62 = rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_43: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_86, torch.bfloat16);  mul_tensor_86 = None
        mul_tensor_87: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_491, convert_element_type_default_43);  add_491 = convert_element_type_default_43 = None
        sum_dim_int_list_106: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_87, [0, 1], True);  mul_tensor_87 = None
        reshape_default_148: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, _shape_param_148);  sum_dim_int_list_106 = _shape_param_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_171: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_1024, [1, 0]);  view_1024 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_172: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_18, [0, 2, 1, 3]);  getitem_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_149: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_172, _shape_param_149);  permute_default_172 = _shape_param_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_150: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_149, _shape_param_150);  reshape_default_149 = _shape_param_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_173: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_1030, [1, 0])
        sum_dim_int_list_107: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1030, [0], True);  view_1030 = None
        reshape_default_151: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_151);  sum_dim_int_list_107 = _shape_param_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_174: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_1034, [1, 0])
        sum_dim_int_list_108: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True);  view_1034 = None
        reshape_default_152: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_152);  sum_dim_int_list_108 = _shape_param_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_175: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_1038, [1, 0])
        sum_dim_int_list_109: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_1038, [0], True);  view_1038 = None
        reshape_default_153: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_109, _shape_param_153);  sum_dim_int_list_109 = _shape_param_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_88: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_49, rsqrt_4);  convert_element_type_49 = rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_44: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_88, torch.bfloat16);  mul_tensor_88 = None
        mul_tensor_89: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_501, convert_element_type_default_44);  add_501 = convert_element_type_default_44 = None
        sum_dim_int_list_110: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_89, [0, 1], True);  mul_tensor_89 = None
        reshape_default_154: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_154);  sum_dim_int_list_110 = _shape_param_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_176: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_1042, [1, 0]);  view_1042 = None
        permute_default_177: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_1044, [1, 0]);  view_1044 = None
        permute_default_178: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_1046, [1, 0]);  view_1046 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_90: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_39, rsqrt_3);  convert_element_type_39 = rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_45: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_90, torch.bfloat16);  mul_tensor_90 = None
        mul_tensor_91: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_506, convert_element_type_default_45);  add_506 = convert_element_type_default_45 = None
        sum_dim_int_list_111: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_91, [0, 1], True);  mul_tensor_91 = None
        reshape_default_155: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_155);  sum_dim_int_list_111 = _shape_param_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_179: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_1049, [1, 0]);  view_1049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_180: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem_9, [0, 2, 1, 3]);  getitem_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_156: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_180, _shape_param_156);  permute_default_180 = _shape_param_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_157: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_156, _shape_param_157);  reshape_default_156 = _shape_param_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_181: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_1055, [1, 0])
        sum_dim_int_list_112: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1055, [0], True);  view_1055 = None
        reshape_default_158: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_158);  sum_dim_int_list_112 = _shape_param_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_182: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_1059, [1, 0])
        sum_dim_int_list_113: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1059, [0], True);  view_1059 = None
        reshape_default_159: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_159);  sum_dim_int_list_113 = _shape_param_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_183: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_1063, [1, 0])
        sum_dim_int_list_114: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_1063, [0], True);  view_1063 = None
        reshape_default_160: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_114, _shape_param_160);  sum_dim_int_list_114 = _shape_param_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_92: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_26, rsqrt_2);  convert_element_type_26 = rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_46: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_92, torch.bfloat16);  mul_tensor_92 = None
        mul_tensor_93: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_516, convert_element_type_default_46);  add_516 = convert_element_type_default_46 = None
        sum_dim_int_list_115: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_93, [0, 1], True);  mul_tensor_93 = None
        reshape_default_161: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_115, _shape_param_161);  sum_dim_int_list_115 = _shape_param_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:47 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_default_184: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_1067, [1, 0]);  view_1067 = None
        permute_default_185: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_1069, [1, 0]);  view_1069 = None
        permute_default_186: "bf16[4864, 2048]" = torch.ops.aten.permute.default(view_1071, [1, 0]);  view_1071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_94: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_16, rsqrt_1);  convert_element_type_16 = rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_47: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_94, torch.bfloat16);  mul_tensor_94 = None
        mul_tensor_95: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_521, convert_element_type_default_47);  add_521 = convert_element_type_default_47 = None
        sum_dim_int_list_116: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_95, [0, 1], True);  mul_tensor_95 = None
        reshape_default_162: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_162);  sum_dim_int_list_116 = _shape_param_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        permute_default_187: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_1074, [1, 0]);  view_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_default_188: "bf16[4, 512, 14, 64]" = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:243 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_default_163: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(permute_default_188, _shape_param_163);  permute_default_188 = _shape_param_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:244 in forward, code: attn_output = self.o_proj(attn_output)
        reshape_default_164: "bf16[2048, 896]" = torch.ops.aten.reshape.default(reshape_default_163, _shape_param_164);  reshape_default_163 = _shape_param_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:219 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_189: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_1080, [1, 0])
        sum_dim_int_list_117: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1080, [0], True);  view_1080 = None
        reshape_default_165: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_165);  sum_dim_int_list_117 = _shape_param_165 = None
        reshape_default_166: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(mm_429, _shape_param_166);  mm_429 = _shape_param_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:218 in forward, code: key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_190: "bf16[128, 2048]" = torch.ops.aten.permute.default(view_1084, [1, 0])
        sum_dim_int_list_118: "bf16[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1084, [0], True);  view_1084 = None
        reshape_default_167: "bf16[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_118, _shape_param_167);  sum_dim_int_list_118 = _shape_param_167 = None
        reshape_default_168: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(mm_431, _shape_param_168);  mm_431 = _shape_param_168 = None
        add_tensor: "bf16[4, 512, 896]" = torch.ops.aten.add.Tensor(reshape_default_166, reshape_default_168);  reshape_default_166 = reshape_default_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:217 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_default_191: "bf16[896, 2048]" = torch.ops.aten.permute.default(view_1088, [1, 0])
        sum_dim_int_list_119: "bf16[1, 896]" = torch.ops.aten.sum.dim_IntList(view_1088, [0], True);  view_1088 = None
        reshape_default_169: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, _shape_param_169);  sum_dim_int_list_119 = _shape_param_169 = None
        reshape_default_170: "bf16[4, 512, 896]" = torch.ops.aten.reshape.default(mm_433, _shape_param_170);  mm_433 = _shape_param_170 = None
        add_tensor_1: "bf16[4, 512, 896]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_170);  add_tensor = reshape_default_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        mul_tensor_96: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_tensor_1, primals_4);  primals_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:260 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_48: "f32[4, 512, 896]" = torch.ops.prims.convert_element_type.default(embedding, torch.float32);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_97: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_default_48, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:263 in forward, code: return self.weight * hidden_states.to(input_dtype)
        convert_element_type_default_49: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_97, torch.bfloat16);  mul_tensor_97 = None
        mul_tensor_98: "bf16[4, 512, 896]" = torch.ops.aten.mul.Tensor(add_tensor_1, convert_element_type_default_49);  add_tensor_1 = convert_element_type_default_49 = None
        sum_dim_int_list_120: "bf16[1, 1, 896]" = torch.ops.aten.sum.dim_IntList(mul_tensor_98, [0, 1], True);  mul_tensor_98 = None
        reshape_default_171: "bf16[896]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, _shape_param_171);  sum_dim_int_list_120 = _shape_param_171 = None
        convert_element_type_default_50: "f32[4, 512, 896]" = torch.ops.prims.convert_element_type.default(mul_tensor_96, torch.float32);  mul_tensor_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:262 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_tensor_99: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_default_50, convert_element_type_default_48)
        mul_tensor_100: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(convert_element_type_default_50, rsqrt);  convert_element_type_default_50 = None
        sum_dim_int_list_121: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_99, [2], True);  mul_tensor_99 = None
        pow_tensor_scalar: "f32[4, 512, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_scalar: "f32[4, 512, 1]" = torch.ops.aten.mul.Scalar(sum_dim_int_list_121, -0.5);  sum_dim_int_list_121 = None
        mul_tensor_101: "f32[4, 512, 1]" = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:261 in forward, code: variance = hidden_states.pow(2).mean(-1, keepdim=True)
        expand_default: "f32[4, 512, 896]" = torch.ops.aten.expand.default(mul_tensor_101, _shape_param_172);  mul_tensor_101 = _shape_param_172 = None
        div_scalar: "f32[4, 512, 896]" = torch.ops.aten.div.Scalar(expand_default, 896);  expand_default = None
        pow_tensor_scalar_1: "f32[4, 512, 896]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_48, 1.0);  convert_element_type_default_48 = None
        mul_scalar_1: "f32[4, 512, 896]" = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_102: "f32[4, 512, 896]" = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_2: "f32[4, 512, 896]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_102);  mul_tensor_100 = mul_tensor_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:260 in forward, code: hidden_states = hidden_states.to(torch.float32)
        convert_element_type_default_51: "bf16[4, 512, 896]" = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        add_tensor_3: "bf16[4, 512, 896]" = torch.ops.aten.add.Tensor(add_523, convert_element_type_default_51);  add_523 = convert_element_type_default_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/qwen2/modeling_qwen2.py:367 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        convert_element_type_default_52: "f32[4, 512, 896]" = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32);  add_tensor_3 = None
        eq_scalar: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_default: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[4, 512, 896]" = torch.ops.aten.where.self(unsqueeze_default, full_default_49, convert_element_type_default_52);  unsqueeze_default = full_default_49 = convert_element_type_default_52 = None
        full_default: "f32[151936, 896]" = torch.ops.aten.full.default([151936, 896], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[151936, 896]" = torch.ops.aten.index_put.default(full_default, [primals_1], where_self, True);  full_default = primals_1 = where_self = None
        convert_element_type_default_53: "bf16[151936, 896]" = torch.ops.prims.convert_element_type.default(index_put_default, torch.bfloat16);  index_put_default = None
        add_tensor_4: "bf16[151936, 896]" = torch.ops.aten.add.Tensor(mm_97, convert_element_type_default_53);  mm_97 = convert_element_type_default_53 = None
        return (reshape_default, permute_default, permute_default_1, permute_default_2, reshape_default_1, permute_default_3, reshape_default_3, permute_default_5, reshape_default_4, permute_default_6, reshape_default_5, permute_default_7, reshape_default_6, reshape_default_7, permute_default_8, permute_default_9, permute_default_10, reshape_default_8, permute_default_11, reshape_default_10, permute_default_13, reshape_default_11, permute_default_14, reshape_default_12, permute_default_15, reshape_default_13, reshape_default_14, permute_default_16, permute_default_17, permute_default_18, reshape_default_15, permute_default_19, reshape_default_17, permute_default_21, reshape_default_18, permute_default_22, reshape_default_19, permute_default_23, reshape_default_20, reshape_default_21, permute_default_24, permute_default_25, permute_default_26, reshape_default_22, permute_default_27, reshape_default_24, permute_default_29, reshape_default_25, permute_default_30, reshape_default_26, permute_default_31, reshape_default_27, reshape_default_28, permute_default_32, permute_default_33, permute_default_34, reshape_default_29, permute_default_35, reshape_default_31, permute_default_37, reshape_default_32, permute_default_38, reshape_default_33, permute_default_39, reshape_default_34, reshape_default_35, permute_default_40, permute_default_41, permute_default_42, reshape_default_36, permute_default_43, reshape_default_38, permute_default_45, reshape_default_39, permute_default_46, reshape_default_40, permute_default_47, reshape_default_41, reshape_default_42, permute_default_48, permute_default_49, permute_default_50, reshape_default_43, permute_default_51, reshape_default_45, permute_default_53, reshape_default_46, permute_default_54, reshape_default_47, permute_default_55, reshape_default_48, reshape_default_49, permute_default_56, permute_default_57, permute_default_58, reshape_default_50, permute_default_59, reshape_default_52, permute_default_61, reshape_default_53, permute_default_62, reshape_default_54, permute_default_63, reshape_default_55, reshape_default_56, permute_default_64, permute_default_65, permute_default_66, reshape_default_57, permute_default_67, reshape_default_59, permute_default_69, reshape_default_60, permute_default_70, reshape_default_61, permute_default_71, reshape_default_62, reshape_default_63, permute_default_72, permute_default_73, permute_default_74, reshape_default_64, permute_default_75, reshape_default_66, permute_default_77, reshape_default_67, permute_default_78, reshape_default_68, permute_default_79, reshape_default_69, reshape_default_70, permute_default_80, permute_default_81, permute_default_82, reshape_default_71, permute_default_83, reshape_default_73, permute_default_85, reshape_default_74, permute_default_86, reshape_default_75, permute_default_87, reshape_default_76, reshape_default_77, permute_default_88, permute_default_89, permute_default_90, reshape_default_78, permute_default_91, reshape_default_80, permute_default_93, reshape_default_81, permute_default_94, reshape_default_82, permute_default_95, reshape_default_83, reshape_default_84, permute_default_96, permute_default_97, permute_default_98, reshape_default_85, permute_default_99, reshape_default_87, permute_default_101, reshape_default_88, permute_default_102, reshape_default_89, permute_default_103, reshape_default_90, reshape_default_91, permute_default_104, permute_default_105, permute_default_106, reshape_default_92, permute_default_107, reshape_default_94, permute_default_109, reshape_default_95, permute_default_110, reshape_default_96, permute_default_111, reshape_default_97, reshape_default_98, permute_default_112, permute_default_113, permute_default_114, reshape_default_99, permute_default_115, reshape_default_101, permute_default_117, reshape_default_102, permute_default_118, reshape_default_103, permute_default_119, reshape_default_104, reshape_default_105, permute_default_120, permute_default_121, permute_default_122, reshape_default_106, permute_default_123, reshape_default_108, permute_default_125, reshape_default_109, permute_default_126, reshape_default_110, permute_default_127, reshape_default_111, reshape_default_112, permute_default_128, permute_default_129, permute_default_130, reshape_default_113, permute_default_131, reshape_default_115, permute_default_133, reshape_default_116, permute_default_134, reshape_default_117, permute_default_135, reshape_default_118, reshape_default_119, permute_default_136, permute_default_137, permute_default_138, reshape_default_120, permute_default_139, reshape_default_122, permute_default_141, reshape_default_123, permute_default_142, reshape_default_124, permute_default_143, reshape_default_125, reshape_default_126, permute_default_144, permute_default_145, permute_default_146, reshape_default_127, permute_default_147, reshape_default_129, permute_default_149, reshape_default_130, permute_default_150, reshape_default_131, permute_default_151, reshape_default_132, reshape_default_133, permute_default_152, permute_default_153, permute_default_154, reshape_default_134, permute_default_155, reshape_default_136, permute_default_157, reshape_default_137, permute_default_158, reshape_default_138, permute_default_159, reshape_default_139, reshape_default_140, permute_default_160, permute_default_161, permute_default_162, reshape_default_141, permute_default_163, reshape_default_143, permute_default_165, reshape_default_144, permute_default_166, reshape_default_145, permute_default_167, reshape_default_146, reshape_default_147, permute_default_168, permute_default_169, permute_default_170, reshape_default_148, permute_default_171, reshape_default_150, permute_default_173, reshape_default_151, permute_default_174, reshape_default_152, permute_default_175, reshape_default_153, reshape_default_154, permute_default_176, permute_default_177, permute_default_178, reshape_default_155, permute_default_179, reshape_default_157, permute_default_181, reshape_default_158, permute_default_182, reshape_default_159, permute_default_183, reshape_default_160, reshape_default_161, permute_default_184, permute_default_185, permute_default_186, reshape_default_162, permute_default_187, reshape_default_164, permute_default_189, reshape_default_165, permute_default_190, reshape_default_167, permute_default_191, reshape_default_169, reshape_default_171, add_tensor_4)


def _default_make_inputs():
    return [
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_0
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_1
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_207
    [4, 512, -1],  # _shape_param_2
    [2048, 896],  # _shape_param_3
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_4
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_5
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_6
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_7
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_8
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_198
    [4, 512, -1],  # _shape_param_9
    [2048, 896],  # _shape_param_10
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_11
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_12
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_13
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_14
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_15
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_189
    [4, 512, -1],  # _shape_param_16
    [2048, 896],  # _shape_param_17
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_18
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_19
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_20
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_21
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_22
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_180
    [4, 512, -1],  # _shape_param_23
    [2048, 896],  # _shape_param_24
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_25
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_26
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_27
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_28
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_29
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_171
    [4, 512, -1],  # _shape_param_30
    [2048, 896],  # _shape_param_31
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_32
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_33
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_34
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_35
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_36
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_162
    [4, 512, -1],  # _shape_param_37
    [2048, 896],  # _shape_param_38
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_39
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_40
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_41
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_42
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_43
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_153
    [4, 512, -1],  # _shape_param_44
    [2048, 896],  # _shape_param_45
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_46
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_47
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_48
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_49
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_50
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_144
    [4, 512, -1],  # _shape_param_51
    [2048, 896],  # _shape_param_52
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_53
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_54
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_55
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_56
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_57
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_135
    [4, 512, -1],  # _shape_param_58
    [2048, 896],  # _shape_param_59
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_60
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_61
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_62
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_63
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_64
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_126
    [4, 512, -1],  # _shape_param_65
    [2048, 896],  # _shape_param_66
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_67
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_68
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_69
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_70
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_71
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_117
    [4, 512, -1],  # _shape_param_72
    [2048, 896],  # _shape_param_73
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_74
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_75
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_76
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_77
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_78
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_108
    [4, 512, -1],  # _shape_param_79
    [2048, 896],  # _shape_param_80
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_81
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_82
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_83
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_84
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_85
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_99
    [4, 512, -1],  # _shape_param_86
    [2048, 896],  # _shape_param_87
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_88
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_89
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_90
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_91
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_92
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_90
    [4, 512, -1],  # _shape_param_93
    [2048, 896],  # _shape_param_94
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_95
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_96
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_97
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_98
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_99
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_81
    [4, 512, -1],  # _shape_param_100
    [2048, 896],  # _shape_param_101
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_102
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_103
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_104
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_105
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_106
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_72
    [4, 512, -1],  # _shape_param_107
    [2048, 896],  # _shape_param_108
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_109
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_110
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_111
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_112
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_113
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_63
    [4, 512, -1],  # _shape_param_114
    [2048, 896],  # _shape_param_115
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_116
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_117
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_118
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_119
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_120
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_54
    [4, 512, -1],  # _shape_param_121
    [2048, 896],  # _shape_param_122
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_123
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_124
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_125
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_126
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_127
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_45
    [4, 512, -1],  # _shape_param_128
    [2048, 896],  # _shape_param_129
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_130
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_131
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_132
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_133
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_134
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_36
    [4, 512, -1],  # _shape_param_135
    [2048, 896],  # _shape_param_136
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_137
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_138
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_139
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_140
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_141
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_27
    [4, 512, -1],  # _shape_param_142
    [2048, 896],  # _shape_param_143
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_144
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_145
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_146
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_147
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_148
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_18
    [4, 512, -1],  # _shape_param_149
    [2048, 896],  # _shape_param_150
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_151
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_152
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_153
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_154
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_155
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem_9
    [4, 512, -1],  # _shape_param_156
    [2048, 896],  # _shape_param_157
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_158
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_159
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_160
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_161
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([2048, 4864], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_162
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn(1835008, dtype=torch.bfloat16, device='cuda').as_strided([4, 14, 512, 64], [458752, 64, 896, 1]),  # getitem
    [4, 512, -1],  # _shape_param_163
    [2048, 896],  # _shape_param_164
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_165
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 896],  # _shape_param_166
    torch.randn([2048, 128], dtype=torch.bfloat16, device='cuda'),
    [128],  # _shape_param_167
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 896],  # _shape_param_168
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [896],  # _shape_param_169
    torch.randn([2048, 896], dtype=torch.bfloat16, device='cuda'),
    [4, 512, 896],  # _shape_param_170
    torch.randn([896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randn([4, 512, 1], dtype=torch.float32, device='cuda'),
    [896],  # _shape_param_171
    [4, 512, 896],  # _shape_param_172
    torch.randn([4, 512, 896], dtype=torch.bfloat16, device='cuda'),
    torch.randint(0, 2, [4, 512], dtype=torch.int64, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([151936, 896], dtype=torch.bfloat16, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
