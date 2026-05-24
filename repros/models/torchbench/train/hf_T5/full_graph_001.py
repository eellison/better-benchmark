import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024]", primals_2: "f32[32128, 512]", primals_3: "f32[512]", primals_4: "f32[512, 512]", primals_5: "f32[512, 512]", primals_6: "f32[512, 512]", primals_8: "f32[512, 512]", primals_9: "f32[512]", primals_10: "f32[2048, 512]", primals_11: "f32[512, 2048]", primals_12: "f32[512]", primals_13: "f32[512, 512]", primals_14: "f32[512, 512]", primals_15: "f32[512, 512]", primals_16: "f32[512, 512]", primals_17: "f32[512]", primals_18: "f32[2048, 512]", primals_19: "f32[512, 2048]", primals_20: "f32[512]", primals_21: "f32[512, 512]", primals_22: "f32[512, 512]", primals_23: "f32[512, 512]", primals_24: "f32[512, 512]", primals_25: "f32[512]", primals_26: "f32[2048, 512]", primals_27: "f32[512, 2048]", primals_28: "f32[512]", primals_29: "f32[512, 512]", primals_30: "f32[512, 512]", primals_31: "f32[512, 512]", primals_32: "f32[512, 512]", primals_33: "f32[512]", primals_34: "f32[2048, 512]", primals_35: "f32[512, 2048]", primals_36: "f32[512]", primals_37: "f32[512, 512]", primals_38: "f32[512, 512]", primals_39: "f32[512, 512]", primals_40: "f32[512, 512]", primals_41: "f32[512]", primals_42: "f32[2048, 512]", primals_43: "f32[512, 2048]", primals_44: "f32[512]", primals_45: "f32[512, 512]", primals_46: "f32[512, 512]", primals_47: "f32[512, 512]", primals_48: "f32[512, 512]", primals_49: "f32[512]", primals_50: "f32[2048, 512]", primals_51: "f32[512, 2048]", primals_52: "f32[512]", primals_53: "i64[8, 1024]", primals_54: "f32[512]", primals_55: "f32[512, 512]", primals_56: "f32[512, 512]", primals_57: "f32[512, 512]", primals_59: "f32[512, 512]", primals_60: "f32[512]", primals_61: "f32[512, 512]", primals_62: "f32[512, 512]", primals_63: "f32[512, 512]", primals_64: "f32[512, 512]", primals_65: "f32[512]", primals_66: "f32[2048, 512]", primals_67: "f32[512, 2048]", primals_68: "f32[512]", primals_69: "f32[512, 512]", primals_70: "f32[512, 512]", primals_71: "f32[512, 512]", primals_72: "f32[512, 512]", primals_73: "f32[512]", primals_74: "f32[512, 512]", primals_75: "f32[512, 512]", primals_76: "f32[512, 512]", primals_77: "f32[512, 512]", primals_78: "f32[512]", primals_79: "f32[2048, 512]", primals_80: "f32[512, 2048]", primals_81: "f32[512]", primals_82: "f32[512, 512]", primals_83: "f32[512, 512]", primals_84: "f32[512, 512]", primals_85: "f32[512, 512]", primals_86: "f32[512]", primals_87: "f32[512, 512]", primals_88: "f32[512, 512]", primals_89: "f32[512, 512]", primals_90: "f32[512, 512]", primals_91: "f32[512]", primals_92: "f32[2048, 512]", primals_93: "f32[512, 2048]", primals_94: "f32[512]", primals_95: "f32[512, 512]", primals_96: "f32[512, 512]", primals_97: "f32[512, 512]", primals_98: "f32[512, 512]", primals_99: "f32[512]", primals_100: "f32[512, 512]", primals_101: "f32[512, 512]", primals_102: "f32[512, 512]", primals_103: "f32[512, 512]", primals_104: "f32[512]", primals_105: "f32[2048, 512]", primals_106: "f32[512, 2048]", primals_107: "f32[512]", primals_108: "f32[512, 512]", primals_109: "f32[512, 512]", primals_110: "f32[512, 512]", primals_111: "f32[512, 512]", primals_112: "f32[512]", primals_113: "f32[512, 512]", primals_114: "f32[512, 512]", primals_115: "f32[512, 512]", primals_116: "f32[512, 512]", primals_117: "f32[512]", primals_118: "f32[2048, 512]", primals_119: "f32[512, 2048]", primals_120: "f32[512]", primals_121: "f32[512, 512]", primals_122: "f32[512, 512]", primals_123: "f32[512, 512]", primals_124: "f32[512, 512]", primals_125: "f32[512]", primals_126: "f32[512, 512]", primals_127: "f32[512, 512]", primals_128: "f32[512, 512]", primals_129: "f32[512, 512]", primals_130: "f32[512]", primals_131: "f32[2048, 512]", primals_132: "f32[512, 2048]", primals_133: "f32[512]", embedding: "f32[8, 1024, 512]", ge: "b8[1, 1, 1024, 1]", gt: "b8[8, 1024, 512]", rsqrt: "f32[8, 1024, 1]", view_1: "f32[8192, 512]", bmm: "f32[64, 1024, 1024]", add_6: "i64[1024, 1024]", embedding_1: "f32[1024, 1024, 8]", amax: "f32[8, 8, 1024, 1]", sum_1: "f32[8, 8, 1024, 1]", gt_2: "b8[8, 8, 1024, 1024]", view_20: "f32[8192, 512]", gt_3: "b8[8, 1024, 512]", add_9: "f32[8, 1024, 512]", rsqrt_1: "f32[8, 1024, 1]", view_22: "f32[8192, 512]", gt_4: "b8[8, 1024, 2048]", view_24: "f32[8192, 2048]", gt_5: "b8[8, 1024, 512]", add_11: "f32[8, 1024, 512]", rsqrt_2: "f32[8, 1024, 1]", view_26: "f32[8192, 512]", div_3: "f32[8, 8, 1024, 1024]", gt_6: "b8[8, 8, 1024, 1024]", view_45: "f32[8192, 512]", gt_7: "b8[8, 1024, 512]", add_14: "f32[8, 1024, 512]", rsqrt_3: "f32[8, 1024, 1]", view_47: "f32[8192, 512]", gt_8: "b8[8, 1024, 2048]", view_49: "f32[8192, 2048]", gt_9: "b8[8, 1024, 512]", add_16: "f32[8, 1024, 512]", rsqrt_4: "f32[8, 1024, 1]", view_51: "f32[8192, 512]", div_4: "f32[8, 8, 1024, 1024]", gt_10: "b8[8, 8, 1024, 1024]", view_70: "f32[8192, 512]", gt_11: "b8[8, 1024, 512]", add_19: "f32[8, 1024, 512]", rsqrt_5: "f32[8, 1024, 1]", view_72: "f32[8192, 512]", gt_12: "b8[8, 1024, 2048]", view_74: "f32[8192, 2048]", gt_13: "b8[8, 1024, 512]", add_21: "f32[8, 1024, 512]", rsqrt_6: "f32[8, 1024, 1]", view_76: "f32[8192, 512]", div_5: "f32[8, 8, 1024, 1024]", gt_14: "b8[8, 8, 1024, 1024]", view_95: "f32[8192, 512]", gt_15: "b8[8, 1024, 512]", add_24: "f32[8, 1024, 512]", rsqrt_7: "f32[8, 1024, 1]", view_97: "f32[8192, 512]", gt_16: "b8[8, 1024, 2048]", view_99: "f32[8192, 2048]", gt_17: "b8[8, 1024, 512]", add_26: "f32[8, 1024, 512]", rsqrt_8: "f32[8, 1024, 1]", view_101: "f32[8192, 512]", div_6: "f32[8, 8, 1024, 1024]", gt_18: "b8[8, 8, 1024, 1024]", view_120: "f32[8192, 512]", gt_19: "b8[8, 1024, 512]", add_29: "f32[8, 1024, 512]", rsqrt_9: "f32[8, 1024, 1]", view_122: "f32[8192, 512]", gt_20: "b8[8, 1024, 2048]", view_124: "f32[8192, 2048]", gt_21: "b8[8, 1024, 512]", add_31: "f32[8, 1024, 512]", rsqrt_10: "f32[8, 1024, 1]", view_126: "f32[8192, 512]", div_7: "f32[8, 8, 1024, 1024]", gt_22: "b8[8, 8, 1024, 1024]", view_145: "f32[8192, 512]", gt_23: "b8[8, 1024, 512]", add_34: "f32[8, 1024, 512]", rsqrt_11: "f32[8, 1024, 1]", view_147: "f32[8192, 512]", gt_24: "b8[8, 1024, 2048]", view_149: "f32[8192, 2048]", gt_25: "b8[8, 1024, 512]", add_36: "f32[8, 1024, 512]", rsqrt_12: "f32[8, 1024, 1]", gt_26: "b8[8, 1024, 512]", mul_79: "f32[8, 1024, 512]", where_2: "i64[8, 1024]", embedding_2: "f32[8, 1024, 512]", gt_27: "b8[8, 1024, 512]", rsqrt_13: "f32[8, 1024, 1]", view_153: "f32[8192, 512]", add_45: "i64[1024, 1024]", div_10: "f32[8, 8, 1024, 1024]", gt_28: "b8[8, 8, 1024, 1024]", view_172: "f32[8192, 512]", gt_29: "b8[8, 1024, 512]", add_48: "f32[8, 1024, 512]", rsqrt_14: "f32[8, 1024, 1]", view_174: "f32[8192, 512]", div_11: "f32[8, 8, 1024, 1024]", gt_30: "b8[8, 8, 1024, 1024]", view_193: "f32[8192, 512]", gt_31: "b8[8, 1024, 512]", add_52: "f32[8, 1024, 512]", rsqrt_15: "f32[8, 1024, 1]", view_195: "f32[8192, 512]", gt_32: "b8[8, 1024, 2048]", view_197: "f32[8192, 2048]", gt_33: "b8[8, 1024, 512]", add_54: "f32[8, 1024, 512]", rsqrt_16: "f32[8, 1024, 1]", view_199: "f32[8192, 512]", div_12: "f32[8, 8, 1024, 1024]", gt_34: "b8[8, 8, 1024, 1024]", view_218: "f32[8192, 512]", gt_35: "b8[8, 1024, 512]", add_57: "f32[8, 1024, 512]", rsqrt_17: "f32[8, 1024, 1]", view_220: "f32[8192, 512]", div_13: "f32[8, 8, 1024, 1024]", gt_36: "b8[8, 8, 1024, 1024]", view_239: "f32[8192, 512]", gt_37: "b8[8, 1024, 512]", add_60: "f32[8, 1024, 512]", rsqrt_18: "f32[8, 1024, 1]", view_241: "f32[8192, 512]", gt_38: "b8[8, 1024, 2048]", view_243: "f32[8192, 2048]", gt_39: "b8[8, 1024, 512]", add_62: "f32[8, 1024, 512]", rsqrt_19: "f32[8, 1024, 1]", view_245: "f32[8192, 512]", div_14: "f32[8, 8, 1024, 1024]", gt_40: "b8[8, 8, 1024, 1024]", view_264: "f32[8192, 512]", gt_41: "b8[8, 1024, 512]", add_65: "f32[8, 1024, 512]", rsqrt_20: "f32[8, 1024, 1]", view_266: "f32[8192, 512]", div_15: "f32[8, 8, 1024, 1024]", gt_42: "b8[8, 8, 1024, 1024]", view_285: "f32[8192, 512]", gt_43: "b8[8, 1024, 512]", add_68: "f32[8, 1024, 512]", rsqrt_21: "f32[8, 1024, 1]", view_287: "f32[8192, 512]", gt_44: "b8[8, 1024, 2048]", view_289: "f32[8192, 2048]", gt_45: "b8[8, 1024, 512]", add_70: "f32[8, 1024, 512]", rsqrt_22: "f32[8, 1024, 1]", view_291: "f32[8192, 512]", div_16: "f32[8, 8, 1024, 1024]", gt_46: "b8[8, 8, 1024, 1024]", view_310: "f32[8192, 512]", gt_47: "b8[8, 1024, 512]", add_73: "f32[8, 1024, 512]", rsqrt_23: "f32[8, 1024, 1]", view_312: "f32[8192, 512]", div_17: "f32[8, 8, 1024, 1024]", gt_48: "b8[8, 8, 1024, 1024]", view_331: "f32[8192, 512]", gt_49: "b8[8, 1024, 512]", add_76: "f32[8, 1024, 512]", rsqrt_24: "f32[8, 1024, 1]", view_333: "f32[8192, 512]", gt_50: "b8[8, 1024, 2048]", view_335: "f32[8192, 2048]", gt_51: "b8[8, 1024, 512]", add_78: "f32[8, 1024, 512]", rsqrt_25: "f32[8, 1024, 1]", view_337: "f32[8192, 512]", div_18: "f32[8, 8, 1024, 1024]", gt_52: "b8[8, 8, 1024, 1024]", view_356: "f32[8192, 512]", gt_53: "b8[8, 1024, 512]", add_81: "f32[8, 1024, 512]", rsqrt_26: "f32[8, 1024, 1]", view_358: "f32[8192, 512]", div_19: "f32[8, 8, 1024, 1024]", gt_54: "b8[8, 8, 1024, 1024]", view_377: "f32[8192, 512]", gt_55: "b8[8, 1024, 512]", add_84: "f32[8, 1024, 512]", rsqrt_27: "f32[8, 1024, 1]", view_379: "f32[8192, 512]", gt_56: "b8[8, 1024, 2048]", view_381: "f32[8192, 2048]", gt_57: "b8[8, 1024, 512]", add_86: "f32[8, 1024, 512]", rsqrt_28: "f32[8, 1024, 1]", view_383: "f32[8192, 512]", div_20: "f32[8, 8, 1024, 1024]", gt_58: "b8[8, 8, 1024, 1024]", view_402: "f32[8192, 512]", gt_59: "b8[8, 1024, 512]", add_89: "f32[8, 1024, 512]", rsqrt_29: "f32[8, 1024, 1]", view_404: "f32[8192, 512]", div_21: "f32[8, 8, 1024, 1024]", gt_60: "b8[8, 8, 1024, 1024]", view_423: "f32[8192, 512]", gt_61: "b8[8, 1024, 512]", add_92: "f32[8, 1024, 512]", rsqrt_30: "f32[8, 1024, 1]", view_425: "f32[8192, 512]", gt_62: "b8[8, 1024, 2048]", view_427: "f32[8192, 2048]", gt_63: "b8[8, 1024, 512]", add_94: "f32[8, 1024, 512]", rsqrt_31: "f32[8, 1024, 1]", gt_64: "b8[8, 1024, 512]", view_429: "f32[8192, 512]", view_430: "f32[8, 1024, 32128]", amax_18: "f32[8192, 1]", log_2: "f32[8192, 1]", convert_element_type_5: "f32[]", le_1: "b8[8, 1024, 2048]", permute_206: "f32[64, 1024, 1024]", permute_207: "f32[64, 64, 1024]", permute_208: "f32[64, 64, 1024]", permute_209: "f32[64, 1024, 64]", permute_231: "f32[64, 1024, 1024]", permute_232: "f32[64, 64, 1024]", permute_233: "f32[64, 64, 1024]", permute_234: "f32[64, 1024, 64]", le_2: "b8[8, 1024, 2048]", permute_264: "f32[64, 1024, 1024]", permute_265: "f32[64, 64, 1024]", permute_266: "f32[64, 64, 1024]", permute_267: "f32[64, 1024, 64]", permute_289: "f32[64, 1024, 1024]", permute_290: "f32[64, 64, 1024]", permute_291: "f32[64, 64, 1024]", permute_292: "f32[64, 1024, 64]", le_3: "b8[8, 1024, 2048]", permute_322: "f32[64, 1024, 1024]", permute_323: "f32[64, 64, 1024]", permute_324: "f32[64, 64, 1024]", permute_325: "f32[64, 1024, 64]", permute_347: "f32[64, 1024, 1024]", permute_348: "f32[64, 64, 1024]", permute_349: "f32[64, 64, 1024]", permute_350: "f32[64, 1024, 64]", le_4: "b8[8, 1024, 2048]", permute_380: "f32[64, 1024, 1024]", permute_381: "f32[64, 64, 1024]", permute_382: "f32[64, 64, 1024]", permute_383: "f32[64, 1024, 64]", permute_405: "f32[64, 1024, 1024]", permute_406: "f32[64, 64, 1024]", permute_407: "f32[64, 64, 1024]", permute_408: "f32[64, 1024, 64]", le_5: "b8[8, 1024, 2048]", permute_438: "f32[64, 1024, 1024]", permute_439: "f32[64, 64, 1024]", permute_440: "f32[64, 64, 1024]", permute_441: "f32[64, 1024, 64]", permute_463: "f32[64, 1024, 1024]", permute_464: "f32[64, 64, 1024]", permute_465: "f32[64, 64, 1024]", permute_466: "f32[64, 1024, 64]", le_6: "b8[8, 1024, 2048]", permute_496: "f32[64, 1024, 1024]", permute_497: "f32[64, 64, 1024]", permute_498: "f32[64, 64, 1024]", permute_499: "f32[64, 1024, 64]", permute_521: "f32[64, 1024, 1024]", permute_522: "f32[64, 64, 1024]", permute_524: "f32[64, 64, 1024]", permute_525: "f32[64, 1024, 64]", le_7: "b8[8, 1024, 2048]", permute_555: "f32[64, 1024, 1024]", permute_556: "f32[64, 64, 1024]", permute_557: "f32[64, 64, 1024]", permute_558: "f32[64, 1024, 64]", le_8: "b8[8, 1024, 2048]", permute_588: "f32[64, 1024, 1024]", permute_589: "f32[64, 64, 1024]", permute_590: "f32[64, 64, 1024]", permute_591: "f32[64, 1024, 64]", le_9: "b8[8, 1024, 2048]", permute_621: "f32[64, 1024, 1024]", permute_622: "f32[64, 64, 1024]", permute_623: "f32[64, 64, 1024]", permute_624: "f32[64, 1024, 64]", le_10: "b8[8, 1024, 2048]", permute_654: "f32[64, 1024, 1024]", permute_655: "f32[64, 64, 1024]", permute_656: "f32[64, 64, 1024]", permute_657: "f32[64, 1024, 64]", le_11: "b8[8, 1024, 2048]", permute_687: "f32[64, 1024, 1024]", permute_688: "f32[64, 64, 1024]", permute_689: "f32[64, 64, 1024]", permute_690: "f32[64, 1024, 64]", le_12: "b8[8, 1024, 2048]", permute_720: "f32[64, 1024, 1024]", permute_721: "f32[64, 64, 1024]", permute_723: "f32[64, 64, 1024]", permute_724: "f32[64, 1024, 64]", tangents_1: "f32[]", tangents_2: "f32[8, 1024, 32128]", tangents_3: "f32[8, 1024, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        div_23: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_5);  tangents_1 = convert_element_type_5 = None
        view_432: "i64[8192]" = torch.ops.aten.reshape.default(primals_53, [-1]);  primals_53 = None
        unsqueeze_19: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(view_432, 1);  view_432 = None
        ne_3: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_19, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:599 in _shift_right, code: shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        full_default_4: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_8: "i64[8192, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_19, full_default_4);  unsqueeze_19 = full_default_4 = None

        # No stacktrace found for following nodes
        iota_default: "i64[32128]" = torch.ops.prims.iota.default(32128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 32128]" = torch.ops.aten.reshape.default(iota_default, [1, 32128]);  iota_default = None
        expand_default: "i64[8192, 32128]" = torch.ops.aten.expand.default(where_8, [8192, 32128]);  where_8 = None
        eq_tensor: "b8[8192, 32128]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_self: "f32[8192, 32128]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_9: "f32[8192, 1]" = torch.ops.aten.where.self(ne_3, div_23, full_default);  ne_3 = div_23 = None
        mul_196: "f32[8192, 32128]" = torch.ops.aten.mul.Tensor(where_self, where_9);  where_self = where_9 = None
        view_431: "f32[8192, 32128]" = torch.ops.aten.reshape.default(view_430, [-1, 32128]);  view_430 = None
        sub_20: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(view_431, amax_18);  view_431 = amax_18 = None
        sub_21: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(sub_20, log_2);  sub_20 = log_2 = None
        exp_19: "f32[8192, 32128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_22: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [1], True)
        mul_197: "f32[8192, 32128]" = torch.ops.aten.mul.Tensor(exp_19, sum_22);  exp_19 = sum_22 = None
        sub_22: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        view_433: "f32[8, 1024, 32128]" = torch.ops.aten.reshape.default(sub_22, [8, 1024, 32128]);  sub_22 = None
        add_96: "f32[8, 1024, 32128]" = torch.ops.aten.add.Tensor(tangents_2, view_433);  tangents_2 = view_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        view_434: "f32[8192, 32128]" = torch.ops.aten.reshape.default(add_96, [8192, 32128]);  add_96 = None
        permute_189: "f32[32128, 8192]" = torch.ops.aten.permute.default(view_434, [1, 0])
        mm_97: "f32[32128, 512]" = torch.ops.aten.mm.default(permute_189, view_429);  permute_189 = view_429 = None
        permute_188: "f32[512, 32128]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_191: "f32[32128, 512]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
        mm_98: "f32[8192, 512]" = torch.ops.aten.mm.default(view_434, permute_191);  view_434 = permute_191 = None
        view_435: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_98, [8, 1024, 512]);  mm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_198: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_435, 0.04419417382415922);  view_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_6: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_64, torch.float32);  gt_64 = None
        mul_199: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_200: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_201: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_200, primals_133);  primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_191: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_94, rsqrt_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_202: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_200, mul_191);  mul_200 = mul_191 = None
        sum_23: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1], True);  mul_202 = None
        view_436: "f32[512]" = torch.ops.aten.reshape.default(sum_23, [512]);  sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_203: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_201, add_94)
        mul_204: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_201, rsqrt_31);  mul_201 = None
        sum_24: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True);  mul_203 = None
        pow_33: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_31, 3);  rsqrt_31 = None
        mul_205: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_24, -0.5);  sum_24 = None
        mul_206: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_205, pow_33);  mul_205 = pow_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_75: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_206, [8, 1024, 512]);  mul_206 = None
        div_24: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_75, 512);  expand_75 = None
        pow_34: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_94, 1.0);  add_94 = None
        mul_207: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_34, 2.0);  pow_34 = None
        mul_208: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_24, mul_207);  div_24 = mul_207 = None
        add_97: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_204, mul_208);  mul_204 = mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_7: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_63, torch.float32);  gt_63 = None
        mul_209: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_210: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_97, mul_209);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_437: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_210, [8192, 512]);  mul_210 = None
        permute_193: "f32[512, 8192]" = torch.ops.aten.permute.default(view_437, [1, 0])
        mm_99: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_193, view_427);  permute_193 = view_427 = None
        permute_187: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        permute_195: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        mm_100: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_437, permute_195);  view_437 = permute_195 = None
        view_438: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_100, [8, 1024, 2048]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_62, torch.float32);  gt_62 = None
        mul_211: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_212: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_438, mul_211);  view_438 = mul_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_10: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_1, full_default, mul_212);  le_1 = mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_439: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_10, [8192, 2048]);  where_10 = None
        permute_197: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_439, [1, 0])
        mm_101: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_197, view_425);  permute_197 = view_425 = None
        permute_186: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        permute_199: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        mm_102: "f32[8192, 512]" = torch.ops.aten.mm.default(view_439, permute_199);  view_439 = permute_199 = None
        view_440: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_102, [8, 1024, 512]);  mm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_213: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_440, primals_130);  primals_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_185: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_92, rsqrt_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_214: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_440, mul_185);  view_440 = mul_185 = None
        sum_25: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1], True);  mul_214 = None
        view_441: "f32[512]" = torch.ops.aten.reshape.default(sum_25, [512]);  sum_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_215: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_213, add_92)
        mul_216: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_213, rsqrt_30);  mul_213 = None
        sum_26: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
        add_98: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_97, mul_216);  add_97 = mul_216 = None
        pow_35: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_30, 3);  rsqrt_30 = None
        mul_217: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_26, -0.5);  sum_26 = None
        mul_218: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_217, pow_35);  mul_217 = pow_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_76: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_218, [8, 1024, 512]);  mul_218 = None
        div_25: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_76, 512);  expand_76 = None
        pow_36: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_92, 1.0);  add_92 = None
        mul_219: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_36, 2.0);  pow_36 = None
        mul_220: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_25, mul_219);  div_25 = mul_219 = None
        add_99: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_98, mul_220);  add_98 = mul_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_9: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_61, torch.float32);  gt_61 = None
        mul_221: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_222: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_99, mul_221);  mul_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_442: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_222, [8192, 512]);  mul_222 = None
        permute_201: "f32[512, 8192]" = torch.ops.aten.permute.default(view_442, [1, 0])
        mm_103: "f32[512, 512]" = torch.ops.aten.mm.default(permute_201, view_423);  permute_201 = view_423 = None
        permute_185: "f32[512, 512]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        permute_203: "f32[512, 512]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        mm_104: "f32[8192, 512]" = torch.ops.aten.mm.default(view_442, permute_203);  view_442 = permute_203 = None
        view_443: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_104, [8, 1024, 512]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_444: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_443, [8, 1024, 8, 64]);  view_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_205: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_77: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_445: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_77, [64, 1024, 64]);  clone_77 = None
        bmm_36: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_206, view_445);  permute_206 = None
        bmm_37: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_445, permute_207);  view_445 = permute_207 = None
        view_446: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_36, [8, 8, 1024, 64]);  bmm_36 = None
        view_447: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_37, [8, 8, 1024, 1024]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_10: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_60, torch.float32);  gt_60 = None
        mul_223: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_224: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_447, mul_223);  view_447 = mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_225: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_224, div_21);  mul_224 = None
        sum_27: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_225, [-1], True)
        neg_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_2, sum_27, mul_225);  neg_2 = sum_27 = mul_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_448: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma, [64, 1024, 1024]);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_38: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_208, view_448);  permute_208 = None
        bmm_39: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_448, permute_209);  view_448 = permute_209 = None
        view_453: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_38, [8, 8, 64, 1024]);  bmm_38 = None
        view_454: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_39, [8, 8, 1024, 64]);  bmm_39 = None
        permute_210: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_453, [0, 1, 3, 2]);  view_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_211: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
        clone_80: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
        view_455: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_80, [8, 1024, 512]);  clone_80 = None
        view_456: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_455, [8192, 512]);  view_455 = None
        permute_212: "f32[512, 8192]" = torch.ops.aten.permute.default(view_456, [1, 0])
        view_410: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_105: "f32[512, 512]" = torch.ops.aten.mm.default(permute_212, view_410);  permute_212 = view_410 = None
        permute_181: "f32[512, 512]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_214: "f32[512, 512]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        mm_106: "f32[8192, 512]" = torch.ops.aten.mm.default(view_456, permute_214);  view_456 = permute_214 = None
        view_457: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_106, [8, 1024, 512]);  mm_106 = None
        add_100: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(tangents_3, view_457);  tangents_3 = view_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_216: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_210, [0, 2, 1, 3]);  permute_210 = None
        view_458: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_216, [8, 1024, 512]);  permute_216 = None
        clone_81: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_458, memory_format = torch.contiguous_format);  view_458 = None
        view_459: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_81, [8192, 512]);  clone_81 = None
        permute_217: "f32[512, 8192]" = torch.ops.aten.permute.default(view_459, [1, 0])
        view_407: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_107: "f32[512, 512]" = torch.ops.aten.mm.default(permute_217, view_407);  permute_217 = view_407 = None
        permute_179: "f32[512, 512]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_219: "f32[512, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        mm_108: "f32[8192, 512]" = torch.ops.aten.mm.default(view_459, permute_219);  view_459 = permute_219 = None
        view_460: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_108, [8, 1024, 512]);  mm_108 = None
        add_101: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_100, view_460);  add_100 = view_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_221: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        clone_82: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_461: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_82, [8, 1024, 512]);  clone_82 = None
        view_462: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_461, [8192, 512]);  view_461 = None
        permute_222: "f32[512, 8192]" = torch.ops.aten.permute.default(view_462, [1, 0])
        mm_109: "f32[512, 512]" = torch.ops.aten.mm.default(permute_222, view_404);  permute_222 = view_404 = None
        permute_177: "f32[512, 512]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
        permute_224: "f32[512, 512]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        mm_110: "f32[8192, 512]" = torch.ops.aten.mm.default(view_462, permute_224);  view_462 = permute_224 = None
        view_463: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_110, [8, 1024, 512]);  mm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_226: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_463, primals_125);  primals_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_179: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_89, rsqrt_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_227: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_463, mul_179);  view_463 = mul_179 = None
        sum_28: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1], True);  mul_227 = None
        view_464: "f32[512]" = torch.ops.aten.reshape.default(sum_28, [512]);  sum_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_228: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_226, add_89)
        mul_229: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_226, rsqrt_29);  mul_226 = None
        sum_29: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True);  mul_228 = None
        add_102: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_99, mul_229);  add_99 = mul_229 = None
        pow_37: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_29, 3);  rsqrt_29 = None
        mul_230: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_29, -0.5);  sum_29 = None
        mul_231: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_230, pow_37);  mul_230 = pow_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_77: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_231, [8, 1024, 512]);  mul_231 = None
        div_26: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_77, 512);  expand_77 = None
        pow_38: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_89, 1.0);  add_89 = None
        mul_232: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
        mul_233: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_26, mul_232);  div_26 = mul_232 = None
        add_103: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_102, mul_233);  add_102 = mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_11: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_59, torch.float32);  gt_59 = None
        mul_234: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_235: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_103, mul_234);  mul_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_465: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_235, [8192, 512]);  mul_235 = None
        permute_226: "f32[512, 8192]" = torch.ops.aten.permute.default(view_465, [1, 0])
        mm_111: "f32[512, 512]" = torch.ops.aten.mm.default(permute_226, view_402);  permute_226 = view_402 = None
        permute_176: "f32[512, 512]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        permute_228: "f32[512, 512]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_112: "f32[8192, 512]" = torch.ops.aten.mm.default(view_465, permute_228);  view_465 = permute_228 = None
        view_466: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_112, [8, 1024, 512]);  mm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_467: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_466, [8, 1024, 8, 64]);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_230: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_84: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
        view_468: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_84, [64, 1024, 64]);  clone_84 = None
        bmm_40: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_231, view_468);  permute_231 = None
        bmm_41: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_468, permute_232);  view_468 = permute_232 = None
        view_469: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_40, [8, 8, 1024, 64]);  bmm_40 = None
        view_470: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_41, [8, 8, 1024, 1024]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_12: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_58, torch.float32);  gt_58 = None
        mul_236: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_237: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_470, mul_236);  view_470 = mul_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_238: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_237, div_20);  mul_237 = None
        sum_30: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [-1], True)
        neg_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_20);  div_20 = None
        fma_1: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_3, sum_30, mul_238);  neg_3 = sum_30 = mul_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_471: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_1, [64, 1024, 1024]);  fma_1 = None
        view_473: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_471, [8, 8, 1024, 1024]);  view_471 = None
        view_474: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_473, [64, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_42: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_233, view_474);  permute_233 = None
        bmm_43: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_474, permute_234);  view_474 = permute_234 = None
        view_476: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_42, [8, 8, 64, 1024]);  bmm_42 = None
        view_477: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_43, [8, 8, 1024, 64]);  bmm_43 = None
        permute_235: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_476, [0, 1, 3, 2]);  view_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_236: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
        clone_87: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_478: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_87, [8, 1024, 512]);  clone_87 = None
        view_479: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_478, [8192, 512]);  view_478 = None
        permute_237: "f32[512, 8192]" = torch.ops.aten.permute.default(view_479, [1, 0])
        mm_113: "f32[512, 512]" = torch.ops.aten.mm.default(permute_237, view_383);  permute_237 = None
        permute_172: "f32[512, 512]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        permute_239: "f32[512, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        mm_114: "f32[8192, 512]" = torch.ops.aten.mm.default(view_479, permute_239);  view_479 = permute_239 = None
        view_480: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_114, [8, 1024, 512]);  mm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_241: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_235, [0, 2, 1, 3]);  permute_235 = None
        view_481: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_241, [8, 1024, 512]);  permute_241 = None
        clone_88: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_481, memory_format = torch.contiguous_format);  view_481 = None
        view_482: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_88, [8192, 512]);  clone_88 = None
        permute_242: "f32[512, 8192]" = torch.ops.aten.permute.default(view_482, [1, 0])
        mm_115: "f32[512, 512]" = torch.ops.aten.mm.default(permute_242, view_383);  permute_242 = None
        permute_170: "f32[512, 512]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        permute_244: "f32[512, 512]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        mm_116: "f32[8192, 512]" = torch.ops.aten.mm.default(view_482, permute_244);  view_482 = permute_244 = None
        view_483: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_116, [8, 1024, 512]);  mm_116 = None
        add_104: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_480, view_483);  view_480 = view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_246: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_477, [0, 2, 1, 3]);  view_477 = None
        clone_89: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        view_484: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_89, [8, 1024, 512]);  clone_89 = None
        view_485: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_484, [8192, 512]);  view_484 = None
        permute_247: "f32[512, 8192]" = torch.ops.aten.permute.default(view_485, [1, 0])
        mm_117: "f32[512, 512]" = torch.ops.aten.mm.default(permute_247, view_383);  permute_247 = view_383 = None
        permute_168: "f32[512, 512]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        permute_249: "f32[512, 512]" = torch.ops.aten.permute.default(permute_168, [1, 0]);  permute_168 = None
        mm_118: "f32[8192, 512]" = torch.ops.aten.mm.default(view_485, permute_249);  view_485 = permute_249 = None
        view_486: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_118, [8, 1024, 512]);  mm_118 = None
        add_105: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_104, view_486);  add_104 = view_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_239: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_105, primals_120);  primals_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_173: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_86, rsqrt_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_240: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_105, mul_173);  add_105 = mul_173 = None
        sum_31: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_240, [0, 1], True);  mul_240 = None
        view_487: "f32[512]" = torch.ops.aten.reshape.default(sum_31, [512]);  sum_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_241: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_239, add_86)
        mul_242: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_239, rsqrt_28);  mul_239 = None
        sum_32: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True);  mul_241 = None
        add_106: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_103, mul_242);  add_103 = mul_242 = None
        pow_39: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_28, 3);  rsqrt_28 = None
        mul_243: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_32, -0.5);  sum_32 = None
        mul_244: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_243, pow_39);  mul_243 = pow_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_78: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_244, [8, 1024, 512]);  mul_244 = None
        div_27: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_78, 512);  expand_78 = None
        pow_40: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_86, 1.0);  add_86 = None
        mul_245: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
        mul_246: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_27, mul_245);  div_27 = mul_245 = None
        add_107: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_106, mul_246);  add_106 = mul_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_13: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_57, torch.float32);  gt_57 = None
        mul_247: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_248: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_107, mul_247);  mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_488: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_248, [8192, 512]);  mul_248 = None
        permute_251: "f32[512, 8192]" = torch.ops.aten.permute.default(view_488, [1, 0])
        mm_119: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_251, view_381);  permute_251 = view_381 = None
        permute_167: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
        permute_253: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        mm_120: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_488, permute_253);  view_488 = permute_253 = None
        view_489: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_120, [8, 1024, 2048]);  mm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_56, torch.float32);  gt_56 = None
        mul_249: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_250: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_489, mul_249);  view_489 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_11: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_2, full_default, mul_250);  le_2 = mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_490: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_11, [8192, 2048]);  where_11 = None
        permute_255: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_121: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_255, view_379);  permute_255 = view_379 = None
        permute_166: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        permute_257: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
        mm_122: "f32[8192, 512]" = torch.ops.aten.mm.default(view_490, permute_257);  view_490 = permute_257 = None
        view_491: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_122, [8, 1024, 512]);  mm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_251: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_491, primals_117);  primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_167: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_252: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_491, mul_167);  view_491 = mul_167 = None
        sum_33: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 1], True);  mul_252 = None
        view_492: "f32[512]" = torch.ops.aten.reshape.default(sum_33, [512]);  sum_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_253: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_251, add_84)
        mul_254: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_251, rsqrt_27);  mul_251 = None
        sum_34: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
        add_108: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_107, mul_254);  add_107 = mul_254 = None
        pow_41: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_27, 3);  rsqrt_27 = None
        mul_255: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_34, -0.5);  sum_34 = None
        mul_256: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_255, pow_41);  mul_255 = pow_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_79: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_256, [8, 1024, 512]);  mul_256 = None
        div_28: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_79, 512);  expand_79 = None
        pow_42: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 1.0);  add_84 = None
        mul_257: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
        mul_258: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_28, mul_257);  div_28 = mul_257 = None
        add_109: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_108, mul_258);  add_108 = mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_15: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_55, torch.float32);  gt_55 = None
        mul_259: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_260: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_109, mul_259);  mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_493: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_260, [8192, 512]);  mul_260 = None
        permute_259: "f32[512, 8192]" = torch.ops.aten.permute.default(view_493, [1, 0])
        mm_123: "f32[512, 512]" = torch.ops.aten.mm.default(permute_259, view_377);  permute_259 = view_377 = None
        permute_165: "f32[512, 512]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        permute_261: "f32[512, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_124: "f32[8192, 512]" = torch.ops.aten.mm.default(view_493, permute_261);  view_493 = permute_261 = None
        view_494: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_124, [8, 1024, 512]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_495: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_494, [8, 1024, 8, 64]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_263: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_495, [0, 2, 1, 3]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_93: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
        view_496: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_93, [64, 1024, 64]);  clone_93 = None
        bmm_44: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_264, view_496);  permute_264 = None
        bmm_45: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_496, permute_265);  view_496 = permute_265 = None
        view_497: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_44, [8, 8, 1024, 64]);  bmm_44 = None
        view_498: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_45, [8, 8, 1024, 1024]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_16: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_54, torch.float32);  gt_54 = None
        mul_261: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_262: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_498, mul_261);  view_498 = mul_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_263: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_262, div_19);  mul_262 = None
        sum_35: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_263, [-1], True)
        neg_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_2: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_4, sum_35, mul_263);  neg_4 = sum_35 = mul_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_499: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_2, [64, 1024, 1024]);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_46: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_266, view_499);  permute_266 = None
        bmm_47: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_499, permute_267);  view_499 = permute_267 = None
        view_504: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_46, [8, 8, 64, 1024]);  bmm_46 = None
        view_505: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_47, [8, 8, 1024, 64]);  bmm_47 = None
        permute_268: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_504, [0, 1, 3, 2]);  view_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_269: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
        clone_96: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_506: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_96, [8, 1024, 512]);  clone_96 = None
        view_507: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_506, [8192, 512]);  view_506 = None
        permute_270: "f32[512, 8192]" = torch.ops.aten.permute.default(view_507, [1, 0])
        view_364: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_125: "f32[512, 512]" = torch.ops.aten.mm.default(permute_270, view_364);  permute_270 = view_364 = None
        permute_161: "f32[512, 512]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_272: "f32[512, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        mm_126: "f32[8192, 512]" = torch.ops.aten.mm.default(view_507, permute_272);  view_507 = permute_272 = None
        view_508: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_126, [8, 1024, 512]);  mm_126 = None
        add_110: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_101, view_508);  add_101 = view_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_274: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_268, [0, 2, 1, 3]);  permute_268 = None
        view_509: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_274, [8, 1024, 512]);  permute_274 = None
        clone_97: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_509, memory_format = torch.contiguous_format);  view_509 = None
        view_510: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_97, [8192, 512]);  clone_97 = None
        permute_275: "f32[512, 8192]" = torch.ops.aten.permute.default(view_510, [1, 0])
        view_361: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_127: "f32[512, 512]" = torch.ops.aten.mm.default(permute_275, view_361);  permute_275 = view_361 = None
        permute_159: "f32[512, 512]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_277: "f32[512, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        mm_128: "f32[8192, 512]" = torch.ops.aten.mm.default(view_510, permute_277);  view_510 = permute_277 = None
        view_511: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_128, [8, 1024, 512]);  mm_128 = None
        add_111: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_110, view_511);  add_110 = view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_279: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None
        clone_98: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
        view_512: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_98, [8, 1024, 512]);  clone_98 = None
        view_513: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_512, [8192, 512]);  view_512 = None
        permute_280: "f32[512, 8192]" = torch.ops.aten.permute.default(view_513, [1, 0])
        mm_129: "f32[512, 512]" = torch.ops.aten.mm.default(permute_280, view_358);  permute_280 = view_358 = None
        permute_157: "f32[512, 512]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        permute_282: "f32[512, 512]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        mm_130: "f32[8192, 512]" = torch.ops.aten.mm.default(view_513, permute_282);  view_513 = permute_282 = None
        view_514: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_130, [8, 1024, 512]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_264: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_514, primals_112);  primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_161: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, rsqrt_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_265: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_514, mul_161);  view_514 = mul_161 = None
        sum_36: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_265, [0, 1], True);  mul_265 = None
        view_515: "f32[512]" = torch.ops.aten.reshape.default(sum_36, [512]);  sum_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_266: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_264, add_81)
        mul_267: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_264, rsqrt_26);  mul_264 = None
        sum_37: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True);  mul_266 = None
        add_112: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_109, mul_267);  add_109 = mul_267 = None
        pow_43: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_26, 3);  rsqrt_26 = None
        mul_268: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_37, -0.5);  sum_37 = None
        mul_269: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_268, pow_43);  mul_268 = pow_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_80: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_269, [8, 1024, 512]);  mul_269 = None
        div_29: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_80, 512);  expand_80 = None
        pow_44: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_81, 1.0);  add_81 = None
        mul_270: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_44, 2.0);  pow_44 = None
        mul_271: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_29, mul_270);  div_29 = mul_270 = None
        add_113: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_112, mul_271);  add_112 = mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_17: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_53, torch.float32);  gt_53 = None
        mul_272: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_273: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_113, mul_272);  mul_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_516: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_273, [8192, 512]);  mul_273 = None
        permute_284: "f32[512, 8192]" = torch.ops.aten.permute.default(view_516, [1, 0])
        mm_131: "f32[512, 512]" = torch.ops.aten.mm.default(permute_284, view_356);  permute_284 = view_356 = None
        permute_156: "f32[512, 512]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        permute_286: "f32[512, 512]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_132: "f32[8192, 512]" = torch.ops.aten.mm.default(view_516, permute_286);  view_516 = permute_286 = None
        view_517: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_132, [8, 1024, 512]);  mm_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_518: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_517, [8, 1024, 8, 64]);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_288: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_100: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
        view_519: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_100, [64, 1024, 64]);  clone_100 = None
        bmm_48: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_289, view_519);  permute_289 = None
        bmm_49: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_519, permute_290);  view_519 = permute_290 = None
        view_520: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_48, [8, 8, 1024, 64]);  bmm_48 = None
        view_521: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_49, [8, 8, 1024, 1024]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_18: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_52, torch.float32);  gt_52 = None
        mul_274: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_275: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_521, mul_274);  view_521 = mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_276: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_275, div_18);  mul_275 = None
        sum_38: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
        neg_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_3: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_5, sum_38, mul_276);  neg_5 = sum_38 = mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_522: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_3, [64, 1024, 1024]);  fma_3 = None
        view_524: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_522, [8, 8, 1024, 1024]);  view_522 = None
        view_525: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_524, [64, 1024, 1024])
        add_114: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_473, view_524);  view_473 = view_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_50: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_291, view_525);  permute_291 = None
        bmm_51: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_525, permute_292);  view_525 = permute_292 = None
        view_527: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_50, [8, 8, 64, 1024]);  bmm_50 = None
        view_528: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_51, [8, 8, 1024, 64]);  bmm_51 = None
        permute_293: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_527, [0, 1, 3, 2]);  view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_294: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_520, [0, 2, 1, 3]);  view_520 = None
        clone_103: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
        view_529: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_103, [8, 1024, 512]);  clone_103 = None
        view_530: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_529, [8192, 512]);  view_529 = None
        permute_295: "f32[512, 8192]" = torch.ops.aten.permute.default(view_530, [1, 0])
        mm_133: "f32[512, 512]" = torch.ops.aten.mm.default(permute_295, view_337);  permute_295 = None
        permute_152: "f32[512, 512]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        permute_297: "f32[512, 512]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_134: "f32[8192, 512]" = torch.ops.aten.mm.default(view_530, permute_297);  view_530 = permute_297 = None
        view_531: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_134, [8, 1024, 512]);  mm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_299: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_293, [0, 2, 1, 3]);  permute_293 = None
        view_532: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_299, [8, 1024, 512]);  permute_299 = None
        clone_104: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_532, memory_format = torch.contiguous_format);  view_532 = None
        view_533: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_104, [8192, 512]);  clone_104 = None
        permute_300: "f32[512, 8192]" = torch.ops.aten.permute.default(view_533, [1, 0])
        mm_135: "f32[512, 512]" = torch.ops.aten.mm.default(permute_300, view_337);  permute_300 = None
        permute_150: "f32[512, 512]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_302: "f32[512, 512]" = torch.ops.aten.permute.default(permute_150, [1, 0]);  permute_150 = None
        mm_136: "f32[8192, 512]" = torch.ops.aten.mm.default(view_533, permute_302);  view_533 = permute_302 = None
        view_534: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_136, [8, 1024, 512]);  mm_136 = None
        add_115: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_531, view_534);  view_531 = view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_304: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None
        clone_105: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
        view_535: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_105, [8, 1024, 512]);  clone_105 = None
        view_536: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_535, [8192, 512]);  view_535 = None
        permute_305: "f32[512, 8192]" = torch.ops.aten.permute.default(view_536, [1, 0])
        mm_137: "f32[512, 512]" = torch.ops.aten.mm.default(permute_305, view_337);  permute_305 = view_337 = None
        permute_148: "f32[512, 512]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        permute_307: "f32[512, 512]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        mm_138: "f32[8192, 512]" = torch.ops.aten.mm.default(view_536, permute_307);  view_536 = permute_307 = None
        view_537: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_138, [8, 1024, 512]);  mm_138 = None
        add_116: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_115, view_537);  add_115 = view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_277: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_116, primals_107);  primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_155: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_278: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_116, mul_155);  add_116 = mul_155 = None
        sum_39: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1], True);  mul_278 = None
        view_538: "f32[512]" = torch.ops.aten.reshape.default(sum_39, [512]);  sum_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_279: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_277, add_78)
        mul_280: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_277, rsqrt_25);  mul_277 = None
        sum_40: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
        add_117: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_113, mul_280);  add_113 = mul_280 = None
        pow_45: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_25, 3);  rsqrt_25 = None
        mul_281: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_40, -0.5);  sum_40 = None
        mul_282: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_281, pow_45);  mul_281 = pow_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_81: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_282, [8, 1024, 512]);  mul_282 = None
        div_30: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_81, 512);  expand_81 = None
        pow_46: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 1.0);  add_78 = None
        mul_283: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
        mul_284: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_30, mul_283);  div_30 = mul_283 = None
        add_118: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_117, mul_284);  add_117 = mul_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_19: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_51, torch.float32);  gt_51 = None
        mul_285: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_286: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_118, mul_285);  mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_539: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_286, [8192, 512]);  mul_286 = None
        permute_309: "f32[512, 8192]" = torch.ops.aten.permute.default(view_539, [1, 0])
        mm_139: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_309, view_335);  permute_309 = view_335 = None
        permute_147: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        permute_311: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        mm_140: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_539, permute_311);  view_539 = permute_311 = None
        view_540: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_140, [8, 1024, 2048]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_50, torch.float32);  gt_50 = None
        mul_287: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_288: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_540, mul_287);  view_540 = mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_12: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_3, full_default, mul_288);  le_3 = mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_541: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_12, [8192, 2048]);  where_12 = None
        permute_313: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_541, [1, 0])
        mm_141: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_313, view_333);  permute_313 = view_333 = None
        permute_146: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_315: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_146, [1, 0]);  permute_146 = None
        mm_142: "f32[8192, 512]" = torch.ops.aten.mm.default(view_541, permute_315);  view_541 = permute_315 = None
        view_542: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_142, [8, 1024, 512]);  mm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_289: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_542, primals_104);  primals_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_149: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_76, rsqrt_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_290: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_542, mul_149);  view_542 = mul_149 = None
        sum_41: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1], True);  mul_290 = None
        view_543: "f32[512]" = torch.ops.aten.reshape.default(sum_41, [512]);  sum_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_291: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_289, add_76)
        mul_292: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_289, rsqrt_24);  mul_289 = None
        sum_42: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True);  mul_291 = None
        add_119: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_118, mul_292);  add_118 = mul_292 = None
        pow_47: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_24, 3);  rsqrt_24 = None
        mul_293: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_42, -0.5);  sum_42 = None
        mul_294: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_293, pow_47);  mul_293 = pow_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_82: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_294, [8, 1024, 512]);  mul_294 = None
        div_31: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_82, 512);  expand_82 = None
        pow_48: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_76, 1.0);  add_76 = None
        mul_295: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
        mul_296: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_31, mul_295);  div_31 = mul_295 = None
        add_120: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_119, mul_296);  add_119 = mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_21: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_49, torch.float32);  gt_49 = None
        mul_297: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_298: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_120, mul_297);  mul_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_544: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_298, [8192, 512]);  mul_298 = None
        permute_317: "f32[512, 8192]" = torch.ops.aten.permute.default(view_544, [1, 0])
        mm_143: "f32[512, 512]" = torch.ops.aten.mm.default(permute_317, view_331);  permute_317 = view_331 = None
        permute_145: "f32[512, 512]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        permute_319: "f32[512, 512]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        mm_144: "f32[8192, 512]" = torch.ops.aten.mm.default(view_544, permute_319);  view_544 = permute_319 = None
        view_545: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_144, [8, 1024, 512]);  mm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_546: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_545, [8, 1024, 8, 64]);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_321: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_109: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
        view_547: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_109, [64, 1024, 64]);  clone_109 = None
        bmm_52: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_322, view_547);  permute_322 = None
        bmm_53: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_547, permute_323);  view_547 = permute_323 = None
        view_548: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_52, [8, 8, 1024, 64]);  bmm_52 = None
        view_549: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_53, [8, 8, 1024, 1024]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_22: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_48, torch.float32);  gt_48 = None
        mul_299: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_300: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_549, mul_299);  view_549 = mul_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_301: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_300, div_17);  mul_300 = None
        sum_43: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
        neg_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_4: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_6, sum_43, mul_301);  neg_6 = sum_43 = mul_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_550: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_4, [64, 1024, 1024]);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_54: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_324, view_550);  permute_324 = None
        bmm_55: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_550, permute_325);  view_550 = permute_325 = None
        view_555: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_54, [8, 8, 64, 1024]);  bmm_54 = None
        view_556: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_55, [8, 8, 1024, 64]);  bmm_55 = None
        permute_326: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_555, [0, 1, 3, 2]);  view_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_327: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None
        clone_112: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
        view_557: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_112, [8, 1024, 512]);  clone_112 = None
        view_558: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_557, [8192, 512]);  view_557 = None
        permute_328: "f32[512, 8192]" = torch.ops.aten.permute.default(view_558, [1, 0])
        view_318: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_145: "f32[512, 512]" = torch.ops.aten.mm.default(permute_328, view_318);  permute_328 = view_318 = None
        permute_141: "f32[512, 512]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        permute_330: "f32[512, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_146: "f32[8192, 512]" = torch.ops.aten.mm.default(view_558, permute_330);  view_558 = permute_330 = None
        view_559: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_146, [8, 1024, 512]);  mm_146 = None
        add_121: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_111, view_559);  add_111 = view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_332: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_326, [0, 2, 1, 3]);  permute_326 = None
        view_560: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_332, [8, 1024, 512]);  permute_332 = None
        clone_113: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_560, memory_format = torch.contiguous_format);  view_560 = None
        view_561: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_113, [8192, 512]);  clone_113 = None
        permute_333: "f32[512, 8192]" = torch.ops.aten.permute.default(view_561, [1, 0])
        view_315: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_147: "f32[512, 512]" = torch.ops.aten.mm.default(permute_333, view_315);  permute_333 = view_315 = None
        permute_139: "f32[512, 512]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_335: "f32[512, 512]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        mm_148: "f32[8192, 512]" = torch.ops.aten.mm.default(view_561, permute_335);  view_561 = permute_335 = None
        view_562: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_148, [8, 1024, 512]);  mm_148 = None
        add_122: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_121, view_562);  add_121 = view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_337: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None
        clone_114: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_563: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_114, [8, 1024, 512]);  clone_114 = None
        view_564: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_563, [8192, 512]);  view_563 = None
        permute_338: "f32[512, 8192]" = torch.ops.aten.permute.default(view_564, [1, 0])
        mm_149: "f32[512, 512]" = torch.ops.aten.mm.default(permute_338, view_312);  permute_338 = view_312 = None
        permute_137: "f32[512, 512]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        permute_340: "f32[512, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        mm_150: "f32[8192, 512]" = torch.ops.aten.mm.default(view_564, permute_340);  view_564 = permute_340 = None
        view_565: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_150, [8, 1024, 512]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_302: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_565, primals_99);  primals_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_143: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_73, rsqrt_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_303: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_565, mul_143);  view_565 = mul_143 = None
        sum_44: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1], True);  mul_303 = None
        view_566: "f32[512]" = torch.ops.aten.reshape.default(sum_44, [512]);  sum_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_304: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_302, add_73)
        mul_305: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_302, rsqrt_23);  mul_302 = None
        sum_45: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
        add_123: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_120, mul_305);  add_120 = mul_305 = None
        pow_49: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_23, 3);  rsqrt_23 = None
        mul_306: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_45, -0.5);  sum_45 = None
        mul_307: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_306, pow_49);  mul_306 = pow_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_83: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_307, [8, 1024, 512]);  mul_307 = None
        div_32: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_83, 512);  expand_83 = None
        pow_50: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_73, 1.0);  add_73 = None
        mul_308: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
        mul_309: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_32, mul_308);  div_32 = mul_308 = None
        add_124: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_123, mul_309);  add_123 = mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_23: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_47, torch.float32);  gt_47 = None
        mul_310: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_311: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_124, mul_310);  mul_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_567: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_311, [8192, 512]);  mul_311 = None
        permute_342: "f32[512, 8192]" = torch.ops.aten.permute.default(view_567, [1, 0])
        mm_151: "f32[512, 512]" = torch.ops.aten.mm.default(permute_342, view_310);  permute_342 = view_310 = None
        permute_136: "f32[512, 512]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
        permute_344: "f32[512, 512]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        mm_152: "f32[8192, 512]" = torch.ops.aten.mm.default(view_567, permute_344);  view_567 = permute_344 = None
        view_568: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_152, [8, 1024, 512]);  mm_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_569: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_568, [8, 1024, 8, 64]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_346: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_116: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
        view_570: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_116, [64, 1024, 64]);  clone_116 = None
        bmm_56: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_347, view_570);  permute_347 = None
        bmm_57: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_570, permute_348);  view_570 = permute_348 = None
        view_571: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_56, [8, 8, 1024, 64]);  bmm_56 = None
        view_572: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_57, [8, 8, 1024, 1024]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_24: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_46, torch.float32);  gt_46 = None
        mul_312: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_313: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_572, mul_312);  view_572 = mul_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_314: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_313, div_16);  mul_313 = None
        sum_46: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [-1], True)
        neg_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_5: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_7, sum_46, mul_314);  neg_7 = sum_46 = mul_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_573: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_5, [64, 1024, 1024]);  fma_5 = None
        view_575: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_573, [8, 8, 1024, 1024]);  view_573 = None
        view_576: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_575, [64, 1024, 1024])
        add_125: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_114, view_575);  add_114 = view_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_58: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_349, view_576);  permute_349 = None
        bmm_59: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_576, permute_350);  view_576 = permute_350 = None
        view_578: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_58, [8, 8, 64, 1024]);  bmm_58 = None
        view_579: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_59, [8, 8, 1024, 64]);  bmm_59 = None
        permute_351: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_578, [0, 1, 3, 2]);  view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_352: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_571, [0, 2, 1, 3]);  view_571 = None
        clone_119: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
        view_580: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_119, [8, 1024, 512]);  clone_119 = None
        view_581: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_580, [8192, 512]);  view_580 = None
        permute_353: "f32[512, 8192]" = torch.ops.aten.permute.default(view_581, [1, 0])
        mm_153: "f32[512, 512]" = torch.ops.aten.mm.default(permute_353, view_291);  permute_353 = None
        permute_132: "f32[512, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        permute_355: "f32[512, 512]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_154: "f32[8192, 512]" = torch.ops.aten.mm.default(view_581, permute_355);  view_581 = permute_355 = None
        view_582: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_154, [8, 1024, 512]);  mm_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_357: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_351, [0, 2, 1, 3]);  permute_351 = None
        view_583: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_357, [8, 1024, 512]);  permute_357 = None
        clone_120: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_583, memory_format = torch.contiguous_format);  view_583 = None
        view_584: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_120, [8192, 512]);  clone_120 = None
        permute_358: "f32[512, 8192]" = torch.ops.aten.permute.default(view_584, [1, 0])
        mm_155: "f32[512, 512]" = torch.ops.aten.mm.default(permute_358, view_291);  permute_358 = None
        permute_130: "f32[512, 512]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        permute_360: "f32[512, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_156: "f32[8192, 512]" = torch.ops.aten.mm.default(view_584, permute_360);  view_584 = permute_360 = None
        view_585: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_156, [8, 1024, 512]);  mm_156 = None
        add_126: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_582, view_585);  view_582 = view_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_362: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_579, [0, 2, 1, 3]);  view_579 = None
        clone_121: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_362, memory_format = torch.contiguous_format);  permute_362 = None
        view_586: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_121, [8, 1024, 512]);  clone_121 = None
        view_587: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_586, [8192, 512]);  view_586 = None
        permute_363: "f32[512, 8192]" = torch.ops.aten.permute.default(view_587, [1, 0])
        mm_157: "f32[512, 512]" = torch.ops.aten.mm.default(permute_363, view_291);  permute_363 = view_291 = None
        permute_128: "f32[512, 512]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        permute_365: "f32[512, 512]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
        mm_158: "f32[8192, 512]" = torch.ops.aten.mm.default(view_587, permute_365);  view_587 = permute_365 = None
        view_588: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_158, [8, 1024, 512]);  mm_158 = None
        add_127: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_126, view_588);  add_126 = view_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_315: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_127, primals_94);  primals_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_137: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_316: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_127, mul_137);  add_127 = mul_137 = None
        sum_47: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1], True);  mul_316 = None
        view_589: "f32[512]" = torch.ops.aten.reshape.default(sum_47, [512]);  sum_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_317: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_315, add_70)
        mul_318: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_315, rsqrt_22);  mul_315 = None
        sum_48: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True);  mul_317 = None
        add_128: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_124, mul_318);  add_124 = mul_318 = None
        pow_51: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_22, 3);  rsqrt_22 = None
        mul_319: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_48, -0.5);  sum_48 = None
        mul_320: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_319, pow_51);  mul_319 = pow_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_84: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_320, [8, 1024, 512]);  mul_320 = None
        div_33: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_84, 512);  expand_84 = None
        pow_52: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 1.0);  add_70 = None
        mul_321: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_52, 2.0);  pow_52 = None
        mul_322: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_33, mul_321);  div_33 = mul_321 = None
        add_129: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_128, mul_322);  add_128 = mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_25: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_45, torch.float32);  gt_45 = None
        mul_323: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_324: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_129, mul_323);  mul_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_590: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_324, [8192, 512]);  mul_324 = None
        permute_367: "f32[512, 8192]" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_159: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_367, view_289);  permute_367 = view_289 = None
        permute_127: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_369: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
        mm_160: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_590, permute_369);  view_590 = permute_369 = None
        view_591: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_160, [8, 1024, 2048]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_44, torch.float32);  gt_44 = None
        mul_325: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_326: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_591, mul_325);  view_591 = mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_13: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_4, full_default, mul_326);  le_4 = mul_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_592: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_13, [8192, 2048]);  where_13 = None
        permute_371: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_592, [1, 0])
        mm_161: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_371, view_287);  permute_371 = view_287 = None
        permute_126: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        permute_373: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_126, [1, 0]);  permute_126 = None
        mm_162: "f32[8192, 512]" = torch.ops.aten.mm.default(view_592, permute_373);  view_592 = permute_373 = None
        view_593: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_162, [8, 1024, 512]);  mm_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_327: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_593, primals_91);  primals_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_131: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_68, rsqrt_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_328: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_593, mul_131);  view_593 = mul_131 = None
        sum_49: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1], True);  mul_328 = None
        view_594: "f32[512]" = torch.ops.aten.reshape.default(sum_49, [512]);  sum_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_329: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_327, add_68)
        mul_330: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_327, rsqrt_21);  mul_327 = None
        sum_50: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
        add_130: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_129, mul_330);  add_129 = mul_330 = None
        pow_53: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_21, 3);  rsqrt_21 = None
        mul_331: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_50, -0.5);  sum_50 = None
        mul_332: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_331, pow_53);  mul_331 = pow_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_85: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_332, [8, 1024, 512]);  mul_332 = None
        div_34: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_85, 512);  expand_85 = None
        pow_54: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_68, 1.0);  add_68 = None
        mul_333: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_54, 2.0);  pow_54 = None
        mul_334: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_34, mul_333);  div_34 = mul_333 = None
        add_131: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_130, mul_334);  add_130 = mul_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_27: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_43, torch.float32);  gt_43 = None
        mul_335: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_336: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_131, mul_335);  mul_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_595: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_336, [8192, 512]);  mul_336 = None
        permute_375: "f32[512, 8192]" = torch.ops.aten.permute.default(view_595, [1, 0])
        mm_163: "f32[512, 512]" = torch.ops.aten.mm.default(permute_375, view_285);  permute_375 = view_285 = None
        permute_125: "f32[512, 512]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        permute_377: "f32[512, 512]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        mm_164: "f32[8192, 512]" = torch.ops.aten.mm.default(view_595, permute_377);  view_595 = permute_377 = None
        view_596: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_164, [8, 1024, 512]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_597: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_596, [8, 1024, 8, 64]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_379: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_125: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_379, memory_format = torch.contiguous_format);  permute_379 = None
        view_598: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_125, [64, 1024, 64]);  clone_125 = None
        bmm_60: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_380, view_598);  permute_380 = None
        bmm_61: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_598, permute_381);  view_598 = permute_381 = None
        view_599: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_60, [8, 8, 1024, 64]);  bmm_60 = None
        view_600: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_61, [8, 8, 1024, 1024]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_28: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_42, torch.float32);  gt_42 = None
        mul_337: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_338: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_600, mul_337);  view_600 = mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_339: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_338, div_15);  mul_338 = None
        sum_51: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [-1], True)
        neg_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_6: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_8, sum_51, mul_339);  neg_8 = sum_51 = mul_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_601: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_6, [64, 1024, 1024]);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_62: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_382, view_601);  permute_382 = None
        bmm_63: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_601, permute_383);  view_601 = permute_383 = None
        view_606: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_62, [8, 8, 64, 1024]);  bmm_62 = None
        view_607: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_63, [8, 8, 1024, 64]);  bmm_63 = None
        permute_384: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_606, [0, 1, 3, 2]);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_385: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
        clone_128: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_608: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_128, [8, 1024, 512]);  clone_128 = None
        view_609: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_608, [8192, 512]);  view_608 = None
        permute_386: "f32[512, 8192]" = torch.ops.aten.permute.default(view_609, [1, 0])
        view_272: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_165: "f32[512, 512]" = torch.ops.aten.mm.default(permute_386, view_272);  permute_386 = view_272 = None
        permute_121: "f32[512, 512]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_388: "f32[512, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_166: "f32[8192, 512]" = torch.ops.aten.mm.default(view_609, permute_388);  view_609 = permute_388 = None
        view_610: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_166, [8, 1024, 512]);  mm_166 = None
        add_132: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_122, view_610);  add_122 = view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_390: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_384, [0, 2, 1, 3]);  permute_384 = None
        view_611: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_390, [8, 1024, 512]);  permute_390 = None
        clone_129: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_611, memory_format = torch.contiguous_format);  view_611 = None
        view_612: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_129, [8192, 512]);  clone_129 = None
        permute_391: "f32[512, 8192]" = torch.ops.aten.permute.default(view_612, [1, 0])
        view_269: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_167: "f32[512, 512]" = torch.ops.aten.mm.default(permute_391, view_269);  permute_391 = view_269 = None
        permute_119: "f32[512, 512]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_393: "f32[512, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_168: "f32[8192, 512]" = torch.ops.aten.mm.default(view_612, permute_393);  view_612 = permute_393 = None
        view_613: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_168, [8, 1024, 512]);  mm_168 = None
        add_133: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_132, view_613);  add_132 = view_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_395: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_607, [0, 2, 1, 3]);  view_607 = None
        clone_130: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
        view_614: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_130, [8, 1024, 512]);  clone_130 = None
        view_615: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_614, [8192, 512]);  view_614 = None
        permute_396: "f32[512, 8192]" = torch.ops.aten.permute.default(view_615, [1, 0])
        mm_169: "f32[512, 512]" = torch.ops.aten.mm.default(permute_396, view_266);  permute_396 = view_266 = None
        permute_117: "f32[512, 512]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_398: "f32[512, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
        mm_170: "f32[8192, 512]" = torch.ops.aten.mm.default(view_615, permute_398);  view_615 = permute_398 = None
        view_616: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_170, [8, 1024, 512]);  mm_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_340: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_616, primals_86);  primals_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_125: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_65, rsqrt_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_341: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_616, mul_125);  view_616 = mul_125 = None
        sum_52: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 1], True);  mul_341 = None
        view_617: "f32[512]" = torch.ops.aten.reshape.default(sum_52, [512]);  sum_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_342: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_340, add_65)
        mul_343: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_340, rsqrt_20);  mul_340 = None
        sum_53: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True);  mul_342 = None
        add_134: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_131, mul_343);  add_131 = mul_343 = None
        pow_55: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_20, 3);  rsqrt_20 = None
        mul_344: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_53, -0.5);  sum_53 = None
        mul_345: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_344, pow_55);  mul_344 = pow_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_86: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_345, [8, 1024, 512]);  mul_345 = None
        div_35: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_86, 512);  expand_86 = None
        pow_56: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_65, 1.0);  add_65 = None
        mul_346: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_56, 2.0);  pow_56 = None
        mul_347: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_35, mul_346);  div_35 = mul_346 = None
        add_135: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_134, mul_347);  add_134 = mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_29: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_41, torch.float32);  gt_41 = None
        mul_348: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_349: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_135, mul_348);  mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_618: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_349, [8192, 512]);  mul_349 = None
        permute_400: "f32[512, 8192]" = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_171: "f32[512, 512]" = torch.ops.aten.mm.default(permute_400, view_264);  permute_400 = view_264 = None
        permute_116: "f32[512, 512]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        permute_402: "f32[512, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
        mm_172: "f32[8192, 512]" = torch.ops.aten.mm.default(view_618, permute_402);  view_618 = permute_402 = None
        view_619: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_172, [8, 1024, 512]);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_620: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_619, [8, 1024, 8, 64]);  view_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_404: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_620, [0, 2, 1, 3]);  view_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_132: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_404, memory_format = torch.contiguous_format);  permute_404 = None
        view_621: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_132, [64, 1024, 64]);  clone_132 = None
        bmm_64: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_405, view_621);  permute_405 = None
        bmm_65: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_621, permute_406);  view_621 = permute_406 = None
        view_622: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_64, [8, 8, 1024, 64]);  bmm_64 = None
        view_623: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_65, [8, 8, 1024, 1024]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_30: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_40, torch.float32);  gt_40 = None
        mul_350: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_351: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_623, mul_350);  view_623 = mul_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_352: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_351, div_14);  mul_351 = None
        sum_54: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [-1], True)
        neg_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_14);  div_14 = None
        fma_7: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_9, sum_54, mul_352);  neg_9 = sum_54 = mul_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_624: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_7, [64, 1024, 1024]);  fma_7 = None
        view_626: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_624, [8, 8, 1024, 1024]);  view_624 = None
        view_627: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_626, [64, 1024, 1024])
        add_136: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_125, view_626);  add_125 = view_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_66: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_407, view_627);  permute_407 = None
        bmm_67: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_627, permute_408);  view_627 = permute_408 = None
        view_629: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_66, [8, 8, 64, 1024]);  bmm_66 = None
        view_630: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_67, [8, 8, 1024, 64]);  bmm_67 = None
        permute_409: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_629, [0, 1, 3, 2]);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_410: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_622, [0, 2, 1, 3]);  view_622 = None
        clone_135: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
        view_631: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_135, [8, 1024, 512]);  clone_135 = None
        view_632: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_631, [8192, 512]);  view_631 = None
        permute_411: "f32[512, 8192]" = torch.ops.aten.permute.default(view_632, [1, 0])
        mm_173: "f32[512, 512]" = torch.ops.aten.mm.default(permute_411, view_245);  permute_411 = None
        permute_112: "f32[512, 512]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
        permute_413: "f32[512, 512]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_174: "f32[8192, 512]" = torch.ops.aten.mm.default(view_632, permute_413);  view_632 = permute_413 = None
        view_633: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_174, [8, 1024, 512]);  mm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_415: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_409, [0, 2, 1, 3]);  permute_409 = None
        view_634: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_415, [8, 1024, 512]);  permute_415 = None
        clone_136: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_634, memory_format = torch.contiguous_format);  view_634 = None
        view_635: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_136, [8192, 512]);  clone_136 = None
        permute_416: "f32[512, 8192]" = torch.ops.aten.permute.default(view_635, [1, 0])
        mm_175: "f32[512, 512]" = torch.ops.aten.mm.default(permute_416, view_245);  permute_416 = None
        permute_110: "f32[512, 512]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_418: "f32[512, 512]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_176: "f32[8192, 512]" = torch.ops.aten.mm.default(view_635, permute_418);  view_635 = permute_418 = None
        view_636: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_176, [8, 1024, 512]);  mm_176 = None
        add_137: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_633, view_636);  view_633 = view_636 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_420: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
        clone_137: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
        view_637: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_137, [8, 1024, 512]);  clone_137 = None
        view_638: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_637, [8192, 512]);  view_637 = None
        permute_421: "f32[512, 8192]" = torch.ops.aten.permute.default(view_638, [1, 0])
        mm_177: "f32[512, 512]" = torch.ops.aten.mm.default(permute_421, view_245);  permute_421 = view_245 = None
        permute_108: "f32[512, 512]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
        permute_423: "f32[512, 512]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_178: "f32[8192, 512]" = torch.ops.aten.mm.default(view_638, permute_423);  view_638 = permute_423 = None
        view_639: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_178, [8, 1024, 512]);  mm_178 = None
        add_138: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_137, view_639);  add_137 = view_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_353: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_138, primals_81);  primals_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_119: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_354: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_138, mul_119);  add_138 = mul_119 = None
        sum_55: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 1], True);  mul_354 = None
        view_640: "f32[512]" = torch.ops.aten.reshape.default(sum_55, [512]);  sum_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_355: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_353, add_62)
        mul_356: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_353, rsqrt_19);  mul_353 = None
        sum_56: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
        add_139: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_135, mul_356);  add_135 = mul_356 = None
        pow_57: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_19, 3);  rsqrt_19 = None
        mul_357: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_56, -0.5);  sum_56 = None
        mul_358: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_357, pow_57);  mul_357 = pow_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_87: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_358, [8, 1024, 512]);  mul_358 = None
        div_36: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_87, 512);  expand_87 = None
        pow_58: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_62, 1.0);  add_62 = None
        mul_359: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_58, 2.0);  pow_58 = None
        mul_360: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_36, mul_359);  div_36 = mul_359 = None
        add_140: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_139, mul_360);  add_139 = mul_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_31: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_39, torch.float32);  gt_39 = None
        mul_361: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_362: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_140, mul_361);  mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_641: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_362, [8192, 512]);  mul_362 = None
        permute_425: "f32[512, 8192]" = torch.ops.aten.permute.default(view_641, [1, 0])
        mm_179: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_425, view_243);  permute_425 = view_243 = None
        permute_107: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_427: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_180: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_641, permute_427);  view_641 = permute_427 = None
        view_642: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_180, [8, 1024, 2048]);  mm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_38, torch.float32);  gt_38 = None
        mul_363: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_364: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_642, mul_363);  view_642 = mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_14: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_5, full_default, mul_364);  le_5 = mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_643: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_14, [8192, 2048]);  where_14 = None
        permute_429: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_643, [1, 0])
        mm_181: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_429, view_241);  permute_429 = view_241 = None
        permute_106: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        permute_431: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        mm_182: "f32[8192, 512]" = torch.ops.aten.mm.default(view_643, permute_431);  view_643 = permute_431 = None
        view_644: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_182, [8, 1024, 512]);  mm_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_365: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_644, primals_78);  primals_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_113: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_60, rsqrt_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_366: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_644, mul_113);  view_644 = mul_113 = None
        sum_57: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 1], True);  mul_366 = None
        view_645: "f32[512]" = torch.ops.aten.reshape.default(sum_57, [512]);  sum_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_367: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_365, add_60)
        mul_368: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_365, rsqrt_18);  mul_365 = None
        sum_58: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True);  mul_367 = None
        add_141: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_140, mul_368);  add_140 = mul_368 = None
        pow_59: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_18, 3);  rsqrt_18 = None
        mul_369: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_58, -0.5);  sum_58 = None
        mul_370: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_369, pow_59);  mul_369 = pow_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_88: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_370, [8, 1024, 512]);  mul_370 = None
        div_37: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_88, 512);  expand_88 = None
        pow_60: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_60, 1.0);  add_60 = None
        mul_371: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
        mul_372: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_37, mul_371);  div_37 = mul_371 = None
        add_142: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_141, mul_372);  add_141 = mul_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_33: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_37, torch.float32);  gt_37 = None
        mul_373: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_374: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_142, mul_373);  mul_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_646: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_374, [8192, 512]);  mul_374 = None
        permute_433: "f32[512, 8192]" = torch.ops.aten.permute.default(view_646, [1, 0])
        mm_183: "f32[512, 512]" = torch.ops.aten.mm.default(permute_433, view_239);  permute_433 = view_239 = None
        permute_105: "f32[512, 512]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_435: "f32[512, 512]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
        mm_184: "f32[8192, 512]" = torch.ops.aten.mm.default(view_646, permute_435);  view_646 = permute_435 = None
        view_647: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_184, [8, 1024, 512]);  mm_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_648: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_647, [8, 1024, 8, 64]);  view_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_437: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_141: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_437, memory_format = torch.contiguous_format);  permute_437 = None
        view_649: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_141, [64, 1024, 64]);  clone_141 = None
        bmm_68: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_438, view_649);  permute_438 = None
        bmm_69: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_649, permute_439);  view_649 = permute_439 = None
        view_650: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_68, [8, 8, 1024, 64]);  bmm_68 = None
        view_651: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_69, [8, 8, 1024, 1024]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_34: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_375: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_376: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_651, mul_375);  view_651 = mul_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_377: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_376, div_13);  mul_376 = None
        sum_59: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [-1], True)
        neg_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_8: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_10, sum_59, mul_377);  neg_10 = sum_59 = mul_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_652: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_8, [64, 1024, 1024]);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_70: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_440, view_652);  permute_440 = None
        bmm_71: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_652, permute_441);  view_652 = permute_441 = None
        view_657: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_70, [8, 8, 64, 1024]);  bmm_70 = None
        view_658: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_71, [8, 8, 1024, 64]);  bmm_71 = None
        permute_442: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_657, [0, 1, 3, 2]);  view_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_443: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_650, [0, 2, 1, 3]);  view_650 = None
        clone_144: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_443, memory_format = torch.contiguous_format);  permute_443 = None
        view_659: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_144, [8, 1024, 512]);  clone_144 = None
        view_660: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_659, [8192, 512]);  view_659 = None
        permute_444: "f32[512, 8192]" = torch.ops.aten.permute.default(view_660, [1, 0])
        view_226: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_185: "f32[512, 512]" = torch.ops.aten.mm.default(permute_444, view_226);  permute_444 = view_226 = None
        permute_101: "f32[512, 512]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
        permute_446: "f32[512, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_186: "f32[8192, 512]" = torch.ops.aten.mm.default(view_660, permute_446);  view_660 = permute_446 = None
        view_661: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_186, [8, 1024, 512]);  mm_186 = None
        add_143: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_133, view_661);  add_133 = view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_448: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_442, [0, 2, 1, 3]);  permute_442 = None
        view_662: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_448, [8, 1024, 512]);  permute_448 = None
        clone_145: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_662, memory_format = torch.contiguous_format);  view_662 = None
        view_663: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_145, [8192, 512]);  clone_145 = None
        permute_449: "f32[512, 8192]" = torch.ops.aten.permute.default(view_663, [1, 0])
        view_223: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_187: "f32[512, 512]" = torch.ops.aten.mm.default(permute_449, view_223);  permute_449 = view_223 = None
        permute_99: "f32[512, 512]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_451: "f32[512, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_188: "f32[8192, 512]" = torch.ops.aten.mm.default(view_663, permute_451);  view_663 = permute_451 = None
        view_664: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_188, [8, 1024, 512]);  mm_188 = None
        add_144: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_143, view_664);  add_143 = view_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_453: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_658, [0, 2, 1, 3]);  view_658 = None
        clone_146: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
        view_665: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_146, [8, 1024, 512]);  clone_146 = None
        view_666: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_665, [8192, 512]);  view_665 = None
        permute_454: "f32[512, 8192]" = torch.ops.aten.permute.default(view_666, [1, 0])
        mm_189: "f32[512, 512]" = torch.ops.aten.mm.default(permute_454, view_220);  permute_454 = view_220 = None
        permute_97: "f32[512, 512]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_456: "f32[512, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_190: "f32[8192, 512]" = torch.ops.aten.mm.default(view_666, permute_456);  view_666 = permute_456 = None
        view_667: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_190, [8, 1024, 512]);  mm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_378: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_667, primals_73);  primals_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_107: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_57, rsqrt_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_379: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_667, mul_107);  view_667 = mul_107 = None
        sum_60: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1], True);  mul_379 = None
        view_668: "f32[512]" = torch.ops.aten.reshape.default(sum_60, [512]);  sum_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_380: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_378, add_57)
        mul_381: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_378, rsqrt_17);  mul_378 = None
        sum_61: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True);  mul_380 = None
        add_145: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_142, mul_381);  add_142 = mul_381 = None
        pow_61: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_17, 3);  rsqrt_17 = None
        mul_382: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_61, -0.5);  sum_61 = None
        mul_383: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_382, pow_61);  mul_382 = pow_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_89: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_383, [8, 1024, 512]);  mul_383 = None
        div_38: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_89, 512);  expand_89 = None
        pow_62: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_57, 1.0);  add_57 = None
        mul_384: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_62, 2.0);  pow_62 = None
        mul_385: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_38, mul_384);  div_38 = mul_384 = None
        add_146: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_145, mul_385);  add_145 = mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_35: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_386: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_387: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_146, mul_386);  mul_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_669: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_387, [8192, 512]);  mul_387 = None
        permute_458: "f32[512, 8192]" = torch.ops.aten.permute.default(view_669, [1, 0])
        mm_191: "f32[512, 512]" = torch.ops.aten.mm.default(permute_458, view_218);  permute_458 = view_218 = None
        permute_96: "f32[512, 512]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
        permute_460: "f32[512, 512]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_192: "f32[8192, 512]" = torch.ops.aten.mm.default(view_669, permute_460);  view_669 = permute_460 = None
        view_670: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_192, [8, 1024, 512]);  mm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_671: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_670, [8, 1024, 8, 64]);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_462: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_671, [0, 2, 1, 3]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_148: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_462, memory_format = torch.contiguous_format);  permute_462 = None
        view_672: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_148, [64, 1024, 64]);  clone_148 = None
        bmm_72: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_463, view_672);  permute_463 = None
        bmm_73: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_672, permute_464);  view_672 = permute_464 = None
        view_673: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_72, [8, 8, 1024, 64]);  bmm_72 = None
        view_674: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_73, [8, 8, 1024, 1024]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_36: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_388: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_389: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_674, mul_388);  view_674 = mul_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_390: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_389, div_12);  mul_389 = None
        sum_62: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [-1], True)
        neg_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_9: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_11, sum_62, mul_390);  neg_11 = sum_62 = mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_675: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_9, [64, 1024, 1024]);  fma_9 = None
        view_677: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_675, [8, 8, 1024, 1024]);  view_675 = None
        view_678: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_677, [64, 1024, 1024])
        add_147: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_136, view_677);  add_136 = view_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_74: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_465, view_678);  permute_465 = None
        bmm_75: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_678, permute_466);  view_678 = permute_466 = None
        view_680: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_74, [8, 8, 64, 1024]);  bmm_74 = None
        view_681: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_75, [8, 8, 1024, 64]);  bmm_75 = None
        permute_467: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_680, [0, 1, 3, 2]);  view_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_468: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
        clone_151: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
        view_682: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_151, [8, 1024, 512]);  clone_151 = None
        view_683: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_682, [8192, 512]);  view_682 = None
        permute_469: "f32[512, 8192]" = torch.ops.aten.permute.default(view_683, [1, 0])
        mm_193: "f32[512, 512]" = torch.ops.aten.mm.default(permute_469, view_199);  permute_469 = None
        permute_92: "f32[512, 512]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_471: "f32[512, 512]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        mm_194: "f32[8192, 512]" = torch.ops.aten.mm.default(view_683, permute_471);  view_683 = permute_471 = None
        view_684: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_194, [8, 1024, 512]);  mm_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_473: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_467, [0, 2, 1, 3]);  permute_467 = None
        view_685: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_473, [8, 1024, 512]);  permute_473 = None
        clone_152: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_685, memory_format = torch.contiguous_format);  view_685 = None
        view_686: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_152, [8192, 512]);  clone_152 = None
        permute_474: "f32[512, 8192]" = torch.ops.aten.permute.default(view_686, [1, 0])
        mm_195: "f32[512, 512]" = torch.ops.aten.mm.default(permute_474, view_199);  permute_474 = None
        permute_90: "f32[512, 512]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        permute_476: "f32[512, 512]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_196: "f32[8192, 512]" = torch.ops.aten.mm.default(view_686, permute_476);  view_686 = permute_476 = None
        view_687: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_196, [8, 1024, 512]);  mm_196 = None
        add_148: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_684, view_687);  view_684 = view_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_478: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_681, [0, 2, 1, 3]);  view_681 = None
        clone_153: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
        view_688: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_153, [8, 1024, 512]);  clone_153 = None
        view_689: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_688, [8192, 512]);  view_688 = None
        permute_479: "f32[512, 8192]" = torch.ops.aten.permute.default(view_689, [1, 0])
        mm_197: "f32[512, 512]" = torch.ops.aten.mm.default(permute_479, view_199);  permute_479 = view_199 = None
        permute_88: "f32[512, 512]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_481: "f32[512, 512]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_198: "f32[8192, 512]" = torch.ops.aten.mm.default(view_689, permute_481);  view_689 = permute_481 = None
        view_690: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_198, [8, 1024, 512]);  mm_198 = None
        add_149: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_148, view_690);  add_148 = view_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_391: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_149, primals_68);  primals_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_101: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_54, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_392: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_149, mul_101);  add_149 = mul_101 = None
        sum_63: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 1], True);  mul_392 = None
        view_691: "f32[512]" = torch.ops.aten.reshape.default(sum_63, [512]);  sum_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_393: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_391, add_54)
        mul_394: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_391, rsqrt_16);  mul_391 = None
        sum_64: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True);  mul_393 = None
        add_150: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_146, mul_394);  add_146 = mul_394 = None
        pow_63: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_16, 3);  rsqrt_16 = None
        mul_395: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_64, -0.5);  sum_64 = None
        mul_396: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_395, pow_63);  mul_395 = pow_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_90: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_396, [8, 1024, 512]);  mul_396 = None
        div_39: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_90, 512);  expand_90 = None
        pow_64: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_54, 1.0);  add_54 = None
        mul_397: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
        mul_398: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_39, mul_397);  div_39 = mul_397 = None
        add_151: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_150, mul_398);  add_150 = mul_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_37: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_399: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_400: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_151, mul_399);  mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_692: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_400, [8192, 512]);  mul_400 = None
        permute_483: "f32[512, 8192]" = torch.ops.aten.permute.default(view_692, [1, 0])
        mm_199: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_483, view_197);  permute_483 = view_197 = None
        permute_87: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        permute_485: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_200: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_692, permute_485);  view_692 = permute_485 = None
        view_693: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_200, [8, 1024, 2048]);  mm_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_38: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_401: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
        mul_402: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_693, mul_401);  view_693 = mul_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_15: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_6, full_default, mul_402);  le_6 = mul_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_694: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_15, [8192, 2048]);  where_15 = None
        permute_487: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_694, [1, 0])
        mm_201: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_487, view_195);  permute_487 = view_195 = None
        permute_86: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_489: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_202: "f32[8192, 512]" = torch.ops.aten.mm.default(view_694, permute_489);  view_694 = permute_489 = None
        view_695: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_202, [8, 1024, 512]);  mm_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_403: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_695, primals_65);  primals_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_95: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_404: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_695, mul_95);  view_695 = mul_95 = None
        sum_65: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 1], True);  mul_404 = None
        view_696: "f32[512]" = torch.ops.aten.reshape.default(sum_65, [512]);  sum_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_405: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_403, add_52)
        mul_406: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_403, rsqrt_15);  mul_403 = None
        sum_66: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [2], True);  mul_405 = None
        add_152: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_151, mul_406);  add_151 = mul_406 = None
        pow_65: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_15, 3);  rsqrt_15 = None
        mul_407: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_66, -0.5);  sum_66 = None
        mul_408: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_407, pow_65);  mul_407 = pow_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_91: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_408, [8, 1024, 512]);  mul_408 = None
        div_40: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_91, 512);  expand_91 = None
        pow_66: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 1.0);  add_52 = None
        mul_409: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_66, 2.0);  pow_66 = None
        mul_410: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_40, mul_409);  div_40 = mul_409 = None
        add_153: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_152, mul_410);  add_152 = mul_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_39: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_411: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
        mul_412: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_153, mul_411);  mul_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_697: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_412, [8192, 512]);  mul_412 = None
        permute_491: "f32[512, 8192]" = torch.ops.aten.permute.default(view_697, [1, 0])
        mm_203: "f32[512, 512]" = torch.ops.aten.mm.default(permute_491, view_193);  permute_491 = view_193 = None
        permute_85: "f32[512, 512]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_493: "f32[512, 512]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_204: "f32[8192, 512]" = torch.ops.aten.mm.default(view_697, permute_493);  view_697 = permute_493 = None
        view_698: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_204, [8, 1024, 512]);  mm_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_699: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_698, [8, 1024, 8, 64]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_495: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_699, [0, 2, 1, 3]);  view_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_157: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_495, memory_format = torch.contiguous_format);  permute_495 = None
        view_700: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_157, [64, 1024, 64]);  clone_157 = None
        bmm_76: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_496, view_700);  permute_496 = None
        bmm_77: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_700, permute_497);  view_700 = permute_497 = None
        view_701: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_76, [8, 8, 1024, 64]);  bmm_76 = None
        view_702: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_77, [8, 8, 1024, 1024]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_40: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_413: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
        mul_414: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_702, mul_413);  view_702 = mul_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_415: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_414, div_11);  mul_414 = None
        sum_67: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_415, [-1], True)
        neg_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_10: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_12, sum_67, mul_415);  neg_12 = sum_67 = mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_703: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_10, [64, 1024, 1024]);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_78: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_498, view_703);  permute_498 = None
        bmm_79: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_703, permute_499);  view_703 = permute_499 = None
        view_708: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_78, [8, 8, 64, 1024]);  bmm_78 = None
        view_709: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_79, [8, 8, 1024, 64]);  bmm_79 = None
        permute_500: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_708, [0, 1, 3, 2]);  view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_501: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_701, [0, 2, 1, 3]);  view_701 = None
        clone_160: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
        view_710: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_160, [8, 1024, 512]);  clone_160 = None
        view_711: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_710, [8192, 512]);  view_710 = None
        permute_502: "f32[512, 8192]" = torch.ops.aten.permute.default(view_711, [1, 0])
        view_180: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512])
        mm_205: "f32[512, 512]" = torch.ops.aten.mm.default(permute_502, view_180);  permute_502 = view_180 = None
        permute_81: "f32[512, 512]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_504: "f32[512, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_206: "f32[8192, 512]" = torch.ops.aten.mm.default(view_711, permute_504);  view_711 = permute_504 = None
        view_712: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_206, [8, 1024, 512]);  mm_206 = None
        add_154: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_144, view_712);  add_144 = view_712 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_506: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_500, [0, 2, 1, 3]);  permute_500 = None
        view_713: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_506, [8, 1024, 512]);  permute_506 = None
        clone_161: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_713, memory_format = torch.contiguous_format);  view_713 = None
        view_714: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_161, [8192, 512]);  clone_161 = None
        permute_507: "f32[512, 8192]" = torch.ops.aten.permute.default(view_714, [1, 0])
        view_177: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_79, [8192, 512]);  mul_79 = None
        mm_207: "f32[512, 512]" = torch.ops.aten.mm.default(permute_507, view_177);  permute_507 = view_177 = None
        permute_79: "f32[512, 512]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_509: "f32[512, 512]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_208: "f32[8192, 512]" = torch.ops.aten.mm.default(view_714, permute_509);  view_714 = permute_509 = None
        view_715: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_208, [8, 1024, 512]);  mm_208 = None
        add_155: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_154, view_715);  add_154 = view_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_511: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_709, [0, 2, 1, 3]);  view_709 = None
        clone_162: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_511, memory_format = torch.contiguous_format);  permute_511 = None
        view_716: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_162, [8, 1024, 512]);  clone_162 = None
        view_717: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_716, [8192, 512]);  view_716 = None
        permute_512: "f32[512, 8192]" = torch.ops.aten.permute.default(view_717, [1, 0])
        mm_209: "f32[512, 512]" = torch.ops.aten.mm.default(permute_512, view_174);  permute_512 = view_174 = None
        permute_77: "f32[512, 512]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_514: "f32[512, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_210: "f32[8192, 512]" = torch.ops.aten.mm.default(view_717, permute_514);  view_717 = permute_514 = None
        view_718: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_210, [8, 1024, 512]);  mm_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_416: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_718, primals_60);  primals_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_89: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_417: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_718, mul_89);  view_718 = mul_89 = None
        sum_68: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1], True);  mul_417 = None
        view_719: "f32[512]" = torch.ops.aten.reshape.default(sum_68, [512]);  sum_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_418: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_416, add_48)
        mul_419: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_416, rsqrt_14);  mul_416 = None
        sum_69: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
        add_156: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_153, mul_419);  add_153 = mul_419 = None
        pow_67: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_14, 3);  rsqrt_14 = None
        mul_420: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_69, -0.5);  sum_69 = None
        mul_421: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_420, pow_67);  mul_420 = pow_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_92: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_421, [8, 1024, 512]);  mul_421 = None
        div_41: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_92, 512);  expand_92 = None
        pow_68: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_48, 1.0);  add_48 = None
        mul_422: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_68, 2.0);  pow_68 = None
        mul_423: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_41, mul_422);  div_41 = mul_422 = None
        add_157: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_156, mul_423);  add_156 = mul_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_41: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_424: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
        mul_425: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_157, mul_424);  mul_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_720: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_425, [8192, 512]);  mul_425 = None
        permute_516: "f32[512, 8192]" = torch.ops.aten.permute.default(view_720, [1, 0])
        mm_211: "f32[512, 512]" = torch.ops.aten.mm.default(permute_516, view_172);  permute_516 = view_172 = None
        permute_76: "f32[512, 512]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_518: "f32[512, 512]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_212: "f32[8192, 512]" = torch.ops.aten.mm.default(view_720, permute_518);  view_720 = permute_518 = None
        view_721: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_212, [8, 1024, 512]);  mm_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_722: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_721, [8, 1024, 8, 64]);  view_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_520: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_722, [0, 2, 1, 3]);  view_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_164: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
        view_723: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_164, [64, 1024, 64]);  clone_164 = None
        bmm_80: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_521, view_723);  permute_521 = None
        bmm_81: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_723, permute_522);  view_723 = permute_522 = None
        view_724: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_80, [8, 8, 1024, 64]);  bmm_80 = None
        view_725: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_81, [8, 8, 1024, 1024]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_42: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_426: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
        mul_427: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_725, mul_426);  view_725 = mul_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_428: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_427, div_10);  mul_427 = None
        sum_70: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_428, [-1], True)
        neg_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_11: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_13, sum_70, mul_428);  neg_13 = sum_70 = mul_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_726: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_11, [64, 1024, 1024]);  fma_11 = None
        view_728: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_726, [8, 8, 1024, 1024]);  view_726 = None
        view_729: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_728, [64, 1024, 1024])
        add_158: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_147, view_728);  add_147 = view_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_71: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_158, [0], True);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_1: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_71, 0);  sum_71 = None
        permute_523: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_1: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_45, -1)
        unsqueeze_20: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_16: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_20, full_default, permute_523);  unsqueeze_20 = permute_523 = None
        clone_167: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_16, memory_format = torch.contiguous_format);  where_16 = None
        full_default_24: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default_24, [add_45], clone_167, True);  add_45 = clone_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_82: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_524, view_729);  permute_524 = None
        bmm_83: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_729, permute_525);  view_729 = permute_525 = None
        view_731: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_82, [8, 8, 64, 1024]);  bmm_82 = None
        view_732: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_83, [8, 8, 1024, 64]);  bmm_83 = None
        permute_526: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_731, [0, 1, 3, 2]);  view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_527: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_724, [0, 2, 1, 3]);  view_724 = None
        clone_168: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_527, memory_format = torch.contiguous_format);  permute_527 = None
        view_733: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_168, [8, 1024, 512]);  clone_168 = None
        view_734: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_733, [8192, 512]);  view_733 = None
        permute_528: "f32[512, 8192]" = torch.ops.aten.permute.default(view_734, [1, 0])
        mm_213: "f32[512, 512]" = torch.ops.aten.mm.default(permute_528, view_153);  permute_528 = None
        permute_71: "f32[512, 512]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_530: "f32[512, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_214: "f32[8192, 512]" = torch.ops.aten.mm.default(view_734, permute_530);  view_734 = permute_530 = None
        view_735: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_214, [8, 1024, 512]);  mm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_532: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_526, [0, 2, 1, 3]);  permute_526 = None
        view_736: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_532, [8, 1024, 512]);  permute_532 = None
        clone_169: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_736, memory_format = torch.contiguous_format);  view_736 = None
        view_737: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_169, [8192, 512]);  clone_169 = None
        permute_533: "f32[512, 8192]" = torch.ops.aten.permute.default(view_737, [1, 0])
        mm_215: "f32[512, 512]" = torch.ops.aten.mm.default(permute_533, view_153);  permute_533 = None
        permute_69: "f32[512, 512]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_535: "f32[512, 512]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        mm_216: "f32[8192, 512]" = torch.ops.aten.mm.default(view_737, permute_535);  view_737 = permute_535 = None
        view_738: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_216, [8, 1024, 512]);  mm_216 = None
        add_159: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_735, view_738);  view_735 = view_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_537: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None
        clone_170: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_537, memory_format = torch.contiguous_format);  permute_537 = None
        view_739: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_170, [8, 1024, 512]);  clone_170 = None
        view_740: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_739, [8192, 512]);  view_739 = None
        permute_538: "f32[512, 8192]" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_217: "f32[512, 512]" = torch.ops.aten.mm.default(permute_538, view_153);  permute_538 = view_153 = None
        permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_540: "f32[512, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        mm_218: "f32[8192, 512]" = torch.ops.aten.mm.default(view_740, permute_540);  view_740 = permute_540 = None
        view_741: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_218, [8, 1024, 512]);  mm_218 = None
        add_160: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_159, view_741);  add_159 = view_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_429: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_160, primals_54);  primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_80: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt_27, embedding_2);  embedding_2 = None
        mul_81: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_80, 1.1111111111111112);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_82: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_81, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_430: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_160, mul_82);  add_160 = mul_82 = None
        sum_72: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1], True);  mul_430 = None
        view_742: "f32[512]" = torch.ops.aten.reshape.default(sum_72, [512]);  sum_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_431: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_429, mul_81)
        mul_432: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_429, rsqrt_13);  mul_429 = None
        sum_73: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True);  mul_431 = None
        add_161: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_157, mul_432);  add_157 = mul_432 = None
        pow_69: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
        mul_433: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_73, -0.5);  sum_73 = None
        mul_434: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_433, pow_69);  mul_433 = pow_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_93: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_434, [8, 1024, 512]);  mul_434 = None
        div_42: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_93, 512);  expand_93 = None
        pow_70: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_81, 1.0);  mul_81 = None
        mul_435: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
        mul_436: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_42, mul_435);  div_42 = mul_435 = None
        add_162: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_161, mul_436);  add_161 = mul_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_43: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_437: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
        mul_438: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_162, mul_437);  add_162 = mul_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_2: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(where_2, -1)
        unsqueeze_21: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_17: "f32[8, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_21, full_default, mul_438);  unsqueeze_21 = mul_438 = None
        full_default_26: "f32[32128, 512]" = torch.ops.aten.full.default([32128, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[32128, 512]" = torch.ops.aten.index_put.default(full_default_26, [where_2], where_17, True);  where_2 = where_17 = None
        add_163: "f32[32128, 512]" = torch.ops.aten.add.Tensor(mm_97, index_put_1);  mm_97 = index_put_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_44: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_439: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
        mul_440: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_155, mul_439);  add_155 = mul_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_441: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_440, primals_52);  primals_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_76: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_36, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_442: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_440, mul_76);  mul_440 = mul_76 = None
        sum_74: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 1], True);  mul_442 = None
        view_743: "f32[512]" = torch.ops.aten.reshape.default(sum_74, [512]);  sum_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_443: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_441, add_36)
        mul_444: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_441, rsqrt_12);  mul_441 = None
        sum_75: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True);  mul_443 = None
        pow_71: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
        mul_445: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_75, -0.5);  sum_75 = None
        mul_446: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_445, pow_71);  mul_445 = pow_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_94: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_446, [8, 1024, 512]);  mul_446 = None
        div_43: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_94, 512);  expand_94 = None
        pow_72: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_36, 1.0);  add_36 = None
        mul_447: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
        mul_448: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_43, mul_447);  div_43 = mul_447 = None
        add_164: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_444, mul_448);  mul_444 = mul_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_45: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_449: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
        mul_450: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_164, mul_449);  mul_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_744: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_450, [8192, 512]);  mul_450 = None
        permute_542: "f32[512, 8192]" = torch.ops.aten.permute.default(view_744, [1, 0])
        mm_219: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_542, view_149);  permute_542 = view_149 = None
        permute_66: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_544: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_220: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_744, permute_544);  view_744 = permute_544 = None
        view_745: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_220, [8, 1024, 2048]);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_46: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_451: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
        mul_452: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_745, mul_451);  view_745 = mul_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_18: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_7, full_default, mul_452);  le_7 = mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_746: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_18, [8192, 2048]);  where_18 = None
        permute_546: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_746, [1, 0])
        mm_221: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_546, view_147);  permute_546 = view_147 = None
        permute_65: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_548: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_222: "f32[8192, 512]" = torch.ops.aten.mm.default(view_746, permute_548);  view_746 = permute_548 = None
        view_747: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_222, [8, 1024, 512]);  mm_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_453: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_747, primals_49);  primals_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_70: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_454: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_747, mul_70);  view_747 = mul_70 = None
        sum_76: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 1], True);  mul_454 = None
        view_748: "f32[512]" = torch.ops.aten.reshape.default(sum_76, [512]);  sum_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_455: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_453, add_34)
        mul_456: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_453, rsqrt_11);  mul_453 = None
        sum_77: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True);  mul_455 = None
        add_165: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_164, mul_456);  add_164 = mul_456 = None
        pow_73: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_11, 3);  rsqrt_11 = None
        mul_457: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_77, -0.5);  sum_77 = None
        mul_458: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_457, pow_73);  mul_457 = pow_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_95: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_458, [8, 1024, 512]);  mul_458 = None
        div_44: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_95, 512);  expand_95 = None
        pow_74: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 1.0);  add_34 = None
        mul_459: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
        mul_460: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_44, mul_459);  div_44 = mul_459 = None
        add_166: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_165, mul_460);  add_165 = mul_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_47: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_461: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
        mul_462: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_166, mul_461);  mul_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_749: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_462, [8192, 512]);  mul_462 = None
        permute_550: "f32[512, 8192]" = torch.ops.aten.permute.default(view_749, [1, 0])
        mm_223: "f32[512, 512]" = torch.ops.aten.mm.default(permute_550, view_145);  permute_550 = view_145 = None
        permute_64: "f32[512, 512]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_552: "f32[512, 512]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_224: "f32[8192, 512]" = torch.ops.aten.mm.default(view_749, permute_552);  view_749 = permute_552 = None
        view_750: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_224, [8, 1024, 512]);  mm_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_751: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_750, [8, 1024, 8, 64]);  view_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_554: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_751, [0, 2, 1, 3]);  view_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_176: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_554, memory_format = torch.contiguous_format);  permute_554 = None
        view_752: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_176, [64, 1024, 64]);  clone_176 = None
        bmm_84: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_555, view_752);  permute_555 = None
        bmm_85: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_752, permute_556);  view_752 = permute_556 = None
        view_753: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_84, [8, 8, 1024, 64]);  bmm_84 = None
        view_754: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_85, [8, 8, 1024, 1024]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_48: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_463: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
        mul_464: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_754, mul_463);  view_754 = mul_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_465: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_464, div_7);  mul_464 = None
        sum_78: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_465, [-1], True)
        neg_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_12: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_14, sum_78, mul_465);  neg_14 = sum_78 = mul_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_755: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_12, [64, 1024, 1024]);  fma_12 = None
        view_757: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_755, [8, 8, 1024, 1024]);  view_755 = None
        view_758: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_757, [64, 1024, 1024])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_86: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_557, view_758);  permute_557 = None
        bmm_87: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_758, permute_558);  view_758 = permute_558 = None
        view_760: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_86, [8, 8, 64, 1024]);  bmm_86 = None
        view_761: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_87, [8, 8, 1024, 64]);  bmm_87 = None
        permute_559: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_760, [0, 1, 3, 2]);  view_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_560: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_753, [0, 2, 1, 3]);  view_753 = None
        clone_179: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format);  permute_560 = None
        view_762: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_179, [8, 1024, 512]);  clone_179 = None
        view_763: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_762, [8192, 512]);  view_762 = None
        permute_561: "f32[512, 8192]" = torch.ops.aten.permute.default(view_763, [1, 0])
        mm_225: "f32[512, 512]" = torch.ops.aten.mm.default(permute_561, view_126);  permute_561 = None
        permute_60: "f32[512, 512]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
        permute_563: "f32[512, 512]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_226: "f32[8192, 512]" = torch.ops.aten.mm.default(view_763, permute_563);  view_763 = permute_563 = None
        view_764: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_226, [8, 1024, 512]);  mm_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_565: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_559, [0, 2, 1, 3]);  permute_559 = None
        view_765: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_565, [8, 1024, 512]);  permute_565 = None
        clone_180: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_765, memory_format = torch.contiguous_format);  view_765 = None
        view_766: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_180, [8192, 512]);  clone_180 = None
        permute_566: "f32[512, 8192]" = torch.ops.aten.permute.default(view_766, [1, 0])
        mm_227: "f32[512, 512]" = torch.ops.aten.mm.default(permute_566, view_126);  permute_566 = None
        permute_58: "f32[512, 512]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        permute_568: "f32[512, 512]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        mm_228: "f32[8192, 512]" = torch.ops.aten.mm.default(view_766, permute_568);  view_766 = permute_568 = None
        view_767: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_228, [8, 1024, 512]);  mm_228 = None
        add_167: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_764, view_767);  view_764 = view_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_570: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_761, [0, 2, 1, 3]);  view_761 = None
        clone_181: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_570, memory_format = torch.contiguous_format);  permute_570 = None
        view_768: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_181, [8, 1024, 512]);  clone_181 = None
        view_769: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_768, [8192, 512]);  view_768 = None
        permute_571: "f32[512, 8192]" = torch.ops.aten.permute.default(view_769, [1, 0])
        mm_229: "f32[512, 512]" = torch.ops.aten.mm.default(permute_571, view_126);  permute_571 = view_126 = None
        permute_56: "f32[512, 512]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_573: "f32[512, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_230: "f32[8192, 512]" = torch.ops.aten.mm.default(view_769, permute_573);  view_769 = permute_573 = None
        view_770: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_230, [8, 1024, 512]);  mm_230 = None
        add_168: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_167, view_770);  add_167 = view_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_466: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_168, primals_44);  primals_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_64: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_467: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_168, mul_64);  add_168 = mul_64 = None
        sum_79: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 1], True);  mul_467 = None
        view_771: "f32[512]" = torch.ops.aten.reshape.default(sum_79, [512]);  sum_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_468: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_466, add_31)
        mul_469: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_466, rsqrt_10);  mul_466 = None
        sum_80: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_468, [2], True);  mul_468 = None
        add_169: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_166, mul_469);  add_166 = mul_469 = None
        pow_75: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_10, 3);  rsqrt_10 = None
        mul_470: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_80, -0.5);  sum_80 = None
        mul_471: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_470, pow_75);  mul_470 = pow_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_96: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_471, [8, 1024, 512]);  mul_471 = None
        div_45: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_96, 512);  expand_96 = None
        pow_76: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 1.0);  add_31 = None
        mul_472: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_76, 2.0);  pow_76 = None
        mul_473: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_45, mul_472);  div_45 = mul_472 = None
        add_170: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_169, mul_473);  add_169 = mul_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_49: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_474: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
        mul_475: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_170, mul_474);  mul_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_772: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_475, [8192, 512]);  mul_475 = None
        permute_575: "f32[512, 8192]" = torch.ops.aten.permute.default(view_772, [1, 0])
        mm_231: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_575, view_124);  permute_575 = view_124 = None
        permute_55: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_577: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_232: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_772, permute_577);  view_772 = permute_577 = None
        view_773: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_232, [8, 1024, 2048]);  mm_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_50: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_476: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
        mul_477: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_773, mul_476);  view_773 = mul_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_19: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_8, full_default, mul_477);  le_8 = mul_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_774: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_19, [8192, 2048]);  where_19 = None
        permute_579: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_774, [1, 0])
        mm_233: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_579, view_122);  permute_579 = view_122 = None
        permute_54: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_581: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_234: "f32[8192, 512]" = torch.ops.aten.mm.default(view_774, permute_581);  view_774 = permute_581 = None
        view_775: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_234, [8, 1024, 512]);  mm_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_478: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_775, primals_41);  primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_58: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_29, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_479: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_775, mul_58);  view_775 = mul_58 = None
        sum_81: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1], True);  mul_479 = None
        view_776: "f32[512]" = torch.ops.aten.reshape.default(sum_81, [512]);  sum_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_480: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_478, add_29)
        mul_481: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_478, rsqrt_9);  mul_478 = None
        sum_82: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [2], True);  mul_480 = None
        add_171: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_170, mul_481);  add_170 = mul_481 = None
        pow_77: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_9, 3);  rsqrt_9 = None
        mul_482: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_82, -0.5);  sum_82 = None
        mul_483: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_482, pow_77);  mul_482 = pow_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_97: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_483, [8, 1024, 512]);  mul_483 = None
        div_46: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_97, 512);  expand_97 = None
        pow_78: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_29, 1.0);  add_29 = None
        mul_484: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_78, 2.0);  pow_78 = None
        mul_485: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_46, mul_484);  div_46 = mul_484 = None
        add_172: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_171, mul_485);  add_171 = mul_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_51: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_486: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
        mul_487: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_172, mul_486);  mul_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_777: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_487, [8192, 512]);  mul_487 = None
        permute_583: "f32[512, 8192]" = torch.ops.aten.permute.default(view_777, [1, 0])
        mm_235: "f32[512, 512]" = torch.ops.aten.mm.default(permute_583, view_120);  permute_583 = view_120 = None
        permute_53: "f32[512, 512]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
        permute_585: "f32[512, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_236: "f32[8192, 512]" = torch.ops.aten.mm.default(view_777, permute_585);  view_777 = permute_585 = None
        view_778: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_236, [8, 1024, 512]);  mm_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_779: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_778, [8, 1024, 8, 64]);  view_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_587: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_779, [0, 2, 1, 3]);  view_779 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_185: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
        view_780: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_185, [64, 1024, 64]);  clone_185 = None
        bmm_88: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_588, view_780);  permute_588 = None
        bmm_89: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_780, permute_589);  view_780 = permute_589 = None
        view_781: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_88, [8, 8, 1024, 64]);  bmm_88 = None
        view_782: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_89, [8, 8, 1024, 1024]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_52: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_488: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_489: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_782, mul_488);  view_782 = mul_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_490: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_489, div_6);  mul_489 = None
        sum_83: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_490, [-1], True)
        neg_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_13: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_15, sum_83, mul_490);  neg_15 = sum_83 = mul_490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_783: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_13, [64, 1024, 1024]);  fma_13 = None
        view_785: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_783, [8, 8, 1024, 1024]);  view_783 = None
        view_786: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_785, [64, 1024, 1024])
        add_173: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_757, view_785);  view_757 = view_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_90: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_590, view_786);  permute_590 = None
        bmm_91: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_786, permute_591);  view_786 = permute_591 = None
        view_788: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_90, [8, 8, 64, 1024]);  bmm_90 = None
        view_789: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_91, [8, 8, 1024, 64]);  bmm_91 = None
        permute_592: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_788, [0, 1, 3, 2]);  view_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_593: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_781, [0, 2, 1, 3]);  view_781 = None
        clone_188: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_593, memory_format = torch.contiguous_format);  permute_593 = None
        view_790: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_188, [8, 1024, 512]);  clone_188 = None
        view_791: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_790, [8192, 512]);  view_790 = None
        permute_594: "f32[512, 8192]" = torch.ops.aten.permute.default(view_791, [1, 0])
        mm_237: "f32[512, 512]" = torch.ops.aten.mm.default(permute_594, view_101);  permute_594 = None
        permute_49: "f32[512, 512]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_596: "f32[512, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        mm_238: "f32[8192, 512]" = torch.ops.aten.mm.default(view_791, permute_596);  view_791 = permute_596 = None
        view_792: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_238, [8, 1024, 512]);  mm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_598: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_592, [0, 2, 1, 3]);  permute_592 = None
        view_793: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_598, [8, 1024, 512]);  permute_598 = None
        clone_189: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_793, memory_format = torch.contiguous_format);  view_793 = None
        view_794: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_189, [8192, 512]);  clone_189 = None
        permute_599: "f32[512, 8192]" = torch.ops.aten.permute.default(view_794, [1, 0])
        mm_239: "f32[512, 512]" = torch.ops.aten.mm.default(permute_599, view_101);  permute_599 = None
        permute_47: "f32[512, 512]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_601: "f32[512, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_240: "f32[8192, 512]" = torch.ops.aten.mm.default(view_794, permute_601);  view_794 = permute_601 = None
        view_795: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_240, [8, 1024, 512]);  mm_240 = None
        add_174: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_792, view_795);  view_792 = view_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_603: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_789, [0, 2, 1, 3]);  view_789 = None
        clone_190: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
        view_796: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_190, [8, 1024, 512]);  clone_190 = None
        view_797: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_796, [8192, 512]);  view_796 = None
        permute_604: "f32[512, 8192]" = torch.ops.aten.permute.default(view_797, [1, 0])
        mm_241: "f32[512, 512]" = torch.ops.aten.mm.default(permute_604, view_101);  permute_604 = view_101 = None
        permute_45: "f32[512, 512]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_606: "f32[512, 512]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_242: "f32[8192, 512]" = torch.ops.aten.mm.default(view_797, permute_606);  view_797 = permute_606 = None
        view_798: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_242, [8, 1024, 512]);  mm_242 = None
        add_175: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_174, view_798);  add_174 = view_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_491: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_175, primals_36);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_52: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_26, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_492: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_175, mul_52);  add_175 = mul_52 = None
        sum_84: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_492, [0, 1], True);  mul_492 = None
        view_799: "f32[512]" = torch.ops.aten.reshape.default(sum_84, [512]);  sum_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_493: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_491, add_26)
        mul_494: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_491, rsqrt_8);  mul_491 = None
        sum_85: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_493, [2], True);  mul_493 = None
        add_176: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_172, mul_494);  add_172 = mul_494 = None
        pow_79: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_8, 3);  rsqrt_8 = None
        mul_495: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_85, -0.5);  sum_85 = None
        mul_496: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_495, pow_79);  mul_495 = pow_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_98: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_496, [8, 1024, 512]);  mul_496 = None
        div_47: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_98, 512);  expand_98 = None
        pow_80: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_26, 1.0);  add_26 = None
        mul_497: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_80, 2.0);  pow_80 = None
        mul_498: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_47, mul_497);  div_47 = mul_497 = None
        add_177: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_176, mul_498);  add_176 = mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_53: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_499: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
        mul_500: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_177, mul_499);  mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_800: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_500, [8192, 512]);  mul_500 = None
        permute_608: "f32[512, 8192]" = torch.ops.aten.permute.default(view_800, [1, 0])
        mm_243: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_608, view_99);  permute_608 = view_99 = None
        permute_44: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_610: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_244: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_800, permute_610);  view_800 = permute_610 = None
        view_801: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_244, [8, 1024, 2048]);  mm_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_54: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_501: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
        mul_502: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_801, mul_501);  view_801 = mul_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_20: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_9, full_default, mul_502);  le_9 = mul_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_802: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_20, [8192, 2048]);  where_20 = None
        permute_612: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_802, [1, 0])
        mm_245: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_612, view_97);  permute_612 = view_97 = None
        permute_43: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_614: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_246: "f32[8192, 512]" = torch.ops.aten.mm.default(view_802, permute_614);  view_802 = permute_614 = None
        view_803: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_246, [8, 1024, 512]);  mm_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_503: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_803, primals_33);  primals_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_46: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_504: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_803, mul_46);  view_803 = mul_46 = None
        sum_86: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 1], True);  mul_504 = None
        view_804: "f32[512]" = torch.ops.aten.reshape.default(sum_86, [512]);  sum_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_505: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_503, add_24)
        mul_506: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_503, rsqrt_7);  mul_503 = None
        sum_87: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [2], True);  mul_505 = None
        add_178: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_177, mul_506);  add_177 = mul_506 = None
        pow_81: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_7, 3);  rsqrt_7 = None
        mul_507: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_87, -0.5);  sum_87 = None
        mul_508: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_507, pow_81);  mul_507 = pow_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_99: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_508, [8, 1024, 512]);  mul_508 = None
        div_48: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_99, 512);  expand_99 = None
        pow_82: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 1.0);  add_24 = None
        mul_509: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_82, 2.0);  pow_82 = None
        mul_510: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_48, mul_509);  div_48 = mul_509 = None
        add_179: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_178, mul_510);  add_178 = mul_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_55: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_511: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
        mul_512: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_179, mul_511);  mul_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_805: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_512, [8192, 512]);  mul_512 = None
        permute_616: "f32[512, 8192]" = torch.ops.aten.permute.default(view_805, [1, 0])
        mm_247: "f32[512, 512]" = torch.ops.aten.mm.default(permute_616, view_95);  permute_616 = view_95 = None
        permute_42: "f32[512, 512]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_618: "f32[512, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_248: "f32[8192, 512]" = torch.ops.aten.mm.default(view_805, permute_618);  view_805 = permute_618 = None
        view_806: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_248, [8, 1024, 512]);  mm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_807: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_806, [8, 1024, 8, 64]);  view_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_620: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_807, [0, 2, 1, 3]);  view_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_194: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_620, memory_format = torch.contiguous_format);  permute_620 = None
        view_808: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_194, [64, 1024, 64]);  clone_194 = None
        bmm_92: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_621, view_808);  permute_621 = None
        bmm_93: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_808, permute_622);  view_808 = permute_622 = None
        view_809: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_92, [8, 8, 1024, 64]);  bmm_92 = None
        view_810: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_93, [8, 8, 1024, 1024]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_56: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_513: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
        mul_514: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_810, mul_513);  view_810 = mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_515: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_514, div_5);  mul_514 = None
        sum_88: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_515, [-1], True)
        neg_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_14: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_16, sum_88, mul_515);  neg_16 = sum_88 = mul_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_811: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_14, [64, 1024, 1024]);  fma_14 = None
        view_813: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_811, [8, 8, 1024, 1024]);  view_811 = None
        view_814: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_813, [64, 1024, 1024])
        add_180: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_173, view_813);  add_173 = view_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_94: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_623, view_814);  permute_623 = None
        bmm_95: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_814, permute_624);  view_814 = permute_624 = None
        view_816: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_94, [8, 8, 64, 1024]);  bmm_94 = None
        view_817: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_95, [8, 8, 1024, 64]);  bmm_95 = None
        permute_625: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_816, [0, 1, 3, 2]);  view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_626: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_809, [0, 2, 1, 3]);  view_809 = None
        clone_197: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_626, memory_format = torch.contiguous_format);  permute_626 = None
        view_818: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_197, [8, 1024, 512]);  clone_197 = None
        view_819: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_818, [8192, 512]);  view_818 = None
        permute_627: "f32[512, 8192]" = torch.ops.aten.permute.default(view_819, [1, 0])
        mm_249: "f32[512, 512]" = torch.ops.aten.mm.default(permute_627, view_76);  permute_627 = None
        permute_38: "f32[512, 512]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_629: "f32[512, 512]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        mm_250: "f32[8192, 512]" = torch.ops.aten.mm.default(view_819, permute_629);  view_819 = permute_629 = None
        view_820: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_250, [8, 1024, 512]);  mm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_631: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_625, [0, 2, 1, 3]);  permute_625 = None
        view_821: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_631, [8, 1024, 512]);  permute_631 = None
        clone_198: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_821, memory_format = torch.contiguous_format);  view_821 = None
        view_822: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_198, [8192, 512]);  clone_198 = None
        permute_632: "f32[512, 8192]" = torch.ops.aten.permute.default(view_822, [1, 0])
        mm_251: "f32[512, 512]" = torch.ops.aten.mm.default(permute_632, view_76);  permute_632 = None
        permute_36: "f32[512, 512]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_634: "f32[512, 512]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_252: "f32[8192, 512]" = torch.ops.aten.mm.default(view_822, permute_634);  view_822 = permute_634 = None
        view_823: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_252, [8, 1024, 512]);  mm_252 = None
        add_181: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_820, view_823);  view_820 = view_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_636: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_817, [0, 2, 1, 3]);  view_817 = None
        clone_199: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_636, memory_format = torch.contiguous_format);  permute_636 = None
        view_824: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_199, [8, 1024, 512]);  clone_199 = None
        view_825: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_824, [8192, 512]);  view_824 = None
        permute_637: "f32[512, 8192]" = torch.ops.aten.permute.default(view_825, [1, 0])
        mm_253: "f32[512, 512]" = torch.ops.aten.mm.default(permute_637, view_76);  permute_637 = view_76 = None
        permute_34: "f32[512, 512]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_639: "f32[512, 512]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_254: "f32[8192, 512]" = torch.ops.aten.mm.default(view_825, permute_639);  view_825 = permute_639 = None
        view_826: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_254, [8, 1024, 512]);  mm_254 = None
        add_182: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_181, view_826);  add_181 = view_826 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_516: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_182, primals_28);  primals_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_40: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_21, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_517: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_182, mul_40);  add_182 = mul_40 = None
        sum_89: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1], True);  mul_517 = None
        view_827: "f32[512]" = torch.ops.aten.reshape.default(sum_89, [512]);  sum_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_518: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_516, add_21)
        mul_519: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_516, rsqrt_6);  mul_516 = None
        sum_90: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_518, [2], True);  mul_518 = None
        add_183: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_179, mul_519);  add_179 = mul_519 = None
        pow_83: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_6, 3);  rsqrt_6 = None
        mul_520: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_90, -0.5);  sum_90 = None
        mul_521: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_520, pow_83);  mul_520 = pow_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_100: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_521, [8, 1024, 512]);  mul_521 = None
        div_49: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_100, 512);  expand_100 = None
        pow_84: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_21, 1.0);  add_21 = None
        mul_522: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_84, 2.0);  pow_84 = None
        mul_523: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_49, mul_522);  div_49 = mul_522 = None
        add_184: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_183, mul_523);  add_183 = mul_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_57: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_524: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
        mul_525: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_184, mul_524);  mul_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_828: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_525, [8192, 512]);  mul_525 = None
        permute_641: "f32[512, 8192]" = torch.ops.aten.permute.default(view_828, [1, 0])
        mm_255: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_641, view_74);  permute_641 = view_74 = None
        permute_33: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_643: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_256: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_828, permute_643);  view_828 = permute_643 = None
        view_829: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_256, [8, 1024, 2048]);  mm_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_58: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_526: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
        mul_527: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_829, mul_526);  view_829 = mul_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_21: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_10, full_default, mul_527);  le_10 = mul_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_830: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_21, [8192, 2048]);  where_21 = None
        permute_645: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_830, [1, 0])
        mm_257: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_645, view_72);  permute_645 = view_72 = None
        permute_32: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_647: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_258: "f32[8192, 512]" = torch.ops.aten.mm.default(view_830, permute_647);  view_830 = permute_647 = None
        view_831: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_258, [8, 1024, 512]);  mm_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_528: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_831, primals_25);  primals_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_34: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_19, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_529: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_831, mul_34);  view_831 = mul_34 = None
        sum_91: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 1], True);  mul_529 = None
        view_832: "f32[512]" = torch.ops.aten.reshape.default(sum_91, [512]);  sum_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_530: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_528, add_19)
        mul_531: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_528, rsqrt_5);  mul_528 = None
        sum_92: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_530, [2], True);  mul_530 = None
        add_185: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_184, mul_531);  add_184 = mul_531 = None
        pow_85: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_5, 3);  rsqrt_5 = None
        mul_532: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_92, -0.5);  sum_92 = None
        mul_533: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_532, pow_85);  mul_532 = pow_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_101: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_533, [8, 1024, 512]);  mul_533 = None
        div_50: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_101, 512);  expand_101 = None
        pow_86: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_19, 1.0);  add_19 = None
        mul_534: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_86, 2.0);  pow_86 = None
        mul_535: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_50, mul_534);  div_50 = mul_534 = None
        add_186: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_185, mul_535);  add_185 = mul_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_59: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_536: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
        mul_537: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_186, mul_536);  mul_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_833: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_537, [8192, 512]);  mul_537 = None
        permute_649: "f32[512, 8192]" = torch.ops.aten.permute.default(view_833, [1, 0])
        mm_259: "f32[512, 512]" = torch.ops.aten.mm.default(permute_649, view_70);  permute_649 = view_70 = None
        permute_31: "f32[512, 512]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_651: "f32[512, 512]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_260: "f32[8192, 512]" = torch.ops.aten.mm.default(view_833, permute_651);  view_833 = permute_651 = None
        view_834: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_260, [8, 1024, 512]);  mm_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_835: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_834, [8, 1024, 8, 64]);  view_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_653: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_835, [0, 2, 1, 3]);  view_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_203: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_653, memory_format = torch.contiguous_format);  permute_653 = None
        view_836: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_203, [64, 1024, 64]);  clone_203 = None
        bmm_96: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_654, view_836);  permute_654 = None
        bmm_97: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_836, permute_655);  view_836 = permute_655 = None
        view_837: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_96, [8, 8, 1024, 64]);  bmm_96 = None
        view_838: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_97, [8, 8, 1024, 1024]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_60: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_538: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
        mul_539: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_838, mul_538);  view_838 = mul_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_540: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_539, div_4);  mul_539 = None
        sum_93: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_540, [-1], True)
        neg_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_15: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_17, sum_93, mul_540);  neg_17 = sum_93 = mul_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_839: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_15, [64, 1024, 1024]);  fma_15 = None
        view_841: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_839, [8, 8, 1024, 1024]);  view_839 = None
        view_842: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_841, [64, 1024, 1024])
        add_187: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_180, view_841);  add_180 = view_841 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_98: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_656, view_842);  permute_656 = None
        bmm_99: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_842, permute_657);  view_842 = permute_657 = None
        view_844: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_98, [8, 8, 64, 1024]);  bmm_98 = None
        view_845: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_99, [8, 8, 1024, 64]);  bmm_99 = None
        permute_658: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_844, [0, 1, 3, 2]);  view_844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_659: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_837, [0, 2, 1, 3]);  view_837 = None
        clone_206: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_659, memory_format = torch.contiguous_format);  permute_659 = None
        view_846: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_206, [8, 1024, 512]);  clone_206 = None
        view_847: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_846, [8192, 512]);  view_846 = None
        permute_660: "f32[512, 8192]" = torch.ops.aten.permute.default(view_847, [1, 0])
        mm_261: "f32[512, 512]" = torch.ops.aten.mm.default(permute_660, view_51);  permute_660 = None
        permute_27: "f32[512, 512]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_662: "f32[512, 512]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        mm_262: "f32[8192, 512]" = torch.ops.aten.mm.default(view_847, permute_662);  view_847 = permute_662 = None
        view_848: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_262, [8, 1024, 512]);  mm_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_664: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_658, [0, 2, 1, 3]);  permute_658 = None
        view_849: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_664, [8, 1024, 512]);  permute_664 = None
        clone_207: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_849, memory_format = torch.contiguous_format);  view_849 = None
        view_850: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_207, [8192, 512]);  clone_207 = None
        permute_665: "f32[512, 8192]" = torch.ops.aten.permute.default(view_850, [1, 0])
        mm_263: "f32[512, 512]" = torch.ops.aten.mm.default(permute_665, view_51);  permute_665 = None
        permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
        permute_667: "f32[512, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        mm_264: "f32[8192, 512]" = torch.ops.aten.mm.default(view_850, permute_667);  view_850 = permute_667 = None
        view_851: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_264, [8, 1024, 512]);  mm_264 = None
        add_188: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_848, view_851);  view_848 = view_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_669: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_845, [0, 2, 1, 3]);  view_845 = None
        clone_208: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_669, memory_format = torch.contiguous_format);  permute_669 = None
        view_852: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_208, [8, 1024, 512]);  clone_208 = None
        view_853: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_852, [8192, 512]);  view_852 = None
        permute_670: "f32[512, 8192]" = torch.ops.aten.permute.default(view_853, [1, 0])
        mm_265: "f32[512, 512]" = torch.ops.aten.mm.default(permute_670, view_51);  permute_670 = view_51 = None
        permute_23: "f32[512, 512]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_672: "f32[512, 512]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_266: "f32[8192, 512]" = torch.ops.aten.mm.default(view_853, permute_672);  view_853 = permute_672 = None
        view_854: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_266, [8, 1024, 512]);  mm_266 = None
        add_189: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_188, view_854);  add_188 = view_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_541: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_189, primals_20);  primals_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_28: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_542: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_189, mul_28);  add_189 = mul_28 = None
        sum_94: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_542, [0, 1], True);  mul_542 = None
        view_855: "f32[512]" = torch.ops.aten.reshape.default(sum_94, [512]);  sum_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_543: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_541, add_16)
        mul_544: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_541, rsqrt_4);  mul_541 = None
        sum_95: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_543, [2], True);  mul_543 = None
        add_190: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_186, mul_544);  add_186 = mul_544 = None
        pow_87: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_4, 3);  rsqrt_4 = None
        mul_545: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_95, -0.5);  sum_95 = None
        mul_546: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_545, pow_87);  mul_545 = pow_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_102: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_546, [8, 1024, 512]);  mul_546 = None
        div_51: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_102, 512);  expand_102 = None
        pow_88: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 1.0);  add_16 = None
        mul_547: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_88, 2.0);  pow_88 = None
        mul_548: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_51, mul_547);  div_51 = mul_547 = None
        add_191: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_190, mul_548);  add_190 = mul_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_61: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_549: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
        mul_550: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_191, mul_549);  mul_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_856: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_550, [8192, 512]);  mul_550 = None
        permute_674: "f32[512, 8192]" = torch.ops.aten.permute.default(view_856, [1, 0])
        mm_267: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_674, view_49);  permute_674 = view_49 = None
        permute_22: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_676: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_268: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_856, permute_676);  view_856 = permute_676 = None
        view_857: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_268, [8, 1024, 2048]);  mm_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_62: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_551: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
        mul_552: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_857, mul_551);  view_857 = mul_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_22: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_11, full_default, mul_552);  le_11 = mul_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_858: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_22, [8192, 2048]);  where_22 = None
        permute_678: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_858, [1, 0])
        mm_269: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_678, view_47);  permute_678 = view_47 = None
        permute_21: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_18, [1, 0]);  primals_18 = None
        permute_680: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_270: "f32[8192, 512]" = torch.ops.aten.mm.default(view_858, permute_680);  view_858 = permute_680 = None
        view_859: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_270, [8, 1024, 512]);  mm_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_553: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_859, primals_17);  primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_22: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_14, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_554: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_859, mul_22);  view_859 = mul_22 = None
        sum_96: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 1], True);  mul_554 = None
        view_860: "f32[512]" = torch.ops.aten.reshape.default(sum_96, [512]);  sum_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_555: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_553, add_14)
        mul_556: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_553, rsqrt_3);  mul_553 = None
        sum_97: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_555, [2], True);  mul_555 = None
        add_192: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_191, mul_556);  add_191 = mul_556 = None
        pow_89: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_557: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_97, -0.5);  sum_97 = None
        mul_558: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_557, pow_89);  mul_557 = pow_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_103: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_558, [8, 1024, 512]);  mul_558 = None
        div_52: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_103, 512);  expand_103 = None
        pow_90: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_14, 1.0);  add_14 = None
        mul_559: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_90, 2.0);  pow_90 = None
        mul_560: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_52, mul_559);  div_52 = mul_559 = None
        add_193: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_192, mul_560);  add_192 = mul_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_63: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_561: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
        mul_562: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_193, mul_561);  mul_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_861: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_562, [8192, 512]);  mul_562 = None
        permute_682: "f32[512, 8192]" = torch.ops.aten.permute.default(view_861, [1, 0])
        mm_271: "f32[512, 512]" = torch.ops.aten.mm.default(permute_682, view_45);  permute_682 = view_45 = None
        permute_20: "f32[512, 512]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_684: "f32[512, 512]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_272: "f32[8192, 512]" = torch.ops.aten.mm.default(view_861, permute_684);  view_861 = permute_684 = None
        view_862: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_272, [8, 1024, 512]);  mm_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_863: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_862, [8, 1024, 8, 64]);  view_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_686: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_863, [0, 2, 1, 3]);  view_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_212: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_686, memory_format = torch.contiguous_format);  permute_686 = None
        view_864: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_212, [64, 1024, 64]);  clone_212 = None
        bmm_100: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_687, view_864);  permute_687 = None
        bmm_101: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_864, permute_688);  view_864 = permute_688 = None
        view_865: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_100, [8, 8, 1024, 64]);  bmm_100 = None
        view_866: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_101, [8, 8, 1024, 1024]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_64: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_563: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
        mul_564: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_866, mul_563);  view_866 = mul_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_565: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_564, div_3);  mul_564 = None
        sum_98: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_565, [-1], True)
        neg_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_16: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_18, sum_98, mul_565);  neg_18 = sum_98 = mul_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_867: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_16, [64, 1024, 1024]);  fma_16 = None
        view_869: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_867, [8, 8, 1024, 1024]);  view_867 = None
        view_870: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_869, [64, 1024, 1024])
        add_194: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_187, view_869);  add_187 = view_869 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_102: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_689, view_870);  permute_689 = None
        bmm_103: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_870, permute_690);  view_870 = permute_690 = None
        view_872: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_102, [8, 8, 64, 1024]);  bmm_102 = None
        view_873: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_103, [8, 8, 1024, 64]);  bmm_103 = None
        permute_691: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_872, [0, 1, 3, 2]);  view_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_692: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_865, [0, 2, 1, 3]);  view_865 = None
        clone_215: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_692, memory_format = torch.contiguous_format);  permute_692 = None
        view_874: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_215, [8, 1024, 512]);  clone_215 = None
        view_875: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_874, [8192, 512]);  view_874 = None
        permute_693: "f32[512, 8192]" = torch.ops.aten.permute.default(view_875, [1, 0])
        mm_273: "f32[512, 512]" = torch.ops.aten.mm.default(permute_693, view_26);  permute_693 = None
        permute_16: "f32[512, 512]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_695: "f32[512, 512]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm_274: "f32[8192, 512]" = torch.ops.aten.mm.default(view_875, permute_695);  view_875 = permute_695 = None
        view_876: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_274, [8, 1024, 512]);  mm_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_697: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_691, [0, 2, 1, 3]);  permute_691 = None
        view_877: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_697, [8, 1024, 512]);  permute_697 = None
        clone_216: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_877, memory_format = torch.contiguous_format);  view_877 = None
        view_878: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_216, [8192, 512]);  clone_216 = None
        permute_698: "f32[512, 8192]" = torch.ops.aten.permute.default(view_878, [1, 0])
        mm_275: "f32[512, 512]" = torch.ops.aten.mm.default(permute_698, view_26);  permute_698 = None
        permute_14: "f32[512, 512]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_700: "f32[512, 512]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        mm_276: "f32[8192, 512]" = torch.ops.aten.mm.default(view_878, permute_700);  view_878 = permute_700 = None
        view_879: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_276, [8, 1024, 512]);  mm_276 = None
        add_195: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_876, view_879);  view_876 = view_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_702: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_873, [0, 2, 1, 3]);  view_873 = None
        clone_217: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_702, memory_format = torch.contiguous_format);  permute_702 = None
        view_880: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_217, [8, 1024, 512]);  clone_217 = None
        view_881: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_880, [8192, 512]);  view_880 = None
        permute_703: "f32[512, 8192]" = torch.ops.aten.permute.default(view_881, [1, 0])
        mm_277: "f32[512, 512]" = torch.ops.aten.mm.default(permute_703, view_26);  permute_703 = view_26 = None
        permute_12: "f32[512, 512]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_705: "f32[512, 512]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_278: "f32[8192, 512]" = torch.ops.aten.mm.default(view_881, permute_705);  view_881 = permute_705 = None
        view_882: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_278, [8, 1024, 512]);  mm_278 = None
        add_196: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_195, view_882);  add_195 = view_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_566: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_196, primals_12);  primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_16: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_11, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_567: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_196, mul_16);  add_196 = mul_16 = None
        sum_99: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_567, [0, 1], True);  mul_567 = None
        view_883: "f32[512]" = torch.ops.aten.reshape.default(sum_99, [512]);  sum_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_568: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_566, add_11)
        mul_569: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_566, rsqrt_2);  mul_566 = None
        sum_100: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_568, [2], True);  mul_568 = None
        add_197: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_193, mul_569);  add_193 = mul_569 = None
        pow_91: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_570: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_100, -0.5);  sum_100 = None
        mul_571: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_570, pow_91);  mul_570 = pow_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_104: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_571, [8, 1024, 512]);  mul_571 = None
        div_53: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_104, 512);  expand_104 = None
        pow_92: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_11, 1.0);  add_11 = None
        mul_572: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_92, 2.0);  pow_92 = None
        mul_573: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_53, mul_572);  div_53 = mul_572 = None
        add_198: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_197, mul_573);  add_197 = mul_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_65: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_574: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
        mul_575: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_198, mul_574);  mul_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_884: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_575, [8192, 512]);  mul_575 = None
        permute_707: "f32[512, 8192]" = torch.ops.aten.permute.default(view_884, [1, 0])
        mm_279: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_707, view_24);  permute_707 = view_24 = None
        permute_11: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_709: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_280: "f32[8192, 2048]" = torch.ops.aten.mm.default(view_884, permute_709);  view_884 = permute_709 = None
        view_885: "f32[8, 1024, 2048]" = torch.ops.aten.reshape.default(mm_280, [8, 1024, 2048]);  mm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_66: "f32[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_576: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_577: "f32[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(view_885, mul_576);  view_885 = mul_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_23: "f32[8, 1024, 2048]" = torch.ops.aten.where.self(le_12, full_default, mul_577);  le_12 = mul_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_886: "f32[8192, 2048]" = torch.ops.aten.reshape.default(where_23, [8192, 2048]);  where_23 = None
        permute_711: "f32[2048, 8192]" = torch.ops.aten.permute.default(view_886, [1, 0])
        mm_281: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_711, view_22);  permute_711 = view_22 = None
        permute_10: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_713: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_282: "f32[8192, 512]" = torch.ops.aten.mm.default(view_886, permute_713);  view_886 = permute_713 = None
        view_887: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_282, [8, 1024, 512]);  mm_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_578: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_887, primals_9);  primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_10: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_579: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(view_887, mul_10);  view_887 = mul_10 = None
        sum_101: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 1], True);  mul_579 = None
        view_888: "f32[512]" = torch.ops.aten.reshape.default(sum_101, [512]);  sum_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_580: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_578, add_9)
        mul_581: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_578, rsqrt_1);  mul_578 = None
        sum_102: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_580, [2], True);  mul_580 = None
        add_199: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_198, mul_581);  add_198 = mul_581 = None
        pow_93: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_582: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_102, -0.5);  sum_102 = None
        mul_583: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_582, pow_93);  mul_582 = pow_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_105: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_583, [8, 1024, 512]);  mul_583 = None
        div_54: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_105, 512);  expand_105 = None
        pow_94: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 1.0);  add_9 = None
        mul_584: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_94, 2.0);  pow_94 = None
        mul_585: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_54, mul_584);  div_54 = mul_584 = None
        add_200: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_199, mul_585);  add_199 = mul_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_67: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_586: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
        mul_587: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_200, mul_586);  mul_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_889: "f32[8192, 512]" = torch.ops.aten.reshape.default(mul_587, [8192, 512]);  mul_587 = None
        permute_715: "f32[512, 8192]" = torch.ops.aten.permute.default(view_889, [1, 0])
        mm_283: "f32[512, 512]" = torch.ops.aten.mm.default(permute_715, view_20);  permute_715 = view_20 = None
        permute_9: "f32[512, 512]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_717: "f32[512, 512]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_284: "f32[8192, 512]" = torch.ops.aten.mm.default(view_889, permute_717);  view_889 = permute_717 = None
        view_890: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_284, [8, 1024, 512]);  mm_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_891: "f32[8, 1024, 8, 64]" = torch.ops.aten.reshape.default(view_890, [8, 1024, 8, 64]);  view_890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_719: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_891, [0, 2, 1, 3]);  view_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_221: "f32[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_719, memory_format = torch.contiguous_format);  permute_719 = None
        view_892: "f32[64, 1024, 64]" = torch.ops.aten.reshape.default(clone_221, [64, 1024, 64]);  clone_221 = None
        bmm_104: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(permute_720, view_892);  permute_720 = None
        bmm_105: "f32[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_892, permute_721);  view_892 = permute_721 = None
        view_893: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_104, [8, 8, 1024, 64]);  bmm_104 = None
        view_894: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm_105, [8, 8, 1024, 1024]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_68: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_588: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_589: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_894, mul_588);  view_894 = mul_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(ge, [8, -1, 1024, 1024]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(bmm, [8, 8, 1024, 1024]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_5, where);  unsqueeze_5 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        sub_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(add_8, amax);  add_8 = amax = None
        exp: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_590: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(mul_589, div_2);  mul_589 = None
        sum_103: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [-1], True)
        neg_19: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_17: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_19, sum_103, mul_590);  neg_19 = sum_103 = mul_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_895: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(fma_17, [64, 1024, 1024]);  fma_17 = None
        view_897: "f32[8, 8, 1024, 1024]" = torch.ops.aten.reshape.default(view_895, [8, 8, 1024, 1024]);  view_895 = None
        view_898: "f32[64, 1024, 1024]" = torch.ops.aten.reshape.default(view_897, [64, 1024, 1024])
        add_201: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_194, view_897);  add_194 = view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_104: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_201, [0], True);  add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_2: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_104, 0);  sum_104 = None
        permute_722: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_3: "b8[1024, 1024]" = torch.ops.aten.eq.Scalar(add_6, -1)
        unsqueeze_22: "b8[1024, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_3, -1);  eq_3 = None
        where_24: "f32[1024, 1024, 8]" = torch.ops.aten.where.self(unsqueeze_22, full_default, permute_722);  unsqueeze_22 = permute_722 = None
        clone_224: "f32[1024, 1024, 8]" = torch.ops.aten.clone.default(where_24, memory_format = torch.contiguous_format);  where_24 = None
        index_put_2: "f32[32, 8]" = torch.ops.aten.index_put.default(full_default_24, [add_6], clone_224, True);  full_default_24 = add_6 = clone_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_106: "f32[64, 64, 1024]" = torch.ops.aten.bmm.default(permute_723, view_898);  permute_723 = None
        bmm_107: "f32[64, 1024, 64]" = torch.ops.aten.bmm.default(view_898, permute_724);  view_898 = permute_724 = None
        view_900: "f32[8, 8, 64, 1024]" = torch.ops.aten.reshape.default(bmm_106, [8, 8, 64, 1024]);  bmm_106 = None
        view_901: "f32[8, 8, 1024, 64]" = torch.ops.aten.reshape.default(bmm_107, [8, 8, 1024, 64]);  bmm_107 = None
        permute_725: "f32[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_900, [0, 1, 3, 2]);  view_900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_726: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_893, [0, 2, 1, 3]);  view_893 = None
        clone_225: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_726, memory_format = torch.contiguous_format);  permute_726 = None
        view_902: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_225, [8, 1024, 512]);  clone_225 = None
        view_903: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_902, [8192, 512]);  view_902 = None
        permute_727: "f32[512, 8192]" = torch.ops.aten.permute.default(view_903, [1, 0])
        mm_285: "f32[512, 512]" = torch.ops.aten.mm.default(permute_727, view_1);  permute_727 = None
        permute_4: "f32[512, 512]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_729: "f32[512, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_286: "f32[8192, 512]" = torch.ops.aten.mm.default(view_903, permute_729);  view_903 = permute_729 = None
        view_904: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_286, [8, 1024, 512]);  mm_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_731: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_725, [0, 2, 1, 3]);  permute_725 = None
        view_905: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(permute_731, [8, 1024, 512]);  permute_731 = None
        clone_226: "f32[8, 1024, 512]" = torch.ops.aten.clone.default(view_905, memory_format = torch.contiguous_format);  view_905 = None
        view_906: "f32[8192, 512]" = torch.ops.aten.reshape.default(clone_226, [8192, 512]);  clone_226 = None
        permute_732: "f32[512, 8192]" = torch.ops.aten.permute.default(view_906, [1, 0])
        mm_287: "f32[512, 512]" = torch.ops.aten.mm.default(permute_732, view_1);  permute_732 = None
        permute_2: "f32[512, 512]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_734: "f32[512, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_288: "f32[8192, 512]" = torch.ops.aten.mm.default(view_906, permute_734);  view_906 = permute_734 = None
        view_907: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_288, [8, 1024, 512]);  mm_288 = None
        add_202: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(view_904, view_907);  view_904 = view_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_736: "f32[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_901, [0, 2, 1, 3]);  view_901 = None
        clone_227: "f32[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_736, memory_format = torch.contiguous_format);  permute_736 = None
        view_908: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(clone_227, [8, 1024, 512]);  clone_227 = None
        view_909: "f32[8192, 512]" = torch.ops.aten.reshape.default(view_908, [8192, 512]);  view_908 = None
        permute_737: "f32[512, 8192]" = torch.ops.aten.permute.default(view_909, [1, 0])
        mm_289: "f32[512, 512]" = torch.ops.aten.mm.default(permute_737, view_1);  permute_737 = view_1 = None
        permute: "f32[512, 512]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_739: "f32[512, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_290: "f32[8192, 512]" = torch.ops.aten.mm.default(view_909, permute_739);  view_909 = permute_739 = None
        view_910: "f32[8, 1024, 512]" = torch.ops.aten.reshape.default(mm_290, [8, 1024, 512]);  mm_290 = None
        add_203: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_202, view_910);  add_202 = view_910 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_591: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_203, primals_3);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_1: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_592: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_203, mul_2);  add_203 = mul_2 = None
        sum_105: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_592, [0, 1], True);  mul_592 = None
        view_911: "f32[512]" = torch.ops.aten.reshape.default(sum_105, [512]);  sum_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_593: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_591, mul_1)
        mul_594: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_591, rsqrt);  mul_591 = None
        sum_106: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [2], True);  mul_593 = None
        add_204: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_200, mul_594);  add_200 = mul_594 = None
        pow_95: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_595: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_106, -0.5);  sum_106 = None
        mul_596: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_595, pow_95);  mul_595 = pow_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_106: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_596, [8, 1024, 512]);  mul_596 = None
        div_55: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_106, 512);  expand_106 = None
        pow_96: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 1.0);  mul_1 = None
        mul_597: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_96, 2.0);  pow_96 = None
        mul_598: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_55, mul_597);  div_55 = mul_597 = None
        add_205: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_204, mul_598);  add_204 = mul_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_69: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_599: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
        mul_600: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_205, mul_599);  add_205 = mul_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_4: "b8[8, 1024]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_23: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(eq_4, -1);  eq_4 = None
        where_25: "f32[8, 1024, 512]" = torch.ops.aten.where.self(unsqueeze_23, full_default, mul_600);  unsqueeze_23 = full_default = mul_600 = None
        index_put_3: "f32[32128, 512]" = torch.ops.aten.index_put.default(full_default_26, [primals_1], where_25, True);  full_default_26 = primals_1 = where_25 = None
        add_206: "f32[32128, 512]" = torch.ops.aten.add.Tensor(add_163, index_put_3);  add_163 = index_put_3 = None
        return (None, add_206, view_911, mm_289, mm_287, mm_285, index_put_2, mm_283, view_888, mm_281, mm_279, view_883, mm_277, mm_275, mm_273, mm_271, view_860, mm_269, mm_267, view_855, mm_265, mm_263, mm_261, mm_259, view_832, mm_257, mm_255, view_827, mm_253, mm_251, mm_249, mm_247, view_804, mm_245, mm_243, view_799, mm_241, mm_239, mm_237, mm_235, view_776, mm_233, mm_231, view_771, mm_229, mm_227, mm_225, mm_223, view_748, mm_221, mm_219, view_743, None, view_742, mm_217, mm_215, mm_213, index_put, mm_211, view_719, mm_209, mm_207, mm_205, mm_203, view_696, mm_201, mm_199, view_691, mm_197, mm_195, mm_193, mm_191, view_668, mm_189, mm_187, mm_185, mm_183, view_645, mm_181, mm_179, view_640, mm_177, mm_175, mm_173, mm_171, view_617, mm_169, mm_167, mm_165, mm_163, view_594, mm_161, mm_159, view_589, mm_157, mm_155, mm_153, mm_151, view_566, mm_149, mm_147, mm_145, mm_143, view_543, mm_141, mm_139, view_538, mm_137, mm_135, mm_133, mm_131, view_515, mm_129, mm_127, mm_125, mm_123, view_492, mm_121, mm_119, view_487, mm_117, mm_115, mm_113, mm_111, view_464, mm_109, mm_107, mm_105, mm_103, view_441, mm_101, mm_99, view_436)
