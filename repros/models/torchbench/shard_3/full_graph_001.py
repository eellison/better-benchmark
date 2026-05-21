class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "i64[4, 476][476, 1]cuda:0", primals_4: "i64[4, 476][476, 1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_17: "f32[768][1]cuda:0", primals_23: "f32[768][1]cuda:0", primals_33: "f32[768][1]cuda:0", primals_39: "f32[768][1]cuda:0", primals_49: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_65: "f32[768][1]cuda:0", primals_71: "f32[768][1]cuda:0", primals_81: "f32[768][1]cuda:0", primals_87: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_113: "f32[768][1]cuda:0", primals_119: "f32[768][1]cuda:0", primals_129: "f32[768][1]cuda:0", primals_135: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_151: "f32[768][1]cuda:0", primals_161: "f32[768][1]cuda:0", primals_167: "f32[768][1]cuda:0", primals_177: "f32[768][1]cuda:0", primals_183: "f32[768][1]cuda:0", primals_193: "f32[768][1]cuda:0", primals_199: "f32[768][1]cuda:0", mul: "f32[4, 1, 1, 476][476, 476, 476, 1]cuda:0", unsqueeze_2: "i64[1, 476][476, 1]cuda:0", mul_1: "f32[4, 476, 768][365568, 768, 1]cuda:0", view: "f16[1904, 768][768, 1]cuda:0", bmm: "f16[48, 476, 476][226576, 476, 1]cuda:0", amax: "f32[4, 12, 476, 1][5728, 476, 1, 1]cuda:0", sum_1: "f32[4, 12, 476, 1][5728, 476, 1, 1]cuda:0", view_16: "f16[1904, 768][768, 1]cuda:0", mul_3: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_18: "f16[1904, 768][768, 1]cuda:0", addmm_4: "f16[1904, 3072][3072, 1]cuda:0", view_20: "f16[1904, 3072][3072, 1]cuda:0", mul_7: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_22: "f16[1904, 768][768, 1]cuda:0", div_4: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_38: "f16[1904, 768][768, 1]cuda:0", mul_9: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_40: "f16[1904, 768][768, 1]cuda:0", addmm_10: "f16[1904, 3072][3072, 1]cuda:0", view_42: "f16[1904, 3072][3072, 1]cuda:0", mul_13: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_44: "f16[1904, 768][768, 1]cuda:0", div_7: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_60: "f16[1904, 768][768, 1]cuda:0", mul_15: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_62: "f16[1904, 768][768, 1]cuda:0", addmm_16: "f16[1904, 3072][3072, 1]cuda:0", view_64: "f16[1904, 3072][3072, 1]cuda:0", mul_19: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_66: "f16[1904, 768][768, 1]cuda:0", div_10: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_82: "f16[1904, 768][768, 1]cuda:0", mul_21: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_84: "f16[1904, 768][768, 1]cuda:0", addmm_22: "f16[1904, 3072][3072, 1]cuda:0", view_86: "f16[1904, 3072][3072, 1]cuda:0", mul_25: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_88: "f16[1904, 768][768, 1]cuda:0", div_13: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_104: "f16[1904, 768][768, 1]cuda:0", mul_27: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_106: "f16[1904, 768][768, 1]cuda:0", addmm_28: "f16[1904, 3072][3072, 1]cuda:0", view_108: "f16[1904, 3072][3072, 1]cuda:0", mul_31: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_110: "f16[1904, 768][768, 1]cuda:0", div_16: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_126: "f16[1904, 768][768, 1]cuda:0", mul_33: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_128: "f16[1904, 768][768, 1]cuda:0", addmm_34: "f16[1904, 3072][3072, 1]cuda:0", view_130: "f16[1904, 3072][3072, 1]cuda:0", mul_37: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_132: "f16[1904, 768][768, 1]cuda:0", div_19: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_148: "f16[1904, 768][768, 1]cuda:0", mul_39: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_150: "f16[1904, 768][768, 1]cuda:0", addmm_40: "f16[1904, 3072][3072, 1]cuda:0", view_152: "f16[1904, 3072][3072, 1]cuda:0", mul_43: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_154: "f16[1904, 768][768, 1]cuda:0", div_22: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_170: "f16[1904, 768][768, 1]cuda:0", mul_45: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_172: "f16[1904, 768][768, 1]cuda:0", addmm_46: "f16[1904, 3072][3072, 1]cuda:0", view_174: "f16[1904, 3072][3072, 1]cuda:0", mul_49: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_176: "f16[1904, 768][768, 1]cuda:0", div_25: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_192: "f16[1904, 768][768, 1]cuda:0", mul_51: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_194: "f16[1904, 768][768, 1]cuda:0", addmm_52: "f16[1904, 3072][3072, 1]cuda:0", view_196: "f16[1904, 3072][3072, 1]cuda:0", mul_55: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_198: "f16[1904, 768][768, 1]cuda:0", div_28: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_214: "f16[1904, 768][768, 1]cuda:0", mul_57: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_216: "f16[1904, 768][768, 1]cuda:0", addmm_58: "f16[1904, 3072][3072, 1]cuda:0", view_218: "f16[1904, 3072][3072, 1]cuda:0", mul_61: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_220: "f16[1904, 768][768, 1]cuda:0", div_31: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_236: "f16[1904, 768][768, 1]cuda:0", mul_63: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_238: "f16[1904, 768][768, 1]cuda:0", addmm_64: "f16[1904, 3072][3072, 1]cuda:0", view_240: "f16[1904, 3072][3072, 1]cuda:0", mul_67: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_242: "f16[1904, 768][768, 1]cuda:0", div_34: "f32[4, 12, 476, 476][2719104, 226592, 476, 1]cuda:0", view_258: "f16[1904, 768][768, 1]cuda:0", mul_69: "f32[4, 476, 768][365568, 768, 1]cuda:0", view_260: "f16[1904, 768][768, 1]cuda:0", addmm_70: "f16[1904, 3072][3072, 1]cuda:0", view_262: "f16[1904, 3072][3072, 1]cuda:0", mul_73: "f32[4, 476, 768][365568, 768, 1]cuda:0", convert_element_type_471: "f16[4, 768][768, 1]cuda:0", tanh: "f16[4, 768][768, 1]cuda:0", permute_133: "f16[768, 768][768, 1]cuda:0", div_36: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_137: "f16[768, 3072][3072, 1]cuda:0", permute_141: "f16[3072, 768][768, 1]cuda:0", div_38: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_145: "f16[768, 768][768, 1]cuda:0", permute_150: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_151: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_152: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_153: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_158: "f16[768, 768][768, 1]cuda:0", permute_162: "f16[768, 768][768, 1]cuda:0", permute_166: "f16[768, 768][768, 1]cuda:0", div_40: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_170: "f16[768, 3072][3072, 1]cuda:0", permute_174: "f16[3072, 768][768, 1]cuda:0", div_42: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_178: "f16[768, 768][768, 1]cuda:0", permute_183: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_184: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_185: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_186: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_191: "f16[768, 768][768, 1]cuda:0", permute_195: "f16[768, 768][768, 1]cuda:0", permute_199: "f16[768, 768][768, 1]cuda:0", div_44: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_203: "f16[768, 3072][3072, 1]cuda:0", permute_207: "f16[3072, 768][768, 1]cuda:0", div_46: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_211: "f16[768, 768][768, 1]cuda:0", permute_216: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_217: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_218: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_219: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_224: "f16[768, 768][768, 1]cuda:0", permute_228: "f16[768, 768][768, 1]cuda:0", permute_232: "f16[768, 768][768, 1]cuda:0", div_48: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_236: "f16[768, 3072][3072, 1]cuda:0", permute_240: "f16[3072, 768][768, 1]cuda:0", div_50: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_244: "f16[768, 768][768, 1]cuda:0", permute_249: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_250: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_251: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_252: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_257: "f16[768, 768][768, 1]cuda:0", permute_261: "f16[768, 768][768, 1]cuda:0", permute_265: "f16[768, 768][768, 1]cuda:0", div_52: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_269: "f16[768, 3072][3072, 1]cuda:0", permute_273: "f16[3072, 768][768, 1]cuda:0", div_54: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_277: "f16[768, 768][768, 1]cuda:0", permute_282: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_283: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_284: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_285: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_290: "f16[768, 768][768, 1]cuda:0", permute_294: "f16[768, 768][768, 1]cuda:0", permute_298: "f16[768, 768][768, 1]cuda:0", div_56: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_302: "f16[768, 3072][3072, 1]cuda:0", permute_306: "f16[3072, 768][768, 1]cuda:0", div_58: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_310: "f16[768, 768][768, 1]cuda:0", permute_315: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_316: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_317: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_318: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_323: "f16[768, 768][768, 1]cuda:0", permute_327: "f16[768, 768][768, 1]cuda:0", permute_331: "f16[768, 768][768, 1]cuda:0", div_60: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_335: "f16[768, 3072][3072, 1]cuda:0", permute_339: "f16[3072, 768][768, 1]cuda:0", div_62: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_343: "f16[768, 768][768, 1]cuda:0", permute_348: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_349: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_350: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_351: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_356: "f16[768, 768][768, 1]cuda:0", permute_360: "f16[768, 768][768, 1]cuda:0", permute_364: "f16[768, 768][768, 1]cuda:0", div_64: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_368: "f16[768, 3072][3072, 1]cuda:0", permute_372: "f16[3072, 768][768, 1]cuda:0", div_66: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_376: "f16[768, 768][768, 1]cuda:0", permute_381: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_382: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_383: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_384: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_389: "f16[768, 768][768, 1]cuda:0", permute_393: "f16[768, 768][768, 1]cuda:0", permute_397: "f16[768, 768][768, 1]cuda:0", div_68: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_401: "f16[768, 3072][3072, 1]cuda:0", permute_405: "f16[3072, 768][768, 1]cuda:0", div_70: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_409: "f16[768, 768][768, 1]cuda:0", permute_414: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_415: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_416: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_417: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_422: "f16[768, 768][768, 1]cuda:0", permute_426: "f16[768, 768][768, 1]cuda:0", permute_430: "f16[768, 768][768, 1]cuda:0", div_72: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_434: "f16[768, 3072][3072, 1]cuda:0", permute_438: "f16[3072, 768][768, 1]cuda:0", div_74: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_442: "f16[768, 768][768, 1]cuda:0", permute_447: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_448: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_449: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_450: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_455: "f16[768, 768][768, 1]cuda:0", permute_459: "f16[768, 768][768, 1]cuda:0", permute_463: "f16[768, 768][768, 1]cuda:0", div_76: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_467: "f16[768, 3072][3072, 1]cuda:0", permute_471: "f16[3072, 768][768, 1]cuda:0", div_78: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_475: "f16[768, 768][768, 1]cuda:0", permute_480: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_481: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_482: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_483: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_488: "f16[768, 768][768, 1]cuda:0", permute_492: "f16[768, 768][768, 1]cuda:0", permute_496: "f16[768, 768][768, 1]cuda:0", div_80: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_500: "f16[768, 3072][3072, 1]cuda:0", permute_504: "f16[3072, 768][768, 1]cuda:0", div_82: "f32[4, 476, 1][476, 1, 1]cuda:0", permute_508: "f16[768, 768][768, 1]cuda:0", permute_513: "f16[48, 476, 476][226624, 1, 476]cuda:0", permute_514: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_515: "f16[48, 64, 476][30464, 1, 64]cuda:0", permute_516: "f16[48, 476, 64][30464, 1, 476]cuda:0", permute_521: "f16[768, 768][768, 1]cuda:0", permute_525: "f16[768, 768][768, 1]cuda:0", permute_529: "f16[768, 768][768, 1]cuda:0", div_84: "f32[4, 476, 1][476, 1, 1]cuda:0", tangents_1: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_2: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_3: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_4: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_5: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_6: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_7: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_8: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_9: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_10: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_11: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_12: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_13: "f32[4, 476, 768][365568, 768, 1]cuda:0", tangents_14: "f16[4, 768][768, 1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:405 in forward, code: pooled_output = self.activation(pooled_output)
        convert_element_type_475: "f32[4, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tangents_14, torch.float32);  tangents_14 = None
        convert_element_type_476: "f32[4, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(tanh, torch.float32);  tanh = None
        mul_75: "f32[4, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_476, convert_element_type_476);  convert_element_type_476 = None
        sub_38: "f32[4, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_75);  mul_75 = None
        mul_76: "f32[4, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_475, sub_38);  convert_element_type_475 = sub_38 = None
        convert_element_type_477: "f16[4, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_76, torch.float16);  mul_76 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:404 in forward, code: pooled_output = self.dense(first_token_tensor)
        mm: "f16[4, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(convert_element_type_477, permute_133);  permute_133 = None
        permute_134: "f16[768, 4][1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_477, [1, 0])
        mm_1: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_134, convert_element_type_471);  permute_134 = convert_element_type_471 = None
        sum_13: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_477, [0], True, dtype = torch.float32);  convert_element_type_477 = None
        view_264: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_13, [768]);  sum_13 = None
        convert_element_type_483: "f32[4, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_484: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:403 in forward, code: first_token_tensor = hidden_states[:, 0]
        full_default: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.full.default([4, 476, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # No stacktrace found for following nodes
        select_scatter_default: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default, convert_element_type_483, 1, 0);  full_default = convert_element_type_483 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:403 in forward, code: first_token_tensor = hidden_states[:, 0]
        add_100: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_13, select_scatter_default);  tangents_13 = select_scatter_default = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_78: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_100, primals_199);  primals_199 = None
        mul_79: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, 768)
        sum_14: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_78, [2], True)
        mul_80: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, mul_73);  mul_78 = None
        sum_15: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_80, [2], True);  mul_80 = None
        mul_81: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, sum_15);  sum_15 = None
        sub_40: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_79, sum_14);  mul_79 = sum_14 = None
        sub_41: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_40, mul_81);  sub_40 = mul_81 = None
        mul_82: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_36, sub_41);  div_36 = sub_41 = None
        mul_83: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_100, mul_73);  mul_73 = None
        sum_16: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_83, [0, 1]);  mul_83 = None
        sum_17: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_100, [0, 1]);  add_100 = None
        convert_element_type_486: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_82, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_265: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_486, [1904, 768]);  convert_element_type_486 = None
        mm_2: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_265, permute_137);  permute_137 = None
        permute_138: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_265, [1, 0])
        mm_3: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_138, view_262);  permute_138 = view_262 = None
        sum_18: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_265, [0], True, dtype = torch.float32);  view_265 = None
        view_266: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_18, [768]);  sum_18 = None
        view_267: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [4, 476, 3072]);  mm_2 = None
        convert_element_type_492: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [4, 476, 3072]);  addmm_70 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_71: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        mul_84: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_267, mul_71);  mul_71 = None
        div_35: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_261, 1.4142135623730951);  view_261 = None
        erf_11: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_35)
        add_96: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1.0);  erf_11 = None
        mul_85: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_267, add_96);  view_267 = add_96 = None
        convert_element_type_494: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_35, torch.float32);  div_35 = None
        pow_1: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_494, 2);  convert_element_type_494 = None
        neg: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_1);  pow_1 = None
        exp_12: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg);  neg = None
        mul_86: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_12, 1.1283791670955126);  exp_12 = None
        mul_87: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, mul_84);  mul_86 = mul_84 = None
        convert_element_type_495: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_87, torch.float16);  mul_87 = None
        div_37: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_495, 1.4142135623730951);  convert_element_type_495 = None
        mul_88: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_85, 0.5);  mul_85 = None
        add_101: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_37, mul_88);  div_37 = mul_88 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_268: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_101, [1904, 3072]);  add_101 = None
        mm_4: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_268, permute_141);  permute_141 = None
        permute_142: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_268, [1, 0])
        mm_5: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_142, view_260);  permute_142 = view_260 = None
        sum_19: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_268, [0], True, dtype = torch.float32);  view_268 = None
        view_269: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_19, [3072]);  sum_19 = None
        view_270: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [4, 476, 768]);  mm_4 = None
        convert_element_type_501: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_270, torch.float32);  view_270 = None
        add_102: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_82, convert_element_type_501);  mul_82 = convert_element_type_501 = None
        convert_element_type_502: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_90: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, primals_193);  primals_193 = None
        mul_91: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, 768)
        sum_20: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_90, [2], True)
        mul_92: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, mul_69);  mul_90 = None
        sum_21: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_92, [2], True);  mul_92 = None
        mul_93: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_69, sum_21);  sum_21 = None
        sub_43: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_91, sum_20);  mul_91 = sum_20 = None
        sub_44: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_43, mul_93);  sub_43 = mul_93 = None
        mul_94: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, sub_44);  div_38 = sub_44 = None
        mul_95: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, mul_69);  mul_69 = None
        sum_22: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_95, [0, 1]);  mul_95 = None
        sum_23: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_102, [0, 1]);  add_102 = None
        convert_element_type_504: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_94, torch.float16)
        add_103: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_12, mul_94);  tangents_12 = mul_94 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_271: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_504, [1904, 768]);  convert_element_type_504 = None
        mm_6: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_271, permute_145);  permute_145 = None
        permute_146: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_271, [1, 0])
        mm_7: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_146, view_258);  permute_146 = view_258 = None
        sum_24: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_271, [0], True, dtype = torch.float32);  view_271 = None
        view_272: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_24, [768]);  sum_24 = None
        view_273: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [4, 476, 768]);  mm_6 = None
        convert_element_type_510: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_274: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_273, [4, 476, 12, 64]);  view_273 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_149: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_85: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        view_275: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [48, 476, 64]);  clone_85 = None
        bmm_24: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_150, view_275);  permute_150 = None
        bmm_25: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_275, permute_151);  view_275 = permute_151 = None
        view_276: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [4, 12, 476, 64]);  bmm_24 = None
        view_277: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [4, 12, 476, 476]);  bmm_25 = None
        convert_element_type_516: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_277, torch.float32);  view_277 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_96: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_516, div_34);  convert_element_type_516 = None
        sum_25: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_96, [-1], True)
        neg_1: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_34);  div_34 = None
        fma: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_1, sum_25, mul_96);  neg_1 = sum_25 = mul_96 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_517: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.float16);  fma = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_39: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_517, 8.0);  convert_element_type_517 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_278: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_39, [48, 476, 476]);  div_39 = None
        bmm_26: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_152, view_278);  permute_152 = None
        bmm_27: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_278, permute_153);  view_278 = permute_153 = None
        view_279: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [4, 12, 64, 476]);  bmm_26 = None
        view_280: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [4, 12, 476, 64]);  bmm_27 = None
        permute_154: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_279, [0, 1, 3, 2]);  view_279 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_155: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_276, [0, 2, 1, 3]);  view_276 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_86: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
        view_281: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [4, 476, 768]);  clone_86 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_156: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_154, [0, 2, 1, 3]);  permute_154 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_282: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_156, [4, 476, 768]);  permute_156 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_157: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_87: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_283: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [4, 476, 768]);  clone_87 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_284: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_281, [1904, 768]);  view_281 = None
        mm_8: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_284, permute_158);  permute_158 = None
        permute_159: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_284, [1, 0])
        mm_9: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_159, view_242);  permute_159 = None
        sum_26: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_284, [0], True, dtype = torch.float32);  view_284 = None
        view_285: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_26, [768]);  sum_26 = None
        view_286: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [4, 476, 768]);  mm_8 = None
        convert_element_type_527: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_286, torch.float32);  view_286 = None
        add_104: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_103, convert_element_type_527);  add_103 = convert_element_type_527 = None
        convert_element_type_528: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_88: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_282, memory_format = torch.contiguous_format);  view_282 = None
        view_287: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [1904, 768]);  clone_88 = None
        mm_10: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_287, permute_162);  permute_162 = None
        permute_163: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_287, [1, 0])
        mm_11: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_163, view_242);  permute_163 = None
        sum_27: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_287, [0], True, dtype = torch.float32);  view_287 = None
        view_288: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        view_289: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [4, 476, 768]);  mm_10 = None
        convert_element_type_535: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_289, torch.float32);  view_289 = None
        add_105: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_104, convert_element_type_535);  add_104 = convert_element_type_535 = None
        convert_element_type_536: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_290: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_283, [1904, 768]);  view_283 = None
        mm_12: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_290, permute_166);  permute_166 = None
        permute_167: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_290, [1, 0])
        mm_13: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_167, view_242);  permute_167 = view_242 = None
        sum_28: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_290, [0], True, dtype = torch.float32);  view_290 = None
        view_291: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [768]);  sum_28 = None
        view_292: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [4, 476, 768]);  mm_12 = None
        convert_element_type_543: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_292, torch.float32);  view_292 = None
        add_106: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_105, convert_element_type_543);  add_105 = convert_element_type_543 = None
        convert_element_type_544: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_98: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_106, primals_183);  primals_183 = None
        mul_99: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, 768)
        sum_29: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_98, [2], True)
        mul_100: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, mul_67);  mul_98 = None
        sum_30: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_100, [2], True);  mul_100 = None
        mul_101: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_67, sum_30);  sum_30 = None
        sub_46: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_99, sum_29);  mul_99 = sum_29 = None
        sub_47: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_46, mul_101);  sub_46 = mul_101 = None
        mul_102: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_40, sub_47);  div_40 = sub_47 = None
        mul_103: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_106, mul_67);  mul_67 = None
        sum_31: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_103, [0, 1]);  mul_103 = None
        sum_32: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_106, [0, 1]);  add_106 = None
        convert_element_type_546: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_102, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_293: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_546, [1904, 768]);  convert_element_type_546 = None
        mm_14: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_293, permute_170);  permute_170 = None
        permute_171: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_293, [1, 0])
        mm_15: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_171, view_240);  permute_171 = view_240 = None
        sum_33: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_293, [0], True, dtype = torch.float32);  view_293 = None
        view_294: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [768]);  sum_33 = None
        view_295: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [4, 476, 3072]);  mm_14 = None
        convert_element_type_552: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [4, 476, 3072]);  addmm_64 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_65: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_104: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_295, mul_65);  mul_65 = None
        div_32: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_239, 1.4142135623730951);  view_239 = None
        erf_10: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_32)
        add_88: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1.0);  erf_10 = None
        mul_105: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_295, add_88);  view_295 = add_88 = None
        convert_element_type_554: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_32, torch.float32);  div_32 = None
        pow_2: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_554, 2);  convert_element_type_554 = None
        neg_2: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_2);  pow_2 = None
        exp_13: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        mul_106: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_13, 1.1283791670955126);  exp_13 = None
        mul_107: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_106, mul_104);  mul_106 = mul_104 = None
        convert_element_type_555: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_107, torch.float16);  mul_107 = None
        div_41: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_555, 1.4142135623730951);  convert_element_type_555 = None
        mul_108: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, 0.5);  mul_105 = None
        add_107: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_41, mul_108);  div_41 = mul_108 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_296: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_107, [1904, 3072]);  add_107 = None
        mm_16: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_296, permute_174);  permute_174 = None
        permute_175: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_296, [1, 0])
        mm_17: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_175, view_238);  permute_175 = view_238 = None
        sum_34: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_296, [0], True, dtype = torch.float32);  view_296 = None
        view_297: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [3072]);  sum_34 = None
        view_298: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [4, 476, 768]);  mm_16 = None
        convert_element_type_561: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_298, torch.float32);  view_298 = None
        add_108: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_102, convert_element_type_561);  mul_102 = convert_element_type_561 = None
        convert_element_type_562: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_110: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_108, primals_177);  primals_177 = None
        mul_111: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, 768)
        sum_35: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True)
        mul_112: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, mul_63);  mul_110 = None
        sum_36: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_112, [2], True);  mul_112 = None
        mul_113: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_63, sum_36);  sum_36 = None
        sub_49: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_111, sum_35);  mul_111 = sum_35 = None
        sub_50: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_49, mul_113);  sub_49 = mul_113 = None
        mul_114: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_42, sub_50);  div_42 = sub_50 = None
        mul_115: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_108, mul_63);  mul_63 = None
        sum_37: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_115, [0, 1]);  mul_115 = None
        sum_38: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_108, [0, 1]);  add_108 = None
        convert_element_type_564: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_114, torch.float16)
        add_109: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_11, mul_114);  tangents_11 = mul_114 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_299: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_564, [1904, 768]);  convert_element_type_564 = None
        mm_18: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_299, permute_178);  permute_178 = None
        permute_179: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_299, [1, 0])
        mm_19: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_179, view_236);  permute_179 = view_236 = None
        sum_39: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_299, [0], True, dtype = torch.float32);  view_299 = None
        view_300: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [768]);  sum_39 = None
        view_301: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [4, 476, 768]);  mm_18 = None
        convert_element_type_570: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_302: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_301, [4, 476, 12, 64]);  view_301 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_182: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_89: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_303: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [48, 476, 64]);  clone_89 = None
        bmm_28: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_183, view_303);  permute_183 = None
        bmm_29: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_303, permute_184);  view_303 = permute_184 = None
        view_304: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [4, 12, 476, 64]);  bmm_28 = None
        view_305: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [4, 12, 476, 476]);  bmm_29 = None
        convert_element_type_576: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_116: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_576, div_31);  convert_element_type_576 = None
        sum_40: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_116, [-1], True)
        neg_3: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_31);  div_31 = None
        fma_1: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_40, mul_116);  neg_3 = sum_40 = mul_116 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_577: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.float16);  fma_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_43: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_577, 8.0);  convert_element_type_577 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_306: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_43, [48, 476, 476]);  div_43 = None
        bmm_30: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_185, view_306);  permute_185 = None
        bmm_31: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_306, permute_186);  view_306 = permute_186 = None
        view_307: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [4, 12, 64, 476]);  bmm_30 = None
        view_308: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [4, 12, 476, 64]);  bmm_31 = None
        permute_187: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_307, [0, 1, 3, 2]);  view_307 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_188: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_90: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_188, memory_format = torch.contiguous_format);  permute_188 = None
        view_309: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [4, 476, 768]);  clone_90 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_189: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_187, [0, 2, 1, 3]);  permute_187 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_310: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_189, [4, 476, 768]);  permute_189 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_190: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_91: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_311: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [4, 476, 768]);  clone_91 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_312: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_309, [1904, 768]);  view_309 = None
        mm_20: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_312, permute_191);  permute_191 = None
        permute_192: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_312, [1, 0])
        mm_21: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_192, view_220);  permute_192 = None
        sum_41: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_312, [0], True, dtype = torch.float32);  view_312 = None
        view_313: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
        view_314: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [4, 476, 768]);  mm_20 = None
        convert_element_type_587: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_314, torch.float32);  view_314 = None
        add_110: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, convert_element_type_587);  add_109 = convert_element_type_587 = None
        convert_element_type_588: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_92: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_310, memory_format = torch.contiguous_format);  view_310 = None
        view_315: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [1904, 768]);  clone_92 = None
        mm_22: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_315, permute_195);  permute_195 = None
        permute_196: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_315, [1, 0])
        mm_23: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_196, view_220);  permute_196 = None
        sum_42: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_315, [0], True, dtype = torch.float32);  view_315 = None
        view_316: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        view_317: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [4, 476, 768]);  mm_22 = None
        convert_element_type_595: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_317, torch.float32);  view_317 = None
        add_111: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, convert_element_type_595);  add_110 = convert_element_type_595 = None
        convert_element_type_596: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_318: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_311, [1904, 768]);  view_311 = None
        mm_24: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_318, permute_199);  permute_199 = None
        permute_200: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_25: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_200, view_220);  permute_200 = view_220 = None
        sum_43: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_318, [0], True, dtype = torch.float32);  view_318 = None
        view_319: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_43, [768]);  sum_43 = None
        view_320: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [4, 476, 768]);  mm_24 = None
        convert_element_type_603: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_320, torch.float32);  view_320 = None
        add_112: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_111, convert_element_type_603);  add_111 = convert_element_type_603 = None
        convert_element_type_604: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_118: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_112, primals_167);  primals_167 = None
        mul_119: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, 768)
        sum_44: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_118, [2], True)
        mul_120: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_118, mul_61);  mul_118 = None
        sum_45: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_120, [2], True);  mul_120 = None
        mul_121: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, sum_45);  sum_45 = None
        sub_52: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_119, sum_44);  mul_119 = sum_44 = None
        sub_53: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_52, mul_121);  sub_52 = mul_121 = None
        mul_122: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_44, sub_53);  div_44 = sub_53 = None
        mul_123: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_112, mul_61);  mul_61 = None
        sum_46: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_123, [0, 1]);  mul_123 = None
        sum_47: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_112, [0, 1]);  add_112 = None
        convert_element_type_606: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_122, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_321: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_606, [1904, 768]);  convert_element_type_606 = None
        mm_26: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_321, permute_203);  permute_203 = None
        permute_204: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_27: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_204, view_218);  permute_204 = view_218 = None
        sum_48: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_321, [0], True, dtype = torch.float32);  view_321 = None
        view_322: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [768]);  sum_48 = None
        view_323: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [4, 476, 3072]);  mm_26 = None
        convert_element_type_612: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [4, 476, 3072]);  addmm_58 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_59: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_124: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_323, mul_59);  mul_59 = None
        div_29: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_217, 1.4142135623730951);  view_217 = None
        erf_9: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_29)
        add_80: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1.0);  erf_9 = None
        mul_125: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_323, add_80);  view_323 = add_80 = None
        convert_element_type_614: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_29, torch.float32);  div_29 = None
        pow_3: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_614, 2);  convert_element_type_614 = None
        neg_4: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_3);  pow_3 = None
        exp_14: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        mul_126: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_14, 1.1283791670955126);  exp_14 = None
        mul_127: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, mul_124);  mul_126 = mul_124 = None
        convert_element_type_615: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_127, torch.float16);  mul_127 = None
        div_45: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_615, 1.4142135623730951);  convert_element_type_615 = None
        mul_128: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, 0.5);  mul_125 = None
        add_113: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_45, mul_128);  div_45 = mul_128 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_324: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_113, [1904, 3072]);  add_113 = None
        mm_28: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_324, permute_207);  permute_207 = None
        permute_208: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_324, [1, 0])
        mm_29: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_208, view_216);  permute_208 = view_216 = None
        sum_49: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_324, [0], True, dtype = torch.float32);  view_324 = None
        view_325: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [3072]);  sum_49 = None
        view_326: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [4, 476, 768]);  mm_28 = None
        convert_element_type_621: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_326, torch.float32);  view_326 = None
        add_114: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_122, convert_element_type_621);  mul_122 = convert_element_type_621 = None
        convert_element_type_622: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_130: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_114, primals_161);  primals_161 = None
        mul_131: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, 768)
        sum_50: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_130, [2], True)
        mul_132: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, mul_57);  mul_130 = None
        sum_51: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True);  mul_132 = None
        mul_133: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, sum_51);  sum_51 = None
        sub_55: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_131, sum_50);  mul_131 = sum_50 = None
        sub_56: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_55, mul_133);  sub_55 = mul_133 = None
        mul_134: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_46, sub_56);  div_46 = sub_56 = None
        mul_135: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_114, mul_57);  mul_57 = None
        sum_52: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_135, [0, 1]);  mul_135 = None
        sum_53: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_114, [0, 1]);  add_114 = None
        convert_element_type_624: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_134, torch.float16)
        add_115: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_10, mul_134);  tangents_10 = mul_134 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_327: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_624, [1904, 768]);  convert_element_type_624 = None
        mm_30: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_327, permute_211);  permute_211 = None
        permute_212: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_327, [1, 0])
        mm_31: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_212, view_214);  permute_212 = view_214 = None
        sum_54: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_327, [0], True, dtype = torch.float32);  view_327 = None
        view_328: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_54, [768]);  sum_54 = None
        view_329: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [4, 476, 768]);  mm_30 = None
        convert_element_type_630: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_330: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_329, [4, 476, 12, 64]);  view_329 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_215: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_93: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_331: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_93, [48, 476, 64]);  clone_93 = None
        bmm_32: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_216, view_331);  permute_216 = None
        bmm_33: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_331, permute_217);  view_331 = permute_217 = None
        view_332: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [4, 12, 476, 64]);  bmm_32 = None
        view_333: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [4, 12, 476, 476]);  bmm_33 = None
        convert_element_type_636: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_333, torch.float32);  view_333 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_136: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_636, div_28);  convert_element_type_636 = None
        sum_55: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_136, [-1], True)
        neg_5: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_28);  div_28 = None
        fma_2: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_55, mul_136);  neg_5 = sum_55 = mul_136 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_637: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.float16);  fma_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_47: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_637, 8.0);  convert_element_type_637 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_334: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_47, [48, 476, 476]);  div_47 = None
        bmm_34: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_218, view_334);  permute_218 = None
        bmm_35: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_334, permute_219);  view_334 = permute_219 = None
        view_335: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [4, 12, 64, 476]);  bmm_34 = None
        view_336: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [4, 12, 476, 64]);  bmm_35 = None
        permute_220: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_335, [0, 1, 3, 2]);  view_335 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_221: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_332, [0, 2, 1, 3]);  view_332 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_94: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_337: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_94, [4, 476, 768]);  clone_94 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_222: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_220, [0, 2, 1, 3]);  permute_220 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_338: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_222, [4, 476, 768]);  permute_222 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_223: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_95: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_339: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [4, 476, 768]);  clone_95 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_340: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_337, [1904, 768]);  view_337 = None
        mm_32: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_340, permute_224);  permute_224 = None
        permute_225: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_340, [1, 0])
        mm_33: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_225, view_198);  permute_225 = None
        sum_56: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_340, [0], True, dtype = torch.float32);  view_340 = None
        view_341: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_56, [768]);  sum_56 = None
        view_342: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [4, 476, 768]);  mm_32 = None
        convert_element_type_647: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_342, torch.float32);  view_342 = None
        add_116: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_115, convert_element_type_647);  add_115 = convert_element_type_647 = None
        convert_element_type_648: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_96: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_338, memory_format = torch.contiguous_format);  view_338 = None
        view_343: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [1904, 768]);  clone_96 = None
        mm_34: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_343, permute_228);  permute_228 = None
        permute_229: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_343, [1, 0])
        mm_35: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_229, view_198);  permute_229 = None
        sum_57: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_343, [0], True, dtype = torch.float32);  view_343 = None
        view_344: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        view_345: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [4, 476, 768]);  mm_34 = None
        convert_element_type_655: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_345, torch.float32);  view_345 = None
        add_117: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_116, convert_element_type_655);  add_116 = convert_element_type_655 = None
        convert_element_type_656: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_346: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_339, [1904, 768]);  view_339 = None
        mm_36: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_346, permute_232);  permute_232 = None
        permute_233: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_346, [1, 0])
        mm_37: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_233, view_198);  permute_233 = view_198 = None
        sum_58: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_346, [0], True, dtype = torch.float32);  view_346 = None
        view_347: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [768]);  sum_58 = None
        view_348: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [4, 476, 768]);  mm_36 = None
        convert_element_type_663: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_348, torch.float32);  view_348 = None
        add_118: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, convert_element_type_663);  add_117 = convert_element_type_663 = None
        convert_element_type_664: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_138: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, primals_151);  primals_151 = None
        mul_139: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, 768)
        sum_59: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True)
        mul_140: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, mul_55);  mul_138 = None
        sum_60: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_140, [2], True);  mul_140 = None
        mul_141: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, sum_60);  sum_60 = None
        sub_58: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_139, sum_59);  mul_139 = sum_59 = None
        sub_59: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_58, mul_141);  sub_58 = mul_141 = None
        mul_142: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_48, sub_59);  div_48 = sub_59 = None
        mul_143: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_118, mul_55);  mul_55 = None
        sum_61: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1]);  mul_143 = None
        sum_62: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_118, [0, 1]);  add_118 = None
        convert_element_type_666: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_142, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_349: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_666, [1904, 768]);  convert_element_type_666 = None
        mm_38: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_349, permute_236);  permute_236 = None
        permute_237: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_349, [1, 0])
        mm_39: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_237, view_196);  permute_237 = view_196 = None
        sum_63: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_349, [0], True, dtype = torch.float32);  view_349 = None
        view_350: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [768]);  sum_63 = None
        view_351: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [4, 476, 3072]);  mm_38 = None
        convert_element_type_672: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [4, 476, 3072]);  addmm_52 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_53: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_195, 0.5)
        mul_144: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_351, mul_53);  mul_53 = None
        div_26: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_195, 1.4142135623730951);  view_195 = None
        erf_8: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_26)
        add_72: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1.0);  erf_8 = None
        mul_145: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_351, add_72);  view_351 = add_72 = None
        convert_element_type_674: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_26, torch.float32);  div_26 = None
        pow_4: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_674, 2);  convert_element_type_674 = None
        neg_6: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_4);  pow_4 = None
        exp_15: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        mul_146: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_15, 1.1283791670955126);  exp_15 = None
        mul_147: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, mul_144);  mul_146 = mul_144 = None
        convert_element_type_675: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_147, torch.float16);  mul_147 = None
        div_49: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_675, 1.4142135623730951);  convert_element_type_675 = None
        mul_148: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, 0.5);  mul_145 = None
        add_119: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_49, mul_148);  div_49 = mul_148 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_352: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_119, [1904, 3072]);  add_119 = None
        mm_40: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_352, permute_240);  permute_240 = None
        permute_241: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_352, [1, 0])
        mm_41: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_241, view_194);  permute_241 = view_194 = None
        sum_64: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_352, [0], True, dtype = torch.float32);  view_352 = None
        view_353: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [3072]);  sum_64 = None
        view_354: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [4, 476, 768]);  mm_40 = None
        convert_element_type_681: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_354, torch.float32);  view_354 = None
        add_120: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_142, convert_element_type_681);  mul_142 = convert_element_type_681 = None
        convert_element_type_682: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_150: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_120, primals_145);  primals_145 = None
        mul_151: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, 768)
        sum_65: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
        mul_152: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, mul_51);  mul_150 = None
        sum_66: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
        mul_153: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_51, sum_66);  sum_66 = None
        sub_61: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_151, sum_65);  mul_151 = sum_65 = None
        sub_62: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_61, mul_153);  sub_61 = mul_153 = None
        mul_154: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, sub_62);  div_50 = sub_62 = None
        mul_155: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_120, mul_51);  mul_51 = None
        sum_67: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
        sum_68: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_120, [0, 1]);  add_120 = None
        convert_element_type_684: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_154, torch.float16)
        add_121: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_9, mul_154);  tangents_9 = mul_154 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_355: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_684, [1904, 768]);  convert_element_type_684 = None
        mm_42: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_355, permute_244);  permute_244 = None
        permute_245: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_355, [1, 0])
        mm_43: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_245, view_192);  permute_245 = view_192 = None
        sum_69: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_355, [0], True, dtype = torch.float32);  view_355 = None
        view_356: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [768]);  sum_69 = None
        view_357: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [4, 476, 768]);  mm_42 = None
        convert_element_type_690: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_358: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_357, [4, 476, 12, 64]);  view_357 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_248: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_97: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_359: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [48, 476, 64]);  clone_97 = None
        bmm_36: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_249, view_359);  permute_249 = None
        bmm_37: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_359, permute_250);  view_359 = permute_250 = None
        view_360: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [4, 12, 476, 64]);  bmm_36 = None
        view_361: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [4, 12, 476, 476]);  bmm_37 = None
        convert_element_type_696: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_361, torch.float32);  view_361 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_156: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_696, div_25);  convert_element_type_696 = None
        sum_70: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_156, [-1], True)
        neg_7: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_3: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_70, mul_156);  neg_7 = sum_70 = mul_156 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_697: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.float16);  fma_3 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_51: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_697, 8.0);  convert_element_type_697 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_362: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_51, [48, 476, 476]);  div_51 = None
        bmm_38: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_251, view_362);  permute_251 = None
        bmm_39: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_362, permute_252);  view_362 = permute_252 = None
        view_363: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [4, 12, 64, 476]);  bmm_38 = None
        view_364: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [4, 12, 476, 64]);  bmm_39 = None
        permute_253: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_363, [0, 1, 3, 2]);  view_363 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_254: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_98: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_254, memory_format = torch.contiguous_format);  permute_254 = None
        view_365: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_98, [4, 476, 768]);  clone_98 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_255: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_253, [0, 2, 1, 3]);  permute_253 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_366: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_255, [4, 476, 768]);  permute_255 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_256: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_99: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_256, memory_format = torch.contiguous_format);  permute_256 = None
        view_367: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [4, 476, 768]);  clone_99 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_368: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [1904, 768]);  view_365 = None
        mm_44: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_368, permute_257);  permute_257 = None
        permute_258: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_368, [1, 0])
        mm_45: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_258, view_176);  permute_258 = None
        sum_71: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_368, [0], True, dtype = torch.float32);  view_368 = None
        view_369: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_71, [768]);  sum_71 = None
        view_370: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [4, 476, 768]);  mm_44 = None
        convert_element_type_707: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_370, torch.float32);  view_370 = None
        add_122: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, convert_element_type_707);  add_121 = convert_element_type_707 = None
        convert_element_type_708: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_100: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_366, memory_format = torch.contiguous_format);  view_366 = None
        view_371: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_100, [1904, 768]);  clone_100 = None
        mm_46: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_371, permute_261);  permute_261 = None
        permute_262: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_371, [1, 0])
        mm_47: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_262, view_176);  permute_262 = None
        sum_72: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_371, [0], True, dtype = torch.float32);  view_371 = None
        view_372: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        view_373: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [4, 476, 768]);  mm_46 = None
        convert_element_type_715: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_373, torch.float32);  view_373 = None
        add_123: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, convert_element_type_715);  add_122 = convert_element_type_715 = None
        convert_element_type_716: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_374: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_367, [1904, 768]);  view_367 = None
        mm_48: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_374, permute_265);  permute_265 = None
        permute_266: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_374, [1, 0])
        mm_49: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_266, view_176);  permute_266 = view_176 = None
        sum_73: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_374, [0], True, dtype = torch.float32);  view_374 = None
        view_375: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_73, [768]);  sum_73 = None
        view_376: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [4, 476, 768]);  mm_48 = None
        convert_element_type_723: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_376, torch.float32);  view_376 = None
        add_124: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_123, convert_element_type_723);  add_123 = convert_element_type_723 = None
        convert_element_type_724: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_158: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_124, primals_135);  primals_135 = None
        mul_159: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, 768)
        sum_74: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_158, [2], True)
        mul_160: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_158, mul_49);  mul_158 = None
        sum_75: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_160, [2], True);  mul_160 = None
        mul_161: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, sum_75);  sum_75 = None
        sub_64: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_159, sum_74);  mul_159 = sum_74 = None
        sub_65: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_64, mul_161);  sub_64 = mul_161 = None
        mul_162: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, sub_65);  div_52 = sub_65 = None
        mul_163: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_124, mul_49);  mul_49 = None
        sum_76: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_163, [0, 1]);  mul_163 = None
        sum_77: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_124, [0, 1]);  add_124 = None
        convert_element_type_726: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_162, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_377: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_726, [1904, 768]);  convert_element_type_726 = None
        mm_50: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_377, permute_269);  permute_269 = None
        permute_270: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_377, [1, 0])
        mm_51: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_270, view_174);  permute_270 = view_174 = None
        sum_78: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_377, [0], True, dtype = torch.float32);  view_377 = None
        view_378: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_78, [768]);  sum_78 = None
        view_379: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [4, 476, 3072]);  mm_50 = None
        convert_element_type_732: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [4, 476, 3072]);  addmm_46 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_47: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_173, 0.5)
        mul_164: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_379, mul_47);  mul_47 = None
        div_23: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_173, 1.4142135623730951);  view_173 = None
        erf_7: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_23)
        add_64: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1.0);  erf_7 = None
        mul_165: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_379, add_64);  view_379 = add_64 = None
        convert_element_type_734: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_23, torch.float32);  div_23 = None
        pow_5: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_734, 2);  convert_element_type_734 = None
        neg_8: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_5);  pow_5 = None
        exp_16: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        mul_166: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_16, 1.1283791670955126);  exp_16 = None
        mul_167: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, mul_164);  mul_166 = mul_164 = None
        convert_element_type_735: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_167, torch.float16);  mul_167 = None
        div_53: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_735, 1.4142135623730951);  convert_element_type_735 = None
        mul_168: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, 0.5);  mul_165 = None
        add_125: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_53, mul_168);  div_53 = mul_168 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_380: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_125, [1904, 3072]);  add_125 = None
        mm_52: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_380, permute_273);  permute_273 = None
        permute_274: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_53: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_274, view_172);  permute_274 = view_172 = None
        sum_79: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_380, [0], True, dtype = torch.float32);  view_380 = None
        view_381: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_79, [3072]);  sum_79 = None
        view_382: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [4, 476, 768]);  mm_52 = None
        convert_element_type_741: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        add_126: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_162, convert_element_type_741);  mul_162 = convert_element_type_741 = None
        convert_element_type_742: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_170: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_126, primals_129);  primals_129 = None
        mul_171: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, 768)
        sum_80: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True)
        mul_172: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_170, mul_45);  mul_170 = None
        sum_81: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
        mul_173: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, sum_81);  sum_81 = None
        sub_67: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_171, sum_80);  mul_171 = sum_80 = None
        sub_68: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_67, mul_173);  sub_67 = mul_173 = None
        mul_174: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_54, sub_68);  div_54 = sub_68 = None
        mul_175: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_126, mul_45);  mul_45 = None
        sum_82: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
        sum_83: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
        convert_element_type_744: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_174, torch.float16)
        add_127: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_8, mul_174);  tangents_8 = mul_174 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_383: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_744, [1904, 768]);  convert_element_type_744 = None
        mm_54: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_383, permute_277);  permute_277 = None
        permute_278: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_383, [1, 0])
        mm_55: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_278, view_170);  permute_278 = view_170 = None
        sum_84: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_383, [0], True, dtype = torch.float32);  view_383 = None
        view_384: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_84, [768]);  sum_84 = None
        view_385: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [4, 476, 768]);  mm_54 = None
        convert_element_type_750: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_386: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_385, [4, 476, 12, 64]);  view_385 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_281: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_101: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_281, memory_format = torch.contiguous_format);  permute_281 = None
        view_387: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [48, 476, 64]);  clone_101 = None
        bmm_40: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_282, view_387);  permute_282 = None
        bmm_41: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_387, permute_283);  view_387 = permute_283 = None
        view_388: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [4, 12, 476, 64]);  bmm_40 = None
        view_389: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [4, 12, 476, 476]);  bmm_41 = None
        convert_element_type_756: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_389, torch.float32);  view_389 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_176: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_756, div_22);  convert_element_type_756 = None
        sum_85: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_176, [-1], True)
        neg_9: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_4: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_9, sum_85, mul_176);  neg_9 = sum_85 = mul_176 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_757: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.float16);  fma_4 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_55: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_757, 8.0);  convert_element_type_757 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_390: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_55, [48, 476, 476]);  div_55 = None
        bmm_42: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_284, view_390);  permute_284 = None
        bmm_43: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_390, permute_285);  view_390 = permute_285 = None
        view_391: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [4, 12, 64, 476]);  bmm_42 = None
        view_392: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [4, 12, 476, 64]);  bmm_43 = None
        permute_286: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_391, [0, 1, 3, 2]);  view_391 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_287: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_102: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_393: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [4, 476, 768]);  clone_102 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_288: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_286, [0, 2, 1, 3]);  permute_286 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_394: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_288, [4, 476, 768]);  permute_288 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_289: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_103: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
        view_395: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [4, 476, 768]);  clone_103 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_396: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_393, [1904, 768]);  view_393 = None
        mm_56: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_396, permute_290);  permute_290 = None
        permute_291: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_396, [1, 0])
        mm_57: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_291, view_154);  permute_291 = None
        sum_86: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_396, [0], True, dtype = torch.float32);  view_396 = None
        view_397: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_86, [768]);  sum_86 = None
        view_398: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [4, 476, 768]);  mm_56 = None
        convert_element_type_767: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        add_128: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_127, convert_element_type_767);  add_127 = convert_element_type_767 = None
        convert_element_type_768: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_104: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_394, memory_format = torch.contiguous_format);  view_394 = None
        view_399: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_104, [1904, 768]);  clone_104 = None
        mm_58: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_399, permute_294);  permute_294 = None
        permute_295: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_399, [1, 0])
        mm_59: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_295, view_154);  permute_295 = None
        sum_87: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_399, [0], True, dtype = torch.float32);  view_399 = None
        view_400: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        view_401: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [4, 476, 768]);  mm_58 = None
        convert_element_type_775: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_401, torch.float32);  view_401 = None
        add_129: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_128, convert_element_type_775);  add_128 = convert_element_type_775 = None
        convert_element_type_776: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_402: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_395, [1904, 768]);  view_395 = None
        mm_60: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_402, permute_298);  permute_298 = None
        permute_299: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_402, [1, 0])
        mm_61: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_299, view_154);  permute_299 = view_154 = None
        sum_88: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_402, [0], True, dtype = torch.float32);  view_402 = None
        view_403: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [768]);  sum_88 = None
        view_404: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [4, 476, 768]);  mm_60 = None
        convert_element_type_783: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_404, torch.float32);  view_404 = None
        add_130: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, convert_element_type_783);  add_129 = convert_element_type_783 = None
        convert_element_type_784: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_178: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_130, primals_119);  primals_119 = None
        mul_179: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, 768)
        sum_89: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_178, [2], True)
        mul_180: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, mul_43);  mul_178 = None
        sum_90: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_180, [2], True);  mul_180 = None
        mul_181: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, sum_90);  sum_90 = None
        sub_70: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_179, sum_89);  mul_179 = sum_89 = None
        sub_71: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, mul_181);  sub_70 = mul_181 = None
        mul_182: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_56, sub_71);  div_56 = sub_71 = None
        mul_183: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_130, mul_43);  mul_43 = None
        sum_91: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_183, [0, 1]);  mul_183 = None
        sum_92: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_130, [0, 1]);  add_130 = None
        convert_element_type_786: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_182, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_405: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_786, [1904, 768]);  convert_element_type_786 = None
        mm_62: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_405, permute_302);  permute_302 = None
        permute_303: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_405, [1, 0])
        mm_63: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_303, view_152);  permute_303 = view_152 = None
        sum_93: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_405, [0], True, dtype = torch.float32);  view_405 = None
        view_406: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [768]);  sum_93 = None
        view_407: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [4, 476, 3072]);  mm_62 = None
        convert_element_type_792: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [4, 476, 3072]);  addmm_40 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_41: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_151, 0.5)
        mul_184: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_407, mul_41);  mul_41 = None
        div_20: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_151, 1.4142135623730951);  view_151 = None
        erf_6: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_20)
        add_56: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1.0);  erf_6 = None
        mul_185: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_407, add_56);  view_407 = add_56 = None
        convert_element_type_794: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_20, torch.float32);  div_20 = None
        pow_6: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_794, 2);  convert_element_type_794 = None
        neg_10: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_6);  pow_6 = None
        exp_17: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        mul_186: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_17, 1.1283791670955126);  exp_17 = None
        mul_187: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, mul_184);  mul_186 = mul_184 = None
        convert_element_type_795: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_187, torch.float16);  mul_187 = None
        div_57: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_795, 1.4142135623730951);  convert_element_type_795 = None
        mul_188: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, 0.5);  mul_185 = None
        add_131: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_57, mul_188);  div_57 = mul_188 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_408: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_131, [1904, 3072]);  add_131 = None
        mm_64: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_408, permute_306);  permute_306 = None
        permute_307: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_408, [1, 0])
        mm_65: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_307, view_150);  permute_307 = view_150 = None
        sum_94: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_408, [0], True, dtype = torch.float32);  view_408 = None
        view_409: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [3072]);  sum_94 = None
        view_410: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [4, 476, 768]);  mm_64 = None
        convert_element_type_801: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_410, torch.float32);  view_410 = None
        add_132: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_182, convert_element_type_801);  mul_182 = convert_element_type_801 = None
        convert_element_type_802: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_190: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_132, primals_113);  primals_113 = None
        mul_191: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, 768)
        sum_95: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_190, [2], True)
        mul_192: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, mul_39);  mul_190 = None
        sum_96: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True);  mul_192 = None
        mul_193: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, sum_96);  sum_96 = None
        sub_73: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_191, sum_95);  mul_191 = sum_95 = None
        sub_74: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_73, mul_193);  sub_73 = mul_193 = None
        mul_194: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_58, sub_74);  div_58 = sub_74 = None
        mul_195: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_132, mul_39);  mul_39 = None
        sum_97: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 1]);  mul_195 = None
        sum_98: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1]);  add_132 = None
        convert_element_type_804: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_194, torch.float16)
        add_133: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_7, mul_194);  tangents_7 = mul_194 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_411: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_804, [1904, 768]);  convert_element_type_804 = None
        mm_66: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_411, permute_310);  permute_310 = None
        permute_311: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_411, [1, 0])
        mm_67: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_311, view_148);  permute_311 = view_148 = None
        sum_99: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_411, [0], True, dtype = torch.float32);  view_411 = None
        view_412: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [768]);  sum_99 = None
        view_413: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [4, 476, 768]);  mm_66 = None
        convert_element_type_810: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_414: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_413, [4, 476, 12, 64]);  view_413 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_314: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_105: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_314, memory_format = torch.contiguous_format);  permute_314 = None
        view_415: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_105, [48, 476, 64]);  clone_105 = None
        bmm_44: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_315, view_415);  permute_315 = None
        bmm_45: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_415, permute_316);  view_415 = permute_316 = None
        view_416: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [4, 12, 476, 64]);  bmm_44 = None
        view_417: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [4, 12, 476, 476]);  bmm_45 = None
        convert_element_type_816: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_417, torch.float32);  view_417 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_196: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_816, div_19);  convert_element_type_816 = None
        sum_100: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_196, [-1], True)
        neg_11: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_5: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_11, sum_100, mul_196);  neg_11 = sum_100 = mul_196 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_817: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.float16);  fma_5 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_59: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_817, 8.0);  convert_element_type_817 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_418: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_59, [48, 476, 476]);  div_59 = None
        bmm_46: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_317, view_418);  permute_317 = None
        bmm_47: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_418, permute_318);  view_418 = permute_318 = None
        view_419: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [4, 12, 64, 476]);  bmm_46 = None
        view_420: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [4, 12, 476, 64]);  bmm_47 = None
        permute_319: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_419, [0, 1, 3, 2]);  view_419 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_320: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_106: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_320, memory_format = torch.contiguous_format);  permute_320 = None
        view_421: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_106, [4, 476, 768]);  clone_106 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_321: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_319, [0, 2, 1, 3]);  permute_319 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_422: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_321, [4, 476, 768]);  permute_321 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_322: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_107: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None
        view_423: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_107, [4, 476, 768]);  clone_107 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_424: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_421, [1904, 768]);  view_421 = None
        mm_68: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_424, permute_323);  permute_323 = None
        permute_324: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_424, [1, 0])
        mm_69: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_324, view_132);  permute_324 = None
        sum_101: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_424, [0], True, dtype = torch.float32);  view_424 = None
        view_425: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_101, [768]);  sum_101 = None
        view_426: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [4, 476, 768]);  mm_68 = None
        convert_element_type_827: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_426, torch.float32);  view_426 = None
        add_134: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_133, convert_element_type_827);  add_133 = convert_element_type_827 = None
        convert_element_type_828: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_108: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_422, memory_format = torch.contiguous_format);  view_422 = None
        view_427: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [1904, 768]);  clone_108 = None
        mm_70: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_427, permute_327);  permute_327 = None
        permute_328: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_427, [1, 0])
        mm_71: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_328, view_132);  permute_328 = None
        sum_102: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_427, [0], True, dtype = torch.float32);  view_427 = None
        view_428: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        view_429: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [4, 476, 768]);  mm_70 = None
        convert_element_type_835: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_429, torch.float32);  view_429 = None
        add_135: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, convert_element_type_835);  add_134 = convert_element_type_835 = None
        convert_element_type_836: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_430: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_423, [1904, 768]);  view_423 = None
        mm_72: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_430, permute_331);  permute_331 = None
        permute_332: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_430, [1, 0])
        mm_73: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_332, view_132);  permute_332 = view_132 = None
        sum_103: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_430, [0], True, dtype = torch.float32);  view_430 = None
        view_431: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_103, [768]);  sum_103 = None
        view_432: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [4, 476, 768]);  mm_72 = None
        convert_element_type_843: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_432, torch.float32);  view_432 = None
        add_136: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, convert_element_type_843);  add_135 = convert_element_type_843 = None
        convert_element_type_844: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_198: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_136, primals_103);  primals_103 = None
        mul_199: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, 768)
        sum_104: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_198, [2], True)
        mul_200: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, mul_37);  mul_198 = None
        sum_105: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True);  mul_200 = None
        mul_201: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_37, sum_105);  sum_105 = None
        sub_76: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_199, sum_104);  mul_199 = sum_104 = None
        sub_77: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_76, mul_201);  sub_76 = mul_201 = None
        mul_202: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_60, sub_77);  div_60 = sub_77 = None
        mul_203: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_136, mul_37);  mul_37 = None
        sum_106: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_203, [0, 1]);  mul_203 = None
        sum_107: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_136, [0, 1]);  add_136 = None
        convert_element_type_846: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_202, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_433: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_846, [1904, 768]);  convert_element_type_846 = None
        mm_74: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_433, permute_335);  permute_335 = None
        permute_336: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_75: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_336, view_130);  permute_336 = view_130 = None
        sum_108: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_433, [0], True, dtype = torch.float32);  view_433 = None
        view_434: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_108, [768]);  sum_108 = None
        view_435: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [4, 476, 3072]);  mm_74 = None
        convert_element_type_852: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [4, 476, 3072]);  addmm_34 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_35: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_129, 0.5)
        mul_204: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_435, mul_35);  mul_35 = None
        div_17: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_129, 1.4142135623730951);  view_129 = None
        erf_5: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_17)
        add_48: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1.0);  erf_5 = None
        mul_205: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_435, add_48);  view_435 = add_48 = None
        convert_element_type_854: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_17, torch.float32);  div_17 = None
        pow_7: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_854, 2);  convert_element_type_854 = None
        neg_12: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_7);  pow_7 = None
        exp_18: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        mul_206: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_18, 1.1283791670955126);  exp_18 = None
        mul_207: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, mul_204);  mul_206 = mul_204 = None
        convert_element_type_855: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_207, torch.float16);  mul_207 = None
        div_61: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_855, 1.4142135623730951);  convert_element_type_855 = None
        mul_208: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, 0.5);  mul_205 = None
        add_137: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_61, mul_208);  div_61 = mul_208 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_436: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_137, [1904, 3072]);  add_137 = None
        mm_76: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_436, permute_339);  permute_339 = None
        permute_340: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_77: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_340, view_128);  permute_340 = view_128 = None
        sum_109: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_436, [0], True, dtype = torch.float32);  view_436 = None
        view_437: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_109, [3072]);  sum_109 = None
        view_438: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [4, 476, 768]);  mm_76 = None
        convert_element_type_861: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_438, torch.float32);  view_438 = None
        add_138: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_202, convert_element_type_861);  mul_202 = convert_element_type_861 = None
        convert_element_type_862: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_210: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_138, primals_97);  primals_97 = None
        mul_211: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, 768)
        sum_110: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_210, [2], True)
        mul_212: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, mul_33);  mul_210 = None
        sum_111: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_212, [2], True);  mul_212 = None
        mul_213: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, sum_111);  sum_111 = None
        sub_79: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_211, sum_110);  mul_211 = sum_110 = None
        sub_80: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_79, mul_213);  sub_79 = mul_213 = None
        mul_214: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_62, sub_80);  div_62 = sub_80 = None
        mul_215: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_138, mul_33);  mul_33 = None
        sum_112: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_215, [0, 1]);  mul_215 = None
        sum_113: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_138, [0, 1]);  add_138 = None
        convert_element_type_864: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_214, torch.float16)
        add_139: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_6, mul_214);  tangents_6 = mul_214 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_439: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_864, [1904, 768]);  convert_element_type_864 = None
        mm_78: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_439, permute_343);  permute_343 = None
        permute_344: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_439, [1, 0])
        mm_79: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_344, view_126);  permute_344 = view_126 = None
        sum_114: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_439, [0], True, dtype = torch.float32);  view_439 = None
        view_440: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_114, [768]);  sum_114 = None
        view_441: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [4, 476, 768]);  mm_78 = None
        convert_element_type_870: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_442: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [4, 476, 12, 64]);  view_441 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_347: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_109: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_347, memory_format = torch.contiguous_format);  permute_347 = None
        view_443: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [48, 476, 64]);  clone_109 = None
        bmm_48: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_348, view_443);  permute_348 = None
        bmm_49: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_443, permute_349);  view_443 = permute_349 = None
        view_444: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [4, 12, 476, 64]);  bmm_48 = None
        view_445: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [4, 12, 476, 476]);  bmm_49 = None
        convert_element_type_876: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_445, torch.float32);  view_445 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_216: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_876, div_16);  convert_element_type_876 = None
        sum_115: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_216, [-1], True)
        neg_13: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_6: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_13, sum_115, mul_216);  neg_13 = sum_115 = mul_216 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_877: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.float16);  fma_6 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_63: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_877, 8.0);  convert_element_type_877 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_446: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_63, [48, 476, 476]);  div_63 = None
        bmm_50: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_350, view_446);  permute_350 = None
        bmm_51: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_446, permute_351);  view_446 = permute_351 = None
        view_447: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [4, 12, 64, 476]);  bmm_50 = None
        view_448: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [4, 12, 476, 64]);  bmm_51 = None
        permute_352: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_447, [0, 1, 3, 2]);  view_447 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_353: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_110: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_353, memory_format = torch.contiguous_format);  permute_353 = None
        view_449: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [4, 476, 768]);  clone_110 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_354: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_352, [0, 2, 1, 3]);  permute_352 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_450: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_354, [4, 476, 768]);  permute_354 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_355: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_111: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
        view_451: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_111, [4, 476, 768]);  clone_111 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_452: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_449, [1904, 768]);  view_449 = None
        mm_80: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_452, permute_356);  permute_356 = None
        permute_357: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_452, [1, 0])
        mm_81: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_357, view_110);  permute_357 = None
        sum_116: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_452, [0], True, dtype = torch.float32);  view_452 = None
        view_453: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_116, [768]);  sum_116 = None
        view_454: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [4, 476, 768]);  mm_80 = None
        convert_element_type_887: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_454, torch.float32);  view_454 = None
        add_140: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_139, convert_element_type_887);  add_139 = convert_element_type_887 = None
        convert_element_type_888: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_112: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_450, memory_format = torch.contiguous_format);  view_450 = None
        view_455: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_112, [1904, 768]);  clone_112 = None
        mm_82: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_455, permute_360);  permute_360 = None
        permute_361: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_455, [1, 0])
        mm_83: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_361, view_110);  permute_361 = None
        sum_117: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_455, [0], True, dtype = torch.float32);  view_455 = None
        view_456: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        view_457: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [4, 476, 768]);  mm_82 = None
        convert_element_type_895: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_457, torch.float32);  view_457 = None
        add_141: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_140, convert_element_type_895);  add_140 = convert_element_type_895 = None
        convert_element_type_896: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_458: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_451, [1904, 768]);  view_451 = None
        mm_84: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_458, permute_364);  permute_364 = None
        permute_365: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_458, [1, 0])
        mm_85: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_365, view_110);  permute_365 = view_110 = None
        sum_118: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_458, [0], True, dtype = torch.float32);  view_458 = None
        view_459: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [768]);  sum_118 = None
        view_460: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [4, 476, 768]);  mm_84 = None
        convert_element_type_903: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_460, torch.float32);  view_460 = None
        add_142: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, convert_element_type_903);  add_141 = convert_element_type_903 = None
        convert_element_type_904: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_218: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_142, primals_87);  primals_87 = None
        mul_219: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_218, 768)
        sum_119: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_218, [2], True)
        mul_220: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_218, mul_31);  mul_218 = None
        sum_120: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_220, [2], True);  mul_220 = None
        mul_221: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_31, sum_120);  sum_120 = None
        sub_82: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_219, sum_119);  mul_219 = sum_119 = None
        sub_83: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_82, mul_221);  sub_82 = mul_221 = None
        mul_222: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_64, sub_83);  div_64 = sub_83 = None
        mul_223: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_142, mul_31);  mul_31 = None
        sum_121: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_223, [0, 1]);  mul_223 = None
        sum_122: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1]);  add_142 = None
        convert_element_type_906: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_222, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_461: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_906, [1904, 768]);  convert_element_type_906 = None
        mm_86: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_461, permute_368);  permute_368 = None
        permute_369: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_461, [1, 0])
        mm_87: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_369, view_108);  permute_369 = view_108 = None
        sum_123: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_461, [0], True, dtype = torch.float32);  view_461 = None
        view_462: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [768]);  sum_123 = None
        view_463: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [4, 476, 3072]);  mm_86 = None
        convert_element_type_912: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [4, 476, 3072]);  addmm_28 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_29: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_107, 0.5)
        mul_224: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_463, mul_29);  mul_29 = None
        div_14: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_107, 1.4142135623730951);  view_107 = None
        erf_4: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_14)
        add_40: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1.0);  erf_4 = None
        mul_225: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_463, add_40);  view_463 = add_40 = None
        convert_element_type_914: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_14, torch.float32);  div_14 = None
        pow_8: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_914, 2);  convert_element_type_914 = None
        neg_14: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_8);  pow_8 = None
        exp_19: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        mul_226: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_19, 1.1283791670955126);  exp_19 = None
        mul_227: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_226, mul_224);  mul_226 = mul_224 = None
        convert_element_type_915: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_227, torch.float16);  mul_227 = None
        div_65: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_915, 1.4142135623730951);  convert_element_type_915 = None
        mul_228: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_225, 0.5);  mul_225 = None
        add_143: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_65, mul_228);  div_65 = mul_228 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_464: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_143, [1904, 3072]);  add_143 = None
        mm_88: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_464, permute_372);  permute_372 = None
        permute_373: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_89: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_373, view_106);  permute_373 = view_106 = None
        sum_124: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_464, [0], True, dtype = torch.float32);  view_464 = None
        view_465: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_124, [3072]);  sum_124 = None
        view_466: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [4, 476, 768]);  mm_88 = None
        convert_element_type_921: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_466, torch.float32);  view_466 = None
        add_144: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_222, convert_element_type_921);  mul_222 = convert_element_type_921 = None
        convert_element_type_922: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_230: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_144, primals_81);  primals_81 = None
        mul_231: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, 768)
        sum_125: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_230, [2], True)
        mul_232: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, mul_27);  mul_230 = None
        sum_126: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_232, [2], True);  mul_232 = None
        mul_233: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, sum_126);  sum_126 = None
        sub_85: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_231, sum_125);  mul_231 = sum_125 = None
        sub_86: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_85, mul_233);  sub_85 = mul_233 = None
        mul_234: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_66, sub_86);  div_66 = sub_86 = None
        mul_235: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_144, mul_27);  mul_27 = None
        sum_127: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_235, [0, 1]);  mul_235 = None
        sum_128: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_144, [0, 1]);  add_144 = None
        convert_element_type_924: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_234, torch.float16)
        add_145: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_5, mul_234);  tangents_5 = mul_234 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_467: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_924, [1904, 768]);  convert_element_type_924 = None
        mm_90: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_467, permute_376);  permute_376 = None
        permute_377: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_467, [1, 0])
        mm_91: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_377, view_104);  permute_377 = view_104 = None
        sum_129: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_467, [0], True, dtype = torch.float32);  view_467 = None
        view_468: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [768]);  sum_129 = None
        view_469: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [4, 476, 768]);  mm_90 = None
        convert_element_type_930: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_470: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_469, [4, 476, 12, 64]);  view_469 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_380: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_113: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
        view_471: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [48, 476, 64]);  clone_113 = None
        bmm_52: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_381, view_471);  permute_381 = None
        bmm_53: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_471, permute_382);  view_471 = permute_382 = None
        view_472: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [4, 12, 476, 64]);  bmm_52 = None
        view_473: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [4, 12, 476, 476]);  bmm_53 = None
        convert_element_type_936: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_473, torch.float32);  view_473 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_236: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_936, div_13);  convert_element_type_936 = None
        sum_130: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_236, [-1], True)
        neg_15: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_7: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_15, sum_130, mul_236);  neg_15 = sum_130 = mul_236 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_937: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.float16);  fma_7 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_67: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_937, 8.0);  convert_element_type_937 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_474: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_67, [48, 476, 476]);  div_67 = None
        bmm_54: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_383, view_474);  permute_383 = None
        bmm_55: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_474, permute_384);  view_474 = permute_384 = None
        view_475: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [4, 12, 64, 476]);  bmm_54 = None
        view_476: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [4, 12, 476, 64]);  bmm_55 = None
        permute_385: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_475, [0, 1, 3, 2]);  view_475 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_386: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_114: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_386, memory_format = torch.contiguous_format);  permute_386 = None
        view_477: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_114, [4, 476, 768]);  clone_114 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_387: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_385, [0, 2, 1, 3]);  permute_385 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_478: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_387, [4, 476, 768]);  permute_387 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_388: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_115: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_388, memory_format = torch.contiguous_format);  permute_388 = None
        view_479: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [4, 476, 768]);  clone_115 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_480: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_477, [1904, 768]);  view_477 = None
        mm_92: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_480, permute_389);  permute_389 = None
        permute_390: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_480, [1, 0])
        mm_93: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_390, view_88);  permute_390 = None
        sum_131: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_480, [0], True, dtype = torch.float32);  view_480 = None
        view_481: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None
        view_482: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [4, 476, 768]);  mm_92 = None
        convert_element_type_947: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_482, torch.float32);  view_482 = None
        add_146: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, convert_element_type_947);  add_145 = convert_element_type_947 = None
        convert_element_type_948: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_116: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_478, memory_format = torch.contiguous_format);  view_478 = None
        view_483: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [1904, 768]);  clone_116 = None
        mm_94: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_483, permute_393);  permute_393 = None
        permute_394: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_483, [1, 0])
        mm_95: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_394, view_88);  permute_394 = None
        sum_132: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_483, [0], True, dtype = torch.float32);  view_483 = None
        view_484: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        view_485: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [4, 476, 768]);  mm_94 = None
        convert_element_type_955: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_485, torch.float32);  view_485 = None
        add_147: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_146, convert_element_type_955);  add_146 = convert_element_type_955 = None
        convert_element_type_956: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_486: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_479, [1904, 768]);  view_479 = None
        mm_96: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_486, permute_397);  permute_397 = None
        permute_398: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_486, [1, 0])
        mm_97: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_398, view_88);  permute_398 = view_88 = None
        sum_133: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_486, [0], True, dtype = torch.float32);  view_486 = None
        view_487: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_133, [768]);  sum_133 = None
        view_488: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [4, 476, 768]);  mm_96 = None
        convert_element_type_963: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_488, torch.float32);  view_488 = None
        add_148: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_147, convert_element_type_963);  add_147 = convert_element_type_963 = None
        convert_element_type_964: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_238: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_148, primals_71);  primals_71 = None
        mul_239: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, 768)
        sum_134: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
        mul_240: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_238, mul_25);  mul_238 = None
        sum_135: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
        mul_241: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, sum_135);  sum_135 = None
        sub_88: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_239, sum_134);  mul_239 = sum_134 = None
        sub_89: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_88, mul_241);  sub_88 = mul_241 = None
        mul_242: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_68, sub_89);  div_68 = sub_89 = None
        mul_243: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_148, mul_25);  mul_25 = None
        sum_136: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
        sum_137: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_148, [0, 1]);  add_148 = None
        convert_element_type_966: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_242, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_489: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_966, [1904, 768]);  convert_element_type_966 = None
        mm_98: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_489, permute_401);  permute_401 = None
        permute_402: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_489, [1, 0])
        mm_99: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_402, view_86);  permute_402 = view_86 = None
        sum_138: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_489, [0], True, dtype = torch.float32);  view_489 = None
        view_490: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_138, [768]);  sum_138 = None
        view_491: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [4, 476, 3072]);  mm_98 = None
        convert_element_type_972: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [4, 476, 3072]);  addmm_22 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_23: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_85, 0.5)
        mul_244: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_491, mul_23);  mul_23 = None
        div_11: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_85, 1.4142135623730951);  view_85 = None
        erf_3: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_11)
        add_32: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1.0);  erf_3 = None
        mul_245: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_491, add_32);  view_491 = add_32 = None
        convert_element_type_974: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_11, torch.float32);  div_11 = None
        pow_9: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_974, 2);  convert_element_type_974 = None
        neg_16: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_9);  pow_9 = None
        exp_20: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        mul_246: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_20, 1.1283791670955126);  exp_20 = None
        mul_247: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_246, mul_244);  mul_246 = mul_244 = None
        convert_element_type_975: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_247, torch.float16);  mul_247 = None
        div_69: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_975, 1.4142135623730951);  convert_element_type_975 = None
        mul_248: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, 0.5);  mul_245 = None
        add_149: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_69, mul_248);  div_69 = mul_248 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_492: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_149, [1904, 3072]);  add_149 = None
        mm_100: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_492, permute_405);  permute_405 = None
        permute_406: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_492, [1, 0])
        mm_101: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_406, view_84);  permute_406 = view_84 = None
        sum_139: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_492, [0], True, dtype = torch.float32);  view_492 = None
        view_493: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_139, [3072]);  sum_139 = None
        view_494: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [4, 476, 768]);  mm_100 = None
        convert_element_type_981: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_494, torch.float32);  view_494 = None
        add_150: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_242, convert_element_type_981);  mul_242 = convert_element_type_981 = None
        convert_element_type_982: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_250: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_150, primals_65);  primals_65 = None
        mul_251: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, 768)
        sum_140: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True)
        mul_252: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, mul_21);  mul_250 = None
        sum_141: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True);  mul_252 = None
        mul_253: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_21, sum_141);  sum_141 = None
        sub_91: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_251, sum_140);  mul_251 = sum_140 = None
        sub_92: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_91, mul_253);  sub_91 = mul_253 = None
        mul_254: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, sub_92);  div_70 = sub_92 = None
        mul_255: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_150, mul_21);  mul_21 = None
        sum_142: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 1]);  mul_255 = None
        sum_143: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_150, [0, 1]);  add_150 = None
        convert_element_type_984: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_254, torch.float16)
        add_151: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_4, mul_254);  tangents_4 = mul_254 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_495: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_984, [1904, 768]);  convert_element_type_984 = None
        mm_102: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_495, permute_409);  permute_409 = None
        permute_410: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_495, [1, 0])
        mm_103: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_410, view_82);  permute_410 = view_82 = None
        sum_144: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_495, [0], True, dtype = torch.float32);  view_495 = None
        view_496: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_144, [768]);  sum_144 = None
        view_497: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [4, 476, 768]);  mm_102 = None
        convert_element_type_990: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_498: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_497, [4, 476, 12, 64]);  view_497 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_413: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_117: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_413, memory_format = torch.contiguous_format);  permute_413 = None
        view_499: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [48, 476, 64]);  clone_117 = None
        bmm_56: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_414, view_499);  permute_414 = None
        bmm_57: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_499, permute_415);  view_499 = permute_415 = None
        view_500: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [4, 12, 476, 64]);  bmm_56 = None
        view_501: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [4, 12, 476, 476]);  bmm_57 = None
        convert_element_type_996: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_501, torch.float32);  view_501 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_256: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_996, div_10);  convert_element_type_996 = None
        sum_145: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_256, [-1], True)
        neg_17: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_8: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_17, sum_145, mul_256);  neg_17 = sum_145 = mul_256 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_997: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.float16);  fma_8 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_71: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_997, 8.0);  convert_element_type_997 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_502: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_71, [48, 476, 476]);  div_71 = None
        bmm_58: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_416, view_502);  permute_416 = None
        bmm_59: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_502, permute_417);  view_502 = permute_417 = None
        view_503: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [4, 12, 64, 476]);  bmm_58 = None
        view_504: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [4, 12, 476, 64]);  bmm_59 = None
        permute_418: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_503, [0, 1, 3, 2]);  view_503 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_419: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_118: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_505: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_118, [4, 476, 768]);  clone_118 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_420: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_418, [0, 2, 1, 3]);  permute_418 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_506: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_420, [4, 476, 768]);  permute_420 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_421: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_119: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
        view_507: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_119, [4, 476, 768]);  clone_119 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_508: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_505, [1904, 768]);  view_505 = None
        mm_104: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_508, permute_422);  permute_422 = None
        permute_423: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_508, [1, 0])
        mm_105: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_423, view_66);  permute_423 = None
        sum_146: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_508, [0], True, dtype = torch.float32);  view_508 = None
        view_509: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_146, [768]);  sum_146 = None
        view_510: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [4, 476, 768]);  mm_104 = None
        convert_element_type_1007: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_510, torch.float32);  view_510 = None
        add_152: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_151, convert_element_type_1007);  add_151 = convert_element_type_1007 = None
        convert_element_type_1008: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_120: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_506, memory_format = torch.contiguous_format);  view_506 = None
        view_511: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [1904, 768]);  clone_120 = None
        mm_106: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_511, permute_426);  permute_426 = None
        permute_427: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_511, [1, 0])
        mm_107: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_427, view_66);  permute_427 = None
        sum_147: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_511, [0], True, dtype = torch.float32);  view_511 = None
        view_512: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        view_513: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [4, 476, 768]);  mm_106 = None
        convert_element_type_1015: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_513, torch.float32);  view_513 = None
        add_153: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_152, convert_element_type_1015);  add_152 = convert_element_type_1015 = None
        convert_element_type_1016: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_514: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_507, [1904, 768]);  view_507 = None
        mm_108: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_514, permute_430);  permute_430 = None
        permute_431: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_514, [1, 0])
        mm_109: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_431, view_66);  permute_431 = view_66 = None
        sum_148: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_514, [0], True, dtype = torch.float32);  view_514 = None
        view_515: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_148, [768]);  sum_148 = None
        view_516: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [4, 476, 768]);  mm_108 = None
        convert_element_type_1023: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_516, torch.float32);  view_516 = None
        add_154: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_153, convert_element_type_1023);  add_153 = convert_element_type_1023 = None
        convert_element_type_1024: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_258: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_154, primals_55);  primals_55 = None
        mul_259: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_258, 768)
        sum_149: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_258, [2], True)
        mul_260: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_258, mul_19);  mul_258 = None
        sum_150: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_260, [2], True);  mul_260 = None
        mul_261: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, sum_150);  sum_150 = None
        sub_94: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_259, sum_149);  mul_259 = sum_149 = None
        sub_95: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, mul_261);  sub_94 = mul_261 = None
        mul_262: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, sub_95);  div_72 = sub_95 = None
        mul_263: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_154, mul_19);  mul_19 = None
        sum_151: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 1]);  mul_263 = None
        sum_152: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_154, [0, 1]);  add_154 = None
        convert_element_type_1026: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_262, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_517: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1026, [1904, 768]);  convert_element_type_1026 = None
        mm_110: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_517, permute_434);  permute_434 = None
        permute_435: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_517, [1, 0])
        mm_111: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_435, view_64);  permute_435 = view_64 = None
        sum_153: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_517, [0], True, dtype = torch.float32);  view_517 = None
        view_518: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [768]);  sum_153 = None
        view_519: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [4, 476, 3072]);  mm_110 = None
        convert_element_type_1032: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [4, 476, 3072]);  addmm_16 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_17: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_264: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_519, mul_17);  mul_17 = None
        div_8: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_63, 1.4142135623730951);  view_63 = None
        erf_2: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_8)
        add_24: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1.0);  erf_2 = None
        mul_265: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_519, add_24);  view_519 = add_24 = None
        convert_element_type_1034: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.float32);  div_8 = None
        pow_10: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1034, 2);  convert_element_type_1034 = None
        neg_18: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_10);  pow_10 = None
        exp_21: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        mul_266: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_21, 1.1283791670955126);  exp_21 = None
        mul_267: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_266, mul_264);  mul_266 = mul_264 = None
        convert_element_type_1035: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_267, torch.float16);  mul_267 = None
        div_73: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1035, 1.4142135623730951);  convert_element_type_1035 = None
        mul_268: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_265, 0.5);  mul_265 = None
        add_155: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_73, mul_268);  div_73 = mul_268 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_520: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_155, [1904, 3072]);  add_155 = None
        mm_112: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_520, permute_438);  permute_438 = None
        permute_439: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_113: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_439, view_62);  permute_439 = view_62 = None
        sum_154: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_520, [0], True, dtype = torch.float32);  view_520 = None
        view_521: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [3072]);  sum_154 = None
        view_522: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [4, 476, 768]);  mm_112 = None
        convert_element_type_1041: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_522, torch.float32);  view_522 = None
        add_156: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_262, convert_element_type_1041);  mul_262 = convert_element_type_1041 = None
        convert_element_type_1042: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_270: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_156, primals_49);  primals_49 = None
        mul_271: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, 768)
        sum_155: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
        mul_272: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_270, mul_15);  mul_270 = None
        sum_156: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
        mul_273: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_15, sum_156);  sum_156 = None
        sub_97: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_271, sum_155);  mul_271 = sum_155 = None
        sub_98: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_97, mul_273);  sub_97 = mul_273 = None
        mul_274: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_74, sub_98);  div_74 = sub_98 = None
        mul_275: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_156, mul_15);  mul_15 = None
        sum_157: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
        sum_158: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
        convert_element_type_1044: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_274, torch.float16)
        add_157: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_3, mul_274);  tangents_3 = mul_274 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_523: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1044, [1904, 768]);  convert_element_type_1044 = None
        mm_114: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_523, permute_442);  permute_442 = None
        permute_443: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_523, [1, 0])
        mm_115: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_443, view_60);  permute_443 = view_60 = None
        sum_159: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_523, [0], True, dtype = torch.float32);  view_523 = None
        view_524: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_159, [768]);  sum_159 = None
        view_525: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [4, 476, 768]);  mm_114 = None
        convert_element_type_1050: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_526: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_525, [4, 476, 12, 64]);  view_525 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_446: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_121: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_446, memory_format = torch.contiguous_format);  permute_446 = None
        view_527: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_121, [48, 476, 64]);  clone_121 = None
        bmm_60: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_447, view_527);  permute_447 = None
        bmm_61: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_527, permute_448);  view_527 = permute_448 = None
        view_528: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [4, 12, 476, 64]);  bmm_60 = None
        view_529: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [4, 12, 476, 476]);  bmm_61 = None
        convert_element_type_1056: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_276: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1056, div_7);  convert_element_type_1056 = None
        sum_160: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
        neg_19: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_9: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_19, sum_160, mul_276);  neg_19 = sum_160 = mul_276 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_1057: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.float16);  fma_9 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_75: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1057, 8.0);  convert_element_type_1057 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_530: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_75, [48, 476, 476]);  div_75 = None
        bmm_62: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_449, view_530);  permute_449 = None
        bmm_63: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_530, permute_450);  view_530 = permute_450 = None
        view_531: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [4, 12, 64, 476]);  bmm_62 = None
        view_532: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [4, 12, 476, 64]);  bmm_63 = None
        permute_451: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_531, [0, 1, 3, 2]);  view_531 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_452: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_122: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_533: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [4, 476, 768]);  clone_122 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_453: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_451, [0, 2, 1, 3]);  permute_451 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_534: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_453, [4, 476, 768]);  permute_453 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_454: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_123: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_454, memory_format = torch.contiguous_format);  permute_454 = None
        view_535: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [4, 476, 768]);  clone_123 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_536: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_533, [1904, 768]);  view_533 = None
        mm_116: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_536, permute_455);  permute_455 = None
        permute_456: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_536, [1, 0])
        mm_117: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_456, view_44);  permute_456 = None
        sum_161: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_536, [0], True, dtype = torch.float32);  view_536 = None
        view_537: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_161, [768]);  sum_161 = None
        view_538: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [4, 476, 768]);  mm_116 = None
        convert_element_type_1067: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_538, torch.float32);  view_538 = None
        add_158: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_157, convert_element_type_1067);  add_157 = convert_element_type_1067 = None
        convert_element_type_1068: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_124: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_534, memory_format = torch.contiguous_format);  view_534 = None
        view_539: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [1904, 768]);  clone_124 = None
        mm_118: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_539, permute_459);  permute_459 = None
        permute_460: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_539, [1, 0])
        mm_119: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_460, view_44);  permute_460 = None
        sum_162: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_539, [0], True, dtype = torch.float32);  view_539 = None
        view_540: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
        view_541: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [4, 476, 768]);  mm_118 = None
        convert_element_type_1075: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_541, torch.float32);  view_541 = None
        add_159: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_158, convert_element_type_1075);  add_158 = convert_element_type_1075 = None
        convert_element_type_1076: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_542: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_535, [1904, 768]);  view_535 = None
        mm_120: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_542, permute_463);  permute_463 = None
        permute_464: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_542, [1, 0])
        mm_121: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_464, view_44);  permute_464 = view_44 = None
        sum_163: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_542, [0], True, dtype = torch.float32);  view_542 = None
        view_543: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_163, [768]);  sum_163 = None
        view_544: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [4, 476, 768]);  mm_120 = None
        convert_element_type_1083: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_544, torch.float32);  view_544 = None
        add_160: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_159, convert_element_type_1083);  add_159 = convert_element_type_1083 = None
        convert_element_type_1084: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_278: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_160, primals_39);  primals_39 = None
        mul_279: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, 768)
        sum_164: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True)
        mul_280: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, mul_13);  mul_278 = None
        sum_165: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
        mul_281: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, sum_165);  sum_165 = None
        sub_100: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_279, sum_164);  mul_279 = sum_164 = None
        sub_101: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_100, mul_281);  sub_100 = mul_281 = None
        mul_282: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_76, sub_101);  div_76 = sub_101 = None
        mul_283: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_160, mul_13);  mul_13 = None
        sum_166: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
        sum_167: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_160, [0, 1]);  add_160 = None
        convert_element_type_1086: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_282, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_545: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1086, [1904, 768]);  convert_element_type_1086 = None
        mm_122: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_545, permute_467);  permute_467 = None
        permute_468: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_545, [1, 0])
        mm_123: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_468, view_42);  permute_468 = view_42 = None
        sum_168: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_545, [0], True, dtype = torch.float32);  view_545 = None
        view_546: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [768]);  sum_168 = None
        view_547: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [4, 476, 3072]);  mm_122 = None
        convert_element_type_1092: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [4, 476, 3072]);  addmm_10 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_11: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_284: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_547, mul_11);  mul_11 = None
        div_5: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_41, 1.4142135623730951);  view_41 = None
        erf_1: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_5)
        add_16: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1.0);  erf_1 = None
        mul_285: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_547, add_16);  view_547 = add_16 = None
        convert_element_type_1094: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_5, torch.float32);  div_5 = None
        pow_11: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1094, 2);  convert_element_type_1094 = None
        neg_20: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_11);  pow_11 = None
        exp_22: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        mul_286: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_22, 1.1283791670955126);  exp_22 = None
        mul_287: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, mul_284);  mul_286 = mul_284 = None
        convert_element_type_1095: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_287, torch.float16);  mul_287 = None
        div_77: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1095, 1.4142135623730951);  convert_element_type_1095 = None
        mul_288: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_285, 0.5);  mul_285 = None
        add_161: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_77, mul_288);  div_77 = mul_288 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_548: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_161, [1904, 3072]);  add_161 = None
        mm_124: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_548, permute_471);  permute_471 = None
        permute_472: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_125: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_472, view_40);  permute_472 = view_40 = None
        sum_169: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_548, [0], True, dtype = torch.float32);  view_548 = None
        view_549: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_169, [3072]);  sum_169 = None
        view_550: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [4, 476, 768]);  mm_124 = None
        convert_element_type_1101: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_550, torch.float32);  view_550 = None
        add_162: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_282, convert_element_type_1101);  mul_282 = convert_element_type_1101 = None
        convert_element_type_1102: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_290: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_162, primals_33);  primals_33 = None
        mul_291: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, 768)
        sum_170: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
        mul_292: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, mul_9);  mul_290 = None
        sum_171: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
        mul_293: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, sum_171);  sum_171 = None
        sub_103: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_291, sum_170);  mul_291 = sum_170 = None
        sub_104: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_103, mul_293);  sub_103 = mul_293 = None
        mul_294: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_78, sub_104);  div_78 = sub_104 = None
        mul_295: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_162, mul_9);  mul_9 = None
        sum_172: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
        sum_173: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1]);  add_162 = None
        convert_element_type_1104: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_294, torch.float16)
        add_163: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, mul_294);  tangents_2 = mul_294 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_551: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1104, [1904, 768]);  convert_element_type_1104 = None
        mm_126: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_551, permute_475);  permute_475 = None
        permute_476: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_551, [1, 0])
        mm_127: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_476, view_38);  permute_476 = view_38 = None
        sum_174: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_551, [0], True, dtype = torch.float32);  view_551 = None
        view_552: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_174, [768]);  sum_174 = None
        view_553: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [4, 476, 768]);  mm_126 = None
        convert_element_type_1110: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_554: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_553, [4, 476, 12, 64]);  view_553 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_479: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_125: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        view_555: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_125, [48, 476, 64]);  clone_125 = None
        bmm_64: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_480, view_555);  permute_480 = None
        bmm_65: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_555, permute_481);  view_555 = permute_481 = None
        view_556: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [4, 12, 476, 64]);  bmm_64 = None
        view_557: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [4, 12, 476, 476]);  bmm_65 = None
        convert_element_type_1116: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_557, torch.float32);  view_557 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        mul_296: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1116, div_4);  convert_element_type_1116 = None
        sum_175: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_296, [-1], True)
        neg_21: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_10: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_21, sum_175, mul_296);  neg_21 = sum_175 = mul_296 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_1117: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.float16);  fma_10 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_79: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1117, 8.0);  convert_element_type_1117 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_558: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_79, [48, 476, 476]);  div_79 = None
        bmm_66: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_482, view_558);  permute_482 = None
        bmm_67: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_558, permute_483);  view_558 = permute_483 = None
        view_559: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [4, 12, 64, 476]);  bmm_66 = None
        view_560: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [4, 12, 476, 64]);  bmm_67 = None
        permute_484: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_559, [0, 1, 3, 2]);  view_559 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_485: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_126: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_485, memory_format = torch.contiguous_format);  permute_485 = None
        view_561: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_126, [4, 476, 768]);  clone_126 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_486: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_484, [0, 2, 1, 3]);  permute_484 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_562: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_486, [4, 476, 768]);  permute_486 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_487: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_127: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_487, memory_format = torch.contiguous_format);  permute_487 = None
        view_563: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [4, 476, 768]);  clone_127 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_564: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_561, [1904, 768]);  view_561 = None
        mm_128: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_564, permute_488);  permute_488 = None
        permute_489: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_564, [1, 0])
        mm_129: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_489, view_22);  permute_489 = None
        sum_176: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_564, [0], True, dtype = torch.float32);  view_564 = None
        view_565: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_176, [768]);  sum_176 = None
        view_566: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [4, 476, 768]);  mm_128 = None
        convert_element_type_1127: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_566, torch.float32);  view_566 = None
        add_164: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_163, convert_element_type_1127);  add_163 = convert_element_type_1127 = None
        convert_element_type_1128: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_128: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_562, memory_format = torch.contiguous_format);  view_562 = None
        view_567: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [1904, 768]);  clone_128 = None
        mm_130: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_567, permute_492);  permute_492 = None
        permute_493: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_567, [1, 0])
        mm_131: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_493, view_22);  permute_493 = None
        sum_177: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_567, [0], True, dtype = torch.float32);  view_567 = None
        view_568: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
        view_569: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [4, 476, 768]);  mm_130 = None
        convert_element_type_1135: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_569, torch.float32);  view_569 = None
        add_165: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_164, convert_element_type_1135);  add_164 = convert_element_type_1135 = None
        convert_element_type_1136: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_570: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_563, [1904, 768]);  view_563 = None
        mm_132: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_570, permute_496);  permute_496 = None
        permute_497: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_570, [1, 0])
        mm_133: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_497, view_22);  permute_497 = view_22 = None
        sum_178: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_570, [0], True, dtype = torch.float32);  view_570 = None
        view_571: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_178, [768]);  sum_178 = None
        view_572: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [4, 476, 768]);  mm_132 = None
        convert_element_type_1143: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_572, torch.float32);  view_572 = None
        add_166: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_165, convert_element_type_1143);  add_165 = convert_element_type_1143 = None
        convert_element_type_1144: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:350 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_298: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_166, primals_23);  primals_23 = None
        mul_299: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_298, 768)
        sum_179: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True)
        mul_300: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_298, mul_7);  mul_298 = None
        sum_180: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True);  mul_300 = None
        mul_301: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, sum_180);  sum_180 = None
        sub_106: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_299, sum_179);  mul_299 = sum_179 = None
        sub_107: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_106, mul_301);  sub_106 = mul_301 = None
        mul_302: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_80, sub_107);  div_80 = sub_107 = None
        mul_303: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_166, mul_7);  mul_7 = None
        sum_181: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1]);  mul_303 = None
        sum_182: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_166, [0, 1]);  add_166 = None
        convert_element_type_1146: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_302, torch.float16)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:348 in forward, code: hidden_states = self.dense(hidden_states)
        view_573: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1146, [1904, 768]);  convert_element_type_1146 = None
        mm_134: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_573, permute_500);  permute_500 = None
        permute_501: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_573, [1, 0])
        mm_135: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_501, view_20);  permute_501 = view_20 = None
        sum_183: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_573, [0], True, dtype = torch.float32);  view_573 = None
        view_574: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [768]);  sum_183 = None
        view_575: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [4, 476, 3072]);  mm_134 = None
        convert_element_type_1152: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [4, 476, 3072]);  addmm_4 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:150 in gelu, code: return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        mul_5: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_304: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_575, mul_5);  mul_5 = None
        div_2: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(view_19, 1.4142135623730951);  view_19 = None
        erf: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.erf.default(div_2)
        add_8: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1.0);  erf = None
        mul_305: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_575, add_8);  view_575 = add_8 = None
        convert_element_type_1154: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_2, torch.float32);  div_2 = None
        pow_12: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_1154, 2);  convert_element_type_1154 = None
        neg_22: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.neg.default(pow_12);  pow_12 = None
        exp_23: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        mul_306: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Scalar(exp_23, 1.1283791670955126);  exp_23 = None
        mul_307: "f32[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, mul_304);  mul_306 = mul_304 = None
        convert_element_type_1155: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_307, torch.float16);  mul_307 = None
        div_81: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1155, 1.4142135623730951);  convert_element_type_1155 = None
        mul_308: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_305, 0.5);  mul_305 = None
        add_167: "f16[4, 476, 3072][1462272, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(div_81, mul_308);  div_81 = mul_308 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:335 in forward, code: hidden_states = self.dense(hidden_states)
        view_576: "f16[1904, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_167, [1904, 3072]);  add_167 = None
        mm_136: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_576, permute_504);  permute_504 = None
        permute_505: "f16[3072, 1904][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_576, [1, 0])
        mm_137: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_505, view_18);  permute_505 = view_18 = None
        sum_184: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_576, [0], True, dtype = torch.float32);  view_576 = None
        view_577: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_184, [3072]);  sum_184 = None
        view_578: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [4, 476, 768]);  mm_136 = None
        convert_element_type_1161: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_578, torch.float32);  view_578 = None
        add_168: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_302, convert_element_type_1161);  mul_302 = convert_element_type_1161 = None
        convert_element_type_1162: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:309 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_310: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_168, primals_17);  primals_17 = None
        mul_311: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_310, 768)
        sum_185: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_310, [2], True)
        mul_312: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_310, mul_3);  mul_310 = None
        sum_186: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True);  mul_312 = None
        mul_313: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, sum_186);  sum_186 = None
        sub_109: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_311, sum_185);  mul_311 = sum_185 = None
        sub_110: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_109, mul_313);  sub_109 = mul_313 = None
        mul_314: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_82, sub_110);  div_82 = sub_110 = None
        mul_315: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_168, mul_3);  mul_3 = None
        sum_187: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_315, [0, 1]);  mul_315 = None
        sum_188: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_168, [0, 1]);  add_168 = None
        convert_element_type_1164: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_314, torch.float16)
        add_169: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_1, mul_314);  tangents_1 = mul_314 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:307 in forward, code: hidden_states = self.dense(hidden_states)
        view_579: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1164, [1904, 768]);  convert_element_type_1164 = None
        mm_138: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_579, permute_508);  permute_508 = None
        permute_509: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_579, [1, 0])
        mm_139: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_509, view_16);  permute_509 = view_16 = None
        sum_189: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_579, [0], True, dtype = torch.float32);  view_579 = None
        view_580: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_189, [768]);  sum_189 = None
        view_581: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [4, 476, 768]);  mm_138 = None
        convert_element_type_1170: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:295 in forward, code: context_layer = context_layer.view(*new_context_layer_shape)
        view_582: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_581, [4, 476, 12, 64]);  view_581 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:293 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_512: "f16[4, 12, 476, 64][365568, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:292 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_129: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
        view_583: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [48, 476, 64]);  clone_129 = None
        bmm_68: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_513, view_583);  permute_513 = None
        bmm_69: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.bmm.default(view_583, permute_514);  view_583 = permute_514 = None
        view_584: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [4, 12, 476, 64]);  bmm_68 = None
        view_585: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [4, 12, 476, 476]);  bmm_69 = None
        convert_element_type_1176: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_585, torch.float32);  view_585 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_11: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [4, 12, 476, 476]);  bmm = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        add_4: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.add.Tensor(div, mul);  div = mul = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:286 in forward, code: attention_probs = nn.Softmax(dim=-1)(attention_scores)
        sub_2: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_4, amax);  add_4 = amax = None
        exp: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        div_1: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_316: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1176, div_1);  convert_element_type_1176 = None
        sum_190: "f32[4, 12, 476, 1][5712, 476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_316, [-1], True)
        neg_23: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_11: "f32[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.fma.default(neg_23, sum_190, mul_316);  neg_23 = sum_190 = mul_316 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:283 in forward, code: attention_scores = attention_scores + attention_mask
        convert_element_type_1177: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.float16);  fma_11 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:281 in forward, code: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        div_83: "f16[4, 12, 476, 476][2718912, 226576, 476, 1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_1177, 8.0);  convert_element_type_1177 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:280 in forward, code: attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        view_586: "f16[48, 476, 476][226576, 476, 1]cuda:0" = torch.ops.aten.reshape.default(div_83, [48, 476, 476]);  div_83 = None
        bmm_70: "f16[48, 64, 476][30464, 476, 1]cuda:0" = torch.ops.aten.bmm.default(permute_515, view_586);  permute_515 = None
        bmm_71: "f16[48, 476, 64][30464, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_586, permute_516);  view_586 = permute_516 = None
        view_587: "f16[4, 12, 64, 476][365568, 30464, 476, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [4, 12, 64, 476]);  bmm_70 = None
        view_588: "f16[4, 12, 476, 64][365568, 30464, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [4, 12, 476, 64]);  bmm_71 = None
        permute_517: "f16[4, 12, 476, 64][365568, 30464, 1, 476]cuda:0" = torch.ops.aten.permute.default(view_587, [0, 1, 3, 2]);  view_587 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_518: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_584, [0, 2, 1, 3]);  view_584 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_130: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_518, memory_format = torch.contiguous_format);  permute_518 = None
        view_589: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [4, 476, 768]);  clone_130 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_519: "f16[4, 476, 12, 64][365568, 1, 30464, 476]cuda:0" = torch.ops.aten.permute.default(permute_517, [0, 2, 1, 3]);  permute_517 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        view_590: "f16[4, 476, 768][365568, 1, 476]cuda:0" = torch.ops.aten.reshape.default(permute_519, [4, 476, 768]);  permute_519 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:268 in transpose_for_scores, code: return x.permute(0, 2, 1, 3)
        permute_520: "f16[4, 476, 12, 64][365568, 64, 30464, 1]cuda:0" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:267 in transpose_for_scores, code: x = x.view(*new_x_shape)
        clone_131: "f16[4, 476, 12, 64][365568, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
        view_591: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [4, 476, 768]);  clone_131 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:273 in forward, code: mixed_value_layer = self.value(hidden_states)
        view_592: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_589, [1904, 768]);  view_589 = None
        mm_140: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_592, permute_521);  permute_521 = None
        permute_522: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_592, [1, 0])
        mm_141: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_522, view);  permute_522 = None
        sum_191: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_592, [0], True, dtype = torch.float32);  view_592 = None
        view_593: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_191, [768]);  sum_191 = None
        view_594: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [4, 476, 768]);  mm_140 = None
        convert_element_type_1187: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_594, torch.float32);  view_594 = None
        add_170: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_169, convert_element_type_1187);  add_169 = convert_element_type_1187 = None
        convert_element_type_1188: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:272 in forward, code: mixed_key_layer = self.key(hidden_states)
        clone_132: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.clone.default(view_590, memory_format = torch.contiguous_format);  view_590 = None
        view_595: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_132, [1904, 768]);  clone_132 = None
        mm_142: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_595, permute_525);  permute_525 = None
        permute_526: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_595, [1, 0])
        mm_143: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_526, view);  permute_526 = None
        sum_192: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_595, [0], True, dtype = torch.float32);  view_595 = None
        view_596: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
        view_597: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [4, 476, 768]);  mm_142 = None
        convert_element_type_1195: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_597, torch.float32);  view_597 = None
        add_171: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_170, convert_element_type_1195);  add_170 = convert_element_type_1195 = None
        convert_element_type_1196: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:271 in forward, code: mixed_query_layer = self.query(hidden_states)
        view_598: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_591, [1904, 768]);  view_591 = None
        mm_144: "f16[1904, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_598, permute_529);  permute_529 = None
        permute_530: "f16[768, 1904][1, 768]cuda:0" = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_145: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_530, view);  permute_530 = view = None
        sum_193: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_598, [0], True, dtype = torch.float32);  view_598 = None
        view_599: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_193, [768]);  sum_193 = None
        view_600: "f16[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [4, 476, 768]);  mm_144 = None
        convert_element_type_1203: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_600, torch.float32);  view_600 = None
        add_172: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_171, convert_element_type_1203);  add_171 = convert_element_type_1203 = None
        convert_element_type_1204: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:243 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_318: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_172, primals_7);  primals_7 = None
        mul_319: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, 768)
        sum_194: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
        mul_320: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, mul_1);  mul_318 = None
        sum_195: "f32[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
        mul_321: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, sum_195);  sum_195 = None
        sub_112: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_319, sum_194);  mul_319 = sum_194 = None
        sub_113: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_112, mul_321);  sub_112 = mul_321 = None
        mul_322: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_84, sub_113);  div_84 = sub_113 = None
        mul_323: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_172, mul_1);  mul_1 = None
        sum_196: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
        sum_197: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_172, [0, 1]);  add_172 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:240 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        eq: "b8[4, 476][476, 1]cuda:0" = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_3: "b8[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_3, full_default_1, mul_322);  unsqueeze_3 = None
        full_default_2: "f32[2, 768][768, 1]cuda:0" = torch.ops.aten.full.default([2, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[2, 768][768, 1]cuda:0" = torch.ops.aten.index_put_.default(full_default_2, [primals_2], where, True);  full_default_2 = primals_2 = where = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:231 in forward, code: position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        expand: "i64[4, 476][0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_2, [4, 476]);  unsqueeze_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:239 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq_1: "b8[4, 476][476, 1]cuda:0" = torch.ops.aten.eq.Scalar(expand, -1)
        unsqueeze_4: "b8[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_1: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_4, full_default_1, mul_322);  unsqueeze_4 = None
        full_default_4: "f32[512, 768][768, 1]cuda:0" = torch.ops.aten.full.default([512, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[512, 768][768, 1]cuda:0" = torch.ops.aten.index_put_.default(full_default_4, [expand], where_1, True);  full_default_4 = expand = where_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/fastNLP/modules/encoder/bert.py:236 in forward, code: words_embeddings = self.word_embeddings(input_ids)
        eq_2: "b8[4, 476][476, 1]cuda:0" = torch.ops.aten.eq.Scalar(primals_4, 0)
        unsqueeze_5: "b8[4, 476, 1][476, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_2: "f32[4, 476, 768][365568, 768, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_5, full_default_1, mul_322);  unsqueeze_5 = full_default_1 = mul_322 = None
        full_default_6: "f32[21128, 768][768, 1]cuda:0" = torch.ops.aten.full.default([21128, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[21128, 768][768, 1]cuda:0" = torch.ops.aten.index_put_.default(full_default_6, [primals_4], where_2, True);  full_default_6 = primals_4 = where_2 = None
        return (None, None, index_put_2, None, index_put_1, index_put, sum_196, sum_197, convert_element_type_1204, view_599, convert_element_type_1196, view_596, convert_element_type_1188, view_593, convert_element_type_1170, view_580, sum_187, sum_188, convert_element_type_1162, view_577, convert_element_type_1152, view_574, sum_181, sum_182, convert_element_type_1144, view_571, convert_element_type_1136, view_568, convert_element_type_1128, view_565, convert_element_type_1110, view_552, sum_172, sum_173, convert_element_type_1102, view_549, convert_element_type_1092, view_546, sum_166, sum_167, convert_element_type_1084, view_543, convert_element_type_1076, view_540, convert_element_type_1068, view_537, convert_element_type_1050, view_524, sum_157, sum_158, convert_element_type_1042, view_521, convert_element_type_1032, view_518, sum_151, sum_152, convert_element_type_1024, view_515, convert_element_type_1016, view_512, convert_element_type_1008, view_509, convert_element_type_990, view_496, sum_142, sum_143, convert_element_type_982, view_493, convert_element_type_972, view_490, sum_136, sum_137, convert_element_type_964, view_487, convert_element_type_956, view_484, convert_element_type_948, view_481, convert_element_type_930, view_468, sum_127, sum_128, convert_element_type_922, view_465, convert_element_type_912, view_462, sum_121, sum_122, convert_element_type_904, view_459, convert_element_type_896, view_456, convert_element_type_888, view_453, convert_element_type_870, view_440, sum_112, sum_113, convert_element_type_862, view_437, convert_element_type_852, view_434, sum_106, sum_107, convert_element_type_844, view_431, convert_element_type_836, view_428, convert_element_type_828, view_425, convert_element_type_810, view_412, sum_97, sum_98, convert_element_type_802, view_409, convert_element_type_792, view_406, sum_91, sum_92, convert_element_type_784, view_403, convert_element_type_776, view_400, convert_element_type_768, view_397, convert_element_type_750, view_384, sum_82, sum_83, convert_element_type_742, view_381, convert_element_type_732, view_378, sum_76, sum_77, convert_element_type_724, view_375, convert_element_type_716, view_372, convert_element_type_708, view_369, convert_element_type_690, view_356, sum_67, sum_68, convert_element_type_682, view_353, convert_element_type_672, view_350, sum_61, sum_62, convert_element_type_664, view_347, convert_element_type_656, view_344, convert_element_type_648, view_341, convert_element_type_630, view_328, sum_52, sum_53, convert_element_type_622, view_325, convert_element_type_612, view_322, sum_46, sum_47, convert_element_type_604, view_319, convert_element_type_596, view_316, convert_element_type_588, view_313, convert_element_type_570, view_300, sum_37, sum_38, convert_element_type_562, view_297, convert_element_type_552, view_294, sum_31, sum_32, convert_element_type_544, view_291, convert_element_type_536, view_288, convert_element_type_528, view_285, convert_element_type_510, view_272, sum_22, sum_23, convert_element_type_502, view_269, convert_element_type_492, view_266, sum_16, sum_17, convert_element_type_484, view_264)
