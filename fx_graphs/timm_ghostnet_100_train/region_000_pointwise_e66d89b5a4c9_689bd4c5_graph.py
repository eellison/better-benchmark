class GraphModule(torch.nn.Module):
    def forward(self, primals_3: "i64[]", getitem_1: "f32[1, 16, 1, 1]", rsqrt: "f32[1, 16, 1, 1]", primals_4: "f32[16]", getitem: "f32[1, 16, 1, 1]", primals_5: "f32[16]", primals_9: "i64[]", getitem_3: "f32[1, 8, 1, 1]", rsqrt_1: "f32[1, 8, 1, 1]", primals_10: "f32[8]", getitem_2: "f32[1, 8, 1, 1]", primals_11: "f32[8]", primals_15: "i64[]", getitem_5: "f32[1, 8, 1, 1]", primals_16: "f32[8]", getitem_4: "f32[1, 8, 1, 1]", primals_17: "f32[8]", primals_21: "i64[]", getitem_7: "f32[1, 8, 1, 1]", rsqrt_3: "f32[1, 8, 1, 1]", primals_22: "f32[8]", getitem_6: "f32[1, 8, 1, 1]", primals_23: "f32[8]", primals_27: "i64[]", getitem_9: "f32[1, 8, 1, 1]", rsqrt_4: "f32[1, 8, 1, 1]", primals_28: "f32[8]", getitem_8: "f32[1, 8, 1, 1]", primals_29: "f32[8]", primals_33: "i64[]", getitem_11: "f32[1, 24, 1, 1]", rsqrt_5: "f32[1, 24, 1, 1]", primals_34: "f32[24]", getitem_10: "f32[1, 24, 1, 1]", primals_35: "f32[24]", primals_39: "i64[]", getitem_13: "f32[1, 24, 1, 1]", primals_40: "f32[24]", getitem_12: "f32[1, 24, 1, 1]", primals_41: "f32[24]", primals_45: "i64[]", getitem_15: "f32[1, 48, 1, 1]", rsqrt_7: "f32[1, 48, 1, 1]", primals_46: "f32[48]", getitem_14: "f32[1, 48, 1, 1]", primals_47: "f32[48]", primals_51: "i64[]", getitem_17: "f32[1, 12, 1, 1]", rsqrt_8: "f32[1, 12, 1, 1]", primals_52: "f32[12]", getitem_16: "f32[1, 12, 1, 1]", primals_53: "f32[12]", primals_57: "i64[]", getitem_19: "f32[1, 12, 1, 1]", rsqrt_9: "f32[1, 12, 1, 1]", primals_58: "f32[12]", getitem_18: "f32[1, 12, 1, 1]", primals_59: "f32[12]", primals_63: "i64[]", getitem_21: "f32[1, 16, 1, 1]", rsqrt_10: "f32[1, 16, 1, 1]", primals_64: "f32[16]", getitem_20: "f32[1, 16, 1, 1]", primals_65: "f32[16]", primals_69: "i64[]", getitem_23: "f32[1, 24, 1, 1]", rsqrt_11: "f32[1, 24, 1, 1]", primals_70: "f32[24]", getitem_22: "f32[1, 24, 1, 1]", primals_71: "f32[24]", primals_75: "i64[]", getitem_25: "f32[1, 36, 1, 1]", rsqrt_12: "f32[1, 36, 1, 1]", primals_76: "f32[36]", getitem_24: "f32[1, 36, 1, 1]", primals_77: "f32[36]", primals_81: "i64[]", getitem_27: "f32[1, 36, 1, 1]", primals_82: "f32[36]", getitem_26: "f32[1, 36, 1, 1]", primals_83: "f32[36]", primals_87: "i64[]", getitem_29: "f32[1, 12, 1, 1]", rsqrt_14: "f32[1, 12, 1, 1]", primals_88: "f32[12]", getitem_28: "f32[1, 12, 1, 1]", primals_89: "f32[12]", primals_93: "i64[]", getitem_31: "f32[1, 12, 1, 1]", rsqrt_15: "f32[1, 12, 1, 1]", primals_94: "f32[12]", getitem_30: "f32[1, 12, 1, 1]", primals_95: "f32[12]", primals_99: "i64[]", getitem_33: "f32[1, 36, 1, 1]", rsqrt_16: "f32[1, 36, 1, 1]", primals_100: "f32[36]", getitem_32: "f32[1, 36, 1, 1]", primals_101: "f32[36]", primals_105: "i64[]", getitem_35: "f32[1, 36, 1, 1]", primals_106: "f32[36]", getitem_34: "f32[1, 36, 1, 1]", primals_107: "f32[36]", primals_111: "i64[]", getitem_37: "f32[1, 72, 1, 1]", primals_112: "f32[72]", getitem_36: "f32[1, 72, 1, 1]", primals_113: "f32[72]", primals_121: "i64[]", getitem_39: "f32[1, 20, 1, 1]", rsqrt_19: "f32[1, 20, 1, 1]", primals_122: "f32[20]", getitem_38: "f32[1, 20, 1, 1]", primals_123: "f32[20]", primals_127: "i64[]", getitem_41: "f32[1, 20, 1, 1]", rsqrt_20: "f32[1, 20, 1, 1]", primals_128: "f32[20]", getitem_40: "f32[1, 20, 1, 1]", primals_129: "f32[20]", primals_133: "i64[]", getitem_43: "f32[1, 24, 1, 1]", rsqrt_21: "f32[1, 24, 1, 1]", primals_134: "f32[24]", getitem_42: "f32[1, 24, 1, 1]", primals_135: "f32[24]", primals_139: "i64[]", getitem_45: "f32[1, 40, 1, 1]", rsqrt_22: "f32[1, 40, 1, 1]", primals_140: "f32[40]", getitem_44: "f32[1, 40, 1, 1]", primals_141: "f32[40]", primals_145: "i64[]", getitem_47: "f32[1, 60, 1, 1]", rsqrt_23: "f32[1, 60, 1, 1]", primals_146: "f32[60]", getitem_46: "f32[1, 60, 1, 1]", primals_147: "f32[60]", primals_151: "i64[]", getitem_49: "f32[1, 60, 1, 1]", primals_152: "f32[60]", getitem_48: "f32[1, 60, 1, 1]", primals_153: "f32[60]", primals_161: "i64[]", getitem_51: "f32[1, 20, 1, 1]", rsqrt_25: "f32[1, 20, 1, 1]", primals_162: "f32[20]", getitem_50: "f32[1, 20, 1, 1]", primals_163: "f32[20]", primals_167: "i64[]", getitem_53: "f32[1, 20, 1, 1]", rsqrt_26: "f32[1, 20, 1, 1]", primals_168: "f32[20]", getitem_52: "f32[1, 20, 1, 1]", primals_169: "f32[20]", primals_173: "i64[]", getitem_55: "f32[1, 120, 1, 1]", rsqrt_27: "f32[1, 120, 1, 1]", primals_174: "f32[120]", getitem_54: "f32[1, 120, 1, 1]", primals_175: "f32[120]", primals_179: "i64[]", getitem_57: "f32[1, 120, 1, 1]", primals_180: "f32[120]", getitem_56: "f32[1, 120, 1, 1]", primals_181: "f32[120]", primals_185: "i64[]", getitem_59: "f32[1, 240, 1, 1]", rsqrt_29: "f32[1, 240, 1, 1]", primals_186: "f32[240]", getitem_58: "f32[1, 240, 1, 1]", primals_187: "f32[240]", primals_191: "i64[]", getitem_61: "f32[1, 40, 1, 1]", rsqrt_30: "f32[1, 40, 1, 1]", primals_192: "f32[40]", getitem_60: "f32[1, 40, 1, 1]", primals_193: "f32[40]", primals_197: "i64[]", getitem_63: "f32[1, 40, 1, 1]", rsqrt_31: "f32[1, 40, 1, 1]", primals_198: "f32[40]", getitem_62: "f32[1, 40, 1, 1]", primals_199: "f32[40]", primals_203: "i64[]", getitem_65: "f32[1, 40, 1, 1]", rsqrt_32: "f32[1, 40, 1, 1]", primals_204: "f32[40]", getitem_64: "f32[1, 40, 1, 1]", primals_205: "f32[40]", primals_209: "i64[]", getitem_67: "f32[1, 80, 1, 1]", rsqrt_33: "f32[1, 80, 1, 1]", primals_210: "f32[80]", getitem_66: "f32[1, 80, 1, 1]", primals_211: "f32[80]", primals_215: "i64[]", getitem_69: "f32[1, 100, 1, 1]", rsqrt_34: "f32[1, 100, 1, 1]", primals_216: "f32[100]", getitem_68: "f32[1, 100, 1, 1]", primals_217: "f32[100]", primals_221: "i64[]", getitem_71: "f32[1, 100, 1, 1]", primals_222: "f32[100]", getitem_70: "f32[1, 100, 1, 1]", primals_223: "f32[100]", primals_227: "i64[]", getitem_73: "f32[1, 40, 1, 1]", rsqrt_36: "f32[1, 40, 1, 1]", primals_228: "f32[40]", getitem_72: "f32[1, 40, 1, 1]", primals_229: "f32[40]", primals_233: "i64[]", getitem_75: "f32[1, 40, 1, 1]", rsqrt_37: "f32[1, 40, 1, 1]", primals_234: "f32[40]", getitem_74: "f32[1, 40, 1, 1]", primals_235: "f32[40]", primals_239: "i64[]", getitem_77: "f32[1, 92, 1, 1]", rsqrt_38: "f32[1, 92, 1, 1]", primals_240: "f32[92]", getitem_76: "f32[1, 92, 1, 1]", primals_241: "f32[92]", primals_245: "i64[]", getitem_79: "f32[1, 92, 1, 1]", primals_246: "f32[92]", getitem_78: "f32[1, 92, 1, 1]", primals_247: "f32[92]", primals_251: "i64[]", getitem_81: "f32[1, 40, 1, 1]", rsqrt_40: "f32[1, 40, 1, 1]", primals_252: "f32[40]", getitem_80: "f32[1, 40, 1, 1]", primals_253: "f32[40]", primals_257: "i64[]", getitem_83: "f32[1, 40, 1, 1]", rsqrt_41: "f32[1, 40, 1, 1]", primals_258: "f32[40]", getitem_82: "f32[1, 40, 1, 1]", primals_259: "f32[40]", primals_263: "i64[]", getitem_85: "f32[1, 92, 1, 1]", rsqrt_42: "f32[1, 92, 1, 1]", primals_264: "f32[92]", getitem_84: "f32[1, 92, 1, 1]", primals_265: "f32[92]", primals_269: "i64[]", getitem_87: "f32[1, 92, 1, 1]", primals_270: "f32[92]", getitem_86: "f32[1, 92, 1, 1]", primals_271: "f32[92]", primals_275: "i64[]", getitem_89: "f32[1, 40, 1, 1]", rsqrt_44: "f32[1, 40, 1, 1]", primals_276: "f32[40]", getitem_88: "f32[1, 40, 1, 1]", primals_277: "f32[40]", primals_281: "i64[]", getitem_91: "f32[1, 40, 1, 1]", rsqrt_45: "f32[1, 40, 1, 1]", primals_282: "f32[40]", getitem_90: "f32[1, 40, 1, 1]", primals_283: "f32[40]", primals_287: "i64[]", getitem_93: "f32[1, 240, 1, 1]", rsqrt_46: "f32[1, 240, 1, 1]", primals_288: "f32[240]", getitem_92: "f32[1, 240, 1, 1]", primals_289: "f32[240]", primals_293: "i64[]", getitem_95: "f32[1, 240, 1, 1]", primals_294: "f32[240]", getitem_94: "f32[1, 240, 1, 1]", primals_295: "f32[240]", primals_303: "i64[]", getitem_97: "f32[1, 56, 1, 1]", rsqrt_48: "f32[1, 56, 1, 1]", primals_304: "f32[56]", getitem_96: "f32[1, 56, 1, 1]", primals_305: "f32[56]", primals_309: "i64[]", getitem_99: "f32[1, 56, 1, 1]", rsqrt_49: "f32[1, 56, 1, 1]", primals_310: "f32[56]", getitem_98: "f32[1, 56, 1, 1]", primals_311: "f32[56]", primals_315: "i64[]", getitem_101: "f32[1, 80, 1, 1]", rsqrt_50: "f32[1, 80, 1, 1]", primals_316: "f32[80]", getitem_100: "f32[1, 80, 1, 1]", primals_317: "f32[80]", primals_321: "i64[]", getitem_103: "f32[1, 112, 1, 1]", rsqrt_51: "f32[1, 112, 1, 1]", primals_322: "f32[112]", getitem_102: "f32[1, 112, 1, 1]", primals_323: "f32[112]", primals_327: "i64[]", getitem_105: "f32[1, 336, 1, 1]", rsqrt_52: "f32[1, 336, 1, 1]", primals_328: "f32[336]", getitem_104: "f32[1, 336, 1, 1]", primals_329: "f32[336]", primals_333: "i64[]", getitem_107: "f32[1, 336, 1, 1]", primals_334: "f32[336]", getitem_106: "f32[1, 336, 1, 1]", primals_335: "f32[336]", primals_343: "i64[]", getitem_109: "f32[1, 56, 1, 1]", rsqrt_54: "f32[1, 56, 1, 1]", primals_344: "f32[56]", getitem_108: "f32[1, 56, 1, 1]", primals_345: "f32[56]", primals_349: "i64[]", getitem_111: "f32[1, 56, 1, 1]", rsqrt_55: "f32[1, 56, 1, 1]", primals_350: "f32[56]", getitem_110: "f32[1, 56, 1, 1]", primals_351: "f32[56]", primals_355: "i64[]", getitem_113: "f32[1, 336, 1, 1]", rsqrt_56: "f32[1, 336, 1, 1]", primals_356: "f32[336]", getitem_112: "f32[1, 336, 1, 1]", primals_357: "f32[336]", primals_361: "i64[]", getitem_115: "f32[1, 336, 1, 1]", primals_362: "f32[336]", getitem_114: "f32[1, 336, 1, 1]", primals_363: "f32[336]", primals_367: "i64[]", getitem_117: "f32[1, 672, 1, 1]", primals_368: "f32[672]", getitem_116: "f32[1, 672, 1, 1]", primals_369: "f32[672]", primals_377: "i64[]", getitem_119: "f32[1, 80, 1, 1]", rsqrt_59: "f32[1, 80, 1, 1]", primals_378: "f32[80]", getitem_118: "f32[1, 80, 1, 1]", primals_379: "f32[80]", primals_383: "i64[]", getitem_121: "f32[1, 80, 1, 1]", rsqrt_60: "f32[1, 80, 1, 1]", primals_384: "f32[80]", getitem_120: "f32[1, 80, 1, 1]", primals_385: "f32[80]", primals_389: "i64[]", getitem_123: "f32[1, 112, 1, 1]", rsqrt_61: "f32[1, 112, 1, 1]", primals_390: "f32[112]", getitem_122: "f32[1, 112, 1, 1]", primals_391: "f32[112]", primals_395: "i64[]", getitem_125: "f32[1, 160, 1, 1]", rsqrt_62: "f32[1, 160, 1, 1]", primals_396: "f32[160]", getitem_124: "f32[1, 160, 1, 1]", primals_397: "f32[160]", primals_401: "i64[]", getitem_127: "f32[1, 480, 1, 1]", rsqrt_63: "f32[1, 480, 1, 1]", primals_402: "f32[480]", getitem_126: "f32[1, 480, 1, 1]", primals_403: "f32[480]", primals_407: "i64[]", getitem_129: "f32[1, 480, 1, 1]", primals_408: "f32[480]", getitem_128: "f32[1, 480, 1, 1]", primals_409: "f32[480]", primals_413: "i64[]", getitem_131: "f32[1, 80, 1, 1]", rsqrt_65: "f32[1, 80, 1, 1]", primals_414: "f32[80]", getitem_130: "f32[1, 80, 1, 1]", primals_415: "f32[80]", primals_419: "i64[]", getitem_133: "f32[1, 80, 1, 1]", rsqrt_66: "f32[1, 80, 1, 1]", primals_420: "f32[80]", getitem_132: "f32[1, 80, 1, 1]", primals_421: "f32[80]", primals_425: "i64[]", getitem_135: "f32[1, 480, 1, 1]", rsqrt_67: "f32[1, 480, 1, 1]", primals_426: "f32[480]", getitem_134: "f32[1, 480, 1, 1]", primals_427: "f32[480]", primals_431: "i64[]", getitem_137: "f32[1, 480, 1, 1]", primals_432: "f32[480]", getitem_136: "f32[1, 480, 1, 1]", primals_433: "f32[480]", primals_441: "i64[]", getitem_139: "f32[1, 80, 1, 1]", rsqrt_69: "f32[1, 80, 1, 1]", primals_442: "f32[80]", getitem_138: "f32[1, 80, 1, 1]", primals_443: "f32[80]", primals_447: "i64[]", getitem_141: "f32[1, 80, 1, 1]", rsqrt_70: "f32[1, 80, 1, 1]", primals_448: "f32[80]", getitem_140: "f32[1, 80, 1, 1]", primals_449: "f32[80]", primals_453: "i64[]", getitem_143: "f32[1, 480, 1, 1]", rsqrt_71: "f32[1, 480, 1, 1]", primals_454: "f32[480]", getitem_142: "f32[1, 480, 1, 1]", primals_455: "f32[480]", primals_459: "i64[]", getitem_145: "f32[1, 480, 1, 1]", primals_460: "f32[480]", getitem_144: "f32[1, 480, 1, 1]", primals_461: "f32[480]", primals_465: "i64[]", getitem_147: "f32[1, 80, 1, 1]", rsqrt_73: "f32[1, 80, 1, 1]", primals_466: "f32[80]", getitem_146: "f32[1, 80, 1, 1]", primals_467: "f32[80]", primals_471: "i64[]", getitem_149: "f32[1, 80, 1, 1]", rsqrt_74: "f32[1, 80, 1, 1]", primals_472: "f32[80]", getitem_148: "f32[1, 80, 1, 1]", primals_473: "f32[80]", primals_477: "i64[]", getitem_151: "f32[1, 480, 1, 1]", rsqrt_75: "f32[1, 480, 1, 1]", primals_478: "f32[480]", getitem_150: "f32[1, 480, 1, 1]", primals_479: "f32[480]", primals_483: "i64[]", getitem_153: "f32[1, 480, 1, 1]", primals_484: "f32[480]", getitem_152: "f32[1, 480, 1, 1]", primals_485: "f32[480]", primals_493: "i64[]", getitem_155: "f32[1, 80, 1, 1]", rsqrt_77: "f32[1, 80, 1, 1]", primals_494: "f32[80]", getitem_154: "f32[1, 80, 1, 1]", primals_495: "f32[80]", primals_499: "i64[]", getitem_157: "f32[1, 80, 1, 1]", rsqrt_78: "f32[1, 80, 1, 1]", primals_500: "f32[80]", getitem_156: "f32[1, 80, 1, 1]", primals_501: "f32[80]", primals_505: "i64[]", getitem_159: "f32[1, 960, 1, 1]", primals_506: "f32[960]", getitem_158: "f32[1, 960, 1, 1]", primals_507: "f32[960]", convolution_94: "f32[512, 1280, 1, 1]", primals_512: "f32[1000, 1280]", _shape_param_0):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:822 in forward_features, code: x = self.bn1(x)
        add_tensor: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)
        squeeze_dims: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_dims_1: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_tensor: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims, 0.1)
        mul_tensor_1: "f32[16]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_tensor_1: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1);  mul_tensor = mul_tensor_1 = None
        squeeze_dims_2: "f32[16]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_tensor_2: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_2, 1.0000001557019536);  squeeze_dims_2 = None
        mul_tensor_3: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_2, 0.1);  mul_tensor_2 = None
        mul_tensor_4: "f32[16]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_tensor_2: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4);  mul_tensor_3 = mul_tensor_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_3: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)
        squeeze_dims_3: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_dims_4: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_tensor_5: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_3, 0.1)
        mul_tensor_6: "f32[8]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_tensor_4: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_6);  mul_tensor_5 = mul_tensor_6 = None
        squeeze_dims_5: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_tensor_7: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_5, 1.0000001557019536);  squeeze_dims_5 = None
        mul_tensor_8: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.1);  mul_tensor_7 = None
        mul_tensor_9: "f32[8]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_tensor_5: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_9);  mul_tensor_8 = mul_tensor_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_6: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)
        squeeze_dims_6: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        mul_tensor_10: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_6, 0.1);  squeeze_dims_6 = None
        mul_tensor_11: "f32[8]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_tensor_7: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_11);  mul_tensor_10 = mul_tensor_11 = None
        squeeze_dims_7: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_tensor_12: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_7, 1.0000001557019536);  squeeze_dims_7 = None
        mul_tensor_13: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.1);  mul_tensor_12 = None
        mul_tensor_14: "f32[8]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_tensor_8: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_14);  mul_tensor_13 = mul_tensor_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_9: "i64[]" = torch.ops.aten.add.Tensor(primals_21, 1)
        squeeze_dims_8: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_dims_9: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_tensor_15: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_8, 0.1)
        mul_tensor_16: "f32[8]" = torch.ops.aten.mul.Tensor(primals_22, 0.9)
        add_tensor_10: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_15, mul_tensor_16);  mul_tensor_15 = mul_tensor_16 = None
        squeeze_dims_10: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_tensor_17: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_10, 1.0000001557019536);  squeeze_dims_10 = None
        mul_tensor_18: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_17, 0.1);  mul_tensor_17 = None
        mul_tensor_19: "f32[8]" = torch.ops.aten.mul.Tensor(primals_23, 0.9)
        add_tensor_11: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_18, mul_tensor_19);  mul_tensor_18 = mul_tensor_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_12: "i64[]" = torch.ops.aten.add.Tensor(primals_27, 1)
        squeeze_dims_11: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_dims_12: "f32[8]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_tensor_20: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_11, 0.1)
        mul_tensor_21: "f32[8]" = torch.ops.aten.mul.Tensor(primals_28, 0.9)
        add_tensor_13: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_20, mul_tensor_21);  mul_tensor_20 = mul_tensor_21 = None
        squeeze_dims_13: "f32[8]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_tensor_22: "f32[8]" = torch.ops.aten.mul.Tensor(squeeze_dims_13, 1.0000001557019536);  squeeze_dims_13 = None
        mul_tensor_23: "f32[8]" = torch.ops.aten.mul.Tensor(mul_tensor_22, 0.1);  mul_tensor_22 = None
        mul_tensor_24: "f32[8]" = torch.ops.aten.mul.Tensor(primals_29, 0.9)
        add_tensor_14: "f32[8]" = torch.ops.aten.add.Tensor(mul_tensor_23, mul_tensor_24);  mul_tensor_23 = mul_tensor_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_15: "i64[]" = torch.ops.aten.add.Tensor(primals_33, 1)
        squeeze_dims_14: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_dims_15: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_tensor_25: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_14, 0.1)
        mul_tensor_26: "f32[24]" = torch.ops.aten.mul.Tensor(primals_34, 0.9)
        add_tensor_16: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_25, mul_tensor_26);  mul_tensor_25 = mul_tensor_26 = None
        squeeze_dims_16: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_tensor_27: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_16, 1.0000001557019536);  squeeze_dims_16 = None
        mul_tensor_28: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_27, 0.1);  mul_tensor_27 = None
        mul_tensor_29: "f32[24]" = torch.ops.aten.mul.Tensor(primals_35, 0.9)
        add_tensor_17: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_28, mul_tensor_29);  mul_tensor_28 = mul_tensor_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_18: "i64[]" = torch.ops.aten.add.Tensor(primals_39, 1)
        squeeze_dims_17: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        mul_tensor_30: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_17, 0.1);  squeeze_dims_17 = None
        mul_tensor_31: "f32[24]" = torch.ops.aten.mul.Tensor(primals_40, 0.9)
        add_tensor_19: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_30, mul_tensor_31);  mul_tensor_30 = mul_tensor_31 = None
        squeeze_dims_18: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_tensor_32: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_18, 1.0000001557019536);  squeeze_dims_18 = None
        mul_tensor_33: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_32, 0.1);  mul_tensor_32 = None
        mul_tensor_34: "f32[24]" = torch.ops.aten.mul.Tensor(primals_41, 0.9)
        add_tensor_20: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_33, mul_tensor_34);  mul_tensor_33 = mul_tensor_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        add_tensor_21: "i64[]" = torch.ops.aten.add.Tensor(primals_45, 1)
        squeeze_dims_19: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_dims_20: "f32[48]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_tensor_35: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_dims_19, 0.1)
        mul_tensor_36: "f32[48]" = torch.ops.aten.mul.Tensor(primals_46, 0.9)
        add_tensor_22: "f32[48]" = torch.ops.aten.add.Tensor(mul_tensor_35, mul_tensor_36);  mul_tensor_35 = mul_tensor_36 = None
        squeeze_dims_21: "f32[48]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_tensor_37: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_dims_21, 1.0000006228081046);  squeeze_dims_21 = None
        mul_tensor_38: "f32[48]" = torch.ops.aten.mul.Tensor(mul_tensor_37, 0.1);  mul_tensor_37 = None
        mul_tensor_39: "f32[48]" = torch.ops.aten.mul.Tensor(primals_47, 0.9)
        add_tensor_23: "f32[48]" = torch.ops.aten.add.Tensor(mul_tensor_38, mul_tensor_39);  mul_tensor_38 = mul_tensor_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_24: "i64[]" = torch.ops.aten.add.Tensor(primals_51, 1)
        squeeze_dims_22: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_dims_23: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_tensor_40: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_22, 0.1)
        mul_tensor_41: "f32[12]" = torch.ops.aten.mul.Tensor(primals_52, 0.9)
        add_tensor_25: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_40, mul_tensor_41);  mul_tensor_40 = mul_tensor_41 = None
        squeeze_dims_24: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_tensor_42: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_24, 1.0000006228081046);  squeeze_dims_24 = None
        mul_tensor_43: "f32[12]" = torch.ops.aten.mul.Tensor(mul_tensor_42, 0.1);  mul_tensor_42 = None
        mul_tensor_44: "f32[12]" = torch.ops.aten.mul.Tensor(primals_53, 0.9)
        add_tensor_26: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_43, mul_tensor_44);  mul_tensor_43 = mul_tensor_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_27: "i64[]" = torch.ops.aten.add.Tensor(primals_57, 1)
        squeeze_dims_25: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        squeeze_dims_26: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_tensor_45: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_25, 0.1)
        mul_tensor_46: "f32[12]" = torch.ops.aten.mul.Tensor(primals_58, 0.9)
        add_tensor_28: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_45, mul_tensor_46);  mul_tensor_45 = mul_tensor_46 = None
        squeeze_dims_27: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_tensor_47: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_27, 1.0000006228081046);  squeeze_dims_27 = None
        mul_tensor_48: "f32[12]" = torch.ops.aten.mul.Tensor(mul_tensor_47, 0.1);  mul_tensor_47 = None
        mul_tensor_49: "f32[12]" = torch.ops.aten.mul.Tensor(primals_59, 0.9)
        add_tensor_29: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_48, mul_tensor_49);  mul_tensor_48 = mul_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        add_tensor_30: "i64[]" = torch.ops.aten.add.Tensor(primals_63, 1)
        squeeze_dims_28: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_dims_29: "f32[16]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_tensor_50: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_28, 0.1)
        mul_tensor_51: "f32[16]" = torch.ops.aten.mul.Tensor(primals_64, 0.9)
        add_tensor_31: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_50, mul_tensor_51);  mul_tensor_50 = mul_tensor_51 = None
        squeeze_dims_30: "f32[16]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_tensor_52: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_dims_30, 1.0000006228081046);  squeeze_dims_30 = None
        mul_tensor_53: "f32[16]" = torch.ops.aten.mul.Tensor(mul_tensor_52, 0.1);  mul_tensor_52 = None
        mul_tensor_54: "f32[16]" = torch.ops.aten.mul.Tensor(primals_65, 0.9)
        add_tensor_32: "f32[16]" = torch.ops.aten.add.Tensor(mul_tensor_53, mul_tensor_54);  mul_tensor_53 = mul_tensor_54 = None
        add_tensor_33: "i64[]" = torch.ops.aten.add.Tensor(primals_69, 1)
        squeeze_dims_31: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_dims_32: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_tensor_55: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_31, 0.1)
        mul_tensor_56: "f32[24]" = torch.ops.aten.mul.Tensor(primals_70, 0.9)
        add_tensor_34: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_55, mul_tensor_56);  mul_tensor_55 = mul_tensor_56 = None
        squeeze_dims_33: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_tensor_57: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_33, 1.0000006228081046);  squeeze_dims_33 = None
        mul_tensor_58: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_57, 0.1);  mul_tensor_57 = None
        mul_tensor_59: "f32[24]" = torch.ops.aten.mul.Tensor(primals_71, 0.9)
        add_tensor_35: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_58, mul_tensor_59);  mul_tensor_58 = mul_tensor_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_36: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)
        squeeze_dims_34: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_dims_35: "f32[36]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_tensor_60: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_34, 0.1)
        mul_tensor_61: "f32[36]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_tensor_37: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_60, mul_tensor_61);  mul_tensor_60 = mul_tensor_61 = None
        squeeze_dims_36: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_tensor_62: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_36, 1.0000006228081046);  squeeze_dims_36 = None
        mul_tensor_63: "f32[36]" = torch.ops.aten.mul.Tensor(mul_tensor_62, 0.1);  mul_tensor_62 = None
        mul_tensor_64: "f32[36]" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_tensor_38: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_63, mul_tensor_64);  mul_tensor_63 = mul_tensor_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_39: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)
        squeeze_dims_37: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        mul_tensor_65: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_37, 0.1);  squeeze_dims_37 = None
        mul_tensor_66: "f32[36]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_tensor_40: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_65, mul_tensor_66);  mul_tensor_65 = mul_tensor_66 = None
        squeeze_dims_38: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_tensor_67: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_38, 1.0000006228081046);  squeeze_dims_38 = None
        mul_tensor_68: "f32[36]" = torch.ops.aten.mul.Tensor(mul_tensor_67, 0.1);  mul_tensor_67 = None
        mul_tensor_69: "f32[36]" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_tensor_41: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_68, mul_tensor_69);  mul_tensor_68 = mul_tensor_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_42: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)
        squeeze_dims_39: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_dims_40: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_tensor_70: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_39, 0.1)
        mul_tensor_71: "f32[12]" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_tensor_43: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_70, mul_tensor_71);  mul_tensor_70 = mul_tensor_71 = None
        squeeze_dims_41: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_tensor_72: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_41, 1.0000006228081046);  squeeze_dims_41 = None
        mul_tensor_73: "f32[12]" = torch.ops.aten.mul.Tensor(mul_tensor_72, 0.1);  mul_tensor_72 = None
        mul_tensor_74: "f32[12]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_tensor_44: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_73, mul_tensor_74);  mul_tensor_73 = mul_tensor_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_45: "i64[]" = torch.ops.aten.add.Tensor(primals_93, 1)
        squeeze_dims_42: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_dims_43: "f32[12]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_tensor_75: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_42, 0.1)
        mul_tensor_76: "f32[12]" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_tensor_46: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_75, mul_tensor_76);  mul_tensor_75 = mul_tensor_76 = None
        squeeze_dims_44: "f32[12]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_tensor_77: "f32[12]" = torch.ops.aten.mul.Tensor(squeeze_dims_44, 1.0000006228081046);  squeeze_dims_44 = None
        mul_tensor_78: "f32[12]" = torch.ops.aten.mul.Tensor(mul_tensor_77, 0.1);  mul_tensor_77 = None
        mul_tensor_79: "f32[12]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_tensor_47: "f32[12]" = torch.ops.aten.add.Tensor(mul_tensor_78, mul_tensor_79);  mul_tensor_78 = mul_tensor_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_48: "i64[]" = torch.ops.aten.add.Tensor(primals_99, 1)
        squeeze_dims_45: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        squeeze_dims_46: "f32[36]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_tensor_80: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_45, 0.1)
        mul_tensor_81: "f32[36]" = torch.ops.aten.mul.Tensor(primals_100, 0.9)
        add_tensor_49: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_80, mul_tensor_81);  mul_tensor_80 = mul_tensor_81 = None
        squeeze_dims_47: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_tensor_82: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_47, 1.0000006228081046);  squeeze_dims_47 = None
        mul_tensor_83: "f32[36]" = torch.ops.aten.mul.Tensor(mul_tensor_82, 0.1);  mul_tensor_82 = None
        mul_tensor_84: "f32[36]" = torch.ops.aten.mul.Tensor(primals_101, 0.9)
        add_tensor_50: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_83, mul_tensor_84);  mul_tensor_83 = mul_tensor_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_51: "i64[]" = torch.ops.aten.add.Tensor(primals_105, 1)
        squeeze_dims_48: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        mul_tensor_85: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_48, 0.1);  squeeze_dims_48 = None
        mul_tensor_86: "f32[36]" = torch.ops.aten.mul.Tensor(primals_106, 0.9)
        add_tensor_52: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_85, mul_tensor_86);  mul_tensor_85 = mul_tensor_86 = None
        squeeze_dims_49: "f32[36]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_tensor_87: "f32[36]" = torch.ops.aten.mul.Tensor(squeeze_dims_49, 1.0000006228081046);  squeeze_dims_49 = None
        mul_tensor_88: "f32[36]" = torch.ops.aten.mul.Tensor(mul_tensor_87, 0.1);  mul_tensor_87 = None
        mul_tensor_89: "f32[36]" = torch.ops.aten.mul.Tensor(primals_107, 0.9)
        add_tensor_53: "f32[36]" = torch.ops.aten.add.Tensor(mul_tensor_88, mul_tensor_89);  mul_tensor_88 = mul_tensor_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        add_tensor_54: "i64[]" = torch.ops.aten.add.Tensor(primals_111, 1)
        squeeze_dims_50: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        mul_tensor_90: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_50, 0.1);  squeeze_dims_50 = None
        mul_tensor_91: "f32[72]" = torch.ops.aten.mul.Tensor(primals_112, 0.9)
        add_tensor_55: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_90, mul_tensor_91);  mul_tensor_90 = mul_tensor_91 = None
        squeeze_dims_51: "f32[72]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_tensor_92: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_dims_51, 1.0000024912370735);  squeeze_dims_51 = None
        mul_tensor_93: "f32[72]" = torch.ops.aten.mul.Tensor(mul_tensor_92, 0.1);  mul_tensor_92 = None
        mul_tensor_94: "f32[72]" = torch.ops.aten.mul.Tensor(primals_113, 0.9)
        add_tensor_56: "f32[72]" = torch.ops.aten.add.Tensor(mul_tensor_93, mul_tensor_94);  mul_tensor_93 = mul_tensor_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_57: "i64[]" = torch.ops.aten.add.Tensor(primals_121, 1)
        squeeze_dims_52: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        squeeze_dims_53: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_tensor_95: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_52, 0.1)
        mul_tensor_96: "f32[20]" = torch.ops.aten.mul.Tensor(primals_122, 0.9)
        add_tensor_58: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_95, mul_tensor_96);  mul_tensor_95 = mul_tensor_96 = None
        squeeze_dims_54: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_tensor_97: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_54, 1.0000024912370735);  squeeze_dims_54 = None
        mul_tensor_98: "f32[20]" = torch.ops.aten.mul.Tensor(mul_tensor_97, 0.1);  mul_tensor_97 = None
        mul_tensor_99: "f32[20]" = torch.ops.aten.mul.Tensor(primals_123, 0.9)
        add_tensor_59: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_98, mul_tensor_99);  mul_tensor_98 = mul_tensor_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_60: "i64[]" = torch.ops.aten.add.Tensor(primals_127, 1)
        squeeze_dims_55: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_dims_56: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_tensor_100: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_55, 0.1)
        mul_tensor_101: "f32[20]" = torch.ops.aten.mul.Tensor(primals_128, 0.9)
        add_tensor_61: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_100, mul_tensor_101);  mul_tensor_100 = mul_tensor_101 = None
        squeeze_dims_57: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_tensor_102: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_57, 1.0000024912370735);  squeeze_dims_57 = None
        mul_tensor_103: "f32[20]" = torch.ops.aten.mul.Tensor(mul_tensor_102, 0.1);  mul_tensor_102 = None
        mul_tensor_104: "f32[20]" = torch.ops.aten.mul.Tensor(primals_129, 0.9)
        add_tensor_62: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_103, mul_tensor_104);  mul_tensor_103 = mul_tensor_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        add_tensor_63: "i64[]" = torch.ops.aten.add.Tensor(primals_133, 1)
        squeeze_dims_58: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_dims_59: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_tensor_105: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_58, 0.1)
        mul_tensor_106: "f32[24]" = torch.ops.aten.mul.Tensor(primals_134, 0.9)
        add_tensor_64: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_105, mul_tensor_106);  mul_tensor_105 = mul_tensor_106 = None
        squeeze_dims_60: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_tensor_107: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_dims_60, 1.0000024912370735);  squeeze_dims_60 = None
        mul_tensor_108: "f32[24]" = torch.ops.aten.mul.Tensor(mul_tensor_107, 0.1);  mul_tensor_107 = None
        mul_tensor_109: "f32[24]" = torch.ops.aten.mul.Tensor(primals_135, 0.9)
        add_tensor_65: "f32[24]" = torch.ops.aten.add.Tensor(mul_tensor_108, mul_tensor_109);  mul_tensor_108 = mul_tensor_109 = None
        add_tensor_66: "i64[]" = torch.ops.aten.add.Tensor(primals_139, 1)
        squeeze_dims_61: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        squeeze_dims_62: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_tensor_110: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_61, 0.1)
        mul_tensor_111: "f32[40]" = torch.ops.aten.mul.Tensor(primals_140, 0.9)
        add_tensor_67: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_110, mul_tensor_111);  mul_tensor_110 = mul_tensor_111 = None
        squeeze_dims_63: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_tensor_112: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_63, 1.0000024912370735);  squeeze_dims_63 = None
        mul_tensor_113: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_112, 0.1);  mul_tensor_112 = None
        mul_tensor_114: "f32[40]" = torch.ops.aten.mul.Tensor(primals_141, 0.9)
        add_tensor_68: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_113, mul_tensor_114);  mul_tensor_113 = mul_tensor_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_69: "i64[]" = torch.ops.aten.add.Tensor(primals_145, 1)
        squeeze_dims_64: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_dims_65: "f32[60]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_tensor_115: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_dims_64, 0.1)
        mul_tensor_116: "f32[60]" = torch.ops.aten.mul.Tensor(primals_146, 0.9)
        add_tensor_70: "f32[60]" = torch.ops.aten.add.Tensor(mul_tensor_115, mul_tensor_116);  mul_tensor_115 = mul_tensor_116 = None
        squeeze_dims_66: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_tensor_117: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_dims_66, 1.0000024912370735);  squeeze_dims_66 = None
        mul_tensor_118: "f32[60]" = torch.ops.aten.mul.Tensor(mul_tensor_117, 0.1);  mul_tensor_117 = None
        mul_tensor_119: "f32[60]" = torch.ops.aten.mul.Tensor(primals_147, 0.9)
        add_tensor_71: "f32[60]" = torch.ops.aten.add.Tensor(mul_tensor_118, mul_tensor_119);  mul_tensor_118 = mul_tensor_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_72: "i64[]" = torch.ops.aten.add.Tensor(primals_151, 1)
        squeeze_dims_67: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        mul_tensor_120: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_dims_67, 0.1);  squeeze_dims_67 = None
        mul_tensor_121: "f32[60]" = torch.ops.aten.mul.Tensor(primals_152, 0.9)
        add_tensor_73: "f32[60]" = torch.ops.aten.add.Tensor(mul_tensor_120, mul_tensor_121);  mul_tensor_120 = mul_tensor_121 = None
        squeeze_dims_68: "f32[60]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_tensor_122: "f32[60]" = torch.ops.aten.mul.Tensor(squeeze_dims_68, 1.0000024912370735);  squeeze_dims_68 = None
        mul_tensor_123: "f32[60]" = torch.ops.aten.mul.Tensor(mul_tensor_122, 0.1);  mul_tensor_122 = None
        mul_tensor_124: "f32[60]" = torch.ops.aten.mul.Tensor(primals_153, 0.9)
        add_tensor_74: "f32[60]" = torch.ops.aten.add.Tensor(mul_tensor_123, mul_tensor_124);  mul_tensor_123 = mul_tensor_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_75: "i64[]" = torch.ops.aten.add.Tensor(primals_161, 1)
        squeeze_dims_69: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        squeeze_dims_70: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_tensor_125: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_69, 0.1)
        mul_tensor_126: "f32[20]" = torch.ops.aten.mul.Tensor(primals_162, 0.9)
        add_tensor_76: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_125, mul_tensor_126);  mul_tensor_125 = mul_tensor_126 = None
        squeeze_dims_71: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_tensor_127: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_71, 1.0000024912370735);  squeeze_dims_71 = None
        mul_tensor_128: "f32[20]" = torch.ops.aten.mul.Tensor(mul_tensor_127, 0.1);  mul_tensor_127 = None
        mul_tensor_129: "f32[20]" = torch.ops.aten.mul.Tensor(primals_163, 0.9)
        add_tensor_77: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_128, mul_tensor_129);  mul_tensor_128 = mul_tensor_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_78: "i64[]" = torch.ops.aten.add.Tensor(primals_167, 1)
        squeeze_dims_72: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_dims_73: "f32[20]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_tensor_130: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_72, 0.1)
        mul_tensor_131: "f32[20]" = torch.ops.aten.mul.Tensor(primals_168, 0.9)
        add_tensor_79: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_130, mul_tensor_131);  mul_tensor_130 = mul_tensor_131 = None
        squeeze_dims_74: "f32[20]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_tensor_132: "f32[20]" = torch.ops.aten.mul.Tensor(squeeze_dims_74, 1.0000024912370735);  squeeze_dims_74 = None
        mul_tensor_133: "f32[20]" = torch.ops.aten.mul.Tensor(mul_tensor_132, 0.1);  mul_tensor_132 = None
        mul_tensor_134: "f32[20]" = torch.ops.aten.mul.Tensor(primals_169, 0.9)
        add_tensor_80: "f32[20]" = torch.ops.aten.add.Tensor(mul_tensor_133, mul_tensor_134);  mul_tensor_133 = mul_tensor_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_81: "i64[]" = torch.ops.aten.add.Tensor(primals_173, 1)
        squeeze_dims_75: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_dims_76: "f32[120]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_tensor_135: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_75, 0.1)
        mul_tensor_136: "f32[120]" = torch.ops.aten.mul.Tensor(primals_174, 0.9)
        add_tensor_82: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_135, mul_tensor_136);  mul_tensor_135 = mul_tensor_136 = None
        squeeze_dims_77: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_tensor_137: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_77, 1.0000024912370735);  squeeze_dims_77 = None
        mul_tensor_138: "f32[120]" = torch.ops.aten.mul.Tensor(mul_tensor_137, 0.1);  mul_tensor_137 = None
        mul_tensor_139: "f32[120]" = torch.ops.aten.mul.Tensor(primals_175, 0.9)
        add_tensor_83: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_138, mul_tensor_139);  mul_tensor_138 = mul_tensor_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_84: "i64[]" = torch.ops.aten.add.Tensor(primals_179, 1)
        squeeze_dims_78: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        mul_tensor_140: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_78, 0.1);  squeeze_dims_78 = None
        mul_tensor_141: "f32[120]" = torch.ops.aten.mul.Tensor(primals_180, 0.9)
        add_tensor_85: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_140, mul_tensor_141);  mul_tensor_140 = mul_tensor_141 = None
        squeeze_dims_79: "f32[120]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_tensor_142: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_dims_79, 1.0000024912370735);  squeeze_dims_79 = None
        mul_tensor_143: "f32[120]" = torch.ops.aten.mul.Tensor(mul_tensor_142, 0.1);  mul_tensor_142 = None
        mul_tensor_144: "f32[120]" = torch.ops.aten.mul.Tensor(primals_181, 0.9)
        add_tensor_86: "f32[120]" = torch.ops.aten.add.Tensor(mul_tensor_143, mul_tensor_144);  mul_tensor_143 = mul_tensor_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        add_tensor_87: "i64[]" = torch.ops.aten.add.Tensor(primals_185, 1)
        squeeze_dims_80: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        squeeze_dims_81: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_29, [0, 2, 3]);  rsqrt_29 = None
        mul_tensor_145: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_80, 0.1)
        mul_tensor_146: "f32[240]" = torch.ops.aten.mul.Tensor(primals_186, 0.9)
        add_tensor_88: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_145, mul_tensor_146);  mul_tensor_145 = mul_tensor_146 = None
        squeeze_dims_82: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_tensor_147: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_82, 1.00000996502277);  squeeze_dims_82 = None
        mul_tensor_148: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_147, 0.1);  mul_tensor_147 = None
        mul_tensor_149: "f32[240]" = torch.ops.aten.mul.Tensor(primals_187, 0.9)
        add_tensor_89: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_148, mul_tensor_149);  mul_tensor_148 = mul_tensor_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_90: "i64[]" = torch.ops.aten.add.Tensor(primals_191, 1)
        squeeze_dims_83: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_dims_84: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_tensor_150: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_83, 0.1)
        mul_tensor_151: "f32[40]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
        add_tensor_91: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_150, mul_tensor_151);  mul_tensor_150 = mul_tensor_151 = None
        squeeze_dims_85: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_tensor_152: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_85, 1.00000996502277);  squeeze_dims_85 = None
        mul_tensor_153: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_152, 0.1);  mul_tensor_152 = None
        mul_tensor_154: "f32[40]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
        add_tensor_92: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_153, mul_tensor_154);  mul_tensor_153 = mul_tensor_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_93: "i64[]" = torch.ops.aten.add.Tensor(primals_197, 1)
        squeeze_dims_86: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_dims_87: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_tensor_155: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_86, 0.1)
        mul_tensor_156: "f32[40]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
        add_tensor_94: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_155, mul_tensor_156);  mul_tensor_155 = mul_tensor_156 = None
        squeeze_dims_88: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_tensor_157: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_88, 1.00000996502277);  squeeze_dims_88 = None
        mul_tensor_158: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_157, 0.1);  mul_tensor_157 = None
        mul_tensor_159: "f32[40]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
        add_tensor_95: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_158, mul_tensor_159);  mul_tensor_158 = mul_tensor_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        add_tensor_96: "i64[]" = torch.ops.aten.add.Tensor(primals_203, 1)
        squeeze_dims_89: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        squeeze_dims_90: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_32, [0, 2, 3]);  rsqrt_32 = None
        mul_tensor_160: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_89, 0.1)
        mul_tensor_161: "f32[40]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
        add_tensor_97: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_160, mul_tensor_161);  mul_tensor_160 = mul_tensor_161 = None
        squeeze_dims_91: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_tensor_162: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_91, 1.00000996502277);  squeeze_dims_91 = None
        mul_tensor_163: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_162, 0.1);  mul_tensor_162 = None
        mul_tensor_164: "f32[40]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
        add_tensor_98: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_163, mul_tensor_164);  mul_tensor_163 = mul_tensor_164 = None
        add_tensor_99: "i64[]" = torch.ops.aten.add.Tensor(primals_209, 1)
        squeeze_dims_92: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_dims_93: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_tensor_165: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_92, 0.1)
        mul_tensor_166: "f32[80]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
        add_tensor_100: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_165, mul_tensor_166);  mul_tensor_165 = mul_tensor_166 = None
        squeeze_dims_94: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_tensor_167: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_94, 1.00000996502277);  squeeze_dims_94 = None
        mul_tensor_168: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_167, 0.1);  mul_tensor_167 = None
        mul_tensor_169: "f32[80]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
        add_tensor_101: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_168, mul_tensor_169);  mul_tensor_168 = mul_tensor_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_102: "i64[]" = torch.ops.aten.add.Tensor(primals_215, 1)
        squeeze_dims_95: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_dims_96: "f32[100]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_tensor_170: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_dims_95, 0.1)
        mul_tensor_171: "f32[100]" = torch.ops.aten.mul.Tensor(primals_216, 0.9)
        add_tensor_103: "f32[100]" = torch.ops.aten.add.Tensor(mul_tensor_170, mul_tensor_171);  mul_tensor_170 = mul_tensor_171 = None
        squeeze_dims_97: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_tensor_172: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_dims_97, 1.00000996502277);  squeeze_dims_97 = None
        mul_tensor_173: "f32[100]" = torch.ops.aten.mul.Tensor(mul_tensor_172, 0.1);  mul_tensor_172 = None
        mul_tensor_174: "f32[100]" = torch.ops.aten.mul.Tensor(primals_217, 0.9)
        add_tensor_104: "f32[100]" = torch.ops.aten.add.Tensor(mul_tensor_173, mul_tensor_174);  mul_tensor_173 = mul_tensor_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_105: "i64[]" = torch.ops.aten.add.Tensor(primals_221, 1)
        squeeze_dims_98: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        mul_tensor_175: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_dims_98, 0.1);  squeeze_dims_98 = None
        mul_tensor_176: "f32[100]" = torch.ops.aten.mul.Tensor(primals_222, 0.9)
        add_tensor_106: "f32[100]" = torch.ops.aten.add.Tensor(mul_tensor_175, mul_tensor_176);  mul_tensor_175 = mul_tensor_176 = None
        squeeze_dims_99: "f32[100]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_tensor_177: "f32[100]" = torch.ops.aten.mul.Tensor(squeeze_dims_99, 1.00000996502277);  squeeze_dims_99 = None
        mul_tensor_178: "f32[100]" = torch.ops.aten.mul.Tensor(mul_tensor_177, 0.1);  mul_tensor_177 = None
        mul_tensor_179: "f32[100]" = torch.ops.aten.mul.Tensor(primals_223, 0.9)
        add_tensor_107: "f32[100]" = torch.ops.aten.add.Tensor(mul_tensor_178, mul_tensor_179);  mul_tensor_178 = mul_tensor_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_108: "i64[]" = torch.ops.aten.add.Tensor(primals_227, 1)
        squeeze_dims_100: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_dims_101: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_tensor_180: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_100, 0.1)
        mul_tensor_181: "f32[40]" = torch.ops.aten.mul.Tensor(primals_228, 0.9)
        add_tensor_109: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_180, mul_tensor_181);  mul_tensor_180 = mul_tensor_181 = None
        squeeze_dims_102: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_tensor_182: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_102, 1.00000996502277);  squeeze_dims_102 = None
        mul_tensor_183: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_182, 0.1);  mul_tensor_182 = None
        mul_tensor_184: "f32[40]" = torch.ops.aten.mul.Tensor(primals_229, 0.9)
        add_tensor_110: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_183, mul_tensor_184);  mul_tensor_183 = mul_tensor_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_111: "i64[]" = torch.ops.aten.add.Tensor(primals_233, 1)
        squeeze_dims_103: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_dims_104: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_tensor_185: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_103, 0.1)
        mul_tensor_186: "f32[40]" = torch.ops.aten.mul.Tensor(primals_234, 0.9)
        add_tensor_112: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_185, mul_tensor_186);  mul_tensor_185 = mul_tensor_186 = None
        squeeze_dims_105: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_tensor_187: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_105, 1.00000996502277);  squeeze_dims_105 = None
        mul_tensor_188: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_187, 0.1);  mul_tensor_187 = None
        mul_tensor_189: "f32[40]" = torch.ops.aten.mul.Tensor(primals_235, 0.9)
        add_tensor_113: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_188, mul_tensor_189);  mul_tensor_188 = mul_tensor_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_114: "i64[]" = torch.ops.aten.add.Tensor(primals_239, 1)
        squeeze_dims_106: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3]);  getitem_77 = None
        squeeze_dims_107: "f32[92]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_tensor_190: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_106, 0.1)
        mul_tensor_191: "f32[92]" = torch.ops.aten.mul.Tensor(primals_240, 0.9)
        add_tensor_115: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_190, mul_tensor_191);  mul_tensor_190 = mul_tensor_191 = None
        squeeze_dims_108: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_tensor_192: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_108, 1.00000996502277);  squeeze_dims_108 = None
        mul_tensor_193: "f32[92]" = torch.ops.aten.mul.Tensor(mul_tensor_192, 0.1);  mul_tensor_192 = None
        mul_tensor_194: "f32[92]" = torch.ops.aten.mul.Tensor(primals_241, 0.9)
        add_tensor_116: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_193, mul_tensor_194);  mul_tensor_193 = mul_tensor_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_117: "i64[]" = torch.ops.aten.add.Tensor(primals_245, 1)
        squeeze_dims_109: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        mul_tensor_195: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_109, 0.1);  squeeze_dims_109 = None
        mul_tensor_196: "f32[92]" = torch.ops.aten.mul.Tensor(primals_246, 0.9)
        add_tensor_118: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_195, mul_tensor_196);  mul_tensor_195 = mul_tensor_196 = None
        squeeze_dims_110: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_tensor_197: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_110, 1.00000996502277);  squeeze_dims_110 = None
        mul_tensor_198: "f32[92]" = torch.ops.aten.mul.Tensor(mul_tensor_197, 0.1);  mul_tensor_197 = None
        mul_tensor_199: "f32[92]" = torch.ops.aten.mul.Tensor(primals_247, 0.9)
        add_tensor_119: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_198, mul_tensor_199);  mul_tensor_198 = mul_tensor_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_120: "i64[]" = torch.ops.aten.add.Tensor(primals_251, 1)
        squeeze_dims_111: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_dims_112: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_tensor_200: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_111, 0.1)
        mul_tensor_201: "f32[40]" = torch.ops.aten.mul.Tensor(primals_252, 0.9)
        add_tensor_121: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_200, mul_tensor_201);  mul_tensor_200 = mul_tensor_201 = None
        squeeze_dims_113: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_tensor_202: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_113, 1.00000996502277);  squeeze_dims_113 = None
        mul_tensor_203: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_202, 0.1);  mul_tensor_202 = None
        mul_tensor_204: "f32[40]" = torch.ops.aten.mul.Tensor(primals_253, 0.9)
        add_tensor_122: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_203, mul_tensor_204);  mul_tensor_203 = mul_tensor_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_123: "i64[]" = torch.ops.aten.add.Tensor(primals_257, 1)
        squeeze_dims_114: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        squeeze_dims_115: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_41, [0, 2, 3]);  rsqrt_41 = None
        mul_tensor_205: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_114, 0.1)
        mul_tensor_206: "f32[40]" = torch.ops.aten.mul.Tensor(primals_258, 0.9)
        add_tensor_124: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_205, mul_tensor_206);  mul_tensor_205 = mul_tensor_206 = None
        squeeze_dims_116: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_tensor_207: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_116, 1.00000996502277);  squeeze_dims_116 = None
        mul_tensor_208: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_207, 0.1);  mul_tensor_207 = None
        mul_tensor_209: "f32[40]" = torch.ops.aten.mul.Tensor(primals_259, 0.9)
        add_tensor_125: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_208, mul_tensor_209);  mul_tensor_208 = mul_tensor_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_126: "i64[]" = torch.ops.aten.add.Tensor(primals_263, 1)
        squeeze_dims_117: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_dims_118: "f32[92]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_tensor_210: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_117, 0.1)
        mul_tensor_211: "f32[92]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
        add_tensor_127: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_210, mul_tensor_211);  mul_tensor_210 = mul_tensor_211 = None
        squeeze_dims_119: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_tensor_212: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_119, 1.00000996502277);  squeeze_dims_119 = None
        mul_tensor_213: "f32[92]" = torch.ops.aten.mul.Tensor(mul_tensor_212, 0.1);  mul_tensor_212 = None
        mul_tensor_214: "f32[92]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
        add_tensor_128: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_213, mul_tensor_214);  mul_tensor_213 = mul_tensor_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_129: "i64[]" = torch.ops.aten.add.Tensor(primals_269, 1)
        squeeze_dims_120: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        mul_tensor_215: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_120, 0.1);  squeeze_dims_120 = None
        mul_tensor_216: "f32[92]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
        add_tensor_130: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_215, mul_tensor_216);  mul_tensor_215 = mul_tensor_216 = None
        squeeze_dims_121: "f32[92]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_tensor_217: "f32[92]" = torch.ops.aten.mul.Tensor(squeeze_dims_121, 1.00000996502277);  squeeze_dims_121 = None
        mul_tensor_218: "f32[92]" = torch.ops.aten.mul.Tensor(mul_tensor_217, 0.1);  mul_tensor_217 = None
        mul_tensor_219: "f32[92]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
        add_tensor_131: "f32[92]" = torch.ops.aten.add.Tensor(mul_tensor_218, mul_tensor_219);  mul_tensor_218 = mul_tensor_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_132: "i64[]" = torch.ops.aten.add.Tensor(primals_275, 1)
        squeeze_dims_122: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3]);  getitem_89 = None
        squeeze_dims_123: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_44, [0, 2, 3]);  rsqrt_44 = None
        mul_tensor_220: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_122, 0.1)
        mul_tensor_221: "f32[40]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
        add_tensor_133: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_220, mul_tensor_221);  mul_tensor_220 = mul_tensor_221 = None
        squeeze_dims_124: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_tensor_222: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_124, 1.00000996502277);  squeeze_dims_124 = None
        mul_tensor_223: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_222, 0.1);  mul_tensor_222 = None
        mul_tensor_224: "f32[40]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
        add_tensor_134: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_223, mul_tensor_224);  mul_tensor_223 = mul_tensor_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_135: "i64[]" = torch.ops.aten.add.Tensor(primals_281, 1)
        squeeze_dims_125: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_dims_126: "f32[40]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_tensor_225: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_125, 0.1)
        mul_tensor_226: "f32[40]" = torch.ops.aten.mul.Tensor(primals_282, 0.9)
        add_tensor_136: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_225, mul_tensor_226);  mul_tensor_225 = mul_tensor_226 = None
        squeeze_dims_127: "f32[40]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_tensor_227: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_dims_127, 1.00000996502277);  squeeze_dims_127 = None
        mul_tensor_228: "f32[40]" = torch.ops.aten.mul.Tensor(mul_tensor_227, 0.1);  mul_tensor_227 = None
        mul_tensor_229: "f32[40]" = torch.ops.aten.mul.Tensor(primals_283, 0.9)
        add_tensor_137: "f32[40]" = torch.ops.aten.add.Tensor(mul_tensor_228, mul_tensor_229);  mul_tensor_228 = mul_tensor_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_138: "i64[]" = torch.ops.aten.add.Tensor(primals_287, 1)
        squeeze_dims_128: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_dims_129: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_tensor_230: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_128, 0.1)
        mul_tensor_231: "f32[240]" = torch.ops.aten.mul.Tensor(primals_288, 0.9)
        add_tensor_139: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_230, mul_tensor_231);  mul_tensor_230 = mul_tensor_231 = None
        squeeze_dims_130: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_tensor_232: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_130, 1.00000996502277);  squeeze_dims_130 = None
        mul_tensor_233: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_232, 0.1);  mul_tensor_232 = None
        mul_tensor_234: "f32[240]" = torch.ops.aten.mul.Tensor(primals_289, 0.9)
        add_tensor_140: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_233, mul_tensor_234);  mul_tensor_233 = mul_tensor_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_141: "i64[]" = torch.ops.aten.add.Tensor(primals_293, 1)
        squeeze_dims_131: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        mul_tensor_235: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_131, 0.1);  squeeze_dims_131 = None
        mul_tensor_236: "f32[240]" = torch.ops.aten.mul.Tensor(primals_294, 0.9)
        add_tensor_142: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_235, mul_tensor_236);  mul_tensor_235 = mul_tensor_236 = None
        squeeze_dims_132: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_tensor_237: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_dims_132, 1.00000996502277);  squeeze_dims_132 = None
        mul_tensor_238: "f32[240]" = torch.ops.aten.mul.Tensor(mul_tensor_237, 0.1);  mul_tensor_237 = None
        mul_tensor_239: "f32[240]" = torch.ops.aten.mul.Tensor(primals_295, 0.9)
        add_tensor_143: "f32[240]" = torch.ops.aten.add.Tensor(mul_tensor_238, mul_tensor_239);  mul_tensor_238 = mul_tensor_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_144: "i64[]" = torch.ops.aten.add.Tensor(primals_303, 1)
        squeeze_dims_133: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_dims_134: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_tensor_240: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_133, 0.1)
        mul_tensor_241: "f32[56]" = torch.ops.aten.mul.Tensor(primals_304, 0.9)
        add_tensor_145: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_240, mul_tensor_241);  mul_tensor_240 = mul_tensor_241 = None
        squeeze_dims_135: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_tensor_242: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_135, 1.00000996502277);  squeeze_dims_135 = None
        mul_tensor_243: "f32[56]" = torch.ops.aten.mul.Tensor(mul_tensor_242, 0.1);  mul_tensor_242 = None
        mul_tensor_244: "f32[56]" = torch.ops.aten.mul.Tensor(primals_305, 0.9)
        add_tensor_146: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_243, mul_tensor_244);  mul_tensor_243 = mul_tensor_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_147: "i64[]" = torch.ops.aten.add.Tensor(primals_309, 1)
        squeeze_dims_136: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_dims_137: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_tensor_245: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_136, 0.1)
        mul_tensor_246: "f32[56]" = torch.ops.aten.mul.Tensor(primals_310, 0.9)
        add_tensor_148: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_245, mul_tensor_246);  mul_tensor_245 = mul_tensor_246 = None
        squeeze_dims_138: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_tensor_247: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_138, 1.00000996502277);  squeeze_dims_138 = None
        mul_tensor_248: "f32[56]" = torch.ops.aten.mul.Tensor(mul_tensor_247, 0.1);  mul_tensor_247 = None
        mul_tensor_249: "f32[56]" = torch.ops.aten.mul.Tensor(primals_311, 0.9)
        add_tensor_149: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_248, mul_tensor_249);  mul_tensor_248 = mul_tensor_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        add_tensor_150: "i64[]" = torch.ops.aten.add.Tensor(primals_315, 1)
        squeeze_dims_139: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3]);  getitem_101 = None
        squeeze_dims_140: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_tensor_250: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_139, 0.1)
        mul_tensor_251: "f32[80]" = torch.ops.aten.mul.Tensor(primals_316, 0.9)
        add_tensor_151: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_250, mul_tensor_251);  mul_tensor_250 = mul_tensor_251 = None
        squeeze_dims_141: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_tensor_252: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_141, 1.00000996502277);  squeeze_dims_141 = None
        mul_tensor_253: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_252, 0.1);  mul_tensor_252 = None
        mul_tensor_254: "f32[80]" = torch.ops.aten.mul.Tensor(primals_317, 0.9)
        add_tensor_152: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_253, mul_tensor_254);  mul_tensor_253 = mul_tensor_254 = None
        add_tensor_153: "i64[]" = torch.ops.aten.add.Tensor(primals_321, 1)
        squeeze_dims_142: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_dims_143: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_tensor_255: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_142, 0.1)
        mul_tensor_256: "f32[112]" = torch.ops.aten.mul.Tensor(primals_322, 0.9)
        add_tensor_154: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_255, mul_tensor_256);  mul_tensor_255 = mul_tensor_256 = None
        squeeze_dims_144: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_tensor_257: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_144, 1.00000996502277);  squeeze_dims_144 = None
        mul_tensor_258: "f32[112]" = torch.ops.aten.mul.Tensor(mul_tensor_257, 0.1);  mul_tensor_257 = None
        mul_tensor_259: "f32[112]" = torch.ops.aten.mul.Tensor(primals_323, 0.9)
        add_tensor_155: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_258, mul_tensor_259);  mul_tensor_258 = mul_tensor_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_156: "i64[]" = torch.ops.aten.add.Tensor(primals_327, 1)
        squeeze_dims_145: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_dims_146: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_tensor_260: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_145, 0.1)
        mul_tensor_261: "f32[336]" = torch.ops.aten.mul.Tensor(primals_328, 0.9)
        add_tensor_157: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_260, mul_tensor_261);  mul_tensor_260 = mul_tensor_261 = None
        squeeze_dims_147: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_tensor_262: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_147, 1.00000996502277);  squeeze_dims_147 = None
        mul_tensor_263: "f32[336]" = torch.ops.aten.mul.Tensor(mul_tensor_262, 0.1);  mul_tensor_262 = None
        mul_tensor_264: "f32[336]" = torch.ops.aten.mul.Tensor(primals_329, 0.9)
        add_tensor_158: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_263, mul_tensor_264);  mul_tensor_263 = mul_tensor_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_159: "i64[]" = torch.ops.aten.add.Tensor(primals_333, 1)
        squeeze_dims_148: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        mul_tensor_265: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_148, 0.1);  squeeze_dims_148 = None
        mul_tensor_266: "f32[336]" = torch.ops.aten.mul.Tensor(primals_334, 0.9)
        add_tensor_160: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_265, mul_tensor_266);  mul_tensor_265 = mul_tensor_266 = None
        squeeze_dims_149: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_tensor_267: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_149, 1.00000996502277);  squeeze_dims_149 = None
        mul_tensor_268: "f32[336]" = torch.ops.aten.mul.Tensor(mul_tensor_267, 0.1);  mul_tensor_267 = None
        mul_tensor_269: "f32[336]" = torch.ops.aten.mul.Tensor(primals_335, 0.9)
        add_tensor_161: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_268, mul_tensor_269);  mul_tensor_268 = mul_tensor_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_162: "i64[]" = torch.ops.aten.add.Tensor(primals_343, 1)
        squeeze_dims_150: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_dims_151: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_tensor_270: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_150, 0.1)
        mul_tensor_271: "f32[56]" = torch.ops.aten.mul.Tensor(primals_344, 0.9)
        add_tensor_163: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_270, mul_tensor_271);  mul_tensor_270 = mul_tensor_271 = None
        squeeze_dims_152: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_tensor_272: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_152, 1.00000996502277);  squeeze_dims_152 = None
        mul_tensor_273: "f32[56]" = torch.ops.aten.mul.Tensor(mul_tensor_272, 0.1);  mul_tensor_272 = None
        mul_tensor_274: "f32[56]" = torch.ops.aten.mul.Tensor(primals_345, 0.9)
        add_tensor_164: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_273, mul_tensor_274);  mul_tensor_273 = mul_tensor_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_165: "i64[]" = torch.ops.aten.add.Tensor(primals_349, 1)
        squeeze_dims_153: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_dims_154: "f32[56]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_tensor_275: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_153, 0.1)
        mul_tensor_276: "f32[56]" = torch.ops.aten.mul.Tensor(primals_350, 0.9)
        add_tensor_166: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_275, mul_tensor_276);  mul_tensor_275 = mul_tensor_276 = None
        squeeze_dims_155: "f32[56]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_tensor_277: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_dims_155, 1.00000996502277);  squeeze_dims_155 = None
        mul_tensor_278: "f32[56]" = torch.ops.aten.mul.Tensor(mul_tensor_277, 0.1);  mul_tensor_277 = None
        mul_tensor_279: "f32[56]" = torch.ops.aten.mul.Tensor(primals_351, 0.9)
        add_tensor_167: "f32[56]" = torch.ops.aten.add.Tensor(mul_tensor_278, mul_tensor_279);  mul_tensor_278 = mul_tensor_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_168: "i64[]" = torch.ops.aten.add.Tensor(primals_355, 1)
        squeeze_dims_156: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        squeeze_dims_157: "f32[336]" = torch.ops.aten.squeeze.dims(rsqrt_56, [0, 2, 3]);  rsqrt_56 = None
        mul_tensor_280: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_156, 0.1)
        mul_tensor_281: "f32[336]" = torch.ops.aten.mul.Tensor(primals_356, 0.9)
        add_tensor_169: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_280, mul_tensor_281);  mul_tensor_280 = mul_tensor_281 = None
        squeeze_dims_158: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_tensor_282: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_158, 1.00000996502277);  squeeze_dims_158 = None
        mul_tensor_283: "f32[336]" = torch.ops.aten.mul.Tensor(mul_tensor_282, 0.1);  mul_tensor_282 = None
        mul_tensor_284: "f32[336]" = torch.ops.aten.mul.Tensor(primals_357, 0.9)
        add_tensor_170: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_283, mul_tensor_284);  mul_tensor_283 = mul_tensor_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_171: "i64[]" = torch.ops.aten.add.Tensor(primals_361, 1)
        squeeze_dims_159: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        mul_tensor_285: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_159, 0.1);  squeeze_dims_159 = None
        mul_tensor_286: "f32[336]" = torch.ops.aten.mul.Tensor(primals_362, 0.9)
        add_tensor_172: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_285, mul_tensor_286);  mul_tensor_285 = mul_tensor_286 = None
        squeeze_dims_160: "f32[336]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        mul_tensor_287: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_dims_160, 1.00000996502277);  squeeze_dims_160 = None
        mul_tensor_288: "f32[336]" = torch.ops.aten.mul.Tensor(mul_tensor_287, 0.1);  mul_tensor_287 = None
        mul_tensor_289: "f32[336]" = torch.ops.aten.mul.Tensor(primals_363, 0.9)
        add_tensor_173: "f32[336]" = torch.ops.aten.add.Tensor(mul_tensor_288, mul_tensor_289);  mul_tensor_288 = mul_tensor_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        add_tensor_174: "i64[]" = torch.ops.aten.add.Tensor(primals_367, 1)
        squeeze_dims_161: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        mul_tensor_290: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_161, 0.1);  squeeze_dims_161 = None
        mul_tensor_291: "f32[672]" = torch.ops.aten.mul.Tensor(primals_368, 0.9)
        add_tensor_175: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_290, mul_tensor_291);  mul_tensor_290 = mul_tensor_291 = None
        squeeze_dims_162: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_tensor_292: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_dims_162, 1.0000398612827361);  squeeze_dims_162 = None
        mul_tensor_293: "f32[672]" = torch.ops.aten.mul.Tensor(mul_tensor_292, 0.1);  mul_tensor_292 = None
        mul_tensor_294: "f32[672]" = torch.ops.aten.mul.Tensor(primals_369, 0.9)
        add_tensor_176: "f32[672]" = torch.ops.aten.add.Tensor(mul_tensor_293, mul_tensor_294);  mul_tensor_293 = mul_tensor_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_177: "i64[]" = torch.ops.aten.add.Tensor(primals_377, 1)
        squeeze_dims_163: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        squeeze_dims_164: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_59, [0, 2, 3]);  rsqrt_59 = None
        mul_tensor_295: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_163, 0.1)
        mul_tensor_296: "f32[80]" = torch.ops.aten.mul.Tensor(primals_378, 0.9)
        add_tensor_178: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_295, mul_tensor_296);  mul_tensor_295 = mul_tensor_296 = None
        squeeze_dims_165: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_tensor_297: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_165, 1.0000398612827361);  squeeze_dims_165 = None
        mul_tensor_298: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_297, 0.1);  mul_tensor_297 = None
        mul_tensor_299: "f32[80]" = torch.ops.aten.mul.Tensor(primals_379, 0.9)
        add_tensor_179: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_298, mul_tensor_299);  mul_tensor_298 = mul_tensor_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_180: "i64[]" = torch.ops.aten.add.Tensor(primals_383, 1)
        squeeze_dims_166: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        squeeze_dims_167: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_60, [0, 2, 3]);  rsqrt_60 = None
        mul_tensor_300: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_166, 0.1)
        mul_tensor_301: "f32[80]" = torch.ops.aten.mul.Tensor(primals_384, 0.9)
        add_tensor_181: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_300, mul_tensor_301);  mul_tensor_300 = mul_tensor_301 = None
        squeeze_dims_168: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_tensor_302: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_168, 1.0000398612827361);  squeeze_dims_168 = None
        mul_tensor_303: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_302, 0.1);  mul_tensor_302 = None
        mul_tensor_304: "f32[80]" = torch.ops.aten.mul.Tensor(primals_385, 0.9)
        add_tensor_182: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_303, mul_tensor_304);  mul_tensor_303 = mul_tensor_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        add_tensor_183: "i64[]" = torch.ops.aten.add.Tensor(primals_389, 1)
        squeeze_dims_169: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        squeeze_dims_170: "f32[112]" = torch.ops.aten.squeeze.dims(rsqrt_61, [0, 2, 3]);  rsqrt_61 = None
        mul_tensor_305: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_169, 0.1)
        mul_tensor_306: "f32[112]" = torch.ops.aten.mul.Tensor(primals_390, 0.9)
        add_tensor_184: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_305, mul_tensor_306);  mul_tensor_305 = mul_tensor_306 = None
        squeeze_dims_171: "f32[112]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        mul_tensor_307: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_dims_171, 1.0000398612827361);  squeeze_dims_171 = None
        mul_tensor_308: "f32[112]" = torch.ops.aten.mul.Tensor(mul_tensor_307, 0.1);  mul_tensor_307 = None
        mul_tensor_309: "f32[112]" = torch.ops.aten.mul.Tensor(primals_391, 0.9)
        add_tensor_185: "f32[112]" = torch.ops.aten.add.Tensor(mul_tensor_308, mul_tensor_309);  mul_tensor_308 = mul_tensor_309 = None
        add_tensor_186: "i64[]" = torch.ops.aten.add.Tensor(primals_395, 1)
        squeeze_dims_172: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_125, [0, 2, 3]);  getitem_125 = None
        squeeze_dims_173: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_62, [0, 2, 3]);  rsqrt_62 = None
        mul_tensor_310: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_172, 0.1)
        mul_tensor_311: "f32[160]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
        add_tensor_187: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_310, mul_tensor_311);  mul_tensor_310 = mul_tensor_311 = None
        squeeze_dims_174: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_124, [0, 2, 3]);  getitem_124 = None
        mul_tensor_312: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_dims_174, 1.0000398612827361);  squeeze_dims_174 = None
        mul_tensor_313: "f32[160]" = torch.ops.aten.mul.Tensor(mul_tensor_312, 0.1);  mul_tensor_312 = None
        mul_tensor_314: "f32[160]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
        add_tensor_188: "f32[160]" = torch.ops.aten.add.Tensor(mul_tensor_313, mul_tensor_314);  mul_tensor_313 = mul_tensor_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_189: "i64[]" = torch.ops.aten.add.Tensor(primals_401, 1)
        squeeze_dims_175: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        squeeze_dims_176: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_63, [0, 2, 3]);  rsqrt_63 = None
        mul_tensor_315: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_175, 0.1)
        mul_tensor_316: "f32[480]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
        add_tensor_190: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_315, mul_tensor_316);  mul_tensor_315 = mul_tensor_316 = None
        squeeze_dims_177: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_126, [0, 2, 3]);  getitem_126 = None
        mul_tensor_317: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_177, 1.0000398612827361);  squeeze_dims_177 = None
        mul_tensor_318: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_317, 0.1);  mul_tensor_317 = None
        mul_tensor_319: "f32[480]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
        add_tensor_191: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_318, mul_tensor_319);  mul_tensor_318 = mul_tensor_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_192: "i64[]" = torch.ops.aten.add.Tensor(primals_407, 1)
        squeeze_dims_178: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        mul_tensor_320: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_178, 0.1);  squeeze_dims_178 = None
        mul_tensor_321: "f32[480]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
        add_tensor_193: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_320, mul_tensor_321);  mul_tensor_320 = mul_tensor_321 = None
        squeeze_dims_179: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_128, [0, 2, 3]);  getitem_128 = None
        mul_tensor_322: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_179, 1.0000398612827361);  squeeze_dims_179 = None
        mul_tensor_323: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_322, 0.1);  mul_tensor_322 = None
        mul_tensor_324: "f32[480]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
        add_tensor_194: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_323, mul_tensor_324);  mul_tensor_323 = mul_tensor_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_195: "i64[]" = torch.ops.aten.add.Tensor(primals_413, 1)
        squeeze_dims_180: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_131, [0, 2, 3]);  getitem_131 = None
        squeeze_dims_181: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_65, [0, 2, 3]);  rsqrt_65 = None
        mul_tensor_325: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_180, 0.1)
        mul_tensor_326: "f32[80]" = torch.ops.aten.mul.Tensor(primals_414, 0.9)
        add_tensor_196: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_325, mul_tensor_326);  mul_tensor_325 = mul_tensor_326 = None
        squeeze_dims_182: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_130, [0, 2, 3]);  getitem_130 = None
        mul_tensor_327: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_182, 1.0000398612827361);  squeeze_dims_182 = None
        mul_tensor_328: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_327, 0.1);  mul_tensor_327 = None
        mul_tensor_329: "f32[80]" = torch.ops.aten.mul.Tensor(primals_415, 0.9)
        add_tensor_197: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_328, mul_tensor_329);  mul_tensor_328 = mul_tensor_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_198: "i64[]" = torch.ops.aten.add.Tensor(primals_419, 1)
        squeeze_dims_183: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_133, [0, 2, 3]);  getitem_133 = None
        squeeze_dims_184: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_66, [0, 2, 3]);  rsqrt_66 = None
        mul_tensor_330: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_183, 0.1)
        mul_tensor_331: "f32[80]" = torch.ops.aten.mul.Tensor(primals_420, 0.9)
        add_tensor_199: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_330, mul_tensor_331);  mul_tensor_330 = mul_tensor_331 = None
        squeeze_dims_185: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_132, [0, 2, 3]);  getitem_132 = None
        mul_tensor_332: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_185, 1.0000398612827361);  squeeze_dims_185 = None
        mul_tensor_333: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_332, 0.1);  mul_tensor_332 = None
        mul_tensor_334: "f32[80]" = torch.ops.aten.mul.Tensor(primals_421, 0.9)
        add_tensor_200: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_333, mul_tensor_334);  mul_tensor_333 = mul_tensor_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_201: "i64[]" = torch.ops.aten.add.Tensor(primals_425, 1)
        squeeze_dims_186: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_135, [0, 2, 3]);  getitem_135 = None
        squeeze_dims_187: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_67, [0, 2, 3]);  rsqrt_67 = None
        mul_tensor_335: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_186, 0.1)
        mul_tensor_336: "f32[480]" = torch.ops.aten.mul.Tensor(primals_426, 0.9)
        add_tensor_202: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_335, mul_tensor_336);  mul_tensor_335 = mul_tensor_336 = None
        squeeze_dims_188: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_134, [0, 2, 3]);  getitem_134 = None
        mul_tensor_337: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_188, 1.0000398612827361);  squeeze_dims_188 = None
        mul_tensor_338: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_337, 0.1);  mul_tensor_337 = None
        mul_tensor_339: "f32[480]" = torch.ops.aten.mul.Tensor(primals_427, 0.9)
        add_tensor_203: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_338, mul_tensor_339);  mul_tensor_338 = mul_tensor_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_204: "i64[]" = torch.ops.aten.add.Tensor(primals_431, 1)
        squeeze_dims_189: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        mul_tensor_340: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_189, 0.1);  squeeze_dims_189 = None
        mul_tensor_341: "f32[480]" = torch.ops.aten.mul.Tensor(primals_432, 0.9)
        add_tensor_205: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_340, mul_tensor_341);  mul_tensor_340 = mul_tensor_341 = None
        squeeze_dims_190: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_136, [0, 2, 3]);  getitem_136 = None
        mul_tensor_342: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_190, 1.0000398612827361);  squeeze_dims_190 = None
        mul_tensor_343: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_342, 0.1);  mul_tensor_342 = None
        mul_tensor_344: "f32[480]" = torch.ops.aten.mul.Tensor(primals_433, 0.9)
        add_tensor_206: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_343, mul_tensor_344);  mul_tensor_343 = mul_tensor_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_207: "i64[]" = torch.ops.aten.add.Tensor(primals_441, 1)
        squeeze_dims_191: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        squeeze_dims_192: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_69, [0, 2, 3]);  rsqrt_69 = None
        mul_tensor_345: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_191, 0.1)
        mul_tensor_346: "f32[80]" = torch.ops.aten.mul.Tensor(primals_442, 0.9)
        add_tensor_208: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_345, mul_tensor_346);  mul_tensor_345 = mul_tensor_346 = None
        squeeze_dims_193: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_138, [0, 2, 3]);  getitem_138 = None
        mul_tensor_347: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_193, 1.0000398612827361);  squeeze_dims_193 = None
        mul_tensor_348: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_347, 0.1);  mul_tensor_347 = None
        mul_tensor_349: "f32[80]" = torch.ops.aten.mul.Tensor(primals_443, 0.9)
        add_tensor_209: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_348, mul_tensor_349);  mul_tensor_348 = mul_tensor_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_210: "i64[]" = torch.ops.aten.add.Tensor(primals_447, 1)
        squeeze_dims_194: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_141, [0, 2, 3]);  getitem_141 = None
        squeeze_dims_195: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_70, [0, 2, 3]);  rsqrt_70 = None
        mul_tensor_350: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_194, 0.1)
        mul_tensor_351: "f32[80]" = torch.ops.aten.mul.Tensor(primals_448, 0.9)
        add_tensor_211: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_350, mul_tensor_351);  mul_tensor_350 = mul_tensor_351 = None
        squeeze_dims_196: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_140, [0, 2, 3]);  getitem_140 = None
        mul_tensor_352: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_196, 1.0000398612827361);  squeeze_dims_196 = None
        mul_tensor_353: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_352, 0.1);  mul_tensor_352 = None
        mul_tensor_354: "f32[80]" = torch.ops.aten.mul.Tensor(primals_449, 0.9)
        add_tensor_212: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_353, mul_tensor_354);  mul_tensor_353 = mul_tensor_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_213: "i64[]" = torch.ops.aten.add.Tensor(primals_453, 1)
        squeeze_dims_197: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_143, [0, 2, 3]);  getitem_143 = None
        squeeze_dims_198: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_71, [0, 2, 3]);  rsqrt_71 = None
        mul_tensor_355: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_197, 0.1)
        mul_tensor_356: "f32[480]" = torch.ops.aten.mul.Tensor(primals_454, 0.9)
        add_tensor_214: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_355, mul_tensor_356);  mul_tensor_355 = mul_tensor_356 = None
        squeeze_dims_199: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_142, [0, 2, 3]);  getitem_142 = None
        mul_tensor_357: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_199, 1.0000398612827361);  squeeze_dims_199 = None
        mul_tensor_358: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_357, 0.1);  mul_tensor_357 = None
        mul_tensor_359: "f32[480]" = torch.ops.aten.mul.Tensor(primals_455, 0.9)
        add_tensor_215: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_358, mul_tensor_359);  mul_tensor_358 = mul_tensor_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_216: "i64[]" = torch.ops.aten.add.Tensor(primals_459, 1)
        squeeze_dims_200: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_145, [0, 2, 3]);  getitem_145 = None
        mul_tensor_360: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_200, 0.1);  squeeze_dims_200 = None
        mul_tensor_361: "f32[480]" = torch.ops.aten.mul.Tensor(primals_460, 0.9)
        add_tensor_217: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_360, mul_tensor_361);  mul_tensor_360 = mul_tensor_361 = None
        squeeze_dims_201: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_144, [0, 2, 3]);  getitem_144 = None
        mul_tensor_362: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_201, 1.0000398612827361);  squeeze_dims_201 = None
        mul_tensor_363: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_362, 0.1);  mul_tensor_362 = None
        mul_tensor_364: "f32[480]" = torch.ops.aten.mul.Tensor(primals_461, 0.9)
        add_tensor_218: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_363, mul_tensor_364);  mul_tensor_363 = mul_tensor_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_219: "i64[]" = torch.ops.aten.add.Tensor(primals_465, 1)
        squeeze_dims_202: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_147, [0, 2, 3]);  getitem_147 = None
        squeeze_dims_203: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_73, [0, 2, 3]);  rsqrt_73 = None
        mul_tensor_365: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_202, 0.1)
        mul_tensor_366: "f32[80]" = torch.ops.aten.mul.Tensor(primals_466, 0.9)
        add_tensor_220: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_365, mul_tensor_366);  mul_tensor_365 = mul_tensor_366 = None
        squeeze_dims_204: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_146, [0, 2, 3]);  getitem_146 = None
        mul_tensor_367: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_204, 1.0000398612827361);  squeeze_dims_204 = None
        mul_tensor_368: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_367, 0.1);  mul_tensor_367 = None
        mul_tensor_369: "f32[80]" = torch.ops.aten.mul.Tensor(primals_467, 0.9)
        add_tensor_221: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_368, mul_tensor_369);  mul_tensor_368 = mul_tensor_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_222: "i64[]" = torch.ops.aten.add.Tensor(primals_471, 1)
        squeeze_dims_205: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_149, [0, 2, 3]);  getitem_149 = None
        squeeze_dims_206: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_74, [0, 2, 3]);  rsqrt_74 = None
        mul_tensor_370: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_205, 0.1)
        mul_tensor_371: "f32[80]" = torch.ops.aten.mul.Tensor(primals_472, 0.9)
        add_tensor_223: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_370, mul_tensor_371);  mul_tensor_370 = mul_tensor_371 = None
        squeeze_dims_207: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_148, [0, 2, 3]);  getitem_148 = None
        mul_tensor_372: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_207, 1.0000398612827361);  squeeze_dims_207 = None
        mul_tensor_373: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_372, 0.1);  mul_tensor_372 = None
        mul_tensor_374: "f32[80]" = torch.ops.aten.mul.Tensor(primals_473, 0.9)
        add_tensor_224: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_373, mul_tensor_374);  mul_tensor_373 = mul_tensor_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_225: "i64[]" = torch.ops.aten.add.Tensor(primals_477, 1)
        squeeze_dims_208: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_151, [0, 2, 3]);  getitem_151 = None
        squeeze_dims_209: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_75, [0, 2, 3]);  rsqrt_75 = None
        mul_tensor_375: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_208, 0.1)
        mul_tensor_376: "f32[480]" = torch.ops.aten.mul.Tensor(primals_478, 0.9)
        add_tensor_226: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_375, mul_tensor_376);  mul_tensor_375 = mul_tensor_376 = None
        squeeze_dims_210: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_150, [0, 2, 3]);  getitem_150 = None
        mul_tensor_377: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_210, 1.0000398612827361);  squeeze_dims_210 = None
        mul_tensor_378: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_377, 0.1);  mul_tensor_377 = None
        mul_tensor_379: "f32[480]" = torch.ops.aten.mul.Tensor(primals_479, 0.9)
        add_tensor_227: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_378, mul_tensor_379);  mul_tensor_378 = mul_tensor_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_228: "i64[]" = torch.ops.aten.add.Tensor(primals_483, 1)
        squeeze_dims_211: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_153, [0, 2, 3]);  getitem_153 = None
        mul_tensor_380: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_211, 0.1);  squeeze_dims_211 = None
        mul_tensor_381: "f32[480]" = torch.ops.aten.mul.Tensor(primals_484, 0.9)
        add_tensor_229: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_380, mul_tensor_381);  mul_tensor_380 = mul_tensor_381 = None
        squeeze_dims_212: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_152, [0, 2, 3]);  getitem_152 = None
        mul_tensor_382: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_dims_212, 1.0000398612827361);  squeeze_dims_212 = None
        mul_tensor_383: "f32[480]" = torch.ops.aten.mul.Tensor(mul_tensor_382, 0.1);  mul_tensor_382 = None
        mul_tensor_384: "f32[480]" = torch.ops.aten.mul.Tensor(primals_485, 0.9)
        add_tensor_230: "f32[480]" = torch.ops.aten.add.Tensor(mul_tensor_383, mul_tensor_384);  mul_tensor_383 = mul_tensor_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        add_tensor_231: "i64[]" = torch.ops.aten.add.Tensor(primals_493, 1)
        squeeze_dims_213: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_155, [0, 2, 3]);  getitem_155 = None
        squeeze_dims_214: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_77, [0, 2, 3]);  rsqrt_77 = None
        mul_tensor_385: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_213, 0.1)
        mul_tensor_386: "f32[80]" = torch.ops.aten.mul.Tensor(primals_494, 0.9)
        add_tensor_232: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_385, mul_tensor_386);  mul_tensor_385 = mul_tensor_386 = None
        squeeze_dims_215: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_154, [0, 2, 3]);  getitem_154 = None
        mul_tensor_387: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_215, 1.0000398612827361);  squeeze_dims_215 = None
        mul_tensor_388: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_387, 0.1);  mul_tensor_387 = None
        mul_tensor_389: "f32[80]" = torch.ops.aten.mul.Tensor(primals_495, 0.9)
        add_tensor_233: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_388, mul_tensor_389);  mul_tensor_388 = mul_tensor_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        add_tensor_234: "i64[]" = torch.ops.aten.add.Tensor(primals_499, 1)
        squeeze_dims_216: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_157, [0, 2, 3]);  getitem_157 = None
        squeeze_dims_217: "f32[80]" = torch.ops.aten.squeeze.dims(rsqrt_78, [0, 2, 3]);  rsqrt_78 = None
        mul_tensor_390: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_216, 0.1)
        mul_tensor_391: "f32[80]" = torch.ops.aten.mul.Tensor(primals_500, 0.9)
        add_tensor_235: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_390, mul_tensor_391);  mul_tensor_390 = mul_tensor_391 = None
        squeeze_dims_218: "f32[80]" = torch.ops.aten.squeeze.dims(getitem_156, [0, 2, 3]);  getitem_156 = None
        mul_tensor_392: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_dims_218, 1.0000398612827361);  squeeze_dims_218 = None
        mul_tensor_393: "f32[80]" = torch.ops.aten.mul.Tensor(mul_tensor_392, 0.1);  mul_tensor_392 = None
        mul_tensor_394: "f32[80]" = torch.ops.aten.mul.Tensor(primals_501, 0.9)
        add_tensor_236: "f32[80]" = torch.ops.aten.add.Tensor(mul_tensor_393, mul_tensor_394);  mul_tensor_393 = mul_tensor_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_tensor_237: "i64[]" = torch.ops.aten.add.Tensor(primals_505, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_dims_219: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_159, [0, 2, 3]);  getitem_159 = None
        mul_tensor_395: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_219, 0.1);  squeeze_dims_219 = None
        mul_tensor_396: "f32[960]" = torch.ops.aten.mul.Tensor(primals_506, 0.9)
        add_tensor_238: "f32[960]" = torch.ops.aten.add.Tensor(mul_tensor_395, mul_tensor_396);  mul_tensor_395 = mul_tensor_396 = None
        squeeze_dims_220: "f32[960]" = torch.ops.aten.squeeze.dims(getitem_158, [0, 2, 3]);  getitem_158 = None
        mul_tensor_397: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_dims_220, 1.0000398612827361);  squeeze_dims_220 = None
        mul_tensor_398: "f32[960]" = torch.ops.aten.mul.Tensor(mul_tensor_397, 0.1);  mul_tensor_397 = None
        mul_tensor_399: "f32[960]" = torch.ops.aten.mul.Tensor(primals_507, 0.9)
        add_tensor_239: "f32[960]" = torch.ops.aten.add.Tensor(mul_tensor_398, mul_tensor_399);  mul_tensor_398 = mul_tensor_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:833 in forward_head, code: x = self.act2(x)
        relu_default: "f32[512, 1280, 1, 1]" = torch.ops.aten.relu.default(convolution_94);  convolution_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:834 in forward_head, code: x = self.flatten(x)
        reshape_default: "f32[512, 1280]" = torch.ops.aten.reshape.default(relu_default, _shape_param_0);  _shape_param_0 = None

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:836 in forward_head, code: x = F.dropout(x, p=self.drop_rate, training=self.training)
        inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 1280]" = torch.ops.prims.inductor_random.default([512, 1280], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt_scalar: "b8[512, 1280]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.2);  inductor_random_default = None
        mul_tensor_400: "f32[512, 1280]" = torch.ops.aten.mul.Tensor(gt_scalar, reshape_default);  gt_scalar = reshape_default = None
        mul_tensor_401: "f32[512, 1280]" = torch.ops.aten.mul.Tensor(mul_tensor_400, 1.25);  mul_tensor_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/linear.py:19 in forward, code: return F.linear(input, self.weight, self.bias)
        permute_default: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_512, [1, 0]);  primals_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:833 in forward_head, code: x = self.act2(x)
        le_scalar: "b8[512, 1280, 1, 1]" = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_216, 0);  squeeze_dims_216 = None
        unsqueeze_default_1: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_3: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_213, 0);  squeeze_dims_213 = None
        unsqueeze_default_4: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 2);  unsqueeze_default_3 = None
        unsqueeze_default_5: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 3);  unsqueeze_default_4 = None
        unsqueeze_default_6: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_208, 0);  squeeze_dims_208 = None
        unsqueeze_default_7: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_6, 2);  unsqueeze_default_6 = None
        unsqueeze_default_8: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_7, 3);  unsqueeze_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_9: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_205, 0);  squeeze_dims_205 = None
        unsqueeze_default_10: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_9, 2);  unsqueeze_default_9 = None
        unsqueeze_default_11: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_10, 3);  unsqueeze_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_12: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_202, 0);  squeeze_dims_202 = None
        unsqueeze_default_13: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_12, 2);  unsqueeze_default_12 = None
        unsqueeze_default_14: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_13, 3);  unsqueeze_default_13 = None
        unsqueeze_default_15: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_197, 0);  squeeze_dims_197 = None
        unsqueeze_default_16: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_15, 2);  unsqueeze_default_15 = None
        unsqueeze_default_17: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_16, 3);  unsqueeze_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_18: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_194, 0);  squeeze_dims_194 = None
        unsqueeze_default_19: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_18, 2);  unsqueeze_default_18 = None
        unsqueeze_default_20: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_19, 3);  unsqueeze_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_21: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_191, 0);  squeeze_dims_191 = None
        unsqueeze_default_22: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_21, 2);  unsqueeze_default_21 = None
        unsqueeze_default_23: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_22, 3);  unsqueeze_default_22 = None
        unsqueeze_default_24: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_186, 0);  squeeze_dims_186 = None
        unsqueeze_default_25: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_24, 2);  unsqueeze_default_24 = None
        unsqueeze_default_26: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_25, 3);  unsqueeze_default_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_27: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_183, 0);  squeeze_dims_183 = None
        unsqueeze_default_28: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_27, 2);  unsqueeze_default_27 = None
        unsqueeze_default_29: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_28, 3);  unsqueeze_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_30: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_180, 0);  squeeze_dims_180 = None
        unsqueeze_default_31: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_30, 2);  unsqueeze_default_30 = None
        unsqueeze_default_32: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_31, 3);  unsqueeze_default_31 = None
        unsqueeze_default_33: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_dims_175, 0);  squeeze_dims_175 = None
        unsqueeze_default_34: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_33, 2);  unsqueeze_default_33 = None
        unsqueeze_default_35: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_34, 3);  unsqueeze_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        unsqueeze_default_36: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_dims_172, 0);  squeeze_dims_172 = None
        unsqueeze_default_37: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_36, 2);  unsqueeze_default_36 = None
        unsqueeze_default_38: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_37, 3);  unsqueeze_default_37 = None
        unsqueeze_default_39: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_dims_169, 0);  squeeze_dims_169 = None
        unsqueeze_default_40: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_39, 2);  unsqueeze_default_39 = None
        unsqueeze_default_41: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_40, 3);  unsqueeze_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_42: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_166, 0);  squeeze_dims_166 = None
        unsqueeze_default_43: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_42, 2);  unsqueeze_default_42 = None
        unsqueeze_default_44: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_43, 3);  unsqueeze_default_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_45: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_163, 0);  squeeze_dims_163 = None
        unsqueeze_default_46: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_45, 2);  unsqueeze_default_45 = None
        unsqueeze_default_47: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_46, 3);  unsqueeze_default_46 = None
        unsqueeze_default_48: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_dims_156, 0);  squeeze_dims_156 = None
        unsqueeze_default_49: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_48, 2);  unsqueeze_default_48 = None
        unsqueeze_default_50: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_49, 3);  unsqueeze_default_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_51: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_dims_153, 0);  squeeze_dims_153 = None
        unsqueeze_default_52: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_51, 2);  unsqueeze_default_51 = None
        unsqueeze_default_53: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_52, 3);  unsqueeze_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_54: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_dims_150, 0);  squeeze_dims_150 = None
        unsqueeze_default_55: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_54, 2);  unsqueeze_default_54 = None
        unsqueeze_default_56: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_55, 3);  unsqueeze_default_55 = None
        unsqueeze_default_57: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(squeeze_dims_145, 0);  squeeze_dims_145 = None
        unsqueeze_default_58: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_57, 2);  unsqueeze_default_57 = None
        unsqueeze_default_59: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_58, 3);  unsqueeze_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        unsqueeze_default_60: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(squeeze_dims_142, 0);  squeeze_dims_142 = None
        unsqueeze_default_61: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_60, 2);  unsqueeze_default_60 = None
        unsqueeze_default_62: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_61, 3);  unsqueeze_default_61 = None
        unsqueeze_default_63: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_139, 0);  squeeze_dims_139 = None
        unsqueeze_default_64: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_63, 2);  unsqueeze_default_63 = None
        unsqueeze_default_65: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_64, 3);  unsqueeze_default_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_66: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_dims_136, 0);  squeeze_dims_136 = None
        unsqueeze_default_67: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_66, 2);  unsqueeze_default_66 = None
        unsqueeze_default_68: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_67, 3);  unsqueeze_default_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_69: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(squeeze_dims_133, 0);  squeeze_dims_133 = None
        unsqueeze_default_70: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_69, 2);  unsqueeze_default_69 = None
        unsqueeze_default_71: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_70, 3);  unsqueeze_default_70 = None
        unsqueeze_default_72: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_dims_128, 0);  squeeze_dims_128 = None
        unsqueeze_default_73: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_72, 2);  unsqueeze_default_72 = None
        unsqueeze_default_74: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_73, 3);  unsqueeze_default_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_75: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_125, 0);  squeeze_dims_125 = None
        unsqueeze_default_76: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_75, 2);  unsqueeze_default_75 = None
        unsqueeze_default_77: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_76, 3);  unsqueeze_default_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_78: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_122, 0);  squeeze_dims_122 = None
        unsqueeze_default_79: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_78, 2);  unsqueeze_default_78 = None
        unsqueeze_default_80: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_79, 3);  unsqueeze_default_79 = None
        unsqueeze_default_81: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(squeeze_dims_117, 0);  squeeze_dims_117 = None
        unsqueeze_default_82: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_81, 2);  unsqueeze_default_81 = None
        unsqueeze_default_83: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_82, 3);  unsqueeze_default_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_84: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_114, 0);  squeeze_dims_114 = None
        unsqueeze_default_85: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_84, 2);  unsqueeze_default_84 = None
        unsqueeze_default_86: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_85, 3);  unsqueeze_default_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_87: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_111, 0);  squeeze_dims_111 = None
        unsqueeze_default_88: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_87, 2);  unsqueeze_default_87 = None
        unsqueeze_default_89: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_88, 3);  unsqueeze_default_88 = None
        unsqueeze_default_90: "f32[1, 92]" = torch.ops.aten.unsqueeze.default(squeeze_dims_106, 0);  squeeze_dims_106 = None
        unsqueeze_default_91: "f32[1, 92, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_90, 2);  unsqueeze_default_90 = None
        unsqueeze_default_92: "f32[1, 92, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_91, 3);  unsqueeze_default_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_93: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_103, 0);  squeeze_dims_103 = None
        unsqueeze_default_94: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_93, 2);  unsqueeze_default_93 = None
        unsqueeze_default_95: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_94, 3);  unsqueeze_default_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_96: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_100, 0);  squeeze_dims_100 = None
        unsqueeze_default_97: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_96, 2);  unsqueeze_default_96 = None
        unsqueeze_default_98: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_97, 3);  unsqueeze_default_97 = None
        unsqueeze_default_99: "f32[1, 100]" = torch.ops.aten.unsqueeze.default(squeeze_dims_95, 0);  squeeze_dims_95 = None
        unsqueeze_default_100: "f32[1, 100, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_99, 2);  unsqueeze_default_99 = None
        unsqueeze_default_101: "f32[1, 100, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_100, 3);  unsqueeze_default_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        unsqueeze_default_102: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(squeeze_dims_92, 0);  squeeze_dims_92 = None
        unsqueeze_default_103: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_102, 2);  unsqueeze_default_102 = None
        unsqueeze_default_104: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_103, 3);  unsqueeze_default_103 = None
        unsqueeze_default_105: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_89, 0);  squeeze_dims_89 = None
        unsqueeze_default_106: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_105, 2);  unsqueeze_default_105 = None
        unsqueeze_default_107: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_106, 3);  unsqueeze_default_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_108: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_86, 0);  squeeze_dims_86 = None
        unsqueeze_default_109: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_108, 2);  unsqueeze_default_108 = None
        unsqueeze_default_110: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_109, 3);  unsqueeze_default_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_111: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_83, 0);  squeeze_dims_83 = None
        unsqueeze_default_112: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_111, 2);  unsqueeze_default_111 = None
        unsqueeze_default_113: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_112, 3);  unsqueeze_default_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        unsqueeze_default_114: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_dims_80, 0);  squeeze_dims_80 = None
        unsqueeze_default_115: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_114, 2);  unsqueeze_default_114 = None
        unsqueeze_default_116: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_115, 3);  unsqueeze_default_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_117: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(squeeze_dims_75, 0);  squeeze_dims_75 = None
        unsqueeze_default_118: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_117, 2);  unsqueeze_default_117 = None
        unsqueeze_default_119: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_118, 3);  unsqueeze_default_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_120: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_dims_72, 0);  squeeze_dims_72 = None
        unsqueeze_default_121: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_120, 2);  unsqueeze_default_120 = None
        unsqueeze_default_122: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_121, 3);  unsqueeze_default_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_123: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_dims_69, 0);  squeeze_dims_69 = None
        unsqueeze_default_124: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_123, 2);  unsqueeze_default_123 = None
        unsqueeze_default_125: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_124, 3);  unsqueeze_default_124 = None
        unsqueeze_default_126: "f32[1, 60]" = torch.ops.aten.unsqueeze.default(squeeze_dims_64, 0);  squeeze_dims_64 = None
        unsqueeze_default_127: "f32[1, 60, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_126, 2);  unsqueeze_default_126 = None
        unsqueeze_default_128: "f32[1, 60, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_127, 3);  unsqueeze_default_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        unsqueeze_default_129: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(squeeze_dims_61, 0);  squeeze_dims_61 = None
        unsqueeze_default_130: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_129, 2);  unsqueeze_default_129 = None
        unsqueeze_default_131: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_130, 3);  unsqueeze_default_130 = None
        unsqueeze_default_132: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_58, 0);  squeeze_dims_58 = None
        unsqueeze_default_133: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_132, 2);  unsqueeze_default_132 = None
        unsqueeze_default_134: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_133, 3);  unsqueeze_default_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_135: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_dims_55, 0);  squeeze_dims_55 = None
        unsqueeze_default_136: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_135, 2);  unsqueeze_default_135 = None
        unsqueeze_default_137: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_136, 3);  unsqueeze_default_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_138: "f32[1, 20]" = torch.ops.aten.unsqueeze.default(squeeze_dims_52, 0);  squeeze_dims_52 = None
        unsqueeze_default_139: "f32[1, 20, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_138, 2);  unsqueeze_default_138 = None
        unsqueeze_default_140: "f32[1, 20, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_139, 3);  unsqueeze_default_139 = None
        unsqueeze_default_141: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(squeeze_dims_45, 0);  squeeze_dims_45 = None
        unsqueeze_default_142: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_141, 2);  unsqueeze_default_141 = None
        unsqueeze_default_143: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_142, 3);  unsqueeze_default_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_144: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_dims_42, 0);  squeeze_dims_42 = None
        unsqueeze_default_145: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_144, 2);  unsqueeze_default_144 = None
        unsqueeze_default_146: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_145, 3);  unsqueeze_default_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_147: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_dims_39, 0);  squeeze_dims_39 = None
        unsqueeze_default_148: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_147, 2);  unsqueeze_default_147 = None
        unsqueeze_default_149: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_148, 3);  unsqueeze_default_148 = None
        unsqueeze_default_150: "f32[1, 36]" = torch.ops.aten.unsqueeze.default(squeeze_dims_34, 0);  squeeze_dims_34 = None
        unsqueeze_default_151: "f32[1, 36, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_150, 2);  unsqueeze_default_150 = None
        unsqueeze_default_152: "f32[1, 36, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_151, 3);  unsqueeze_default_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:445 in forward, code: x += self.shortcut(shortcut)
        unsqueeze_default_153: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_31, 0);  squeeze_dims_31 = None
        unsqueeze_default_154: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_153, 2);  unsqueeze_default_153 = None
        unsqueeze_default_155: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_154, 3);  unsqueeze_default_154 = None
        unsqueeze_default_156: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_dims_28, 0);  squeeze_dims_28 = None
        unsqueeze_default_157: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_156, 2);  unsqueeze_default_156 = None
        unsqueeze_default_158: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_157, 3);  unsqueeze_default_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_159: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_dims_25, 0);  squeeze_dims_25 = None
        unsqueeze_default_160: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_159, 2);  unsqueeze_default_159 = None
        unsqueeze_default_161: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_160, 3);  unsqueeze_default_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_162: "f32[1, 12]" = torch.ops.aten.unsqueeze.default(squeeze_dims_22, 0);  squeeze_dims_22 = None
        unsqueeze_default_163: "f32[1, 12, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_162, 2);  unsqueeze_default_162 = None
        unsqueeze_default_164: "f32[1, 12, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_163, 3);  unsqueeze_default_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:436 in forward, code: x = self.bn_dw(x)
        unsqueeze_default_165: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(squeeze_dims_19, 0);  squeeze_dims_19 = None
        unsqueeze_default_166: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_165, 2);  unsqueeze_default_165 = None
        unsqueeze_default_167: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_166, 3);  unsqueeze_default_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_168: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze_dims_14, 0);  squeeze_dims_14 = None
        unsqueeze_default_169: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_168, 2);  unsqueeze_default_168 = None
        unsqueeze_default_170: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_169, 3);  unsqueeze_default_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:69 in forward, code: x2 = self.cheap_operation(x1)
        unsqueeze_default_171: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_dims_11, 0);  squeeze_dims_11 = None
        unsqueeze_default_172: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_171, 2);  unsqueeze_default_171 = None
        unsqueeze_default_173: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_172, 3);  unsqueeze_default_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:68 in forward, code: x1 = self.primary_conv(x)
        unsqueeze_default_174: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_dims_8, 0);  squeeze_dims_8 = None
        unsqueeze_default_175: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_174, 2);  unsqueeze_default_174 = None
        unsqueeze_default_176: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_175, 3);  unsqueeze_default_175 = None
        unsqueeze_default_177: "f32[1, 8]" = torch.ops.aten.unsqueeze.default(squeeze_dims_3, 0);  squeeze_dims_3 = None
        unsqueeze_default_178: "f32[1, 8, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_177, 2);  unsqueeze_default_177 = None
        unsqueeze_default_179: "f32[1, 8, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_178, 3);  unsqueeze_default_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/ghostnet.py:822 in forward_features, code: x = self.bn1(x)
        unsqueeze_default_180: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(squeeze_dims, 0);  squeeze_dims = None
        unsqueeze_default_181: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_180, 2);  unsqueeze_default_180 = None
        unsqueeze_default_182: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_default_181, 3);  unsqueeze_default_181 = None

        # No stacktrace found for following nodes
        copy__default: "i64[]" = torch.ops.aten.copy_.default(primals_3, add_tensor);  primals_3 = add_tensor = None
        copy__default_1: "f32[16]" = torch.ops.aten.copy_.default(primals_4, add_tensor_1);  primals_4 = add_tensor_1 = None
        copy__default_2: "f32[16]" = torch.ops.aten.copy_.default(primals_5, add_tensor_2);  primals_5 = add_tensor_2 = None
        copy__default_3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_tensor_3);  primals_9 = add_tensor_3 = None
        copy__default_4: "f32[8]" = torch.ops.aten.copy_.default(primals_10, add_tensor_4);  primals_10 = add_tensor_4 = None
        copy__default_5: "f32[8]" = torch.ops.aten.copy_.default(primals_11, add_tensor_5);  primals_11 = add_tensor_5 = None
        copy__default_6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_tensor_6);  primals_15 = add_tensor_6 = None
        copy__default_7: "f32[8]" = torch.ops.aten.copy_.default(primals_16, add_tensor_7);  primals_16 = add_tensor_7 = None
        copy__default_8: "f32[8]" = torch.ops.aten.copy_.default(primals_17, add_tensor_8);  primals_17 = add_tensor_8 = None
        copy__default_9: "i64[]" = torch.ops.aten.copy_.default(primals_21, add_tensor_9);  primals_21 = add_tensor_9 = None
        copy__default_10: "f32[8]" = torch.ops.aten.copy_.default(primals_22, add_tensor_10);  primals_22 = add_tensor_10 = None
        copy__default_11: "f32[8]" = torch.ops.aten.copy_.default(primals_23, add_tensor_11);  primals_23 = add_tensor_11 = None
        copy__default_12: "i64[]" = torch.ops.aten.copy_.default(primals_27, add_tensor_12);  primals_27 = add_tensor_12 = None
        copy__default_13: "f32[8]" = torch.ops.aten.copy_.default(primals_28, add_tensor_13);  primals_28 = add_tensor_13 = None
        copy__default_14: "f32[8]" = torch.ops.aten.copy_.default(primals_29, add_tensor_14);  primals_29 = add_tensor_14 = None
        copy__default_15: "i64[]" = torch.ops.aten.copy_.default(primals_33, add_tensor_15);  primals_33 = add_tensor_15 = None
        copy__default_16: "f32[24]" = torch.ops.aten.copy_.default(primals_34, add_tensor_16);  primals_34 = add_tensor_16 = None
        copy__default_17: "f32[24]" = torch.ops.aten.copy_.default(primals_35, add_tensor_17);  primals_35 = add_tensor_17 = None
        copy__default_18: "i64[]" = torch.ops.aten.copy_.default(primals_39, add_tensor_18);  primals_39 = add_tensor_18 = None
        copy__default_19: "f32[24]" = torch.ops.aten.copy_.default(primals_40, add_tensor_19);  primals_40 = add_tensor_19 = None
        copy__default_20: "f32[24]" = torch.ops.aten.copy_.default(primals_41, add_tensor_20);  primals_41 = add_tensor_20 = None
        copy__default_21: "i64[]" = torch.ops.aten.copy_.default(primals_45, add_tensor_21);  primals_45 = add_tensor_21 = None
        copy__default_22: "f32[48]" = torch.ops.aten.copy_.default(primals_46, add_tensor_22);  primals_46 = add_tensor_22 = None
        copy__default_23: "f32[48]" = torch.ops.aten.copy_.default(primals_47, add_tensor_23);  primals_47 = add_tensor_23 = None
        copy__default_24: "i64[]" = torch.ops.aten.copy_.default(primals_51, add_tensor_24);  primals_51 = add_tensor_24 = None
        copy__default_25: "f32[12]" = torch.ops.aten.copy_.default(primals_52, add_tensor_25);  primals_52 = add_tensor_25 = None
        copy__default_26: "f32[12]" = torch.ops.aten.copy_.default(primals_53, add_tensor_26);  primals_53 = add_tensor_26 = None
        copy__default_27: "i64[]" = torch.ops.aten.copy_.default(primals_57, add_tensor_27);  primals_57 = add_tensor_27 = None
        copy__default_28: "f32[12]" = torch.ops.aten.copy_.default(primals_58, add_tensor_28);  primals_58 = add_tensor_28 = None
        copy__default_29: "f32[12]" = torch.ops.aten.copy_.default(primals_59, add_tensor_29);  primals_59 = add_tensor_29 = None
        copy__default_30: "i64[]" = torch.ops.aten.copy_.default(primals_63, add_tensor_30);  primals_63 = add_tensor_30 = None
        copy__default_31: "f32[16]" = torch.ops.aten.copy_.default(primals_64, add_tensor_31);  primals_64 = add_tensor_31 = None
        copy__default_32: "f32[16]" = torch.ops.aten.copy_.default(primals_65, add_tensor_32);  primals_65 = add_tensor_32 = None
        copy__default_33: "i64[]" = torch.ops.aten.copy_.default(primals_69, add_tensor_33);  primals_69 = add_tensor_33 = None
        copy__default_34: "f32[24]" = torch.ops.aten.copy_.default(primals_70, add_tensor_34);  primals_70 = add_tensor_34 = None
        copy__default_35: "f32[24]" = torch.ops.aten.copy_.default(primals_71, add_tensor_35);  primals_71 = add_tensor_35 = None
        copy__default_36: "i64[]" = torch.ops.aten.copy_.default(primals_75, add_tensor_36);  primals_75 = add_tensor_36 = None
        copy__default_37: "f32[36]" = torch.ops.aten.copy_.default(primals_76, add_tensor_37);  primals_76 = add_tensor_37 = None
        copy__default_38: "f32[36]" = torch.ops.aten.copy_.default(primals_77, add_tensor_38);  primals_77 = add_tensor_38 = None
        copy__default_39: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_tensor_39);  primals_81 = add_tensor_39 = None
        copy__default_40: "f32[36]" = torch.ops.aten.copy_.default(primals_82, add_tensor_40);  primals_82 = add_tensor_40 = None
        copy__default_41: "f32[36]" = torch.ops.aten.copy_.default(primals_83, add_tensor_41);  primals_83 = add_tensor_41 = None
        copy__default_42: "i64[]" = torch.ops.aten.copy_.default(primals_87, add_tensor_42);  primals_87 = add_tensor_42 = None
        copy__default_43: "f32[12]" = torch.ops.aten.copy_.default(primals_88, add_tensor_43);  primals_88 = add_tensor_43 = None
        copy__default_44: "f32[12]" = torch.ops.aten.copy_.default(primals_89, add_tensor_44);  primals_89 = add_tensor_44 = None
        copy__default_45: "i64[]" = torch.ops.aten.copy_.default(primals_93, add_tensor_45);  primals_93 = add_tensor_45 = None
        copy__default_46: "f32[12]" = torch.ops.aten.copy_.default(primals_94, add_tensor_46);  primals_94 = add_tensor_46 = None
        copy__default_47: "f32[12]" = torch.ops.aten.copy_.default(primals_95, add_tensor_47);  primals_95 = add_tensor_47 = None
        copy__default_48: "i64[]" = torch.ops.aten.copy_.default(primals_99, add_tensor_48);  primals_99 = add_tensor_48 = None
        copy__default_49: "f32[36]" = torch.ops.aten.copy_.default(primals_100, add_tensor_49);  primals_100 = add_tensor_49 = None
        copy__default_50: "f32[36]" = torch.ops.aten.copy_.default(primals_101, add_tensor_50);  primals_101 = add_tensor_50 = None
        copy__default_51: "i64[]" = torch.ops.aten.copy_.default(primals_105, add_tensor_51);  primals_105 = add_tensor_51 = None
        copy__default_52: "f32[36]" = torch.ops.aten.copy_.default(primals_106, add_tensor_52);  primals_106 = add_tensor_52 = None
        copy__default_53: "f32[36]" = torch.ops.aten.copy_.default(primals_107, add_tensor_53);  primals_107 = add_tensor_53 = None
        copy__default_54: "i64[]" = torch.ops.aten.copy_.default(primals_111, add_tensor_54);  primals_111 = add_tensor_54 = None
        copy__default_55: "f32[72]" = torch.ops.aten.copy_.default(primals_112, add_tensor_55);  primals_112 = add_tensor_55 = None
        copy__default_56: "f32[72]" = torch.ops.aten.copy_.default(primals_113, add_tensor_56);  primals_113 = add_tensor_56 = None
        copy__default_57: "i64[]" = torch.ops.aten.copy_.default(primals_121, add_tensor_57);  primals_121 = add_tensor_57 = None
        copy__default_58: "f32[20]" = torch.ops.aten.copy_.default(primals_122, add_tensor_58);  primals_122 = add_tensor_58 = None
        copy__default_59: "f32[20]" = torch.ops.aten.copy_.default(primals_123, add_tensor_59);  primals_123 = add_tensor_59 = None
        copy__default_60: "i64[]" = torch.ops.aten.copy_.default(primals_127, add_tensor_60);  primals_127 = add_tensor_60 = None
        copy__default_61: "f32[20]" = torch.ops.aten.copy_.default(primals_128, add_tensor_61);  primals_128 = add_tensor_61 = None
        copy__default_62: "f32[20]" = torch.ops.aten.copy_.default(primals_129, add_tensor_62);  primals_129 = add_tensor_62 = None
        copy__default_63: "i64[]" = torch.ops.aten.copy_.default(primals_133, add_tensor_63);  primals_133 = add_tensor_63 = None
        copy__default_64: "f32[24]" = torch.ops.aten.copy_.default(primals_134, add_tensor_64);  primals_134 = add_tensor_64 = None
        copy__default_65: "f32[24]" = torch.ops.aten.copy_.default(primals_135, add_tensor_65);  primals_135 = add_tensor_65 = None
        copy__default_66: "i64[]" = torch.ops.aten.copy_.default(primals_139, add_tensor_66);  primals_139 = add_tensor_66 = None
        copy__default_67: "f32[40]" = torch.ops.aten.copy_.default(primals_140, add_tensor_67);  primals_140 = add_tensor_67 = None
        copy__default_68: "f32[40]" = torch.ops.aten.copy_.default(primals_141, add_tensor_68);  primals_141 = add_tensor_68 = None
        copy__default_69: "i64[]" = torch.ops.aten.copy_.default(primals_145, add_tensor_69);  primals_145 = add_tensor_69 = None
        copy__default_70: "f32[60]" = torch.ops.aten.copy_.default(primals_146, add_tensor_70);  primals_146 = add_tensor_70 = None
        copy__default_71: "f32[60]" = torch.ops.aten.copy_.default(primals_147, add_tensor_71);  primals_147 = add_tensor_71 = None
        copy__default_72: "i64[]" = torch.ops.aten.copy_.default(primals_151, add_tensor_72);  primals_151 = add_tensor_72 = None
        copy__default_73: "f32[60]" = torch.ops.aten.copy_.default(primals_152, add_tensor_73);  primals_152 = add_tensor_73 = None
        copy__default_74: "f32[60]" = torch.ops.aten.copy_.default(primals_153, add_tensor_74);  primals_153 = add_tensor_74 = None
        copy__default_75: "i64[]" = torch.ops.aten.copy_.default(primals_161, add_tensor_75);  primals_161 = add_tensor_75 = None
        copy__default_76: "f32[20]" = torch.ops.aten.copy_.default(primals_162, add_tensor_76);  primals_162 = add_tensor_76 = None
        copy__default_77: "f32[20]" = torch.ops.aten.copy_.default(primals_163, add_tensor_77);  primals_163 = add_tensor_77 = None
        copy__default_78: "i64[]" = torch.ops.aten.copy_.default(primals_167, add_tensor_78);  primals_167 = add_tensor_78 = None
        copy__default_79: "f32[20]" = torch.ops.aten.copy_.default(primals_168, add_tensor_79);  primals_168 = add_tensor_79 = None
        copy__default_80: "f32[20]" = torch.ops.aten.copy_.default(primals_169, add_tensor_80);  primals_169 = add_tensor_80 = None
        copy__default_81: "i64[]" = torch.ops.aten.copy_.default(primals_173, add_tensor_81);  primals_173 = add_tensor_81 = None
        copy__default_82: "f32[120]" = torch.ops.aten.copy_.default(primals_174, add_tensor_82);  primals_174 = add_tensor_82 = None
        copy__default_83: "f32[120]" = torch.ops.aten.copy_.default(primals_175, add_tensor_83);  primals_175 = add_tensor_83 = None
        copy__default_84: "i64[]" = torch.ops.aten.copy_.default(primals_179, add_tensor_84);  primals_179 = add_tensor_84 = None
        copy__default_85: "f32[120]" = torch.ops.aten.copy_.default(primals_180, add_tensor_85);  primals_180 = add_tensor_85 = None
        copy__default_86: "f32[120]" = torch.ops.aten.copy_.default(primals_181, add_tensor_86);  primals_181 = add_tensor_86 = None
        copy__default_87: "i64[]" = torch.ops.aten.copy_.default(primals_185, add_tensor_87);  primals_185 = add_tensor_87 = None
        copy__default_88: "f32[240]" = torch.ops.aten.copy_.default(primals_186, add_tensor_88);  primals_186 = add_tensor_88 = None
        copy__default_89: "f32[240]" = torch.ops.aten.copy_.default(primals_187, add_tensor_89);  primals_187 = add_tensor_89 = None
        copy__default_90: "i64[]" = torch.ops.aten.copy_.default(primals_191, add_tensor_90);  primals_191 = add_tensor_90 = None
        copy__default_91: "f32[40]" = torch.ops.aten.copy_.default(primals_192, add_tensor_91);  primals_192 = add_tensor_91 = None
        copy__default_92: "f32[40]" = torch.ops.aten.copy_.default(primals_193, add_tensor_92);  primals_193 = add_tensor_92 = None
        copy__default_93: "i64[]" = torch.ops.aten.copy_.default(primals_197, add_tensor_93);  primals_197 = add_tensor_93 = None
        copy__default_94: "f32[40]" = torch.ops.aten.copy_.default(primals_198, add_tensor_94);  primals_198 = add_tensor_94 = None
        copy__default_95: "f32[40]" = torch.ops.aten.copy_.default(primals_199, add_tensor_95);  primals_199 = add_tensor_95 = None
        copy__default_96: "i64[]" = torch.ops.aten.copy_.default(primals_203, add_tensor_96);  primals_203 = add_tensor_96 = None
        copy__default_97: "f32[40]" = torch.ops.aten.copy_.default(primals_204, add_tensor_97);  primals_204 = add_tensor_97 = None
        copy__default_98: "f32[40]" = torch.ops.aten.copy_.default(primals_205, add_tensor_98);  primals_205 = add_tensor_98 = None
        copy__default_99: "i64[]" = torch.ops.aten.copy_.default(primals_209, add_tensor_99);  primals_209 = add_tensor_99 = None
        copy__default_100: "f32[80]" = torch.ops.aten.copy_.default(primals_210, add_tensor_100);  primals_210 = add_tensor_100 = None
        copy__default_101: "f32[80]" = torch.ops.aten.copy_.default(primals_211, add_tensor_101);  primals_211 = add_tensor_101 = None
        copy__default_102: "i64[]" = torch.ops.aten.copy_.default(primals_215, add_tensor_102);  primals_215 = add_tensor_102 = None
        copy__default_103: "f32[100]" = torch.ops.aten.copy_.default(primals_216, add_tensor_103);  primals_216 = add_tensor_103 = None
        copy__default_104: "f32[100]" = torch.ops.aten.copy_.default(primals_217, add_tensor_104);  primals_217 = add_tensor_104 = None
        copy__default_105: "i64[]" = torch.ops.aten.copy_.default(primals_221, add_tensor_105);  primals_221 = add_tensor_105 = None
        copy__default_106: "f32[100]" = torch.ops.aten.copy_.default(primals_222, add_tensor_106);  primals_222 = add_tensor_106 = None
        copy__default_107: "f32[100]" = torch.ops.aten.copy_.default(primals_223, add_tensor_107);  primals_223 = add_tensor_107 = None
        copy__default_108: "i64[]" = torch.ops.aten.copy_.default(primals_227, add_tensor_108);  primals_227 = add_tensor_108 = None
        copy__default_109: "f32[40]" = torch.ops.aten.copy_.default(primals_228, add_tensor_109);  primals_228 = add_tensor_109 = None
        copy__default_110: "f32[40]" = torch.ops.aten.copy_.default(primals_229, add_tensor_110);  primals_229 = add_tensor_110 = None
        copy__default_111: "i64[]" = torch.ops.aten.copy_.default(primals_233, add_tensor_111);  primals_233 = add_tensor_111 = None
        copy__default_112: "f32[40]" = torch.ops.aten.copy_.default(primals_234, add_tensor_112);  primals_234 = add_tensor_112 = None
        copy__default_113: "f32[40]" = torch.ops.aten.copy_.default(primals_235, add_tensor_113);  primals_235 = add_tensor_113 = None
        copy__default_114: "i64[]" = torch.ops.aten.copy_.default(primals_239, add_tensor_114);  primals_239 = add_tensor_114 = None
        copy__default_115: "f32[92]" = torch.ops.aten.copy_.default(primals_240, add_tensor_115);  primals_240 = add_tensor_115 = None
        copy__default_116: "f32[92]" = torch.ops.aten.copy_.default(primals_241, add_tensor_116);  primals_241 = add_tensor_116 = None
        copy__default_117: "i64[]" = torch.ops.aten.copy_.default(primals_245, add_tensor_117);  primals_245 = add_tensor_117 = None
        copy__default_118: "f32[92]" = torch.ops.aten.copy_.default(primals_246, add_tensor_118);  primals_246 = add_tensor_118 = None
        copy__default_119: "f32[92]" = torch.ops.aten.copy_.default(primals_247, add_tensor_119);  primals_247 = add_tensor_119 = None
        copy__default_120: "i64[]" = torch.ops.aten.copy_.default(primals_251, add_tensor_120);  primals_251 = add_tensor_120 = None
        copy__default_121: "f32[40]" = torch.ops.aten.copy_.default(primals_252, add_tensor_121);  primals_252 = add_tensor_121 = None
        copy__default_122: "f32[40]" = torch.ops.aten.copy_.default(primals_253, add_tensor_122);  primals_253 = add_tensor_122 = None
        copy__default_123: "i64[]" = torch.ops.aten.copy_.default(primals_257, add_tensor_123);  primals_257 = add_tensor_123 = None
        copy__default_124: "f32[40]" = torch.ops.aten.copy_.default(primals_258, add_tensor_124);  primals_258 = add_tensor_124 = None
        copy__default_125: "f32[40]" = torch.ops.aten.copy_.default(primals_259, add_tensor_125);  primals_259 = add_tensor_125 = None
        copy__default_126: "i64[]" = torch.ops.aten.copy_.default(primals_263, add_tensor_126);  primals_263 = add_tensor_126 = None
        copy__default_127: "f32[92]" = torch.ops.aten.copy_.default(primals_264, add_tensor_127);  primals_264 = add_tensor_127 = None
        copy__default_128: "f32[92]" = torch.ops.aten.copy_.default(primals_265, add_tensor_128);  primals_265 = add_tensor_128 = None
        copy__default_129: "i64[]" = torch.ops.aten.copy_.default(primals_269, add_tensor_129);  primals_269 = add_tensor_129 = None
        copy__default_130: "f32[92]" = torch.ops.aten.copy_.default(primals_270, add_tensor_130);  primals_270 = add_tensor_130 = None
        copy__default_131: "f32[92]" = torch.ops.aten.copy_.default(primals_271, add_tensor_131);  primals_271 = add_tensor_131 = None
        copy__default_132: "i64[]" = torch.ops.aten.copy_.default(primals_275, add_tensor_132);  primals_275 = add_tensor_132 = None
        copy__default_133: "f32[40]" = torch.ops.aten.copy_.default(primals_276, add_tensor_133);  primals_276 = add_tensor_133 = None
        copy__default_134: "f32[40]" = torch.ops.aten.copy_.default(primals_277, add_tensor_134);  primals_277 = add_tensor_134 = None
        copy__default_135: "i64[]" = torch.ops.aten.copy_.default(primals_281, add_tensor_135);  primals_281 = add_tensor_135 = None
        copy__default_136: "f32[40]" = torch.ops.aten.copy_.default(primals_282, add_tensor_136);  primals_282 = add_tensor_136 = None
        copy__default_137: "f32[40]" = torch.ops.aten.copy_.default(primals_283, add_tensor_137);  primals_283 = add_tensor_137 = None
        copy__default_138: "i64[]" = torch.ops.aten.copy_.default(primals_287, add_tensor_138);  primals_287 = add_tensor_138 = None
        copy__default_139: "f32[240]" = torch.ops.aten.copy_.default(primals_288, add_tensor_139);  primals_288 = add_tensor_139 = None
        copy__default_140: "f32[240]" = torch.ops.aten.copy_.default(primals_289, add_tensor_140);  primals_289 = add_tensor_140 = None
        copy__default_141: "i64[]" = torch.ops.aten.copy_.default(primals_293, add_tensor_141);  primals_293 = add_tensor_141 = None
        copy__default_142: "f32[240]" = torch.ops.aten.copy_.default(primals_294, add_tensor_142);  primals_294 = add_tensor_142 = None
        copy__default_143: "f32[240]" = torch.ops.aten.copy_.default(primals_295, add_tensor_143);  primals_295 = add_tensor_143 = None
        copy__default_144: "i64[]" = torch.ops.aten.copy_.default(primals_303, add_tensor_144);  primals_303 = add_tensor_144 = None
        copy__default_145: "f32[56]" = torch.ops.aten.copy_.default(primals_304, add_tensor_145);  primals_304 = add_tensor_145 = None
        copy__default_146: "f32[56]" = torch.ops.aten.copy_.default(primals_305, add_tensor_146);  primals_305 = add_tensor_146 = None
        copy__default_147: "i64[]" = torch.ops.aten.copy_.default(primals_309, add_tensor_147);  primals_309 = add_tensor_147 = None
        copy__default_148: "f32[56]" = torch.ops.aten.copy_.default(primals_310, add_tensor_148);  primals_310 = add_tensor_148 = None
        copy__default_149: "f32[56]" = torch.ops.aten.copy_.default(primals_311, add_tensor_149);  primals_311 = add_tensor_149 = None
        copy__default_150: "i64[]" = torch.ops.aten.copy_.default(primals_315, add_tensor_150);  primals_315 = add_tensor_150 = None
        copy__default_151: "f32[80]" = torch.ops.aten.copy_.default(primals_316, add_tensor_151);  primals_316 = add_tensor_151 = None
        copy__default_152: "f32[80]" = torch.ops.aten.copy_.default(primals_317, add_tensor_152);  primals_317 = add_tensor_152 = None
        copy__default_153: "i64[]" = torch.ops.aten.copy_.default(primals_321, add_tensor_153);  primals_321 = add_tensor_153 = None
        copy__default_154: "f32[112]" = torch.ops.aten.copy_.default(primals_322, add_tensor_154);  primals_322 = add_tensor_154 = None
        copy__default_155: "f32[112]" = torch.ops.aten.copy_.default(primals_323, add_tensor_155);  primals_323 = add_tensor_155 = None
        copy__default_156: "i64[]" = torch.ops.aten.copy_.default(primals_327, add_tensor_156);  primals_327 = add_tensor_156 = None
        copy__default_157: "f32[336]" = torch.ops.aten.copy_.default(primals_328, add_tensor_157);  primals_328 = add_tensor_157 = None
        copy__default_158: "f32[336]" = torch.ops.aten.copy_.default(primals_329, add_tensor_158);  primals_329 = add_tensor_158 = None
        copy__default_159: "i64[]" = torch.ops.aten.copy_.default(primals_333, add_tensor_159);  primals_333 = add_tensor_159 = None
        copy__default_160: "f32[336]" = torch.ops.aten.copy_.default(primals_334, add_tensor_160);  primals_334 = add_tensor_160 = None
        copy__default_161: "f32[336]" = torch.ops.aten.copy_.default(primals_335, add_tensor_161);  primals_335 = add_tensor_161 = None
        copy__default_162: "i64[]" = torch.ops.aten.copy_.default(primals_343, add_tensor_162);  primals_343 = add_tensor_162 = None
        copy__default_163: "f32[56]" = torch.ops.aten.copy_.default(primals_344, add_tensor_163);  primals_344 = add_tensor_163 = None
        copy__default_164: "f32[56]" = torch.ops.aten.copy_.default(primals_345, add_tensor_164);  primals_345 = add_tensor_164 = None
        copy__default_165: "i64[]" = torch.ops.aten.copy_.default(primals_349, add_tensor_165);  primals_349 = add_tensor_165 = None
        copy__default_166: "f32[56]" = torch.ops.aten.copy_.default(primals_350, add_tensor_166);  primals_350 = add_tensor_166 = None
        copy__default_167: "f32[56]" = torch.ops.aten.copy_.default(primals_351, add_tensor_167);  primals_351 = add_tensor_167 = None
        copy__default_168: "i64[]" = torch.ops.aten.copy_.default(primals_355, add_tensor_168);  primals_355 = add_tensor_168 = None
        copy__default_169: "f32[336]" = torch.ops.aten.copy_.default(primals_356, add_tensor_169);  primals_356 = add_tensor_169 = None
        copy__default_170: "f32[336]" = torch.ops.aten.copy_.default(primals_357, add_tensor_170);  primals_357 = add_tensor_170 = None
        copy__default_171: "i64[]" = torch.ops.aten.copy_.default(primals_361, add_tensor_171);  primals_361 = add_tensor_171 = None
        copy__default_172: "f32[336]" = torch.ops.aten.copy_.default(primals_362, add_tensor_172);  primals_362 = add_tensor_172 = None
        copy__default_173: "f32[336]" = torch.ops.aten.copy_.default(primals_363, add_tensor_173);  primals_363 = add_tensor_173 = None
        copy__default_174: "i64[]" = torch.ops.aten.copy_.default(primals_367, add_tensor_174);  primals_367 = add_tensor_174 = None
        copy__default_175: "f32[672]" = torch.ops.aten.copy_.default(primals_368, add_tensor_175);  primals_368 = add_tensor_175 = None
        copy__default_176: "f32[672]" = torch.ops.aten.copy_.default(primals_369, add_tensor_176);  primals_369 = add_tensor_176 = None
        copy__default_177: "i64[]" = torch.ops.aten.copy_.default(primals_377, add_tensor_177);  primals_377 = add_tensor_177 = None
        copy__default_178: "f32[80]" = torch.ops.aten.copy_.default(primals_378, add_tensor_178);  primals_378 = add_tensor_178 = None
        copy__default_179: "f32[80]" = torch.ops.aten.copy_.default(primals_379, add_tensor_179);  primals_379 = add_tensor_179 = None
        copy__default_180: "i64[]" = torch.ops.aten.copy_.default(primals_383, add_tensor_180);  primals_383 = add_tensor_180 = None
        copy__default_181: "f32[80]" = torch.ops.aten.copy_.default(primals_384, add_tensor_181);  primals_384 = add_tensor_181 = None
        copy__default_182: "f32[80]" = torch.ops.aten.copy_.default(primals_385, add_tensor_182);  primals_385 = add_tensor_182 = None
        copy__default_183: "i64[]" = torch.ops.aten.copy_.default(primals_389, add_tensor_183);  primals_389 = add_tensor_183 = None
        copy__default_184: "f32[112]" = torch.ops.aten.copy_.default(primals_390, add_tensor_184);  primals_390 = add_tensor_184 = None
        copy__default_185: "f32[112]" = torch.ops.aten.copy_.default(primals_391, add_tensor_185);  primals_391 = add_tensor_185 = None
        copy__default_186: "i64[]" = torch.ops.aten.copy_.default(primals_395, add_tensor_186);  primals_395 = add_tensor_186 = None
        copy__default_187: "f32[160]" = torch.ops.aten.copy_.default(primals_396, add_tensor_187);  primals_396 = add_tensor_187 = None
        copy__default_188: "f32[160]" = torch.ops.aten.copy_.default(primals_397, add_tensor_188);  primals_397 = add_tensor_188 = None
        copy__default_189: "i64[]" = torch.ops.aten.copy_.default(primals_401, add_tensor_189);  primals_401 = add_tensor_189 = None
        copy__default_190: "f32[480]" = torch.ops.aten.copy_.default(primals_402, add_tensor_190);  primals_402 = add_tensor_190 = None
        copy__default_191: "f32[480]" = torch.ops.aten.copy_.default(primals_403, add_tensor_191);  primals_403 = add_tensor_191 = None
        copy__default_192: "i64[]" = torch.ops.aten.copy_.default(primals_407, add_tensor_192);  primals_407 = add_tensor_192 = None
        copy__default_193: "f32[480]" = torch.ops.aten.copy_.default(primals_408, add_tensor_193);  primals_408 = add_tensor_193 = None
        copy__default_194: "f32[480]" = torch.ops.aten.copy_.default(primals_409, add_tensor_194);  primals_409 = add_tensor_194 = None
        copy__default_195: "i64[]" = torch.ops.aten.copy_.default(primals_413, add_tensor_195);  primals_413 = add_tensor_195 = None
        copy__default_196: "f32[80]" = torch.ops.aten.copy_.default(primals_414, add_tensor_196);  primals_414 = add_tensor_196 = None
        copy__default_197: "f32[80]" = torch.ops.aten.copy_.default(primals_415, add_tensor_197);  primals_415 = add_tensor_197 = None
        copy__default_198: "i64[]" = torch.ops.aten.copy_.default(primals_419, add_tensor_198);  primals_419 = add_tensor_198 = None
        copy__default_199: "f32[80]" = torch.ops.aten.copy_.default(primals_420, add_tensor_199);  primals_420 = add_tensor_199 = None
        copy__default_200: "f32[80]" = torch.ops.aten.copy_.default(primals_421, add_tensor_200);  primals_421 = add_tensor_200 = None
        copy__default_201: "i64[]" = torch.ops.aten.copy_.default(primals_425, add_tensor_201);  primals_425 = add_tensor_201 = None
        copy__default_202: "f32[480]" = torch.ops.aten.copy_.default(primals_426, add_tensor_202);  primals_426 = add_tensor_202 = None
        copy__default_203: "f32[480]" = torch.ops.aten.copy_.default(primals_427, add_tensor_203);  primals_427 = add_tensor_203 = None
        copy__default_204: "i64[]" = torch.ops.aten.copy_.default(primals_431, add_tensor_204);  primals_431 = add_tensor_204 = None
        copy__default_205: "f32[480]" = torch.ops.aten.copy_.default(primals_432, add_tensor_205);  primals_432 = add_tensor_205 = None
        copy__default_206: "f32[480]" = torch.ops.aten.copy_.default(primals_433, add_tensor_206);  primals_433 = add_tensor_206 = None
        copy__default_207: "i64[]" = torch.ops.aten.copy_.default(primals_441, add_tensor_207);  primals_441 = add_tensor_207 = None
        copy__default_208: "f32[80]" = torch.ops.aten.copy_.default(primals_442, add_tensor_208);  primals_442 = add_tensor_208 = None
        copy__default_209: "f32[80]" = torch.ops.aten.copy_.default(primals_443, add_tensor_209);  primals_443 = add_tensor_209 = None
        copy__default_210: "i64[]" = torch.ops.aten.copy_.default(primals_447, add_tensor_210);  primals_447 = add_tensor_210 = None
        copy__default_211: "f32[80]" = torch.ops.aten.copy_.default(primals_448, add_tensor_211);  primals_448 = add_tensor_211 = None
        copy__default_212: "f32[80]" = torch.ops.aten.copy_.default(primals_449, add_tensor_212);  primals_449 = add_tensor_212 = None
        copy__default_213: "i64[]" = torch.ops.aten.copy_.default(primals_453, add_tensor_213);  primals_453 = add_tensor_213 = None
        copy__default_214: "f32[480]" = torch.ops.aten.copy_.default(primals_454, add_tensor_214);  primals_454 = add_tensor_214 = None
        copy__default_215: "f32[480]" = torch.ops.aten.copy_.default(primals_455, add_tensor_215);  primals_455 = add_tensor_215 = None
        copy__default_216: "i64[]" = torch.ops.aten.copy_.default(primals_459, add_tensor_216);  primals_459 = add_tensor_216 = None
        copy__default_217: "f32[480]" = torch.ops.aten.copy_.default(primals_460, add_tensor_217);  primals_460 = add_tensor_217 = None
        copy__default_218: "f32[480]" = torch.ops.aten.copy_.default(primals_461, add_tensor_218);  primals_461 = add_tensor_218 = None
        copy__default_219: "i64[]" = torch.ops.aten.copy_.default(primals_465, add_tensor_219);  primals_465 = add_tensor_219 = None
        copy__default_220: "f32[80]" = torch.ops.aten.copy_.default(primals_466, add_tensor_220);  primals_466 = add_tensor_220 = None
        copy__default_221: "f32[80]" = torch.ops.aten.copy_.default(primals_467, add_tensor_221);  primals_467 = add_tensor_221 = None
        copy__default_222: "i64[]" = torch.ops.aten.copy_.default(primals_471, add_tensor_222);  primals_471 = add_tensor_222 = None
        copy__default_223: "f32[80]" = torch.ops.aten.copy_.default(primals_472, add_tensor_223);  primals_472 = add_tensor_223 = None
        copy__default_224: "f32[80]" = torch.ops.aten.copy_.default(primals_473, add_tensor_224);  primals_473 = add_tensor_224 = None
        copy__default_225: "i64[]" = torch.ops.aten.copy_.default(primals_477, add_tensor_225);  primals_477 = add_tensor_225 = None
        copy__default_226: "f32[480]" = torch.ops.aten.copy_.default(primals_478, add_tensor_226);  primals_478 = add_tensor_226 = None
        copy__default_227: "f32[480]" = torch.ops.aten.copy_.default(primals_479, add_tensor_227);  primals_479 = add_tensor_227 = None
        copy__default_228: "i64[]" = torch.ops.aten.copy_.default(primals_483, add_tensor_228);  primals_483 = add_tensor_228 = None
        copy__default_229: "f32[480]" = torch.ops.aten.copy_.default(primals_484, add_tensor_229);  primals_484 = add_tensor_229 = None
        copy__default_230: "f32[480]" = torch.ops.aten.copy_.default(primals_485, add_tensor_230);  primals_485 = add_tensor_230 = None
        copy__default_231: "i64[]" = torch.ops.aten.copy_.default(primals_493, add_tensor_231);  primals_493 = add_tensor_231 = None
        copy__default_232: "f32[80]" = torch.ops.aten.copy_.default(primals_494, add_tensor_232);  primals_494 = add_tensor_232 = None
        copy__default_233: "f32[80]" = torch.ops.aten.copy_.default(primals_495, add_tensor_233);  primals_495 = add_tensor_233 = None
        copy__default_234: "i64[]" = torch.ops.aten.copy_.default(primals_499, add_tensor_234);  primals_499 = add_tensor_234 = None
        copy__default_235: "f32[80]" = torch.ops.aten.copy_.default(primals_500, add_tensor_235);  primals_500 = add_tensor_235 = None
        copy__default_236: "f32[80]" = torch.ops.aten.copy_.default(primals_501, add_tensor_236);  primals_501 = add_tensor_236 = None
        copy__default_237: "i64[]" = torch.ops.aten.copy_.default(primals_505, add_tensor_237);  primals_505 = add_tensor_237 = None
        copy__default_238: "f32[960]" = torch.ops.aten.copy_.default(primals_506, add_tensor_238);  primals_506 = add_tensor_238 = None
        copy__default_239: "f32[960]" = torch.ops.aten.copy_.default(primals_507, add_tensor_239);  primals_507 = add_tensor_239 = None
        return (squeeze_dims_1, squeeze_dims_4, squeeze_dims_9, squeeze_dims_12, squeeze_dims_15, squeeze_dims_20, squeeze_dims_23, squeeze_dims_26, squeeze_dims_29, squeeze_dims_32, squeeze_dims_35, squeeze_dims_40, squeeze_dims_43, squeeze_dims_46, squeeze_dims_53, squeeze_dims_56, squeeze_dims_59, squeeze_dims_62, squeeze_dims_65, squeeze_dims_70, squeeze_dims_73, squeeze_dims_76, squeeze_dims_81, squeeze_dims_84, squeeze_dims_87, squeeze_dims_90, squeeze_dims_93, squeeze_dims_96, squeeze_dims_101, squeeze_dims_104, squeeze_dims_107, squeeze_dims_112, squeeze_dims_115, squeeze_dims_118, squeeze_dims_123, squeeze_dims_126, squeeze_dims_129, squeeze_dims_134, squeeze_dims_137, squeeze_dims_140, squeeze_dims_143, squeeze_dims_146, squeeze_dims_151, squeeze_dims_154, squeeze_dims_157, squeeze_dims_164, squeeze_dims_167, squeeze_dims_170, squeeze_dims_173, squeeze_dims_176, squeeze_dims_181, squeeze_dims_184, squeeze_dims_187, squeeze_dims_192, squeeze_dims_195, squeeze_dims_198, squeeze_dims_203, squeeze_dims_206, squeeze_dims_209, squeeze_dims_214, squeeze_dims_217, mul_tensor_401, permute_default, le_scalar, unsqueeze_default_2, unsqueeze_default_5, unsqueeze_default_8, unsqueeze_default_11, unsqueeze_default_14, unsqueeze_default_17, unsqueeze_default_20, unsqueeze_default_23, unsqueeze_default_26, unsqueeze_default_29, unsqueeze_default_32, unsqueeze_default_35, unsqueeze_default_38, unsqueeze_default_41, unsqueeze_default_44, unsqueeze_default_47, unsqueeze_default_50, unsqueeze_default_53, unsqueeze_default_56, unsqueeze_default_59, unsqueeze_default_62, unsqueeze_default_65, unsqueeze_default_68, unsqueeze_default_71, unsqueeze_default_74, unsqueeze_default_77, unsqueeze_default_80, unsqueeze_default_83, unsqueeze_default_86, unsqueeze_default_89, unsqueeze_default_92, unsqueeze_default_95, unsqueeze_default_98, unsqueeze_default_101, unsqueeze_default_104, unsqueeze_default_107, unsqueeze_default_110, unsqueeze_default_113, unsqueeze_default_116, unsqueeze_default_119, unsqueeze_default_122, unsqueeze_default_125, unsqueeze_default_128, unsqueeze_default_131, unsqueeze_default_134, unsqueeze_default_137, unsqueeze_default_140, unsqueeze_default_143, unsqueeze_default_146, unsqueeze_default_149, unsqueeze_default_152, unsqueeze_default_155, unsqueeze_default_158, unsqueeze_default_161, unsqueeze_default_164, unsqueeze_default_167, unsqueeze_default_170, unsqueeze_default_173, unsqueeze_default_176, unsqueeze_default_179, unsqueeze_default_182, copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95, copy__default_96, copy__default_97, copy__default_98, copy__default_99, copy__default_100, copy__default_101, copy__default_102, copy__default_103, copy__default_104, copy__default_105, copy__default_106, copy__default_107, copy__default_108, copy__default_109, copy__default_110, copy__default_111, copy__default_112, copy__default_113, copy__default_114, copy__default_115, copy__default_116, copy__default_117, copy__default_118, copy__default_119, copy__default_120, copy__default_121, copy__default_122, copy__default_123, copy__default_124, copy__default_125, copy__default_126, copy__default_127, copy__default_128, copy__default_129, copy__default_130, copy__default_131, copy__default_132, copy__default_133, copy__default_134, copy__default_135, copy__default_136, copy__default_137, copy__default_138, copy__default_139, copy__default_140, copy__default_141, copy__default_142, copy__default_143, copy__default_144, copy__default_145, copy__default_146, copy__default_147, copy__default_148, copy__default_149, copy__default_150, copy__default_151, copy__default_152, copy__default_153, copy__default_154, copy__default_155, copy__default_156, copy__default_157, copy__default_158, copy__default_159, copy__default_160, copy__default_161, copy__default_162, copy__default_163, copy__default_164, copy__default_165, copy__default_166, copy__default_167, copy__default_168, copy__default_169, copy__default_170, copy__default_171, copy__default_172, copy__default_173, copy__default_174, copy__default_175, copy__default_176, copy__default_177, copy__default_178, copy__default_179, copy__default_180, copy__default_181, copy__default_182, copy__default_183, copy__default_184, copy__default_185, copy__default_186, copy__default_187, copy__default_188, copy__default_189, copy__default_190, copy__default_191, copy__default_192, copy__default_193, copy__default_194, copy__default_195, copy__default_196, copy__default_197, copy__default_198, copy__default_199, copy__default_200, copy__default_201, copy__default_202, copy__default_203, copy__default_204, copy__default_205, copy__default_206, copy__default_207, copy__default_208, copy__default_209, copy__default_210, copy__default_211, copy__default_212, copy__default_213, copy__default_214, copy__default_215, copy__default_216, copy__default_217, copy__default_218, copy__default_219, copy__default_220, copy__default_221, copy__default_222, copy__default_223, copy__default_224, copy__default_225, copy__default_226, copy__default_227, copy__default_228, copy__default_229, copy__default_230, copy__default_231, copy__default_232, copy__default_233, copy__default_234, copy__default_235, copy__default_236, copy__default_237, copy__default_238, copy__default_239)
