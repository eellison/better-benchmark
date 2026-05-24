import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32, 3, 224, 224]", primals_3: "i64[]", primals_4: "f32[32]", primals_5: "f32[32]", primals_6: "f32[32]", primals_7: "f32[32]", primals_8: "f32[224, 32, 1, 1]", primals_9: "i64[]", primals_10: "f32[224]", primals_11: "f32[224]", primals_12: "f32[224]", primals_13: "f32[224]", primals_14: "f32[224, 112, 3, 3]", primals_15: "i64[]", primals_16: "f32[224]", primals_17: "f32[224]", primals_18: "f32[224]", primals_19: "f32[224]", primals_20: "f32[8, 224, 1, 1]", primals_21: "f32[8]", primals_22: "f32[224, 8, 1, 1]", primals_23: "f32[224]", primals_24: "f32[224, 224, 1, 1]", primals_25: "i64[]", primals_26: "f32[224]", primals_27: "f32[224]", primals_28: "f32[224]", primals_29: "f32[224]", primals_30: "f32[224, 32, 1, 1]", primals_31: "i64[]", primals_32: "f32[224]", primals_33: "f32[224]", primals_34: "f32[224]", primals_35: "f32[224]", primals_36: "f32[224, 224, 1, 1]", primals_37: "i64[]", primals_38: "f32[224]", primals_39: "f32[224]", primals_40: "f32[224]", primals_41: "f32[224]", primals_42: "f32[224, 112, 3, 3]", primals_43: "i64[]", primals_44: "f32[224]", primals_45: "f32[224]", primals_46: "f32[224]", primals_47: "f32[224]", primals_48: "f32[56, 224, 1, 1]", primals_49: "f32[56]", primals_50: "f32[224, 56, 1, 1]", primals_51: "f32[224]", primals_52: "f32[224, 224, 1, 1]", primals_53: "i64[]", primals_54: "f32[224]", primals_55: "f32[224]", primals_56: "f32[224]", primals_57: "f32[224]", primals_58: "f32[448, 224, 1, 1]", primals_59: "i64[]", primals_60: "f32[448]", primals_61: "f32[448]", primals_62: "f32[448]", primals_63: "f32[448]", primals_64: "f32[448, 112, 3, 3]", primals_65: "i64[]", primals_66: "f32[448]", primals_67: "f32[448]", primals_68: "f32[448]", primals_69: "f32[448]", primals_70: "f32[56, 448, 1, 1]", primals_71: "f32[56]", primals_72: "f32[448, 56, 1, 1]", primals_73: "f32[448]", primals_74: "f32[448, 448, 1, 1]", primals_75: "i64[]", primals_76: "f32[448]", primals_77: "f32[448]", primals_78: "f32[448]", primals_79: "f32[448]", primals_80: "f32[448, 224, 1, 1]", primals_81: "i64[]", primals_82: "f32[448]", primals_83: "f32[448]", primals_84: "f32[448]", primals_85: "f32[448]", primals_86: "f32[448, 448, 1, 1]", primals_87: "i64[]", primals_88: "f32[448]", primals_89: "f32[448]", primals_90: "f32[448]", primals_91: "f32[448]", primals_92: "f32[448, 112, 3, 3]", primals_93: "i64[]", primals_94: "f32[448]", primals_95: "f32[448]", primals_96: "f32[448]", primals_97: "f32[448]", primals_98: "f32[112, 448, 1, 1]", primals_99: "f32[112]", primals_100: "f32[448, 112, 1, 1]", primals_101: "f32[448]", primals_102: "f32[448, 448, 1, 1]", primals_103: "i64[]", primals_104: "f32[448]", primals_105: "f32[448]", primals_106: "f32[448]", primals_107: "f32[448]", primals_108: "f32[448, 448, 1, 1]", primals_109: "i64[]", primals_110: "f32[448]", primals_111: "f32[448]", primals_112: "f32[448]", primals_113: "f32[448]", primals_114: "f32[448, 112, 3, 3]", primals_115: "i64[]", primals_116: "f32[448]", primals_117: "f32[448]", primals_118: "f32[448]", primals_119: "f32[448]", primals_120: "f32[112, 448, 1, 1]", primals_121: "f32[112]", primals_122: "f32[448, 112, 1, 1]", primals_123: "f32[448]", primals_124: "f32[448, 448, 1, 1]", primals_125: "i64[]", primals_126: "f32[448]", primals_127: "f32[448]", primals_128: "f32[448]", primals_129: "f32[448]", primals_130: "f32[448, 448, 1, 1]", primals_131: "i64[]", primals_132: "f32[448]", primals_133: "f32[448]", primals_134: "f32[448]", primals_135: "f32[448]", primals_136: "f32[448, 112, 3, 3]", primals_137: "i64[]", primals_138: "f32[448]", primals_139: "f32[448]", primals_140: "f32[448]", primals_141: "f32[448]", primals_142: "f32[112, 448, 1, 1]", primals_143: "f32[112]", primals_144: "f32[448, 112, 1, 1]", primals_145: "f32[448]", primals_146: "f32[448, 448, 1, 1]", primals_147: "i64[]", primals_148: "f32[448]", primals_149: "f32[448]", primals_150: "f32[448]", primals_151: "f32[448]", primals_152: "f32[448, 448, 1, 1]", primals_153: "i64[]", primals_154: "f32[448]", primals_155: "f32[448]", primals_156: "f32[448]", primals_157: "f32[448]", primals_158: "f32[448, 112, 3, 3]", primals_159: "i64[]", primals_160: "f32[448]", primals_161: "f32[448]", primals_162: "f32[448]", primals_163: "f32[448]", primals_164: "f32[112, 448, 1, 1]", primals_165: "f32[112]", primals_166: "f32[448, 112, 1, 1]", primals_167: "f32[448]", primals_168: "f32[448, 448, 1, 1]", primals_169: "i64[]", primals_170: "f32[448]", primals_171: "f32[448]", primals_172: "f32[448]", primals_173: "f32[448]", primals_174: "f32[896, 448, 1, 1]", primals_175: "i64[]", primals_176: "f32[896]", primals_177: "f32[896]", primals_178: "f32[896]", primals_179: "f32[896]", primals_180: "f32[896, 112, 3, 3]", primals_181: "i64[]", primals_182: "f32[896]", primals_183: "f32[896]", primals_184: "f32[896]", primals_185: "f32[896]", primals_186: "f32[112, 896, 1, 1]", primals_187: "f32[112]", primals_188: "f32[896, 112, 1, 1]", primals_189: "f32[896]", primals_190: "f32[896, 896, 1, 1]", primals_191: "i64[]", primals_192: "f32[896]", primals_193: "f32[896]", primals_194: "f32[896]", primals_195: "f32[896]", primals_196: "f32[896, 448, 1, 1]", primals_197: "i64[]", primals_198: "f32[896]", primals_199: "f32[896]", primals_200: "f32[896]", primals_201: "f32[896]", primals_202: "f32[896, 896, 1, 1]", primals_203: "i64[]", primals_204: "f32[896]", primals_205: "f32[896]", primals_206: "f32[896]", primals_207: "f32[896]", primals_208: "f32[896, 112, 3, 3]", primals_209: "i64[]", primals_210: "f32[896]", primals_211: "f32[896]", primals_212: "f32[896]", primals_213: "f32[896]", primals_214: "f32[224, 896, 1, 1]", primals_215: "f32[224]", primals_216: "f32[896, 224, 1, 1]", primals_217: "f32[896]", primals_218: "f32[896, 896, 1, 1]", primals_219: "i64[]", primals_220: "f32[896]", primals_221: "f32[896]", primals_222: "f32[896]", primals_223: "f32[896]", primals_224: "f32[896, 896, 1, 1]", primals_225: "i64[]", primals_226: "f32[896]", primals_227: "f32[896]", primals_228: "f32[896]", primals_229: "f32[896]", primals_230: "f32[896, 112, 3, 3]", primals_231: "i64[]", primals_232: "f32[896]", primals_233: "f32[896]", primals_234: "f32[896]", primals_235: "f32[896]", primals_236: "f32[224, 896, 1, 1]", primals_237: "f32[224]", primals_238: "f32[896, 224, 1, 1]", primals_239: "f32[896]", primals_240: "f32[896, 896, 1, 1]", primals_241: "i64[]", primals_242: "f32[896]", primals_243: "f32[896]", primals_244: "f32[896]", primals_245: "f32[896]", primals_246: "f32[896, 896, 1, 1]", primals_247: "i64[]", primals_248: "f32[896]", primals_249: "f32[896]", primals_250: "f32[896]", primals_251: "f32[896]", primals_252: "f32[896, 112, 3, 3]", primals_253: "i64[]", primals_254: "f32[896]", primals_255: "f32[896]", primals_256: "f32[896]", primals_257: "f32[896]", primals_258: "f32[224, 896, 1, 1]", primals_259: "f32[224]", primals_260: "f32[896, 224, 1, 1]", primals_261: "f32[896]", primals_262: "f32[896, 896, 1, 1]", primals_263: "i64[]", primals_264: "f32[896]", primals_265: "f32[896]", primals_266: "f32[896]", primals_267: "f32[896]", primals_268: "f32[896, 896, 1, 1]", primals_269: "i64[]", primals_270: "f32[896]", primals_271: "f32[896]", primals_272: "f32[896]", primals_273: "f32[896]", primals_274: "f32[896, 112, 3, 3]", primals_275: "i64[]", primals_276: "f32[896]", primals_277: "f32[896]", primals_278: "f32[896]", primals_279: "f32[896]", primals_280: "f32[224, 896, 1, 1]", primals_281: "f32[224]", primals_282: "f32[896, 224, 1, 1]", primals_283: "f32[896]", primals_284: "f32[896, 896, 1, 1]", primals_285: "i64[]", primals_286: "f32[896]", primals_287: "f32[896]", primals_288: "f32[896]", primals_289: "f32[896]", primals_290: "f32[896, 896, 1, 1]", primals_291: "i64[]", primals_292: "f32[896]", primals_293: "f32[896]", primals_294: "f32[896]", primals_295: "f32[896]", primals_296: "f32[896, 112, 3, 3]", primals_297: "i64[]", primals_298: "f32[896]", primals_299: "f32[896]", primals_300: "f32[896]", primals_301: "f32[896]", primals_302: "f32[224, 896, 1, 1]", primals_303: "f32[224]", primals_304: "f32[896, 224, 1, 1]", primals_305: "f32[896]", primals_306: "f32[896, 896, 1, 1]", primals_307: "i64[]", primals_308: "f32[896]", primals_309: "f32[896]", primals_310: "f32[896]", primals_311: "f32[896]", primals_312: "f32[896, 896, 1, 1]", primals_313: "i64[]", primals_314: "f32[896]", primals_315: "f32[896]", primals_316: "f32[896]", primals_317: "f32[896]", primals_318: "f32[896, 112, 3, 3]", primals_319: "i64[]", primals_320: "f32[896]", primals_321: "f32[896]", primals_322: "f32[896]", primals_323: "f32[896]", primals_324: "f32[224, 896, 1, 1]", primals_325: "f32[224]", primals_326: "f32[896, 224, 1, 1]", primals_327: "f32[896]", primals_328: "f32[896, 896, 1, 1]", primals_329: "i64[]", primals_330: "f32[896]", primals_331: "f32[896]", primals_332: "f32[896]", primals_333: "f32[896]", primals_334: "f32[896, 896, 1, 1]", primals_335: "i64[]", primals_336: "f32[896]", primals_337: "f32[896]", primals_338: "f32[896]", primals_339: "f32[896]", primals_340: "f32[896, 112, 3, 3]", primals_341: "i64[]", primals_342: "f32[896]", primals_343: "f32[896]", primals_344: "f32[896]", primals_345: "f32[896]", primals_346: "f32[224, 896, 1, 1]", primals_347: "f32[224]", primals_348: "f32[896, 224, 1, 1]", primals_349: "f32[896]", primals_350: "f32[896, 896, 1, 1]", primals_351: "i64[]", primals_352: "f32[896]", primals_353: "f32[896]", primals_354: "f32[896]", primals_355: "f32[896]", primals_356: "f32[896, 896, 1, 1]", primals_357: "i64[]", primals_358: "f32[896]", primals_359: "f32[896]", primals_360: "f32[896]", primals_361: "f32[896]", primals_362: "f32[896, 112, 3, 3]", primals_363: "i64[]", primals_364: "f32[896]", primals_365: "f32[896]", primals_366: "f32[896]", primals_367: "f32[896]", primals_368: "f32[224, 896, 1, 1]", primals_369: "f32[224]", primals_370: "f32[896, 224, 1, 1]", primals_371: "f32[896]", primals_372: "f32[896, 896, 1, 1]", primals_373: "i64[]", primals_374: "f32[896]", primals_375: "f32[896]", primals_376: "f32[896]", primals_377: "f32[896]", primals_378: "f32[896, 896, 1, 1]", primals_379: "i64[]", primals_380: "f32[896]", primals_381: "f32[896]", primals_382: "f32[896]", primals_383: "f32[896]", primals_384: "f32[896, 112, 3, 3]", primals_385: "i64[]", primals_386: "f32[896]", primals_387: "f32[896]", primals_388: "f32[896]", primals_389: "f32[896]", primals_390: "f32[224, 896, 1, 1]", primals_391: "f32[224]", primals_392: "f32[896, 224, 1, 1]", primals_393: "f32[896]", primals_394: "f32[896, 896, 1, 1]", primals_395: "i64[]", primals_396: "f32[896]", primals_397: "f32[896]", primals_398: "f32[896]", primals_399: "f32[896]", primals_400: "f32[896, 896, 1, 1]", primals_401: "i64[]", primals_402: "f32[896]", primals_403: "f32[896]", primals_404: "f32[896]", primals_405: "f32[896]", primals_406: "f32[896, 112, 3, 3]", primals_407: "i64[]", primals_408: "f32[896]", primals_409: "f32[896]", primals_410: "f32[896]", primals_411: "f32[896]", primals_412: "f32[224, 896, 1, 1]", primals_413: "f32[224]", primals_414: "f32[896, 224, 1, 1]", primals_415: "f32[896]", primals_416: "f32[896, 896, 1, 1]", primals_417: "i64[]", primals_418: "f32[896]", primals_419: "f32[896]", primals_420: "f32[896]", primals_421: "f32[896]", primals_422: "f32[2240, 896, 1, 1]", primals_423: "i64[]", primals_424: "f32[2240]", primals_425: "f32[2240]", primals_426: "f32[2240]", primals_427: "f32[2240]", primals_428: "f32[2240, 112, 3, 3]", primals_429: "i64[]", primals_430: "f32[2240]", primals_431: "f32[2240]", primals_432: "f32[2240]", primals_433: "f32[2240]", primals_434: "f32[224, 2240, 1, 1]", primals_435: "f32[224]", primals_436: "f32[2240, 224, 1, 1]", primals_437: "f32[2240]", primals_438: "f32[2240, 2240, 1, 1]", primals_439: "i64[]", primals_440: "f32[2240]", primals_441: "f32[2240]", primals_442: "f32[2240]", primals_443: "f32[2240]", primals_444: "f32[2240, 896, 1, 1]", primals_445: "i64[]", primals_446: "f32[2240]", primals_447: "f32[2240]", primals_448: "f32[2240]", primals_449: "f32[2240]", primals_450: "f32[1000, 2240]", primals_451: "f32[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution: "f32[32, 32, 112, 112]" = torch.ops.aten.convolution.default(primals_2, primals_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add: "i64[]" = torch.ops.aten.add.Tensor(primals_3, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean = torch.ops.aten.var_mean.correction(convolution, [0, 2, 3], correction = 0, keepdim = True)
        getitem: "f32[1, 32, 1, 1]" = var_mean[0]
        getitem_1: "f32[1, 32, 1, 1]" = var_mean[1];  var_mean = None
        add_1: "f32[1, 32, 1, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05)
        rsqrt: "f32[1, 32, 1, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        sub: "f32[32, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze, 0.1)
        mul_2: "f32[32]" = torch.ops.aten.mul.Tensor(primals_4, 0.9)
        add_2: "f32[32]" = torch.ops.aten.add.Tensor(mul_1, mul_2);  mul_1 = mul_2 = None
        squeeze_2: "f32[32]" = torch.ops.aten.squeeze.dims(getitem, [0, 2, 3]);  getitem = None
        mul_3: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_2, 1.0000024912370735);  squeeze_2 = None
        mul_4: "f32[32]" = torch.ops.aten.mul.Tensor(mul_3, 0.1);  mul_3 = None
        mul_5: "f32[32]" = torch.ops.aten.mul.Tensor(primals_5, 0.9)
        add_3: "f32[32]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[32, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[32, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu: "f32[32, 32, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_1: "f32[32, 224, 112, 112]" = torch.ops.aten.convolution.default(relu, primals_8, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_5: "i64[]" = torch.ops.aten.add.Tensor(primals_9, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_1 = torch.ops.aten.var_mean.correction(convolution_1, [0, 2, 3], correction = 0, keepdim = True)
        getitem_2: "f32[1, 224, 1, 1]" = var_mean_1[0]
        getitem_3: "f32[1, 224, 1, 1]" = var_mean_1[1];  var_mean_1 = None
        add_6: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05)
        rsqrt_1: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_6);  add_6 = None
        sub_1: "f32[32, 224, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[32, 224, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        squeeze_3: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        squeeze_4: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_8: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_3, 0.1)
        mul_9: "f32[224]" = torch.ops.aten.mul.Tensor(primals_10, 0.9)
        add_7: "f32[224]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        squeeze_5: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_2, [0, 2, 3]);  getitem_2 = None
        mul_10: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_5, 1.0000024912370735);  squeeze_5 = None
        mul_11: "f32[224]" = torch.ops.aten.mul.Tensor(mul_10, 0.1);  mul_10 = None
        mul_12: "f32[224]" = torch.ops.aten.mul.Tensor(primals_11, 0.9)
        add_8: "f32[224]" = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        unsqueeze_4: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[32, 224, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_9: "f32[32, 224, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_1: "f32[32, 224, 112, 112]" = torch.ops.aten.relu.default(add_9);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_2: "f32[32, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_1, primals_14, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_10: "i64[]" = torch.ops.aten.add.Tensor(primals_15, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_2 = torch.ops.aten.var_mean.correction(convolution_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_4: "f32[1, 224, 1, 1]" = var_mean_2[0]
        getitem_5: "f32[1, 224, 1, 1]" = var_mean_2[1];  var_mean_2 = None
        add_11: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05)
        rsqrt_2: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        sub_2: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        squeeze_6: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3])
        mul_15: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_6, 0.1);  squeeze_6 = None
        mul_16: "f32[224]" = torch.ops.aten.mul.Tensor(primals_16, 0.9)
        add_12: "f32[224]" = torch.ops.aten.add.Tensor(mul_15, mul_16);  mul_15 = mul_16 = None
        squeeze_8: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_4, [0, 2, 3]);  getitem_4 = None
        mul_17: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_8, 1.00000996502277);  squeeze_8 = None
        mul_18: "f32[224]" = torch.ops.aten.mul.Tensor(mul_17, 0.1);  mul_17 = None
        mul_19: "f32[224]" = torch.ops.aten.mul.Tensor(primals_17, 0.9)
        add_13: "f32[224]" = torch.ops.aten.add.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        unsqueeze_8: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1)
        unsqueeze_11: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_2: "f32[32, 224, 56, 56]" = torch.ops.aten.relu.default(add_14);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean: "f32[32, 224, 1, 1]" = torch.ops.aten.mean.dim(relu_2, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_3: "f32[32, 8, 1, 1]" = torch.ops.aten.convolution.default(mean, primals_20, primals_21, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_3: "f32[32, 8, 1, 1]" = torch.ops.aten.relu.default(convolution_3);  convolution_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_4: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(relu_3, primals_22, primals_23, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid: "f32[32, 224, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_21: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(relu_2, sigmoid);  relu_2 = sigmoid = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_5: "f32[32, 224, 56, 56]" = torch.ops.aten.convolution.default(mul_21, primals_24, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_15: "i64[]" = torch.ops.aten.add.Tensor(primals_25, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_3 = torch.ops.aten.var_mean.correction(convolution_5, [0, 2, 3], correction = 0, keepdim = True)
        getitem_6: "f32[1, 224, 1, 1]" = var_mean_3[0]
        getitem_7: "f32[1, 224, 1, 1]" = var_mean_3[1];  var_mean_3 = None
        add_16: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05)
        rsqrt_3: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_3: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
        mul_22: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        squeeze_9: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        squeeze_10: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_23: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_9, 0.1)
        mul_24: "f32[224]" = torch.ops.aten.mul.Tensor(primals_26, 0.9)
        add_17: "f32[224]" = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
        squeeze_11: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_6, [0, 2, 3]);  getitem_6 = None
        mul_25: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_11, 1.00000996502277);  squeeze_11 = None
        mul_26: "f32[224]" = torch.ops.aten.mul.Tensor(mul_25, 0.1);  mul_25 = None
        mul_27: "f32[224]" = torch.ops.aten.mul.Tensor(primals_27, 0.9)
        add_18: "f32[224]" = torch.ops.aten.add.Tensor(mul_26, mul_27);  mul_26 = mul_27 = None
        unsqueeze_12: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1)
        unsqueeze_13: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_28: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_13);  mul_22 = unsqueeze_13 = None
        unsqueeze_14: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1);  primals_29 = None
        unsqueeze_15: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_19: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_15);  mul_28 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_6: "f32[32, 224, 56, 56]" = torch.ops.aten.convolution.default(relu, primals_30, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_20: "i64[]" = torch.ops.aten.add.Tensor(primals_31, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_4 = torch.ops.aten.var_mean.correction(convolution_6, [0, 2, 3], correction = 0, keepdim = True)
        getitem_8: "f32[1, 224, 1, 1]" = var_mean_4[0]
        getitem_9: "f32[1, 224, 1, 1]" = var_mean_4[1];  var_mean_4 = None
        add_21: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05)
        rsqrt_4: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
        sub_4: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_9)
        mul_29: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        squeeze_12: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        squeeze_13: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_30: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_12, 0.1)
        mul_31: "f32[224]" = torch.ops.aten.mul.Tensor(primals_32, 0.9)
        add_22: "f32[224]" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
        squeeze_14: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_8, [0, 2, 3]);  getitem_8 = None
        mul_32: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_14, 1.00000996502277);  squeeze_14 = None
        mul_33: "f32[224]" = torch.ops.aten.mul.Tensor(mul_32, 0.1);  mul_32 = None
        mul_34: "f32[224]" = torch.ops.aten.mul.Tensor(primals_33, 0.9)
        add_23: "f32[224]" = torch.ops.aten.add.Tensor(mul_33, mul_34);  mul_33 = mul_34 = None
        unsqueeze_16: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1)
        unsqueeze_17: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_35: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_17);  mul_29 = unsqueeze_17 = None
        unsqueeze_18: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1);  primals_35 = None
        unsqueeze_19: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_24: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_19);  mul_35 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_25: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(add_19, add_24);  add_19 = add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_4: "f32[32, 224, 56, 56]" = torch.ops.aten.relu.default(add_25);  add_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_7: "f32[32, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_4, primals_36, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_26: "i64[]" = torch.ops.aten.add.Tensor(primals_37, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_5 = torch.ops.aten.var_mean.correction(convolution_7, [0, 2, 3], correction = 0, keepdim = True)
        getitem_10: "f32[1, 224, 1, 1]" = var_mean_5[0]
        getitem_11: "f32[1, 224, 1, 1]" = var_mean_5[1];  var_mean_5 = None
        add_27: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05)
        rsqrt_5: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_5: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_11)
        mul_36: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        squeeze_15: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        squeeze_16: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_37: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_15, 0.1)
        mul_38: "f32[224]" = torch.ops.aten.mul.Tensor(primals_38, 0.9)
        add_28: "f32[224]" = torch.ops.aten.add.Tensor(mul_37, mul_38);  mul_37 = mul_38 = None
        squeeze_17: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_10, [0, 2, 3]);  getitem_10 = None
        mul_39: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_17, 1.00000996502277);  squeeze_17 = None
        mul_40: "f32[224]" = torch.ops.aten.mul.Tensor(mul_39, 0.1);  mul_39 = None
        mul_41: "f32[224]" = torch.ops.aten.mul.Tensor(primals_39, 0.9)
        add_29: "f32[224]" = torch.ops.aten.add.Tensor(mul_40, mul_41);  mul_40 = mul_41 = None
        unsqueeze_20: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_40, -1)
        unsqueeze_21: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_42: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_36, unsqueeze_21);  mul_36 = unsqueeze_21 = None
        unsqueeze_22: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_41, -1);  primals_41 = None
        unsqueeze_23: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_30: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_42, unsqueeze_23);  mul_42 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_5: "f32[32, 224, 56, 56]" = torch.ops.aten.relu.default(add_30);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_8: "f32[32, 224, 56, 56]" = torch.ops.aten.convolution.default(relu_5, primals_42, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_31: "i64[]" = torch.ops.aten.add.Tensor(primals_43, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_6 = torch.ops.aten.var_mean.correction(convolution_8, [0, 2, 3], correction = 0, keepdim = True)
        getitem_12: "f32[1, 224, 1, 1]" = var_mean_6[0]
        getitem_13: "f32[1, 224, 1, 1]" = var_mean_6[1];  var_mean_6 = None
        add_32: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05)
        rsqrt_6: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        sub_6: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_13)
        mul_43: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        squeeze_18: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3])
        mul_44: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_18, 0.1);  squeeze_18 = None
        mul_45: "f32[224]" = torch.ops.aten.mul.Tensor(primals_44, 0.9)
        add_33: "f32[224]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
        squeeze_20: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_12, [0, 2, 3]);  getitem_12 = None
        mul_46: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_20, 1.00000996502277);  squeeze_20 = None
        mul_47: "f32[224]" = torch.ops.aten.mul.Tensor(mul_46, 0.1);  mul_46 = None
        mul_48: "f32[224]" = torch.ops.aten.mul.Tensor(primals_45, 0.9)
        add_34: "f32[224]" = torch.ops.aten.add.Tensor(mul_47, mul_48);  mul_47 = mul_48 = None
        unsqueeze_24: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_46, -1)
        unsqueeze_25: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_49: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_25);  mul_43 = unsqueeze_25 = None
        unsqueeze_26: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_47, -1)
        unsqueeze_27: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_35: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_49, unsqueeze_27);  mul_49 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_6: "f32[32, 224, 56, 56]" = torch.ops.aten.relu.default(add_35);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_1: "f32[32, 224, 1, 1]" = torch.ops.aten.mean.dim(relu_6, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_9: "f32[32, 56, 1, 1]" = torch.ops.aten.convolution.default(mean_1, primals_48, primals_49, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_7: "f32[32, 56, 1, 1]" = torch.ops.aten.relu.default(convolution_9);  convolution_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_10: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(relu_7, primals_50, primals_51, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_1: "f32[32, 224, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_50: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(relu_6, sigmoid_1);  relu_6 = sigmoid_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_11: "f32[32, 224, 56, 56]" = torch.ops.aten.convolution.default(mul_50, primals_52, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_36: "i64[]" = torch.ops.aten.add.Tensor(primals_53, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_7 = torch.ops.aten.var_mean.correction(convolution_11, [0, 2, 3], correction = 0, keepdim = True)
        getitem_14: "f32[1, 224, 1, 1]" = var_mean_7[0]
        getitem_15: "f32[1, 224, 1, 1]" = var_mean_7[1];  var_mean_7 = None
        add_37: "f32[1, 224, 1, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05)
        rsqrt_7: "f32[1, 224, 1, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
        sub_7: "f32[32, 224, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_15)
        mul_51: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        squeeze_21: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        squeeze_22: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_52: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_21, 0.1)
        mul_53: "f32[224]" = torch.ops.aten.mul.Tensor(primals_54, 0.9)
        add_38: "f32[224]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
        squeeze_23: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_14, [0, 2, 3]);  getitem_14 = None
        mul_54: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_23, 1.00000996502277);  squeeze_23 = None
        mul_55: "f32[224]" = torch.ops.aten.mul.Tensor(mul_54, 0.1);  mul_54 = None
        mul_56: "f32[224]" = torch.ops.aten.mul.Tensor(primals_55, 0.9)
        add_39: "f32[224]" = torch.ops.aten.add.Tensor(mul_55, mul_56);  mul_55 = mul_56 = None
        unsqueeze_28: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
        unsqueeze_29: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_57: "f32[32, 224, 56, 56]" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_29);  mul_51 = unsqueeze_29 = None
        unsqueeze_30: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
        unsqueeze_31: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_40: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_31);  mul_57 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_41: "f32[32, 224, 56, 56]" = torch.ops.aten.add.Tensor(add_40, relu_4);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_8: "f32[32, 224, 56, 56]" = torch.ops.aten.relu.default(add_41);  add_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_12: "f32[32, 448, 56, 56]" = torch.ops.aten.convolution.default(relu_8, primals_58, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_42: "i64[]" = torch.ops.aten.add.Tensor(primals_59, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_8 = torch.ops.aten.var_mean.correction(convolution_12, [0, 2, 3], correction = 0, keepdim = True)
        getitem_16: "f32[1, 448, 1, 1]" = var_mean_8[0]
        getitem_17: "f32[1, 448, 1, 1]" = var_mean_8[1];  var_mean_8 = None
        add_43: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05)
        rsqrt_8: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        sub_8: "f32[32, 448, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, getitem_17)
        mul_58: "f32[32, 448, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        squeeze_24: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        squeeze_25: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_59: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_24, 0.1)
        mul_60: "f32[448]" = torch.ops.aten.mul.Tensor(primals_60, 0.9)
        add_44: "f32[448]" = torch.ops.aten.add.Tensor(mul_59, mul_60);  mul_59 = mul_60 = None
        squeeze_26: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_16, [0, 2, 3]);  getitem_16 = None
        mul_61: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_26, 1.00000996502277);  squeeze_26 = None
        mul_62: "f32[448]" = torch.ops.aten.mul.Tensor(mul_61, 0.1);  mul_61 = None
        mul_63: "f32[448]" = torch.ops.aten.mul.Tensor(primals_61, 0.9)
        add_45: "f32[448]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
        unsqueeze_32: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_62, -1)
        unsqueeze_33: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_64: "f32[32, 448, 56, 56]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_33);  mul_58 = unsqueeze_33 = None
        unsqueeze_34: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_63, -1);  primals_63 = None
        unsqueeze_35: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_46: "f32[32, 448, 56, 56]" = torch.ops.aten.add.Tensor(mul_64, unsqueeze_35);  mul_64 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_9: "f32[32, 448, 56, 56]" = torch.ops.aten.relu.default(add_46);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_13: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_9, primals_64, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_47: "i64[]" = torch.ops.aten.add.Tensor(primals_65, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_9 = torch.ops.aten.var_mean.correction(convolution_13, [0, 2, 3], correction = 0, keepdim = True)
        getitem_18: "f32[1, 448, 1, 1]" = var_mean_9[0]
        getitem_19: "f32[1, 448, 1, 1]" = var_mean_9[1];  var_mean_9 = None
        add_48: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05)
        rsqrt_9: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
        sub_9: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_19)
        mul_65: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        squeeze_27: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3])
        mul_66: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_27, 0.1);  squeeze_27 = None
        mul_67: "f32[448]" = torch.ops.aten.mul.Tensor(primals_66, 0.9)
        add_49: "f32[448]" = torch.ops.aten.add.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
        squeeze_29: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_18, [0, 2, 3]);  getitem_18 = None
        mul_68: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_29, 1.0000398612827361);  squeeze_29 = None
        mul_69: "f32[448]" = torch.ops.aten.mul.Tensor(mul_68, 0.1);  mul_68 = None
        mul_70: "f32[448]" = torch.ops.aten.mul.Tensor(primals_67, 0.9)
        add_50: "f32[448]" = torch.ops.aten.add.Tensor(mul_69, mul_70);  mul_69 = mul_70 = None
        unsqueeze_36: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_68, -1)
        unsqueeze_37: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_71: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_65, unsqueeze_37);  mul_65 = unsqueeze_37 = None
        unsqueeze_38: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_69, -1)
        unsqueeze_39: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_51: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_39);  mul_71 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_10: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_51);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_2: "f32[32, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_10, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_14: "f32[32, 56, 1, 1]" = torch.ops.aten.convolution.default(mean_2, primals_70, primals_71, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_11: "f32[32, 56, 1, 1]" = torch.ops.aten.relu.default(convolution_14);  convolution_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_15: "f32[32, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_11, primals_72, primals_73, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_2: "f32[32, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_72: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_10, sigmoid_2);  relu_10 = sigmoid_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_16: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_72, primals_74, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_52: "i64[]" = torch.ops.aten.add.Tensor(primals_75, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_10 = torch.ops.aten.var_mean.correction(convolution_16, [0, 2, 3], correction = 0, keepdim = True)
        getitem_20: "f32[1, 448, 1, 1]" = var_mean_10[0]
        getitem_21: "f32[1, 448, 1, 1]" = var_mean_10[1];  var_mean_10 = None
        add_53: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05)
        rsqrt_10: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_10: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_21)
        mul_73: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        squeeze_30: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        squeeze_31: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_74: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_30, 0.1)
        mul_75: "f32[448]" = torch.ops.aten.mul.Tensor(primals_76, 0.9)
        add_54: "f32[448]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
        squeeze_32: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_20, [0, 2, 3]);  getitem_20 = None
        mul_76: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_32, 1.0000398612827361);  squeeze_32 = None
        mul_77: "f32[448]" = torch.ops.aten.mul.Tensor(mul_76, 0.1);  mul_76 = None
        mul_78: "f32[448]" = torch.ops.aten.mul.Tensor(primals_77, 0.9)
        add_55: "f32[448]" = torch.ops.aten.add.Tensor(mul_77, mul_78);  mul_77 = mul_78 = None
        unsqueeze_40: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_41: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_79: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_41);  mul_73 = unsqueeze_41 = None
        unsqueeze_42: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_43: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_56: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_43);  mul_79 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_17: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_8, primals_80, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_57: "i64[]" = torch.ops.aten.add.Tensor(primals_81, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_11 = torch.ops.aten.var_mean.correction(convolution_17, [0, 2, 3], correction = 0, keepdim = True)
        getitem_22: "f32[1, 448, 1, 1]" = var_mean_11[0]
        getitem_23: "f32[1, 448, 1, 1]" = var_mean_11[1];  var_mean_11 = None
        add_58: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05)
        rsqrt_11: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
        sub_11: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, getitem_23)
        mul_80: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        squeeze_33: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        squeeze_34: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_81: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_33, 0.1)
        mul_82: "f32[448]" = torch.ops.aten.mul.Tensor(primals_82, 0.9)
        add_59: "f32[448]" = torch.ops.aten.add.Tensor(mul_81, mul_82);  mul_81 = mul_82 = None
        squeeze_35: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_22, [0, 2, 3]);  getitem_22 = None
        mul_83: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_35, 1.0000398612827361);  squeeze_35 = None
        mul_84: "f32[448]" = torch.ops.aten.mul.Tensor(mul_83, 0.1);  mul_83 = None
        mul_85: "f32[448]" = torch.ops.aten.mul.Tensor(primals_83, 0.9)
        add_60: "f32[448]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
        unsqueeze_44: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_45: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_86: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_80, unsqueeze_45);  mul_80 = unsqueeze_45 = None
        unsqueeze_46: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_47: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_61: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_47);  mul_86 = unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_62: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_56, add_61);  add_56 = add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_12: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_62);  add_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_18: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_12, primals_86, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_63: "i64[]" = torch.ops.aten.add.Tensor(primals_87, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_12 = torch.ops.aten.var_mean.correction(convolution_18, [0, 2, 3], correction = 0, keepdim = True)
        getitem_24: "f32[1, 448, 1, 1]" = var_mean_12[0]
        getitem_25: "f32[1, 448, 1, 1]" = var_mean_12[1];  var_mean_12 = None
        add_64: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05)
        rsqrt_12: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_12: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, getitem_25)
        mul_87: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        squeeze_36: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        squeeze_37: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_88: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_36, 0.1)
        mul_89: "f32[448]" = torch.ops.aten.mul.Tensor(primals_88, 0.9)
        add_65: "f32[448]" = torch.ops.aten.add.Tensor(mul_88, mul_89);  mul_88 = mul_89 = None
        squeeze_38: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_24, [0, 2, 3]);  getitem_24 = None
        mul_90: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_38, 1.0000398612827361);  squeeze_38 = None
        mul_91: "f32[448]" = torch.ops.aten.mul.Tensor(mul_90, 0.1);  mul_90 = None
        mul_92: "f32[448]" = torch.ops.aten.mul.Tensor(primals_89, 0.9)
        add_66: "f32[448]" = torch.ops.aten.add.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None
        unsqueeze_48: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_49: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_93: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_87, unsqueeze_49);  mul_87 = unsqueeze_49 = None
        unsqueeze_50: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_51: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_67: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_93, unsqueeze_51);  mul_93 = unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_13: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_67);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_19: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_13, primals_92, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_68: "i64[]" = torch.ops.aten.add.Tensor(primals_93, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_13 = torch.ops.aten.var_mean.correction(convolution_19, [0, 2, 3], correction = 0, keepdim = True)
        getitem_26: "f32[1, 448, 1, 1]" = var_mean_13[0]
        getitem_27: "f32[1, 448, 1, 1]" = var_mean_13[1];  var_mean_13 = None
        add_69: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05)
        rsqrt_13: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
        sub_13: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_27)
        mul_94: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        squeeze_39: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3])
        mul_95: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_39, 0.1);  squeeze_39 = None
        mul_96: "f32[448]" = torch.ops.aten.mul.Tensor(primals_94, 0.9)
        add_70: "f32[448]" = torch.ops.aten.add.Tensor(mul_95, mul_96);  mul_95 = mul_96 = None
        squeeze_41: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_26, [0, 2, 3]);  getitem_26 = None
        mul_97: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_41, 1.0000398612827361);  squeeze_41 = None
        mul_98: "f32[448]" = torch.ops.aten.mul.Tensor(mul_97, 0.1);  mul_97 = None
        mul_99: "f32[448]" = torch.ops.aten.mul.Tensor(primals_95, 0.9)
        add_71: "f32[448]" = torch.ops.aten.add.Tensor(mul_98, mul_99);  mul_98 = mul_99 = None
        unsqueeze_52: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_96, -1)
        unsqueeze_53: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_100: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_53);  mul_94 = unsqueeze_53 = None
        unsqueeze_54: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_97, -1)
        unsqueeze_55: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_72: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_100, unsqueeze_55);  mul_100 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_14: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_72);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_3: "f32[32, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_14, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_20: "f32[32, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_3, primals_98, primals_99, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_15: "f32[32, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_20);  convolution_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_21: "f32[32, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_15, primals_100, primals_101, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_3: "f32[32, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_101: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_14, sigmoid_3);  relu_14 = sigmoid_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_22: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_101, primals_102, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_73: "i64[]" = torch.ops.aten.add.Tensor(primals_103, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_14 = torch.ops.aten.var_mean.correction(convolution_22, [0, 2, 3], correction = 0, keepdim = True)
        getitem_28: "f32[1, 448, 1, 1]" = var_mean_14[0]
        getitem_29: "f32[1, 448, 1, 1]" = var_mean_14[1];  var_mean_14 = None
        add_74: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05)
        rsqrt_14: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        sub_14: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_29)
        mul_102: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        squeeze_42: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        squeeze_43: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_103: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_42, 0.1)
        mul_104: "f32[448]" = torch.ops.aten.mul.Tensor(primals_104, 0.9)
        add_75: "f32[448]" = torch.ops.aten.add.Tensor(mul_103, mul_104);  mul_103 = mul_104 = None
        squeeze_44: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_28, [0, 2, 3]);  getitem_28 = None
        mul_105: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_44, 1.0000398612827361);  squeeze_44 = None
        mul_106: "f32[448]" = torch.ops.aten.mul.Tensor(mul_105, 0.1);  mul_105 = None
        mul_107: "f32[448]" = torch.ops.aten.mul.Tensor(primals_105, 0.9)
        add_76: "f32[448]" = torch.ops.aten.add.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
        unsqueeze_56: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_106, -1)
        unsqueeze_57: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_108: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_102, unsqueeze_57);  mul_102 = unsqueeze_57 = None
        unsqueeze_58: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_107, -1);  primals_107 = None
        unsqueeze_59: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_77: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_108, unsqueeze_59);  mul_108 = unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_78: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_77, relu_12);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_16: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_78);  add_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_23: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_16, primals_108, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_79: "i64[]" = torch.ops.aten.add.Tensor(primals_109, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_15 = torch.ops.aten.var_mean.correction(convolution_23, [0, 2, 3], correction = 0, keepdim = True)
        getitem_30: "f32[1, 448, 1, 1]" = var_mean_15[0]
        getitem_31: "f32[1, 448, 1, 1]" = var_mean_15[1];  var_mean_15 = None
        add_80: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05)
        rsqrt_15: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
        sub_15: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, getitem_31)
        mul_109: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        squeeze_45: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        squeeze_46: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_110: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_45, 0.1)
        mul_111: "f32[448]" = torch.ops.aten.mul.Tensor(primals_110, 0.9)
        add_81: "f32[448]" = torch.ops.aten.add.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
        squeeze_47: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_30, [0, 2, 3]);  getitem_30 = None
        mul_112: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_47, 1.0000398612827361);  squeeze_47 = None
        mul_113: "f32[448]" = torch.ops.aten.mul.Tensor(mul_112, 0.1);  mul_112 = None
        mul_114: "f32[448]" = torch.ops.aten.mul.Tensor(primals_111, 0.9)
        add_82: "f32[448]" = torch.ops.aten.add.Tensor(mul_113, mul_114);  mul_113 = mul_114 = None
        unsqueeze_60: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_112, -1)
        unsqueeze_61: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_115: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_61);  mul_109 = unsqueeze_61 = None
        unsqueeze_62: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_113, -1);  primals_113 = None
        unsqueeze_63: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_83: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_115, unsqueeze_63);  mul_115 = unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_17: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_83);  add_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_24: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_17, primals_114, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_84: "i64[]" = torch.ops.aten.add.Tensor(primals_115, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_16 = torch.ops.aten.var_mean.correction(convolution_24, [0, 2, 3], correction = 0, keepdim = True)
        getitem_32: "f32[1, 448, 1, 1]" = var_mean_16[0]
        getitem_33: "f32[1, 448, 1, 1]" = var_mean_16[1];  var_mean_16 = None
        add_85: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05)
        rsqrt_16: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        sub_16: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, getitem_33)
        mul_116: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        squeeze_48: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3])
        mul_117: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_48, 0.1);  squeeze_48 = None
        mul_118: "f32[448]" = torch.ops.aten.mul.Tensor(primals_116, 0.9)
        add_86: "f32[448]" = torch.ops.aten.add.Tensor(mul_117, mul_118);  mul_117 = mul_118 = None
        squeeze_50: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_32, [0, 2, 3]);  getitem_32 = None
        mul_119: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_50, 1.0000398612827361);  squeeze_50 = None
        mul_120: "f32[448]" = torch.ops.aten.mul.Tensor(mul_119, 0.1);  mul_119 = None
        mul_121: "f32[448]" = torch.ops.aten.mul.Tensor(primals_117, 0.9)
        add_87: "f32[448]" = torch.ops.aten.add.Tensor(mul_120, mul_121);  mul_120 = mul_121 = None
        unsqueeze_64: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_118, -1)
        unsqueeze_65: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_122: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_116, unsqueeze_65);  mul_116 = unsqueeze_65 = None
        unsqueeze_66: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_119, -1)
        unsqueeze_67: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_88: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_67);  mul_122 = unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_18: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_88);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_4: "f32[32, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_18, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_25: "f32[32, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_4, primals_120, primals_121, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_19: "f32[32, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_25);  convolution_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_26: "f32[32, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_19, primals_122, primals_123, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_4: "f32[32, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_123: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_18, sigmoid_4);  relu_18 = sigmoid_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_27: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_123, primals_124, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_89: "i64[]" = torch.ops.aten.add.Tensor(primals_125, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_17 = torch.ops.aten.var_mean.correction(convolution_27, [0, 2, 3], correction = 0, keepdim = True)
        getitem_34: "f32[1, 448, 1, 1]" = var_mean_17[0]
        getitem_35: "f32[1, 448, 1, 1]" = var_mean_17[1];  var_mean_17 = None
        add_90: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05)
        rsqrt_17: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
        sub_17: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, getitem_35)
        mul_124: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        squeeze_51: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_35, [0, 2, 3]);  getitem_35 = None
        squeeze_52: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_17, [0, 2, 3]);  rsqrt_17 = None
        mul_125: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_51, 0.1)
        mul_126: "f32[448]" = torch.ops.aten.mul.Tensor(primals_126, 0.9)
        add_91: "f32[448]" = torch.ops.aten.add.Tensor(mul_125, mul_126);  mul_125 = mul_126 = None
        squeeze_53: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_34, [0, 2, 3]);  getitem_34 = None
        mul_127: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_53, 1.0000398612827361);  squeeze_53 = None
        mul_128: "f32[448]" = torch.ops.aten.mul.Tensor(mul_127, 0.1);  mul_127 = None
        mul_129: "f32[448]" = torch.ops.aten.mul.Tensor(primals_127, 0.9)
        add_92: "f32[448]" = torch.ops.aten.add.Tensor(mul_128, mul_129);  mul_128 = mul_129 = None
        unsqueeze_68: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_128, -1)
        unsqueeze_69: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_130: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_69);  mul_124 = unsqueeze_69 = None
        unsqueeze_70: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_129, -1);  primals_129 = None
        unsqueeze_71: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_93: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_130, unsqueeze_71);  mul_130 = unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_94: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_93, relu_16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_20: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_94);  add_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_28: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_20, primals_130, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_95: "i64[]" = torch.ops.aten.add.Tensor(primals_131, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_18 = torch.ops.aten.var_mean.correction(convolution_28, [0, 2, 3], correction = 0, keepdim = True)
        getitem_36: "f32[1, 448, 1, 1]" = var_mean_18[0]
        getitem_37: "f32[1, 448, 1, 1]" = var_mean_18[1];  var_mean_18 = None
        add_96: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05)
        rsqrt_18: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        sub_18: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_37)
        mul_131: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        squeeze_54: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        squeeze_55: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_132: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_54, 0.1)
        mul_133: "f32[448]" = torch.ops.aten.mul.Tensor(primals_132, 0.9)
        add_97: "f32[448]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
        squeeze_56: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_36, [0, 2, 3]);  getitem_36 = None
        mul_134: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_56, 1.0000398612827361);  squeeze_56 = None
        mul_135: "f32[448]" = torch.ops.aten.mul.Tensor(mul_134, 0.1);  mul_134 = None
        mul_136: "f32[448]" = torch.ops.aten.mul.Tensor(primals_133, 0.9)
        add_98: "f32[448]" = torch.ops.aten.add.Tensor(mul_135, mul_136);  mul_135 = mul_136 = None
        unsqueeze_72: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_134, -1)
        unsqueeze_73: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_137: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_131, unsqueeze_73);  mul_131 = unsqueeze_73 = None
        unsqueeze_74: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_135, -1);  primals_135 = None
        unsqueeze_75: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_99: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_75);  mul_137 = unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_21: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_99);  add_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_29: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_21, primals_136, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_100: "i64[]" = torch.ops.aten.add.Tensor(primals_137, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_19 = torch.ops.aten.var_mean.correction(convolution_29, [0, 2, 3], correction = 0, keepdim = True)
        getitem_38: "f32[1, 448, 1, 1]" = var_mean_19[0]
        getitem_39: "f32[1, 448, 1, 1]" = var_mean_19[1];  var_mean_19 = None
        add_101: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05)
        rsqrt_19: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
        sub_19: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, getitem_39)
        mul_138: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        squeeze_57: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3])
        mul_139: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_57, 0.1);  squeeze_57 = None
        mul_140: "f32[448]" = torch.ops.aten.mul.Tensor(primals_138, 0.9)
        add_102: "f32[448]" = torch.ops.aten.add.Tensor(mul_139, mul_140);  mul_139 = mul_140 = None
        squeeze_59: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_38, [0, 2, 3]);  getitem_38 = None
        mul_141: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_59, 1.0000398612827361);  squeeze_59 = None
        mul_142: "f32[448]" = torch.ops.aten.mul.Tensor(mul_141, 0.1);  mul_141 = None
        mul_143: "f32[448]" = torch.ops.aten.mul.Tensor(primals_139, 0.9)
        add_103: "f32[448]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        unsqueeze_76: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_140, -1)
        unsqueeze_77: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_144: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_138, unsqueeze_77);  mul_138 = unsqueeze_77 = None
        unsqueeze_78: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_141, -1)
        unsqueeze_79: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_104: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_144, unsqueeze_79);  mul_144 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_22: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_104);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_5: "f32[32, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_22, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_30: "f32[32, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_5, primals_142, primals_143, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_23: "f32[32, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_30);  convolution_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_31: "f32[32, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_23, primals_144, primals_145, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_5: "f32[32, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_145: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_22, sigmoid_5);  relu_22 = sigmoid_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_32: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_145, primals_146, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_105: "i64[]" = torch.ops.aten.add.Tensor(primals_147, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_20 = torch.ops.aten.var_mean.correction(convolution_32, [0, 2, 3], correction = 0, keepdim = True)
        getitem_40: "f32[1, 448, 1, 1]" = var_mean_20[0]
        getitem_41: "f32[1, 448, 1, 1]" = var_mean_20[1];  var_mean_20 = None
        add_106: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05)
        rsqrt_20: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        sub_20: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, getitem_41)
        mul_146: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
        squeeze_60: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        squeeze_61: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_20, [0, 2, 3]);  rsqrt_20 = None
        mul_147: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_60, 0.1)
        mul_148: "f32[448]" = torch.ops.aten.mul.Tensor(primals_148, 0.9)
        add_107: "f32[448]" = torch.ops.aten.add.Tensor(mul_147, mul_148);  mul_147 = mul_148 = None
        squeeze_62: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_40, [0, 2, 3]);  getitem_40 = None
        mul_149: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_62, 1.0000398612827361);  squeeze_62 = None
        mul_150: "f32[448]" = torch.ops.aten.mul.Tensor(mul_149, 0.1);  mul_149 = None
        mul_151: "f32[448]" = torch.ops.aten.mul.Tensor(primals_149, 0.9)
        add_108: "f32[448]" = torch.ops.aten.add.Tensor(mul_150, mul_151);  mul_150 = mul_151 = None
        unsqueeze_80: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_150, -1)
        unsqueeze_81: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_152: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_146, unsqueeze_81);  mul_146 = unsqueeze_81 = None
        unsqueeze_82: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_151, -1);  primals_151 = None
        unsqueeze_83: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_109: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_83);  mul_152 = unsqueeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_110: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_109, relu_20);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_24: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_110);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_33: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_24, primals_152, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_111: "i64[]" = torch.ops.aten.add.Tensor(primals_153, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_21 = torch.ops.aten.var_mean.correction(convolution_33, [0, 2, 3], correction = 0, keepdim = True)
        getitem_42: "f32[1, 448, 1, 1]" = var_mean_21[0]
        getitem_43: "f32[1, 448, 1, 1]" = var_mean_21[1];  var_mean_21 = None
        add_112: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05)
        rsqrt_21: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
        sub_21: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, getitem_43)
        mul_153: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        squeeze_63: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        squeeze_64: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_154: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_63, 0.1)
        mul_155: "f32[448]" = torch.ops.aten.mul.Tensor(primals_154, 0.9)
        add_113: "f32[448]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
        squeeze_65: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_42, [0, 2, 3]);  getitem_42 = None
        mul_156: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_65, 1.0000398612827361);  squeeze_65 = None
        mul_157: "f32[448]" = torch.ops.aten.mul.Tensor(mul_156, 0.1);  mul_156 = None
        mul_158: "f32[448]" = torch.ops.aten.mul.Tensor(primals_155, 0.9)
        add_114: "f32[448]" = torch.ops.aten.add.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
        unsqueeze_84: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_85: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_159: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_153, unsqueeze_85);  mul_153 = unsqueeze_85 = None
        unsqueeze_86: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_87: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_115: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_159, unsqueeze_87);  mul_159 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_25: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_115);  add_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_34: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(relu_25, primals_158, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_116: "i64[]" = torch.ops.aten.add.Tensor(primals_159, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_22 = torch.ops.aten.var_mean.correction(convolution_34, [0, 2, 3], correction = 0, keepdim = True)
        getitem_44: "f32[1, 448, 1, 1]" = var_mean_22[0]
        getitem_45: "f32[1, 448, 1, 1]" = var_mean_22[1];  var_mean_22 = None
        add_117: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05)
        rsqrt_22: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
        sub_22: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_45)
        mul_160: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        squeeze_66: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3])
        mul_161: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_66, 0.1);  squeeze_66 = None
        mul_162: "f32[448]" = torch.ops.aten.mul.Tensor(primals_160, 0.9)
        add_118: "f32[448]" = torch.ops.aten.add.Tensor(mul_161, mul_162);  mul_161 = mul_162 = None
        squeeze_68: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_44, [0, 2, 3]);  getitem_44 = None
        mul_163: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_68, 1.0000398612827361);  squeeze_68 = None
        mul_164: "f32[448]" = torch.ops.aten.mul.Tensor(mul_163, 0.1);  mul_163 = None
        mul_165: "f32[448]" = torch.ops.aten.mul.Tensor(primals_161, 0.9)
        add_119: "f32[448]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
        unsqueeze_88: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_89: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_166: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_89);  mul_160 = unsqueeze_89 = None
        unsqueeze_90: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1)
        unsqueeze_91: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_120: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_166, unsqueeze_91);  mul_166 = unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_26: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_120);  add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_6: "f32[32, 448, 1, 1]" = torch.ops.aten.mean.dim(relu_26, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_35: "f32[32, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_6, primals_164, primals_165, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_27: "f32[32, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_35);  convolution_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_36: "f32[32, 448, 1, 1]" = torch.ops.aten.convolution.default(relu_27, primals_166, primals_167, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_6: "f32[32, 448, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_36)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_167: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(relu_26, sigmoid_6);  relu_26 = sigmoid_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_37: "f32[32, 448, 28, 28]" = torch.ops.aten.convolution.default(mul_167, primals_168, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_121: "i64[]" = torch.ops.aten.add.Tensor(primals_169, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_23 = torch.ops.aten.var_mean.correction(convolution_37, [0, 2, 3], correction = 0, keepdim = True)
        getitem_46: "f32[1, 448, 1, 1]" = var_mean_23[0]
        getitem_47: "f32[1, 448, 1, 1]" = var_mean_23[1];  var_mean_23 = None
        add_122: "f32[1, 448, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05)
        rsqrt_23: "f32[1, 448, 1, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        sub_23: "f32[32, 448, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_47)
        mul_168: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        squeeze_69: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        squeeze_70: "f32[448]" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_169: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_69, 0.1)
        mul_170: "f32[448]" = torch.ops.aten.mul.Tensor(primals_170, 0.9)
        add_123: "f32[448]" = torch.ops.aten.add.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        squeeze_71: "f32[448]" = torch.ops.aten.squeeze.dims(getitem_46, [0, 2, 3]);  getitem_46 = None
        mul_171: "f32[448]" = torch.ops.aten.mul.Tensor(squeeze_71, 1.0000398612827361);  squeeze_71 = None
        mul_172: "f32[448]" = torch.ops.aten.mul.Tensor(mul_171, 0.1);  mul_171 = None
        mul_173: "f32[448]" = torch.ops.aten.mul.Tensor(primals_171, 0.9)
        add_124: "f32[448]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
        unsqueeze_92: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_172, -1)
        unsqueeze_93: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_174: "f32[32, 448, 28, 28]" = torch.ops.aten.mul.Tensor(mul_168, unsqueeze_93);  mul_168 = unsqueeze_93 = None
        unsqueeze_94: "f32[448, 1]" = torch.ops.aten.unsqueeze.default(primals_173, -1);  primals_173 = None
        unsqueeze_95: "f32[448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_125: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(mul_174, unsqueeze_95);  mul_174 = unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_126: "f32[32, 448, 28, 28]" = torch.ops.aten.add.Tensor(add_125, relu_24);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_28: "f32[32, 448, 28, 28]" = torch.ops.aten.relu.default(add_126);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_38: "f32[32, 896, 28, 28]" = torch.ops.aten.convolution.default(relu_28, primals_174, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_127: "i64[]" = torch.ops.aten.add.Tensor(primals_175, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_24 = torch.ops.aten.var_mean.correction(convolution_38, [0, 2, 3], correction = 0, keepdim = True)
        getitem_48: "f32[1, 896, 1, 1]" = var_mean_24[0]
        getitem_49: "f32[1, 896, 1, 1]" = var_mean_24[1];  var_mean_24 = None
        add_128: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05)
        rsqrt_24: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
        sub_24: "f32[32, 896, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_49)
        mul_175: "f32[32, 896, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        squeeze_72: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        squeeze_73: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_176: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_72, 0.1)
        mul_177: "f32[896]" = torch.ops.aten.mul.Tensor(primals_176, 0.9)
        add_129: "f32[896]" = torch.ops.aten.add.Tensor(mul_176, mul_177);  mul_176 = mul_177 = None
        squeeze_74: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_48, [0, 2, 3]);  getitem_48 = None
        mul_178: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_74, 1.0000398612827361);  squeeze_74 = None
        mul_179: "f32[896]" = torch.ops.aten.mul.Tensor(mul_178, 0.1);  mul_178 = None
        mul_180: "f32[896]" = torch.ops.aten.mul.Tensor(primals_177, 0.9)
        add_130: "f32[896]" = torch.ops.aten.add.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        unsqueeze_96: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_178, -1)
        unsqueeze_97: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_181: "f32[32, 896, 28, 28]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_97);  mul_175 = unsqueeze_97 = None
        unsqueeze_98: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_179, -1);  primals_179 = None
        unsqueeze_99: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_131: "f32[32, 896, 28, 28]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_99);  mul_181 = unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_29: "f32[32, 896, 28, 28]" = torch.ops.aten.relu.default(add_131);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_39: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_29, primals_180, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_132: "i64[]" = torch.ops.aten.add.Tensor(primals_181, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_25 = torch.ops.aten.var_mean.correction(convolution_39, [0, 2, 3], correction = 0, keepdim = True)
        getitem_50: "f32[1, 896, 1, 1]" = var_mean_25[0]
        getitem_51: "f32[1, 896, 1, 1]" = var_mean_25[1];  var_mean_25 = None
        add_133: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05)
        rsqrt_25: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        sub_25: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, getitem_51)
        mul_182: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        squeeze_75: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3])
        mul_183: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_75, 0.1);  squeeze_75 = None
        mul_184: "f32[896]" = torch.ops.aten.mul.Tensor(primals_182, 0.9)
        add_134: "f32[896]" = torch.ops.aten.add.Tensor(mul_183, mul_184);  mul_183 = mul_184 = None
        squeeze_77: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_50, [0, 2, 3]);  getitem_50 = None
        mul_185: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_77, 1.0001594642002871);  squeeze_77 = None
        mul_186: "f32[896]" = torch.ops.aten.mul.Tensor(mul_185, 0.1);  mul_185 = None
        mul_187: "f32[896]" = torch.ops.aten.mul.Tensor(primals_183, 0.9)
        add_135: "f32[896]" = torch.ops.aten.add.Tensor(mul_186, mul_187);  mul_186 = mul_187 = None
        unsqueeze_100: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_184, -1)
        unsqueeze_101: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_188: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_101);  mul_182 = unsqueeze_101 = None
        unsqueeze_102: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_185, -1)
        unsqueeze_103: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_136: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_103);  mul_188 = unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_30: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_136);  add_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_7: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_30, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_40: "f32[32, 112, 1, 1]" = torch.ops.aten.convolution.default(mean_7, primals_186, primals_187, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_31: "f32[32, 112, 1, 1]" = torch.ops.aten.relu.default(convolution_40);  convolution_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_41: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_31, primals_188, primals_189, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_7: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_41)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_189: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_30, sigmoid_7);  relu_30 = sigmoid_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_42: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_189, primals_190, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_137: "i64[]" = torch.ops.aten.add.Tensor(primals_191, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_26 = torch.ops.aten.var_mean.correction(convolution_42, [0, 2, 3], correction = 0, keepdim = True)
        getitem_52: "f32[1, 896, 1, 1]" = var_mean_26[0]
        getitem_53: "f32[1, 896, 1, 1]" = var_mean_26[1];  var_mean_26 = None
        add_138: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05)
        rsqrt_26: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
        sub_26: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_53)
        mul_190: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        squeeze_78: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_53, [0, 2, 3]);  getitem_53 = None
        squeeze_79: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_191: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_78, 0.1)
        mul_192: "f32[896]" = torch.ops.aten.mul.Tensor(primals_192, 0.9)
        add_139: "f32[896]" = torch.ops.aten.add.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None
        squeeze_80: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_52, [0, 2, 3]);  getitem_52 = None
        mul_193: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_80, 1.0001594642002871);  squeeze_80 = None
        mul_194: "f32[896]" = torch.ops.aten.mul.Tensor(mul_193, 0.1);  mul_193 = None
        mul_195: "f32[896]" = torch.ops.aten.mul.Tensor(primals_193, 0.9)
        add_140: "f32[896]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
        unsqueeze_104: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_194, -1)
        unsqueeze_105: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_196: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_105);  mul_190 = unsqueeze_105 = None
        unsqueeze_106: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_195, -1);  primals_195 = None
        unsqueeze_107: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_141: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_196, unsqueeze_107);  mul_196 = unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_43: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_28, primals_196, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_142: "i64[]" = torch.ops.aten.add.Tensor(primals_197, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_27 = torch.ops.aten.var_mean.correction(convolution_43, [0, 2, 3], correction = 0, keepdim = True)
        getitem_54: "f32[1, 896, 1, 1]" = var_mean_27[0]
        getitem_55: "f32[1, 896, 1, 1]" = var_mean_27[1];  var_mean_27 = None
        add_143: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05)
        rsqrt_27: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
        sub_27: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, getitem_55)
        mul_197: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        squeeze_81: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        squeeze_82: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_198: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_81, 0.1)
        mul_199: "f32[896]" = torch.ops.aten.mul.Tensor(primals_198, 0.9)
        add_144: "f32[896]" = torch.ops.aten.add.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None
        squeeze_83: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_54, [0, 2, 3]);  getitem_54 = None
        mul_200: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_83, 1.0001594642002871);  squeeze_83 = None
        mul_201: "f32[896]" = torch.ops.aten.mul.Tensor(mul_200, 0.1);  mul_200 = None
        mul_202: "f32[896]" = torch.ops.aten.mul.Tensor(primals_199, 0.9)
        add_145: "f32[896]" = torch.ops.aten.add.Tensor(mul_201, mul_202);  mul_201 = mul_202 = None
        unsqueeze_108: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_200, -1)
        unsqueeze_109: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_203: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_197, unsqueeze_109);  mul_197 = unsqueeze_109 = None
        unsqueeze_110: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_201, -1);  primals_201 = None
        unsqueeze_111: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_146: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, unsqueeze_111);  mul_203 = unsqueeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_147: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_141, add_146);  add_141 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_32: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_147);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_44: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_32, primals_202, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_148: "i64[]" = torch.ops.aten.add.Tensor(primals_203, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_28 = torch.ops.aten.var_mean.correction(convolution_44, [0, 2, 3], correction = 0, keepdim = True)
        getitem_56: "f32[1, 896, 1, 1]" = var_mean_28[0]
        getitem_57: "f32[1, 896, 1, 1]" = var_mean_28[1];  var_mean_28 = None
        add_149: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05)
        rsqrt_28: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
        sub_28: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, getitem_57)
        mul_204: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        squeeze_84: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        squeeze_85: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_205: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_84, 0.1)
        mul_206: "f32[896]" = torch.ops.aten.mul.Tensor(primals_204, 0.9)
        add_150: "f32[896]" = torch.ops.aten.add.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
        squeeze_86: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_56, [0, 2, 3]);  getitem_56 = None
        mul_207: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_86, 1.0001594642002871);  squeeze_86 = None
        mul_208: "f32[896]" = torch.ops.aten.mul.Tensor(mul_207, 0.1);  mul_207 = None
        mul_209: "f32[896]" = torch.ops.aten.mul.Tensor(primals_205, 0.9)
        add_151: "f32[896]" = torch.ops.aten.add.Tensor(mul_208, mul_209);  mul_208 = mul_209 = None
        unsqueeze_112: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_206, -1)
        unsqueeze_113: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_210: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_204, unsqueeze_113);  mul_204 = unsqueeze_113 = None
        unsqueeze_114: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_207, -1);  primals_207 = None
        unsqueeze_115: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_152: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_210, unsqueeze_115);  mul_210 = unsqueeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_33: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_152);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_45: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_33, primals_208, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_153: "i64[]" = torch.ops.aten.add.Tensor(primals_209, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_29 = torch.ops.aten.var_mean.correction(convolution_45, [0, 2, 3], correction = 0, keepdim = True)
        getitem_58: "f32[1, 896, 1, 1]" = var_mean_29[0]
        getitem_59: "f32[1, 896, 1, 1]" = var_mean_29[1];  var_mean_29 = None
        add_154: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05)
        rsqrt_29: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
        sub_29: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_59)
        mul_211: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = None
        squeeze_87: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3])
        mul_212: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_87, 0.1);  squeeze_87 = None
        mul_213: "f32[896]" = torch.ops.aten.mul.Tensor(primals_210, 0.9)
        add_155: "f32[896]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
        squeeze_89: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_58, [0, 2, 3]);  getitem_58 = None
        mul_214: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_89, 1.0001594642002871);  squeeze_89 = None
        mul_215: "f32[896]" = torch.ops.aten.mul.Tensor(mul_214, 0.1);  mul_214 = None
        mul_216: "f32[896]" = torch.ops.aten.mul.Tensor(primals_211, 0.9)
        add_156: "f32[896]" = torch.ops.aten.add.Tensor(mul_215, mul_216);  mul_215 = mul_216 = None
        unsqueeze_116: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_212, -1)
        unsqueeze_117: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_217: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_117);  mul_211 = unsqueeze_117 = None
        unsqueeze_118: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_213, -1)
        unsqueeze_119: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_157: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_217, unsqueeze_119);  mul_217 = unsqueeze_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_34: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_157);  add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_8: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_34, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_46: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_8, primals_214, primals_215, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_35: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_46);  convolution_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_47: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_35, primals_216, primals_217, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_8: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_47)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_218: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_34, sigmoid_8);  relu_34 = sigmoid_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_48: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_218, primals_218, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_158: "i64[]" = torch.ops.aten.add.Tensor(primals_219, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_30 = torch.ops.aten.var_mean.correction(convolution_48, [0, 2, 3], correction = 0, keepdim = True)
        getitem_60: "f32[1, 896, 1, 1]" = var_mean_30[0]
        getitem_61: "f32[1, 896, 1, 1]" = var_mean_30[1];  var_mean_30 = None
        add_159: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05)
        rsqrt_30: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_30: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_61)
        mul_219: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        squeeze_90: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        squeeze_91: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_220: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_90, 0.1)
        mul_221: "f32[896]" = torch.ops.aten.mul.Tensor(primals_220, 0.9)
        add_160: "f32[896]" = torch.ops.aten.add.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
        squeeze_92: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_60, [0, 2, 3]);  getitem_60 = None
        mul_222: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_92, 1.0001594642002871);  squeeze_92 = None
        mul_223: "f32[896]" = torch.ops.aten.mul.Tensor(mul_222, 0.1);  mul_222 = None
        mul_224: "f32[896]" = torch.ops.aten.mul.Tensor(primals_221, 0.9)
        add_161: "f32[896]" = torch.ops.aten.add.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
        unsqueeze_120: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_121: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_225: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_219, unsqueeze_121);  mul_219 = unsqueeze_121 = None
        unsqueeze_122: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
        unsqueeze_123: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_162: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_225, unsqueeze_123);  mul_225 = unsqueeze_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_163: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_162, relu_32);  add_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_36: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_163);  add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_49: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_36, primals_224, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_164: "i64[]" = torch.ops.aten.add.Tensor(primals_225, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_31 = torch.ops.aten.var_mean.correction(convolution_49, [0, 2, 3], correction = 0, keepdim = True)
        getitem_62: "f32[1, 896, 1, 1]" = var_mean_31[0]
        getitem_63: "f32[1, 896, 1, 1]" = var_mean_31[1];  var_mean_31 = None
        add_165: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05)
        rsqrt_31: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        sub_31: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, getitem_63)
        mul_226: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        squeeze_93: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        squeeze_94: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_227: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_93, 0.1)
        mul_228: "f32[896]" = torch.ops.aten.mul.Tensor(primals_226, 0.9)
        add_166: "f32[896]" = torch.ops.aten.add.Tensor(mul_227, mul_228);  mul_227 = mul_228 = None
        squeeze_95: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_62, [0, 2, 3]);  getitem_62 = None
        mul_229: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_95, 1.0001594642002871);  squeeze_95 = None
        mul_230: "f32[896]" = torch.ops.aten.mul.Tensor(mul_229, 0.1);  mul_229 = None
        mul_231: "f32[896]" = torch.ops.aten.mul.Tensor(primals_227, 0.9)
        add_167: "f32[896]" = torch.ops.aten.add.Tensor(mul_230, mul_231);  mul_230 = mul_231 = None
        unsqueeze_124: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_125: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_232: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_125);  mul_226 = unsqueeze_125 = None
        unsqueeze_126: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_127: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_168: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_232, unsqueeze_127);  mul_232 = unsqueeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_37: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_168);  add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_50: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_37, primals_230, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_169: "i64[]" = torch.ops.aten.add.Tensor(primals_231, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_32 = torch.ops.aten.var_mean.correction(convolution_50, [0, 2, 3], correction = 0, keepdim = True)
        getitem_64: "f32[1, 896, 1, 1]" = var_mean_32[0]
        getitem_65: "f32[1, 896, 1, 1]" = var_mean_32[1];  var_mean_32 = None
        add_170: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05)
        rsqrt_32: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_32: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_65)
        mul_233: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = None
        squeeze_96: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3])
        mul_234: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_96, 0.1);  squeeze_96 = None
        mul_235: "f32[896]" = torch.ops.aten.mul.Tensor(primals_232, 0.9)
        add_171: "f32[896]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
        squeeze_98: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_64, [0, 2, 3]);  getitem_64 = None
        mul_236: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_98, 1.0001594642002871);  squeeze_98 = None
        mul_237: "f32[896]" = torch.ops.aten.mul.Tensor(mul_236, 0.1);  mul_236 = None
        mul_238: "f32[896]" = torch.ops.aten.mul.Tensor(primals_233, 0.9)
        add_172: "f32[896]" = torch.ops.aten.add.Tensor(mul_237, mul_238);  mul_237 = mul_238 = None
        unsqueeze_128: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_129: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        mul_239: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_233, unsqueeze_129);  mul_233 = unsqueeze_129 = None
        unsqueeze_130: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1)
        unsqueeze_131: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        add_173: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_131);  mul_239 = unsqueeze_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_38: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_173);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_9: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_38, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_51: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_9, primals_236, primals_237, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_39: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_51);  convolution_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_52: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_39, primals_238, primals_239, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_9: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_52)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_240: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_38, sigmoid_9);  relu_38 = sigmoid_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_53: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_240, primals_240, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_174: "i64[]" = torch.ops.aten.add.Tensor(primals_241, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_33 = torch.ops.aten.var_mean.correction(convolution_53, [0, 2, 3], correction = 0, keepdim = True)
        getitem_66: "f32[1, 896, 1, 1]" = var_mean_33[0]
        getitem_67: "f32[1, 896, 1, 1]" = var_mean_33[1];  var_mean_33 = None
        add_175: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05)
        rsqrt_33: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
        sub_33: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, getitem_67)
        mul_241: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        squeeze_99: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        squeeze_100: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_242: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_99, 0.1)
        mul_243: "f32[896]" = torch.ops.aten.mul.Tensor(primals_242, 0.9)
        add_176: "f32[896]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
        squeeze_101: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_66, [0, 2, 3]);  getitem_66 = None
        mul_244: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_101, 1.0001594642002871);  squeeze_101 = None
        mul_245: "f32[896]" = torch.ops.aten.mul.Tensor(mul_244, 0.1);  mul_244 = None
        mul_246: "f32[896]" = torch.ops.aten.mul.Tensor(primals_243, 0.9)
        add_177: "f32[896]" = torch.ops.aten.add.Tensor(mul_245, mul_246);  mul_245 = mul_246 = None
        unsqueeze_132: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_244, -1)
        unsqueeze_133: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_247: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_133);  mul_241 = unsqueeze_133 = None
        unsqueeze_134: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_245, -1);  primals_245 = None
        unsqueeze_135: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_178: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_247, unsqueeze_135);  mul_247 = unsqueeze_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_179: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_178, relu_36);  add_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_40: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_179);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_54: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_40, primals_246, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_180: "i64[]" = torch.ops.aten.add.Tensor(primals_247, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_34 = torch.ops.aten.var_mean.correction(convolution_54, [0, 2, 3], correction = 0, keepdim = True)
        getitem_68: "f32[1, 896, 1, 1]" = var_mean_34[0]
        getitem_69: "f32[1, 896, 1, 1]" = var_mean_34[1];  var_mean_34 = None
        add_181: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05)
        rsqrt_34: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_34: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_69)
        mul_248: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        squeeze_102: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        squeeze_103: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_249: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_102, 0.1)
        mul_250: "f32[896]" = torch.ops.aten.mul.Tensor(primals_248, 0.9)
        add_182: "f32[896]" = torch.ops.aten.add.Tensor(mul_249, mul_250);  mul_249 = mul_250 = None
        squeeze_104: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_68, [0, 2, 3]);  getitem_68 = None
        mul_251: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_104, 1.0001594642002871);  squeeze_104 = None
        mul_252: "f32[896]" = torch.ops.aten.mul.Tensor(mul_251, 0.1);  mul_251 = None
        mul_253: "f32[896]" = torch.ops.aten.mul.Tensor(primals_249, 0.9)
        add_183: "f32[896]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
        unsqueeze_136: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_250, -1)
        unsqueeze_137: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_254: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_248, unsqueeze_137);  mul_248 = unsqueeze_137 = None
        unsqueeze_138: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_251, -1);  primals_251 = None
        unsqueeze_139: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_184: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_139);  mul_254 = unsqueeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_41: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_184);  add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_55: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_41, primals_252, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_185: "i64[]" = torch.ops.aten.add.Tensor(primals_253, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_35 = torch.ops.aten.var_mean.correction(convolution_55, [0, 2, 3], correction = 0, keepdim = True)
        getitem_70: "f32[1, 896, 1, 1]" = var_mean_35[0]
        getitem_71: "f32[1, 896, 1, 1]" = var_mean_35[1];  var_mean_35 = None
        add_186: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05)
        rsqrt_35: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        sub_35: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_71)
        mul_255: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = None
        squeeze_105: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3])
        mul_256: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_105, 0.1);  squeeze_105 = None
        mul_257: "f32[896]" = torch.ops.aten.mul.Tensor(primals_254, 0.9)
        add_187: "f32[896]" = torch.ops.aten.add.Tensor(mul_256, mul_257);  mul_256 = mul_257 = None
        squeeze_107: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_70, [0, 2, 3]);  getitem_70 = None
        mul_258: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_107, 1.0001594642002871);  squeeze_107 = None
        mul_259: "f32[896]" = torch.ops.aten.mul.Tensor(mul_258, 0.1);  mul_258 = None
        mul_260: "f32[896]" = torch.ops.aten.mul.Tensor(primals_255, 0.9)
        add_188: "f32[896]" = torch.ops.aten.add.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
        unsqueeze_140: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_256, -1)
        unsqueeze_141: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_261: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_255, unsqueeze_141);  mul_255 = unsqueeze_141 = None
        unsqueeze_142: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_257, -1)
        unsqueeze_143: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_189: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_261, unsqueeze_143);  mul_261 = unsqueeze_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_42: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_189);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_10: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_42, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_56: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_10, primals_258, primals_259, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_43: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_56);  convolution_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_57: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_43, primals_260, primals_261, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_10: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_57)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_262: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_42, sigmoid_10);  relu_42 = sigmoid_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_58: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_262, primals_262, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_190: "i64[]" = torch.ops.aten.add.Tensor(primals_263, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_36 = torch.ops.aten.var_mean.correction(convolution_58, [0, 2, 3], correction = 0, keepdim = True)
        getitem_72: "f32[1, 896, 1, 1]" = var_mean_36[0]
        getitem_73: "f32[1, 896, 1, 1]" = var_mean_36[1];  var_mean_36 = None
        add_191: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05)
        rsqrt_36: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
        sub_36: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, getitem_73)
        mul_263: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        squeeze_108: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        squeeze_109: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_264: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_108, 0.1)
        mul_265: "f32[896]" = torch.ops.aten.mul.Tensor(primals_264, 0.9)
        add_192: "f32[896]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
        squeeze_110: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_72, [0, 2, 3]);  getitem_72 = None
        mul_266: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_110, 1.0001594642002871);  squeeze_110 = None
        mul_267: "f32[896]" = torch.ops.aten.mul.Tensor(mul_266, 0.1);  mul_266 = None
        mul_268: "f32[896]" = torch.ops.aten.mul.Tensor(primals_265, 0.9)
        add_193: "f32[896]" = torch.ops.aten.add.Tensor(mul_267, mul_268);  mul_267 = mul_268 = None
        unsqueeze_144: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_266, -1)
        unsqueeze_145: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_269: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_263, unsqueeze_145);  mul_263 = unsqueeze_145 = None
        unsqueeze_146: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_267, -1);  primals_267 = None
        unsqueeze_147: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_194: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_147);  mul_269 = unsqueeze_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_195: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_194, relu_40);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_44: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_195);  add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_59: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_44, primals_268, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_196: "i64[]" = torch.ops.aten.add.Tensor(primals_269, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_37 = torch.ops.aten.var_mean.correction(convolution_59, [0, 2, 3], correction = 0, keepdim = True)
        getitem_74: "f32[1, 896, 1, 1]" = var_mean_37[0]
        getitem_75: "f32[1, 896, 1, 1]" = var_mean_37[1];  var_mean_37 = None
        add_197: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05)
        rsqrt_37: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        sub_37: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, getitem_75)
        mul_270: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        squeeze_111: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        squeeze_112: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_271: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_111, 0.1)
        mul_272: "f32[896]" = torch.ops.aten.mul.Tensor(primals_270, 0.9)
        add_198: "f32[896]" = torch.ops.aten.add.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
        squeeze_113: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_74, [0, 2, 3]);  getitem_74 = None
        mul_273: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_113, 1.0001594642002871);  squeeze_113 = None
        mul_274: "f32[896]" = torch.ops.aten.mul.Tensor(mul_273, 0.1);  mul_273 = None
        mul_275: "f32[896]" = torch.ops.aten.mul.Tensor(primals_271, 0.9)
        add_199: "f32[896]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
        unsqueeze_148: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_272, -1)
        unsqueeze_149: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_276: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_270, unsqueeze_149);  mul_270 = unsqueeze_149 = None
        unsqueeze_150: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_273, -1);  primals_273 = None
        unsqueeze_151: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_200: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_276, unsqueeze_151);  mul_276 = unsqueeze_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_45: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_200);  add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_60: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_45, primals_274, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_201: "i64[]" = torch.ops.aten.add.Tensor(primals_275, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_38 = torch.ops.aten.var_mean.correction(convolution_60, [0, 2, 3], correction = 0, keepdim = True)
        getitem_76: "f32[1, 896, 1, 1]" = var_mean_38[0]
        getitem_77: "f32[1, 896, 1, 1]" = var_mean_38[1];  var_mean_38 = None
        add_202: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05)
        rsqrt_38: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
        sub_38: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_77)
        mul_277: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        squeeze_114: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_77, [0, 2, 3])
        mul_278: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_114, 0.1);  squeeze_114 = None
        mul_279: "f32[896]" = torch.ops.aten.mul.Tensor(primals_276, 0.9)
        add_203: "f32[896]" = torch.ops.aten.add.Tensor(mul_278, mul_279);  mul_278 = mul_279 = None
        squeeze_116: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_76, [0, 2, 3]);  getitem_76 = None
        mul_280: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_116, 1.0001594642002871);  squeeze_116 = None
        mul_281: "f32[896]" = torch.ops.aten.mul.Tensor(mul_280, 0.1);  mul_280 = None
        mul_282: "f32[896]" = torch.ops.aten.mul.Tensor(primals_277, 0.9)
        add_204: "f32[896]" = torch.ops.aten.add.Tensor(mul_281, mul_282);  mul_281 = mul_282 = None
        unsqueeze_152: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_278, -1)
        unsqueeze_153: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_283: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_153);  mul_277 = unsqueeze_153 = None
        unsqueeze_154: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_279, -1)
        unsqueeze_155: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_205: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_283, unsqueeze_155);  mul_283 = unsqueeze_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_46: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_205);  add_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_11: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_46, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_61: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_11, primals_280, primals_281, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_47: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_61);  convolution_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_62: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_47, primals_282, primals_283, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_11: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_62)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_284: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_46, sigmoid_11);  relu_46 = sigmoid_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_63: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_284, primals_284, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_206: "i64[]" = torch.ops.aten.add.Tensor(primals_285, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_39 = torch.ops.aten.var_mean.correction(convolution_63, [0, 2, 3], correction = 0, keepdim = True)
        getitem_78: "f32[1, 896, 1, 1]" = var_mean_39[0]
        getitem_79: "f32[1, 896, 1, 1]" = var_mean_39[1];  var_mean_39 = None
        add_207: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05)
        rsqrt_39: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_39: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, getitem_79)
        mul_285: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        squeeze_117: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        squeeze_118: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_286: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_117, 0.1)
        mul_287: "f32[896]" = torch.ops.aten.mul.Tensor(primals_286, 0.9)
        add_208: "f32[896]" = torch.ops.aten.add.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
        squeeze_119: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_78, [0, 2, 3]);  getitem_78 = None
        mul_288: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_119, 1.0001594642002871);  squeeze_119 = None
        mul_289: "f32[896]" = torch.ops.aten.mul.Tensor(mul_288, 0.1);  mul_288 = None
        mul_290: "f32[896]" = torch.ops.aten.mul.Tensor(primals_287, 0.9)
        add_209: "f32[896]" = torch.ops.aten.add.Tensor(mul_289, mul_290);  mul_289 = mul_290 = None
        unsqueeze_156: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_288, -1)
        unsqueeze_157: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_291: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_285, unsqueeze_157);  mul_285 = unsqueeze_157 = None
        unsqueeze_158: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_289, -1);  primals_289 = None
        unsqueeze_159: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_210: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_291, unsqueeze_159);  mul_291 = unsqueeze_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_211: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_210, relu_44);  add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_48: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_211);  add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_64: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_48, primals_290, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_212: "i64[]" = torch.ops.aten.add.Tensor(primals_291, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_40 = torch.ops.aten.var_mean.correction(convolution_64, [0, 2, 3], correction = 0, keepdim = True)
        getitem_80: "f32[1, 896, 1, 1]" = var_mean_40[0]
        getitem_81: "f32[1, 896, 1, 1]" = var_mean_40[1];  var_mean_40 = None
        add_213: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05)
        rsqrt_40: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_213);  add_213 = None
        sub_40: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, getitem_81)
        mul_292: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        squeeze_120: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        squeeze_121: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_293: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_120, 0.1)
        mul_294: "f32[896]" = torch.ops.aten.mul.Tensor(primals_292, 0.9)
        add_214: "f32[896]" = torch.ops.aten.add.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
        squeeze_122: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_80, [0, 2, 3]);  getitem_80 = None
        mul_295: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_122, 1.0001594642002871);  squeeze_122 = None
        mul_296: "f32[896]" = torch.ops.aten.mul.Tensor(mul_295, 0.1);  mul_295 = None
        mul_297: "f32[896]" = torch.ops.aten.mul.Tensor(primals_293, 0.9)
        add_215: "f32[896]" = torch.ops.aten.add.Tensor(mul_296, mul_297);  mul_296 = mul_297 = None
        unsqueeze_160: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_161: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_298: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_161);  mul_292 = unsqueeze_161 = None
        unsqueeze_162: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_295, -1);  primals_295 = None
        unsqueeze_163: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_216: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_298, unsqueeze_163);  mul_298 = unsqueeze_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_49: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_216);  add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_65: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_49, primals_296, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_217: "i64[]" = torch.ops.aten.add.Tensor(primals_297, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_41 = torch.ops.aten.var_mean.correction(convolution_65, [0, 2, 3], correction = 0, keepdim = True)
        getitem_82: "f32[1, 896, 1, 1]" = var_mean_41[0]
        getitem_83: "f32[1, 896, 1, 1]" = var_mean_41[1];  var_mean_41 = None
        add_218: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05)
        rsqrt_41: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_41: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_83)
        mul_299: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = None
        squeeze_123: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3])
        mul_300: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_123, 0.1);  squeeze_123 = None
        mul_301: "f32[896]" = torch.ops.aten.mul.Tensor(primals_298, 0.9)
        add_219: "f32[896]" = torch.ops.aten.add.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
        squeeze_125: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_82, [0, 2, 3]);  getitem_82 = None
        mul_302: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_125, 1.0001594642002871);  squeeze_125 = None
        mul_303: "f32[896]" = torch.ops.aten.mul.Tensor(mul_302, 0.1);  mul_302 = None
        mul_304: "f32[896]" = torch.ops.aten.mul.Tensor(primals_299, 0.9)
        add_220: "f32[896]" = torch.ops.aten.add.Tensor(mul_303, mul_304);  mul_303 = mul_304 = None
        unsqueeze_164: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_300, -1)
        unsqueeze_165: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_305: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_299, unsqueeze_165);  mul_299 = unsqueeze_165 = None
        unsqueeze_166: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_301, -1)
        unsqueeze_167: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_221: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_167);  mul_305 = unsqueeze_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_50: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_221);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_12: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_50, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_66: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_12, primals_302, primals_303, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_51: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_66);  convolution_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_67: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_51, primals_304, primals_305, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_12: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_67)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_306: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_50, sigmoid_12);  relu_50 = sigmoid_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_68: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_306, primals_306, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_222: "i64[]" = torch.ops.aten.add.Tensor(primals_307, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_42 = torch.ops.aten.var_mean.correction(convolution_68, [0, 2, 3], correction = 0, keepdim = True)
        getitem_84: "f32[1, 896, 1, 1]" = var_mean_42[0]
        getitem_85: "f32[1, 896, 1, 1]" = var_mean_42[1];  var_mean_42 = None
        add_223: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05)
        rsqrt_42: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_223);  add_223 = None
        sub_42: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, getitem_85)
        mul_307: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        squeeze_126: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        squeeze_127: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_308: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_126, 0.1)
        mul_309: "f32[896]" = torch.ops.aten.mul.Tensor(primals_308, 0.9)
        add_224: "f32[896]" = torch.ops.aten.add.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None
        squeeze_128: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_84, [0, 2, 3]);  getitem_84 = None
        mul_310: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_128, 1.0001594642002871);  squeeze_128 = None
        mul_311: "f32[896]" = torch.ops.aten.mul.Tensor(mul_310, 0.1);  mul_310 = None
        mul_312: "f32[896]" = torch.ops.aten.mul.Tensor(primals_309, 0.9)
        add_225: "f32[896]" = torch.ops.aten.add.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
        unsqueeze_168: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_310, -1)
        unsqueeze_169: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_313: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_169);  mul_307 = unsqueeze_169 = None
        unsqueeze_170: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_311, -1);  primals_311 = None
        unsqueeze_171: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_226: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_313, unsqueeze_171);  mul_313 = unsqueeze_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_227: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_226, relu_48);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_52: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_227);  add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_69: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_52, primals_312, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_228: "i64[]" = torch.ops.aten.add.Tensor(primals_313, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_43 = torch.ops.aten.var_mean.correction(convolution_69, [0, 2, 3], correction = 0, keepdim = True)
        getitem_86: "f32[1, 896, 1, 1]" = var_mean_43[0]
        getitem_87: "f32[1, 896, 1, 1]" = var_mean_43[1];  var_mean_43 = None
        add_229: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05)
        rsqrt_43: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_43: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, getitem_87)
        mul_314: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        squeeze_129: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        squeeze_130: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_315: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_129, 0.1)
        mul_316: "f32[896]" = torch.ops.aten.mul.Tensor(primals_314, 0.9)
        add_230: "f32[896]" = torch.ops.aten.add.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
        squeeze_131: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_86, [0, 2, 3]);  getitem_86 = None
        mul_317: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_131, 1.0001594642002871);  squeeze_131 = None
        mul_318: "f32[896]" = torch.ops.aten.mul.Tensor(mul_317, 0.1);  mul_317 = None
        mul_319: "f32[896]" = torch.ops.aten.mul.Tensor(primals_315, 0.9)
        add_231: "f32[896]" = torch.ops.aten.add.Tensor(mul_318, mul_319);  mul_318 = mul_319 = None
        unsqueeze_172: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_316, -1)
        unsqueeze_173: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_320: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_314, unsqueeze_173);  mul_314 = unsqueeze_173 = None
        unsqueeze_174: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_317, -1);  primals_317 = None
        unsqueeze_175: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_232: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_320, unsqueeze_175);  mul_320 = unsqueeze_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_53: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_232);  add_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_70: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_53, primals_318, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_233: "i64[]" = torch.ops.aten.add.Tensor(primals_319, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_44 = torch.ops.aten.var_mean.correction(convolution_70, [0, 2, 3], correction = 0, keepdim = True)
        getitem_88: "f32[1, 896, 1, 1]" = var_mean_44[0]
        getitem_89: "f32[1, 896, 1, 1]" = var_mean_44[1];  var_mean_44 = None
        add_234: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05)
        rsqrt_44: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_234);  add_234 = None
        sub_44: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_89)
        mul_321: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = None
        squeeze_132: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_89, [0, 2, 3])
        mul_322: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_132, 0.1);  squeeze_132 = None
        mul_323: "f32[896]" = torch.ops.aten.mul.Tensor(primals_320, 0.9)
        add_235: "f32[896]" = torch.ops.aten.add.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None
        squeeze_134: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_88, [0, 2, 3]);  getitem_88 = None
        mul_324: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_134, 1.0001594642002871);  squeeze_134 = None
        mul_325: "f32[896]" = torch.ops.aten.mul.Tensor(mul_324, 0.1);  mul_324 = None
        mul_326: "f32[896]" = torch.ops.aten.mul.Tensor(primals_321, 0.9)
        add_236: "f32[896]" = torch.ops.aten.add.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
        unsqueeze_176: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_322, -1)
        unsqueeze_177: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        mul_327: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_321, unsqueeze_177);  mul_321 = unsqueeze_177 = None
        unsqueeze_178: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_323, -1)
        unsqueeze_179: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        add_237: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_327, unsqueeze_179);  mul_327 = unsqueeze_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_54: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_237);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_13: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_54, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_71: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_13, primals_324, primals_325, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_55: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_71);  convolution_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_72: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_55, primals_326, primals_327, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_13: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_72)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_328: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_54, sigmoid_13);  relu_54 = sigmoid_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_73: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_328, primals_328, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_238: "i64[]" = torch.ops.aten.add.Tensor(primals_329, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_45 = torch.ops.aten.var_mean.correction(convolution_73, [0, 2, 3], correction = 0, keepdim = True)
        getitem_90: "f32[1, 896, 1, 1]" = var_mean_45[0]
        getitem_91: "f32[1, 896, 1, 1]" = var_mean_45[1];  var_mean_45 = None
        add_239: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05)
        rsqrt_45: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        sub_45: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, getitem_91)
        mul_329: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        squeeze_135: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        squeeze_136: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_330: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_135, 0.1)
        mul_331: "f32[896]" = torch.ops.aten.mul.Tensor(primals_330, 0.9)
        add_240: "f32[896]" = torch.ops.aten.add.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
        squeeze_137: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_90, [0, 2, 3]);  getitem_90 = None
        mul_332: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_137, 1.0001594642002871);  squeeze_137 = None
        mul_333: "f32[896]" = torch.ops.aten.mul.Tensor(mul_332, 0.1);  mul_332 = None
        mul_334: "f32[896]" = torch.ops.aten.mul.Tensor(primals_331, 0.9)
        add_241: "f32[896]" = torch.ops.aten.add.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
        unsqueeze_180: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_332, -1)
        unsqueeze_181: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_335: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_329, unsqueeze_181);  mul_329 = unsqueeze_181 = None
        unsqueeze_182: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_333, -1);  primals_333 = None
        unsqueeze_183: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_242: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_183);  mul_335 = unsqueeze_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_243: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_242, relu_52);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_56: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_243);  add_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_74: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_56, primals_334, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_244: "i64[]" = torch.ops.aten.add.Tensor(primals_335, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_46 = torch.ops.aten.var_mean.correction(convolution_74, [0, 2, 3], correction = 0, keepdim = True)
        getitem_92: "f32[1, 896, 1, 1]" = var_mean_46[0]
        getitem_93: "f32[1, 896, 1, 1]" = var_mean_46[1];  var_mean_46 = None
        add_245: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05)
        rsqrt_46: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_245);  add_245 = None
        sub_46: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, getitem_93)
        mul_336: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        squeeze_138: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        squeeze_139: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_337: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_138, 0.1)
        mul_338: "f32[896]" = torch.ops.aten.mul.Tensor(primals_336, 0.9)
        add_246: "f32[896]" = torch.ops.aten.add.Tensor(mul_337, mul_338);  mul_337 = mul_338 = None
        squeeze_140: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_92, [0, 2, 3]);  getitem_92 = None
        mul_339: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_140, 1.0001594642002871);  squeeze_140 = None
        mul_340: "f32[896]" = torch.ops.aten.mul.Tensor(mul_339, 0.1);  mul_339 = None
        mul_341: "f32[896]" = torch.ops.aten.mul.Tensor(primals_337, 0.9)
        add_247: "f32[896]" = torch.ops.aten.add.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        unsqueeze_184: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_338, -1)
        unsqueeze_185: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_342: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_185);  mul_336 = unsqueeze_185 = None
        unsqueeze_186: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_339, -1);  primals_339 = None
        unsqueeze_187: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_248: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_187);  mul_342 = unsqueeze_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_57: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_248);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_75: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_57, primals_340, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_249: "i64[]" = torch.ops.aten.add.Tensor(primals_341, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_47 = torch.ops.aten.var_mean.correction(convolution_75, [0, 2, 3], correction = 0, keepdim = True)
        getitem_94: "f32[1, 896, 1, 1]" = var_mean_47[0]
        getitem_95: "f32[1, 896, 1, 1]" = var_mean_47[1];  var_mean_47 = None
        add_250: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05)
        rsqrt_47: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        sub_47: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_95)
        mul_343: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = None
        squeeze_141: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3])
        mul_344: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_141, 0.1);  squeeze_141 = None
        mul_345: "f32[896]" = torch.ops.aten.mul.Tensor(primals_342, 0.9)
        add_251: "f32[896]" = torch.ops.aten.add.Tensor(mul_344, mul_345);  mul_344 = mul_345 = None
        squeeze_143: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_94, [0, 2, 3]);  getitem_94 = None
        mul_346: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_143, 1.0001594642002871);  squeeze_143 = None
        mul_347: "f32[896]" = torch.ops.aten.mul.Tensor(mul_346, 0.1);  mul_346 = None
        mul_348: "f32[896]" = torch.ops.aten.mul.Tensor(primals_343, 0.9)
        add_252: "f32[896]" = torch.ops.aten.add.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_188: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_344, -1)
        unsqueeze_189: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_349: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_189);  mul_343 = unsqueeze_189 = None
        unsqueeze_190: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_345, -1)
        unsqueeze_191: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_253: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_349, unsqueeze_191);  mul_349 = unsqueeze_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_58: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_253);  add_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_14: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_58, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_76: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_14, primals_346, primals_347, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_59: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_76);  convolution_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_77: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_59, primals_348, primals_349, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_14: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_77)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_350: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_58, sigmoid_14);  relu_58 = sigmoid_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_78: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_350, primals_350, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_254: "i64[]" = torch.ops.aten.add.Tensor(primals_351, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_48 = torch.ops.aten.var_mean.correction(convolution_78, [0, 2, 3], correction = 0, keepdim = True)
        getitem_96: "f32[1, 896, 1, 1]" = var_mean_48[0]
        getitem_97: "f32[1, 896, 1, 1]" = var_mean_48[1];  var_mean_48 = None
        add_255: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05)
        rsqrt_48: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_255);  add_255 = None
        sub_48: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, getitem_97)
        mul_351: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        squeeze_144: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        squeeze_145: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_352: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_144, 0.1)
        mul_353: "f32[896]" = torch.ops.aten.mul.Tensor(primals_352, 0.9)
        add_256: "f32[896]" = torch.ops.aten.add.Tensor(mul_352, mul_353);  mul_352 = mul_353 = None
        squeeze_146: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_96, [0, 2, 3]);  getitem_96 = None
        mul_354: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_146, 1.0001594642002871);  squeeze_146 = None
        mul_355: "f32[896]" = torch.ops.aten.mul.Tensor(mul_354, 0.1);  mul_354 = None
        mul_356: "f32[896]" = torch.ops.aten.mul.Tensor(primals_353, 0.9)
        add_257: "f32[896]" = torch.ops.aten.add.Tensor(mul_355, mul_356);  mul_355 = mul_356 = None
        unsqueeze_192: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_354, -1)
        unsqueeze_193: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_357: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_351, unsqueeze_193);  mul_351 = unsqueeze_193 = None
        unsqueeze_194: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_355, -1);  primals_355 = None
        unsqueeze_195: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_258: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_357, unsqueeze_195);  mul_357 = unsqueeze_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_259: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_258, relu_56);  add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_60: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_259);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_79: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_60, primals_356, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_260: "i64[]" = torch.ops.aten.add.Tensor(primals_357, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_49 = torch.ops.aten.var_mean.correction(convolution_79, [0, 2, 3], correction = 0, keepdim = True)
        getitem_98: "f32[1, 896, 1, 1]" = var_mean_49[0]
        getitem_99: "f32[1, 896, 1, 1]" = var_mean_49[1];  var_mean_49 = None
        add_261: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05)
        rsqrt_49: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        sub_49: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, getitem_99)
        mul_358: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = None
        squeeze_147: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_99, [0, 2, 3]);  getitem_99 = None
        squeeze_148: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_49, [0, 2, 3]);  rsqrt_49 = None
        mul_359: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_147, 0.1)
        mul_360: "f32[896]" = torch.ops.aten.mul.Tensor(primals_358, 0.9)
        add_262: "f32[896]" = torch.ops.aten.add.Tensor(mul_359, mul_360);  mul_359 = mul_360 = None
        squeeze_149: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_98, [0, 2, 3]);  getitem_98 = None
        mul_361: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_149, 1.0001594642002871);  squeeze_149 = None
        mul_362: "f32[896]" = torch.ops.aten.mul.Tensor(mul_361, 0.1);  mul_361 = None
        mul_363: "f32[896]" = torch.ops.aten.mul.Tensor(primals_359, 0.9)
        add_263: "f32[896]" = torch.ops.aten.add.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None
        unsqueeze_196: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_360, -1)
        unsqueeze_197: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_364: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_197);  mul_358 = unsqueeze_197 = None
        unsqueeze_198: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_361, -1);  primals_361 = None
        unsqueeze_199: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_264: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_364, unsqueeze_199);  mul_364 = unsqueeze_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_61: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_264);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_80: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_61, primals_362, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_265: "i64[]" = torch.ops.aten.add.Tensor(primals_363, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_50 = torch.ops.aten.var_mean.correction(convolution_80, [0, 2, 3], correction = 0, keepdim = True)
        getitem_100: "f32[1, 896, 1, 1]" = var_mean_50[0]
        getitem_101: "f32[1, 896, 1, 1]" = var_mean_50[1];  var_mean_50 = None
        add_266: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05)
        rsqrt_50: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_266);  add_266 = None
        sub_50: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_101)
        mul_365: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        squeeze_150: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_101, [0, 2, 3])
        mul_366: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_150, 0.1);  squeeze_150 = None
        mul_367: "f32[896]" = torch.ops.aten.mul.Tensor(primals_364, 0.9)
        add_267: "f32[896]" = torch.ops.aten.add.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
        squeeze_152: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_100, [0, 2, 3]);  getitem_100 = None
        mul_368: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_152, 1.0001594642002871);  squeeze_152 = None
        mul_369: "f32[896]" = torch.ops.aten.mul.Tensor(mul_368, 0.1);  mul_368 = None
        mul_370: "f32[896]" = torch.ops.aten.mul.Tensor(primals_365, 0.9)
        add_268: "f32[896]" = torch.ops.aten.add.Tensor(mul_369, mul_370);  mul_369 = mul_370 = None
        unsqueeze_200: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_366, -1)
        unsqueeze_201: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        mul_371: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_365, unsqueeze_201);  mul_365 = unsqueeze_201 = None
        unsqueeze_202: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_367, -1)
        unsqueeze_203: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        add_269: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_371, unsqueeze_203);  mul_371 = unsqueeze_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_62: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_269);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_15: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_62, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_81: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_15, primals_368, primals_369, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_63: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_81);  convolution_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_82: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_63, primals_370, primals_371, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_15: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_82)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_372: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_62, sigmoid_15);  relu_62 = sigmoid_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_83: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_372, primals_372, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_270: "i64[]" = torch.ops.aten.add.Tensor(primals_373, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_51 = torch.ops.aten.var_mean.correction(convolution_83, [0, 2, 3], correction = 0, keepdim = True)
        getitem_102: "f32[1, 896, 1, 1]" = var_mean_51[0]
        getitem_103: "f32[1, 896, 1, 1]" = var_mean_51[1];  var_mean_51 = None
        add_271: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05)
        rsqrt_51: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_271);  add_271 = None
        sub_51: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, getitem_103)
        mul_373: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        squeeze_153: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        squeeze_154: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_374: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_153, 0.1)
        mul_375: "f32[896]" = torch.ops.aten.mul.Tensor(primals_374, 0.9)
        add_272: "f32[896]" = torch.ops.aten.add.Tensor(mul_374, mul_375);  mul_374 = mul_375 = None
        squeeze_155: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_102, [0, 2, 3]);  getitem_102 = None
        mul_376: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_155, 1.0001594642002871);  squeeze_155 = None
        mul_377: "f32[896]" = torch.ops.aten.mul.Tensor(mul_376, 0.1);  mul_376 = None
        mul_378: "f32[896]" = torch.ops.aten.mul.Tensor(primals_375, 0.9)
        add_273: "f32[896]" = torch.ops.aten.add.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
        unsqueeze_204: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_376, -1)
        unsqueeze_205: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_379: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_205);  mul_373 = unsqueeze_205 = None
        unsqueeze_206: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_377, -1);  primals_377 = None
        unsqueeze_207: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_274: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_379, unsqueeze_207);  mul_379 = unsqueeze_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_275: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_274, relu_60);  add_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_64: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_275);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_84: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_64, primals_378, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_276: "i64[]" = torch.ops.aten.add.Tensor(primals_379, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_52 = torch.ops.aten.var_mean.correction(convolution_84, [0, 2, 3], correction = 0, keepdim = True)
        getitem_104: "f32[1, 896, 1, 1]" = var_mean_52[0]
        getitem_105: "f32[1, 896, 1, 1]" = var_mean_52[1];  var_mean_52 = None
        add_277: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05)
        rsqrt_52: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
        sub_52: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, getitem_105)
        mul_380: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        squeeze_156: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_105, [0, 2, 3]);  getitem_105 = None
        squeeze_157: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_381: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_156, 0.1)
        mul_382: "f32[896]" = torch.ops.aten.mul.Tensor(primals_380, 0.9)
        add_278: "f32[896]" = torch.ops.aten.add.Tensor(mul_381, mul_382);  mul_381 = mul_382 = None
        squeeze_158: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_104, [0, 2, 3]);  getitem_104 = None
        mul_383: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_158, 1.0001594642002871);  squeeze_158 = None
        mul_384: "f32[896]" = torch.ops.aten.mul.Tensor(mul_383, 0.1);  mul_383 = None
        mul_385: "f32[896]" = torch.ops.aten.mul.Tensor(primals_381, 0.9)
        add_279: "f32[896]" = torch.ops.aten.add.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
        unsqueeze_208: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_382, -1)
        unsqueeze_209: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        mul_386: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_380, unsqueeze_209);  mul_380 = unsqueeze_209 = None
        unsqueeze_210: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_383, -1);  primals_383 = None
        unsqueeze_211: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        add_280: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_386, unsqueeze_211);  mul_386 = unsqueeze_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_65: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_280);  add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_85: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_65, primals_384, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_281: "i64[]" = torch.ops.aten.add.Tensor(primals_385, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_53 = torch.ops.aten.var_mean.correction(convolution_85, [0, 2, 3], correction = 0, keepdim = True)
        getitem_106: "f32[1, 896, 1, 1]" = var_mean_53[0]
        getitem_107: "f32[1, 896, 1, 1]" = var_mean_53[1];  var_mean_53 = None
        add_282: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05)
        rsqrt_53: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_282);  add_282 = None
        sub_53: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, getitem_107)
        mul_387: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = None
        squeeze_159: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3])
        mul_388: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_159, 0.1);  squeeze_159 = None
        mul_389: "f32[896]" = torch.ops.aten.mul.Tensor(primals_386, 0.9)
        add_283: "f32[896]" = torch.ops.aten.add.Tensor(mul_388, mul_389);  mul_388 = mul_389 = None
        squeeze_161: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_106, [0, 2, 3]);  getitem_106 = None
        mul_390: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_161, 1.0001594642002871);  squeeze_161 = None
        mul_391: "f32[896]" = torch.ops.aten.mul.Tensor(mul_390, 0.1);  mul_390 = None
        mul_392: "f32[896]" = torch.ops.aten.mul.Tensor(primals_387, 0.9)
        add_284: "f32[896]" = torch.ops.aten.add.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
        unsqueeze_212: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_388, -1)
        unsqueeze_213: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_393: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_387, unsqueeze_213);  mul_387 = unsqueeze_213 = None
        unsqueeze_214: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_389, -1)
        unsqueeze_215: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_285: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_393, unsqueeze_215);  mul_393 = unsqueeze_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_66: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_285);  add_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_16: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_66, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_86: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_16, primals_390, primals_391, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_67: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_86);  convolution_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_87: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_67, primals_392, primals_393, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_16: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_87)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_394: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_66, sigmoid_16);  relu_66 = sigmoid_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_88: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_394, primals_394, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_286: "i64[]" = torch.ops.aten.add.Tensor(primals_395, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_54 = torch.ops.aten.var_mean.correction(convolution_88, [0, 2, 3], correction = 0, keepdim = True)
        getitem_108: "f32[1, 896, 1, 1]" = var_mean_54[0]
        getitem_109: "f32[1, 896, 1, 1]" = var_mean_54[1];  var_mean_54 = None
        add_287: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05)
        rsqrt_54: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_287);  add_287 = None
        sub_54: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, getitem_109)
        mul_395: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        squeeze_162: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_109, [0, 2, 3]);  getitem_109 = None
        squeeze_163: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_396: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_162, 0.1)
        mul_397: "f32[896]" = torch.ops.aten.mul.Tensor(primals_396, 0.9)
        add_288: "f32[896]" = torch.ops.aten.add.Tensor(mul_396, mul_397);  mul_396 = mul_397 = None
        squeeze_164: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_108, [0, 2, 3]);  getitem_108 = None
        mul_398: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_164, 1.0001594642002871);  squeeze_164 = None
        mul_399: "f32[896]" = torch.ops.aten.mul.Tensor(mul_398, 0.1);  mul_398 = None
        mul_400: "f32[896]" = torch.ops.aten.mul.Tensor(primals_397, 0.9)
        add_289: "f32[896]" = torch.ops.aten.add.Tensor(mul_399, mul_400);  mul_399 = mul_400 = None
        unsqueeze_216: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_398, -1)
        unsqueeze_217: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        mul_401: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_395, unsqueeze_217);  mul_395 = unsqueeze_217 = None
        unsqueeze_218: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_399, -1);  primals_399 = None
        unsqueeze_219: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        add_290: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_401, unsqueeze_219);  mul_401 = unsqueeze_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_291: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_290, relu_64);  add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_68: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_291);  add_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_89: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_68, primals_400, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_292: "i64[]" = torch.ops.aten.add.Tensor(primals_401, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_55 = torch.ops.aten.var_mean.correction(convolution_89, [0, 2, 3], correction = 0, keepdim = True)
        getitem_110: "f32[1, 896, 1, 1]" = var_mean_55[0]
        getitem_111: "f32[1, 896, 1, 1]" = var_mean_55[1];  var_mean_55 = None
        add_293: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05)
        rsqrt_55: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_293);  add_293 = None
        sub_55: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, getitem_111)
        mul_402: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        squeeze_165: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_111, [0, 2, 3]);  getitem_111 = None
        squeeze_166: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_403: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_165, 0.1)
        mul_404: "f32[896]" = torch.ops.aten.mul.Tensor(primals_402, 0.9)
        add_294: "f32[896]" = torch.ops.aten.add.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
        squeeze_167: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_110, [0, 2, 3]);  getitem_110 = None
        mul_405: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_167, 1.0001594642002871);  squeeze_167 = None
        mul_406: "f32[896]" = torch.ops.aten.mul.Tensor(mul_405, 0.1);  mul_405 = None
        mul_407: "f32[896]" = torch.ops.aten.mul.Tensor(primals_403, 0.9)
        add_295: "f32[896]" = torch.ops.aten.add.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
        unsqueeze_220: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_404, -1)
        unsqueeze_221: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_408: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_402, unsqueeze_221);  mul_402 = unsqueeze_221 = None
        unsqueeze_222: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_405, -1);  primals_405 = None
        unsqueeze_223: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_296: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_408, unsqueeze_223);  mul_408 = unsqueeze_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_69: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_296);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_90: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(relu_69, primals_406, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_297: "i64[]" = torch.ops.aten.add.Tensor(primals_407, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_56 = torch.ops.aten.var_mean.correction(convolution_90, [0, 2, 3], correction = 0, keepdim = True)
        getitem_112: "f32[1, 896, 1, 1]" = var_mean_56[0]
        getitem_113: "f32[1, 896, 1, 1]" = var_mean_56[1];  var_mean_56 = None
        add_298: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05)
        rsqrt_56: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_298);  add_298 = None
        sub_56: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, getitem_113)
        mul_409: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = None
        squeeze_168: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3])
        mul_410: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_168, 0.1);  squeeze_168 = None
        mul_411: "f32[896]" = torch.ops.aten.mul.Tensor(primals_408, 0.9)
        add_299: "f32[896]" = torch.ops.aten.add.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
        squeeze_170: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_112, [0, 2, 3]);  getitem_112 = None
        mul_412: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_170, 1.0001594642002871);  squeeze_170 = None
        mul_413: "f32[896]" = torch.ops.aten.mul.Tensor(mul_412, 0.1);  mul_412 = None
        mul_414: "f32[896]" = torch.ops.aten.mul.Tensor(primals_409, 0.9)
        add_300: "f32[896]" = torch.ops.aten.add.Tensor(mul_413, mul_414);  mul_413 = mul_414 = None
        unsqueeze_224: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_410, -1)
        unsqueeze_225: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        mul_415: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_225);  mul_409 = unsqueeze_225 = None
        unsqueeze_226: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_411, -1)
        unsqueeze_227: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        add_301: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_415, unsqueeze_227);  mul_415 = unsqueeze_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_70: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_301);  add_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_17: "f32[32, 896, 1, 1]" = torch.ops.aten.mean.dim(relu_70, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_91: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_17, primals_412, primals_413, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_71: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_91);  convolution_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_92: "f32[32, 896, 1, 1]" = torch.ops.aten.convolution.default(relu_71, primals_414, primals_415, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_17: "f32[32, 896, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_92)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_416: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(relu_70, sigmoid_17);  relu_70 = sigmoid_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_93: "f32[32, 896, 14, 14]" = torch.ops.aten.convolution.default(mul_416, primals_416, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_302: "i64[]" = torch.ops.aten.add.Tensor(primals_417, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_57 = torch.ops.aten.var_mean.correction(convolution_93, [0, 2, 3], correction = 0, keepdim = True)
        getitem_114: "f32[1, 896, 1, 1]" = var_mean_57[0]
        getitem_115: "f32[1, 896, 1, 1]" = var_mean_57[1];  var_mean_57 = None
        add_303: "f32[1, 896, 1, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05)
        rsqrt_57: "f32[1, 896, 1, 1]" = torch.ops.aten.rsqrt.default(add_303);  add_303 = None
        sub_57: "f32[32, 896, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, getitem_115)
        mul_417: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = None
        squeeze_171: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_115, [0, 2, 3]);  getitem_115 = None
        squeeze_172: "f32[896]" = torch.ops.aten.squeeze.dims(rsqrt_57, [0, 2, 3]);  rsqrt_57 = None
        mul_418: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_171, 0.1)
        mul_419: "f32[896]" = torch.ops.aten.mul.Tensor(primals_418, 0.9)
        add_304: "f32[896]" = torch.ops.aten.add.Tensor(mul_418, mul_419);  mul_418 = mul_419 = None
        squeeze_173: "f32[896]" = torch.ops.aten.squeeze.dims(getitem_114, [0, 2, 3]);  getitem_114 = None
        mul_420: "f32[896]" = torch.ops.aten.mul.Tensor(squeeze_173, 1.0001594642002871);  squeeze_173 = None
        mul_421: "f32[896]" = torch.ops.aten.mul.Tensor(mul_420, 0.1);  mul_420 = None
        mul_422: "f32[896]" = torch.ops.aten.mul.Tensor(primals_419, 0.9)
        add_305: "f32[896]" = torch.ops.aten.add.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
        unsqueeze_228: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_420, -1)
        unsqueeze_229: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_423: "f32[32, 896, 14, 14]" = torch.ops.aten.mul.Tensor(mul_417, unsqueeze_229);  mul_417 = unsqueeze_229 = None
        unsqueeze_230: "f32[896, 1]" = torch.ops.aten.unsqueeze.default(primals_421, -1);  primals_421 = None
        unsqueeze_231: "f32[896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_306: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(mul_423, unsqueeze_231);  mul_423 = unsqueeze_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_307: "f32[32, 896, 14, 14]" = torch.ops.aten.add.Tensor(add_306, relu_68);  add_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_72: "f32[32, 896, 14, 14]" = torch.ops.aten.relu.default(add_307);  add_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_94: "f32[32, 2240, 14, 14]" = torch.ops.aten.convolution.default(relu_72, primals_422, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_308: "i64[]" = torch.ops.aten.add.Tensor(primals_423, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_58 = torch.ops.aten.var_mean.correction(convolution_94, [0, 2, 3], correction = 0, keepdim = True)
        getitem_116: "f32[1, 2240, 1, 1]" = var_mean_58[0]
        getitem_117: "f32[1, 2240, 1, 1]" = var_mean_58[1];  var_mean_58 = None
        add_309: "f32[1, 2240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05)
        rsqrt_58: "f32[1, 2240, 1, 1]" = torch.ops.aten.rsqrt.default(add_309);  add_309 = None
        sub_58: "f32[32, 2240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, getitem_117)
        mul_424: "f32[32, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = None
        squeeze_174: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        squeeze_175: "f32[2240]" = torch.ops.aten.squeeze.dims(rsqrt_58, [0, 2, 3]);  rsqrt_58 = None
        mul_425: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_174, 0.1)
        mul_426: "f32[2240]" = torch.ops.aten.mul.Tensor(primals_424, 0.9)
        add_310: "f32[2240]" = torch.ops.aten.add.Tensor(mul_425, mul_426);  mul_425 = mul_426 = None
        squeeze_176: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_116, [0, 2, 3]);  getitem_116 = None
        mul_427: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_176, 1.0001594642002871);  squeeze_176 = None
        mul_428: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_427, 0.1);  mul_427 = None
        mul_429: "f32[2240]" = torch.ops.aten.mul.Tensor(primals_425, 0.9)
        add_311: "f32[2240]" = torch.ops.aten.add.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
        unsqueeze_232: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_426, -1)
        unsqueeze_233: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        mul_430: "f32[32, 2240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_424, unsqueeze_233);  mul_424 = unsqueeze_233 = None
        unsqueeze_234: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_427, -1);  primals_427 = None
        unsqueeze_235: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        add_312: "f32[32, 2240, 14, 14]" = torch.ops.aten.add.Tensor(mul_430, unsqueeze_235);  mul_430 = unsqueeze_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_73: "f32[32, 2240, 14, 14]" = torch.ops.aten.relu.default(add_312);  add_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_95: "f32[32, 2240, 7, 7]" = torch.ops.aten.convolution.default(relu_73, primals_428, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_313: "i64[]" = torch.ops.aten.add.Tensor(primals_429, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_59 = torch.ops.aten.var_mean.correction(convolution_95, [0, 2, 3], correction = 0, keepdim = True)
        getitem_118: "f32[1, 2240, 1, 1]" = var_mean_59[0]
        getitem_119: "f32[1, 2240, 1, 1]" = var_mean_59[1];  var_mean_59 = None
        add_314: "f32[1, 2240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05)
        rsqrt_59: "f32[1, 2240, 1, 1]" = torch.ops.aten.rsqrt.default(add_314);  add_314 = None
        sub_59: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_95, getitem_119)
        mul_431: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = None
        squeeze_177: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3])
        mul_432: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_177, 0.1);  squeeze_177 = None
        mul_433: "f32[2240]" = torch.ops.aten.mul.Tensor(primals_430, 0.9)
        add_315: "f32[2240]" = torch.ops.aten.add.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
        squeeze_179: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_118, [0, 2, 3]);  getitem_118 = None
        mul_434: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_179, 1.0006381620931717);  squeeze_179 = None
        mul_435: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_434, 0.1);  mul_434 = None
        mul_436: "f32[2240]" = torch.ops.aten.mul.Tensor(primals_431, 0.9)
        add_316: "f32[2240]" = torch.ops.aten.add.Tensor(mul_435, mul_436);  mul_435 = mul_436 = None
        unsqueeze_236: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_432, -1)
        unsqueeze_237: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_437: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_431, unsqueeze_237);  mul_431 = unsqueeze_237 = None
        unsqueeze_238: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_433, -1)
        unsqueeze_239: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_317: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_437, unsqueeze_239);  mul_437 = unsqueeze_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_74: "f32[32, 2240, 7, 7]" = torch.ops.aten.relu.default(add_317);  add_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:56 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        mean_18: "f32[32, 2240, 1, 1]" = torch.ops.aten.mean.dim(relu_74, [2, 3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:60 in forward, code: x_se = self.fc1(x_se)
        convolution_96: "f32[32, 224, 1, 1]" = torch.ops.aten.convolution.default(mean_18, primals_434, primals_435, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:61 in forward, code: x_se = self.act(self.bn(x_se))
        relu_75: "f32[32, 224, 1, 1]" = torch.ops.aten.relu.default(convolution_96);  convolution_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:62 in forward, code: x_se = self.fc2(x_se)
        convolution_97: "f32[32, 2240, 1, 1]" = torch.ops.aten.convolution.default(relu_75, primals_436, primals_437, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  primals_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:57 in forward, code: return x.sigmoid_() if self.inplace else x.sigmoid()
        sigmoid_18: "f32[32, 2240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_97)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/squeeze_excite.py:63 in forward, code: return x * self.gate(x_se)
        mul_438: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(relu_74, sigmoid_18);  relu_74 = sigmoid_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_98: "f32[32, 2240, 7, 7]" = torch.ops.aten.convolution.default(mul_438, primals_438, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_318: "i64[]" = torch.ops.aten.add.Tensor(primals_439, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_60 = torch.ops.aten.var_mean.correction(convolution_98, [0, 2, 3], correction = 0, keepdim = True)
        getitem_120: "f32[1, 2240, 1, 1]" = var_mean_60[0]
        getitem_121: "f32[1, 2240, 1, 1]" = var_mean_60[1];  var_mean_60 = None
        add_319: "f32[1, 2240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05)
        rsqrt_60: "f32[1, 2240, 1, 1]" = torch.ops.aten.rsqrt.default(add_319);  add_319 = None
        sub_60: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_98, getitem_121)
        mul_439: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = None
        squeeze_180: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3])
        mul_440: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_180, 0.1);  squeeze_180 = None
        mul_441: "f32[2240]" = torch.ops.aten.mul.Tensor(primals_440, 0.9)
        add_320: "f32[2240]" = torch.ops.aten.add.Tensor(mul_440, mul_441);  mul_440 = mul_441 = None
        squeeze_182: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_120, [0, 2, 3]);  getitem_120 = None
        mul_442: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_182, 1.0006381620931717);  squeeze_182 = None
        mul_443: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_442, 0.1);  mul_442 = None
        mul_444: "f32[2240]" = torch.ops.aten.mul.Tensor(primals_441, 0.9)
        add_321: "f32[2240]" = torch.ops.aten.add.Tensor(mul_443, mul_444);  mul_443 = mul_444 = None
        unsqueeze_240: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_442, -1)
        unsqueeze_241: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        mul_445: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_241);  mul_439 = unsqueeze_241 = None
        unsqueeze_242: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_443, -1)
        unsqueeze_243: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        add_322: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_445, unsqueeze_243);  mul_445 = unsqueeze_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_99: "f32[32, 2240, 7, 7]" = torch.ops.aten.convolution.default(relu_72, primals_444, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:116 in forward, code: self.num_batches_tracked.add_(1)  # type: ignore[has-type]
        add_323: "i64[]" = torch.ops.aten.add.Tensor(primals_445, 1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        var_mean_61 = torch.ops.aten.var_mean.correction(convolution_99, [0, 2, 3], correction = 0, keepdim = True)
        getitem_122: "f32[1, 2240, 1, 1]" = var_mean_61[0]
        getitem_123: "f32[1, 2240, 1, 1]" = var_mean_61[1];  var_mean_61 = None
        add_324: "f32[1, 2240, 1, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05)
        rsqrt_61: "f32[1, 2240, 1, 1]" = torch.ops.aten.rsqrt.default(add_324);  add_324 = None
        sub_61: "f32[32, 2240, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_99, getitem_123)
        mul_446: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = None
        squeeze_183: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3])
        mul_447: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_183, 0.1);  squeeze_183 = None
        mul_448: "f32[2240]" = torch.ops.aten.mul.Tensor(primals_446, 0.9)
        add_325: "f32[2240]" = torch.ops.aten.add.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
        squeeze_185: "f32[2240]" = torch.ops.aten.squeeze.dims(getitem_122, [0, 2, 3]);  getitem_122 = None
        mul_449: "f32[2240]" = torch.ops.aten.mul.Tensor(squeeze_185, 1.0006381620931717);  squeeze_185 = None
        mul_450: "f32[2240]" = torch.ops.aten.mul.Tensor(mul_449, 0.1);  mul_449 = None
        mul_451: "f32[2240]" = torch.ops.aten.mul.Tensor(primals_447, 0.9)
        add_326: "f32[2240]" = torch.ops.aten.add.Tensor(mul_450, mul_451);  mul_450 = mul_451 = None
        unsqueeze_244: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_448, -1)
        unsqueeze_245: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_452: "f32[32, 2240, 7, 7]" = torch.ops.aten.mul.Tensor(mul_446, unsqueeze_245);  mul_446 = unsqueeze_245 = None
        unsqueeze_246: "f32[2240, 1]" = torch.ops.aten.unsqueeze.default(primals_449, -1)
        unsqueeze_247: "f32[2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_327: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_247);  mul_452 = unsqueeze_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:373 in forward, code: x = self.drop_path(x) + self.downsample(shortcut)
        add_328: "f32[32, 2240, 7, 7]" = torch.ops.aten.add.Tensor(add_322, add_327);  add_322 = add_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/regnet.py:374 in forward, code: x = self.act3(x)
        relu_76: "f32[32, 2240, 7, 7]" = torch.ops.aten.relu.default(add_328);  add_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        mean_19: "f32[32, 2240, 1, 1]" = torch.ops.aten.mean.dim(relu_76, [-1, -2], True);  relu_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view: "f32[32, 2240]" = torch.ops.aten.reshape.default(mean_19, [32, 2240]);  mean_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute: "f32[2240, 1000]" = torch.ops.aten.permute.default(primals_450, [1, 0])
        addmm: "f32[32, 1000]" = torch.ops.aten.addmm.default(primals_451, view, permute);  primals_451 = permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        unsqueeze_284: "f32[1, 2240]" = torch.ops.aten.unsqueeze.default(squeeze_174, 0);  squeeze_174 = None
        unsqueeze_285: "f32[1, 2240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
        unsqueeze_286: "f32[1, 2240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
        unsqueeze_296: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_171, 0);  squeeze_171 = None
        unsqueeze_297: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
        unsqueeze_298: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
        unsqueeze_320: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
        unsqueeze_321: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 2);  unsqueeze_320 = None
        unsqueeze_322: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 3);  unsqueeze_321 = None
        unsqueeze_332: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
        unsqueeze_333: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
        unsqueeze_334: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
        unsqueeze_356: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_156, 0);  squeeze_156 = None
        unsqueeze_357: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 2);  unsqueeze_356 = None
        unsqueeze_358: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 3);  unsqueeze_357 = None
        unsqueeze_368: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_369: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
        unsqueeze_370: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
        unsqueeze_392: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_147, 0);  squeeze_147 = None
        unsqueeze_393: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
        unsqueeze_394: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
        unsqueeze_404: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_405: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
        unsqueeze_406: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
        unsqueeze_428: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_429: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
        unsqueeze_430: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
        unsqueeze_440: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_441: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
        unsqueeze_442: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
        unsqueeze_464: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_465: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
        unsqueeze_466: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
        unsqueeze_476: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_477: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
        unsqueeze_478: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
        unsqueeze_500: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_501: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
        unsqueeze_502: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
        unsqueeze_512: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_513: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
        unsqueeze_514: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
        unsqueeze_536: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_537: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
        unsqueeze_538: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
        unsqueeze_548: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_549: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
        unsqueeze_550: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
        unsqueeze_572: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_573: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
        unsqueeze_574: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
        unsqueeze_584: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_585: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
        unsqueeze_586: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
        unsqueeze_608: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_609: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
        unsqueeze_610: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
        unsqueeze_620: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_621: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
        unsqueeze_622: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
        unsqueeze_644: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_645: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
        unsqueeze_646: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
        unsqueeze_656: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_657: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
        unsqueeze_658: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
        unsqueeze_668: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_669: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
        unsqueeze_670: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
        unsqueeze_692: "f32[1, 896]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_693: "f32[1, 896, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
        unsqueeze_694: "f32[1, 896, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
        unsqueeze_704: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_705: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
        unsqueeze_706: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
        unsqueeze_728: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_729: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
        unsqueeze_730: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
        unsqueeze_740: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_741: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
        unsqueeze_742: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
        unsqueeze_764: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_765: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
        unsqueeze_766: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
        unsqueeze_776: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_777: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
        unsqueeze_778: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
        unsqueeze_800: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_801: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
        unsqueeze_802: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
        unsqueeze_812: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_813: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
        unsqueeze_814: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
        unsqueeze_836: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_837: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
        unsqueeze_838: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
        unsqueeze_848: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_849: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
        unsqueeze_850: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
        unsqueeze_860: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_861: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
        unsqueeze_862: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
        unsqueeze_884: "f32[1, 448]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_885: "f32[1, 448, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
        unsqueeze_886: "f32[1, 448, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
        unsqueeze_896: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_897: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
        unsqueeze_898: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
        unsqueeze_920: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_921: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 2);  unsqueeze_920 = None
        unsqueeze_922: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_921, 3);  unsqueeze_921 = None
        unsqueeze_932: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_933: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 2);  unsqueeze_932 = None
        unsqueeze_934: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_933, 3);  unsqueeze_933 = None
        unsqueeze_944: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_945: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 2);  unsqueeze_944 = None
        unsqueeze_946: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_945, 3);  unsqueeze_945 = None
        unsqueeze_968: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_969: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 2);  unsqueeze_968 = None
        unsqueeze_970: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_969, 3);  unsqueeze_969 = None
        unsqueeze_980: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_981: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 2);  unsqueeze_980 = None
        unsqueeze_982: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_981, 3);  unsqueeze_981 = None

        # No stacktrace found for following nodes
        copy_: "i64[]" = torch.ops.aten.copy_.default(primals_3, add);  primals_3 = add = copy_ = None
        copy__1: "f32[32]" = torch.ops.aten.copy_.default(primals_4, add_2);  primals_4 = add_2 = copy__1 = None
        copy__2: "f32[32]" = torch.ops.aten.copy_.default(primals_5, add_3);  primals_5 = add_3 = copy__2 = None
        copy__3: "i64[]" = torch.ops.aten.copy_.default(primals_9, add_5);  primals_9 = add_5 = copy__3 = None
        copy__4: "f32[224]" = torch.ops.aten.copy_.default(primals_10, add_7);  primals_10 = add_7 = copy__4 = None
        copy__5: "f32[224]" = torch.ops.aten.copy_.default(primals_11, add_8);  primals_11 = add_8 = copy__5 = None
        copy__6: "i64[]" = torch.ops.aten.copy_.default(primals_15, add_10);  primals_15 = add_10 = copy__6 = None
        copy__7: "f32[224]" = torch.ops.aten.copy_.default(primals_16, add_12);  primals_16 = add_12 = copy__7 = None
        copy__8: "f32[224]" = torch.ops.aten.copy_.default(primals_17, add_13);  primals_17 = add_13 = copy__8 = None
        copy__9: "i64[]" = torch.ops.aten.copy_.default(primals_25, add_15);  primals_25 = add_15 = copy__9 = None
        copy__10: "f32[224]" = torch.ops.aten.copy_.default(primals_26, add_17);  primals_26 = add_17 = copy__10 = None
        copy__11: "f32[224]" = torch.ops.aten.copy_.default(primals_27, add_18);  primals_27 = add_18 = copy__11 = None
        copy__12: "i64[]" = torch.ops.aten.copy_.default(primals_31, add_20);  primals_31 = add_20 = copy__12 = None
        copy__13: "f32[224]" = torch.ops.aten.copy_.default(primals_32, add_22);  primals_32 = add_22 = copy__13 = None
        copy__14: "f32[224]" = torch.ops.aten.copy_.default(primals_33, add_23);  primals_33 = add_23 = copy__14 = None
        copy__15: "i64[]" = torch.ops.aten.copy_.default(primals_37, add_26);  primals_37 = add_26 = copy__15 = None
        copy__16: "f32[224]" = torch.ops.aten.copy_.default(primals_38, add_28);  primals_38 = add_28 = copy__16 = None
        copy__17: "f32[224]" = torch.ops.aten.copy_.default(primals_39, add_29);  primals_39 = add_29 = copy__17 = None
        copy__18: "i64[]" = torch.ops.aten.copy_.default(primals_43, add_31);  primals_43 = add_31 = copy__18 = None
        copy__19: "f32[224]" = torch.ops.aten.copy_.default(primals_44, add_33);  primals_44 = add_33 = copy__19 = None
        copy__20: "f32[224]" = torch.ops.aten.copy_.default(primals_45, add_34);  primals_45 = add_34 = copy__20 = None
        copy__21: "i64[]" = torch.ops.aten.copy_.default(primals_53, add_36);  primals_53 = add_36 = copy__21 = None
        copy__22: "f32[224]" = torch.ops.aten.copy_.default(primals_54, add_38);  primals_54 = add_38 = copy__22 = None
        copy__23: "f32[224]" = torch.ops.aten.copy_.default(primals_55, add_39);  primals_55 = add_39 = copy__23 = None
        copy__24: "i64[]" = torch.ops.aten.copy_.default(primals_59, add_42);  primals_59 = add_42 = copy__24 = None
        copy__25: "f32[448]" = torch.ops.aten.copy_.default(primals_60, add_44);  primals_60 = add_44 = copy__25 = None
        copy__26: "f32[448]" = torch.ops.aten.copy_.default(primals_61, add_45);  primals_61 = add_45 = copy__26 = None
        copy__27: "i64[]" = torch.ops.aten.copy_.default(primals_65, add_47);  primals_65 = add_47 = copy__27 = None
        copy__28: "f32[448]" = torch.ops.aten.copy_.default(primals_66, add_49);  primals_66 = add_49 = copy__28 = None
        copy__29: "f32[448]" = torch.ops.aten.copy_.default(primals_67, add_50);  primals_67 = add_50 = copy__29 = None
        copy__30: "i64[]" = torch.ops.aten.copy_.default(primals_75, add_52);  primals_75 = add_52 = copy__30 = None
        copy__31: "f32[448]" = torch.ops.aten.copy_.default(primals_76, add_54);  primals_76 = add_54 = copy__31 = None
        copy__32: "f32[448]" = torch.ops.aten.copy_.default(primals_77, add_55);  primals_77 = add_55 = copy__32 = None
        copy__33: "i64[]" = torch.ops.aten.copy_.default(primals_81, add_57);  primals_81 = add_57 = copy__33 = None
        copy__34: "f32[448]" = torch.ops.aten.copy_.default(primals_82, add_59);  primals_82 = add_59 = copy__34 = None
        copy__35: "f32[448]" = torch.ops.aten.copy_.default(primals_83, add_60);  primals_83 = add_60 = copy__35 = None
        copy__36: "i64[]" = torch.ops.aten.copy_.default(primals_87, add_63);  primals_87 = add_63 = copy__36 = None
        copy__37: "f32[448]" = torch.ops.aten.copy_.default(primals_88, add_65);  primals_88 = add_65 = copy__37 = None
        copy__38: "f32[448]" = torch.ops.aten.copy_.default(primals_89, add_66);  primals_89 = add_66 = copy__38 = None
        copy__39: "i64[]" = torch.ops.aten.copy_.default(primals_93, add_68);  primals_93 = add_68 = copy__39 = None
        copy__40: "f32[448]" = torch.ops.aten.copy_.default(primals_94, add_70);  primals_94 = add_70 = copy__40 = None
        copy__41: "f32[448]" = torch.ops.aten.copy_.default(primals_95, add_71);  primals_95 = add_71 = copy__41 = None
        copy__42: "i64[]" = torch.ops.aten.copy_.default(primals_103, add_73);  primals_103 = add_73 = copy__42 = None
        copy__43: "f32[448]" = torch.ops.aten.copy_.default(primals_104, add_75);  primals_104 = add_75 = copy__43 = None
        copy__44: "f32[448]" = torch.ops.aten.copy_.default(primals_105, add_76);  primals_105 = add_76 = copy__44 = None
        copy__45: "i64[]" = torch.ops.aten.copy_.default(primals_109, add_79);  primals_109 = add_79 = copy__45 = None
        copy__46: "f32[448]" = torch.ops.aten.copy_.default(primals_110, add_81);  primals_110 = add_81 = copy__46 = None
        copy__47: "f32[448]" = torch.ops.aten.copy_.default(primals_111, add_82);  primals_111 = add_82 = copy__47 = None
        copy__48: "i64[]" = torch.ops.aten.copy_.default(primals_115, add_84);  primals_115 = add_84 = copy__48 = None
        copy__49: "f32[448]" = torch.ops.aten.copy_.default(primals_116, add_86);  primals_116 = add_86 = copy__49 = None
        copy__50: "f32[448]" = torch.ops.aten.copy_.default(primals_117, add_87);  primals_117 = add_87 = copy__50 = None
        copy__51: "i64[]" = torch.ops.aten.copy_.default(primals_125, add_89);  primals_125 = add_89 = copy__51 = None
        copy__52: "f32[448]" = torch.ops.aten.copy_.default(primals_126, add_91);  primals_126 = add_91 = copy__52 = None
        copy__53: "f32[448]" = torch.ops.aten.copy_.default(primals_127, add_92);  primals_127 = add_92 = copy__53 = None
        copy__54: "i64[]" = torch.ops.aten.copy_.default(primals_131, add_95);  primals_131 = add_95 = copy__54 = None
        copy__55: "f32[448]" = torch.ops.aten.copy_.default(primals_132, add_97);  primals_132 = add_97 = copy__55 = None
        copy__56: "f32[448]" = torch.ops.aten.copy_.default(primals_133, add_98);  primals_133 = add_98 = copy__56 = None
        copy__57: "i64[]" = torch.ops.aten.copy_.default(primals_137, add_100);  primals_137 = add_100 = copy__57 = None
        copy__58: "f32[448]" = torch.ops.aten.copy_.default(primals_138, add_102);  primals_138 = add_102 = copy__58 = None
        copy__59: "f32[448]" = torch.ops.aten.copy_.default(primals_139, add_103);  primals_139 = add_103 = copy__59 = None
        copy__60: "i64[]" = torch.ops.aten.copy_.default(primals_147, add_105);  primals_147 = add_105 = copy__60 = None
        copy__61: "f32[448]" = torch.ops.aten.copy_.default(primals_148, add_107);  primals_148 = add_107 = copy__61 = None
        copy__62: "f32[448]" = torch.ops.aten.copy_.default(primals_149, add_108);  primals_149 = add_108 = copy__62 = None
        copy__63: "i64[]" = torch.ops.aten.copy_.default(primals_153, add_111);  primals_153 = add_111 = copy__63 = None
        copy__64: "f32[448]" = torch.ops.aten.copy_.default(primals_154, add_113);  primals_154 = add_113 = copy__64 = None
        copy__65: "f32[448]" = torch.ops.aten.copy_.default(primals_155, add_114);  primals_155 = add_114 = copy__65 = None
        copy__66: "i64[]" = torch.ops.aten.copy_.default(primals_159, add_116);  primals_159 = add_116 = copy__66 = None
        copy__67: "f32[448]" = torch.ops.aten.copy_.default(primals_160, add_118);  primals_160 = add_118 = copy__67 = None
        copy__68: "f32[448]" = torch.ops.aten.copy_.default(primals_161, add_119);  primals_161 = add_119 = copy__68 = None
        copy__69: "i64[]" = torch.ops.aten.copy_.default(primals_169, add_121);  primals_169 = add_121 = copy__69 = None
        copy__70: "f32[448]" = torch.ops.aten.copy_.default(primals_170, add_123);  primals_170 = add_123 = copy__70 = None
        copy__71: "f32[448]" = torch.ops.aten.copy_.default(primals_171, add_124);  primals_171 = add_124 = copy__71 = None
        copy__72: "i64[]" = torch.ops.aten.copy_.default(primals_175, add_127);  primals_175 = add_127 = copy__72 = None
        copy__73: "f32[896]" = torch.ops.aten.copy_.default(primals_176, add_129);  primals_176 = add_129 = copy__73 = None
        copy__74: "f32[896]" = torch.ops.aten.copy_.default(primals_177, add_130);  primals_177 = add_130 = copy__74 = None
        copy__75: "i64[]" = torch.ops.aten.copy_.default(primals_181, add_132);  primals_181 = add_132 = copy__75 = None
        copy__76: "f32[896]" = torch.ops.aten.copy_.default(primals_182, add_134);  primals_182 = add_134 = copy__76 = None
        copy__77: "f32[896]" = torch.ops.aten.copy_.default(primals_183, add_135);  primals_183 = add_135 = copy__77 = None
        copy__78: "i64[]" = torch.ops.aten.copy_.default(primals_191, add_137);  primals_191 = add_137 = copy__78 = None
        copy__79: "f32[896]" = torch.ops.aten.copy_.default(primals_192, add_139);  primals_192 = add_139 = copy__79 = None
        copy__80: "f32[896]" = torch.ops.aten.copy_.default(primals_193, add_140);  primals_193 = add_140 = copy__80 = None
        copy__81: "i64[]" = torch.ops.aten.copy_.default(primals_197, add_142);  primals_197 = add_142 = copy__81 = None
        copy__82: "f32[896]" = torch.ops.aten.copy_.default(primals_198, add_144);  primals_198 = add_144 = copy__82 = None
        copy__83: "f32[896]" = torch.ops.aten.copy_.default(primals_199, add_145);  primals_199 = add_145 = copy__83 = None
        copy__84: "i64[]" = torch.ops.aten.copy_.default(primals_203, add_148);  primals_203 = add_148 = copy__84 = None
        copy__85: "f32[896]" = torch.ops.aten.copy_.default(primals_204, add_150);  primals_204 = add_150 = copy__85 = None
        copy__86: "f32[896]" = torch.ops.aten.copy_.default(primals_205, add_151);  primals_205 = add_151 = copy__86 = None
        copy__87: "i64[]" = torch.ops.aten.copy_.default(primals_209, add_153);  primals_209 = add_153 = copy__87 = None
        copy__88: "f32[896]" = torch.ops.aten.copy_.default(primals_210, add_155);  primals_210 = add_155 = copy__88 = None
        copy__89: "f32[896]" = torch.ops.aten.copy_.default(primals_211, add_156);  primals_211 = add_156 = copy__89 = None
        copy__90: "i64[]" = torch.ops.aten.copy_.default(primals_219, add_158);  primals_219 = add_158 = copy__90 = None
        copy__91: "f32[896]" = torch.ops.aten.copy_.default(primals_220, add_160);  primals_220 = add_160 = copy__91 = None
        copy__92: "f32[896]" = torch.ops.aten.copy_.default(primals_221, add_161);  primals_221 = add_161 = copy__92 = None
        copy__93: "i64[]" = torch.ops.aten.copy_.default(primals_225, add_164);  primals_225 = add_164 = copy__93 = None
        copy__94: "f32[896]" = torch.ops.aten.copy_.default(primals_226, add_166);  primals_226 = add_166 = copy__94 = None
        copy__95: "f32[896]" = torch.ops.aten.copy_.default(primals_227, add_167);  primals_227 = add_167 = copy__95 = None
        copy__96: "i64[]" = torch.ops.aten.copy_.default(primals_231, add_169);  primals_231 = add_169 = copy__96 = None
        copy__97: "f32[896]" = torch.ops.aten.copy_.default(primals_232, add_171);  primals_232 = add_171 = copy__97 = None
        copy__98: "f32[896]" = torch.ops.aten.copy_.default(primals_233, add_172);  primals_233 = add_172 = copy__98 = None
        copy__99: "i64[]" = torch.ops.aten.copy_.default(primals_241, add_174);  primals_241 = add_174 = copy__99 = None
        copy__100: "f32[896]" = torch.ops.aten.copy_.default(primals_242, add_176);  primals_242 = add_176 = copy__100 = None
        copy__101: "f32[896]" = torch.ops.aten.copy_.default(primals_243, add_177);  primals_243 = add_177 = copy__101 = None
        copy__102: "i64[]" = torch.ops.aten.copy_.default(primals_247, add_180);  primals_247 = add_180 = copy__102 = None
        copy__103: "f32[896]" = torch.ops.aten.copy_.default(primals_248, add_182);  primals_248 = add_182 = copy__103 = None
        copy__104: "f32[896]" = torch.ops.aten.copy_.default(primals_249, add_183);  primals_249 = add_183 = copy__104 = None
        copy__105: "i64[]" = torch.ops.aten.copy_.default(primals_253, add_185);  primals_253 = add_185 = copy__105 = None
        copy__106: "f32[896]" = torch.ops.aten.copy_.default(primals_254, add_187);  primals_254 = add_187 = copy__106 = None
        copy__107: "f32[896]" = torch.ops.aten.copy_.default(primals_255, add_188);  primals_255 = add_188 = copy__107 = None
        copy__108: "i64[]" = torch.ops.aten.copy_.default(primals_263, add_190);  primals_263 = add_190 = copy__108 = None
        copy__109: "f32[896]" = torch.ops.aten.copy_.default(primals_264, add_192);  primals_264 = add_192 = copy__109 = None
        copy__110: "f32[896]" = torch.ops.aten.copy_.default(primals_265, add_193);  primals_265 = add_193 = copy__110 = None
        copy__111: "i64[]" = torch.ops.aten.copy_.default(primals_269, add_196);  primals_269 = add_196 = copy__111 = None
        copy__112: "f32[896]" = torch.ops.aten.copy_.default(primals_270, add_198);  primals_270 = add_198 = copy__112 = None
        copy__113: "f32[896]" = torch.ops.aten.copy_.default(primals_271, add_199);  primals_271 = add_199 = copy__113 = None
        copy__114: "i64[]" = torch.ops.aten.copy_.default(primals_275, add_201);  primals_275 = add_201 = copy__114 = None
        copy__115: "f32[896]" = torch.ops.aten.copy_.default(primals_276, add_203);  primals_276 = add_203 = copy__115 = None
        copy__116: "f32[896]" = torch.ops.aten.copy_.default(primals_277, add_204);  primals_277 = add_204 = copy__116 = None
        copy__117: "i64[]" = torch.ops.aten.copy_.default(primals_285, add_206);  primals_285 = add_206 = copy__117 = None
        copy__118: "f32[896]" = torch.ops.aten.copy_.default(primals_286, add_208);  primals_286 = add_208 = copy__118 = None
        copy__119: "f32[896]" = torch.ops.aten.copy_.default(primals_287, add_209);  primals_287 = add_209 = copy__119 = None
        copy__120: "i64[]" = torch.ops.aten.copy_.default(primals_291, add_212);  primals_291 = add_212 = copy__120 = None
        copy__121: "f32[896]" = torch.ops.aten.copy_.default(primals_292, add_214);  primals_292 = add_214 = copy__121 = None
        copy__122: "f32[896]" = torch.ops.aten.copy_.default(primals_293, add_215);  primals_293 = add_215 = copy__122 = None
        copy__123: "i64[]" = torch.ops.aten.copy_.default(primals_297, add_217);  primals_297 = add_217 = copy__123 = None
        copy__124: "f32[896]" = torch.ops.aten.copy_.default(primals_298, add_219);  primals_298 = add_219 = copy__124 = None
        copy__125: "f32[896]" = torch.ops.aten.copy_.default(primals_299, add_220);  primals_299 = add_220 = copy__125 = None
        copy__126: "i64[]" = torch.ops.aten.copy_.default(primals_307, add_222);  primals_307 = add_222 = copy__126 = None
        copy__127: "f32[896]" = torch.ops.aten.copy_.default(primals_308, add_224);  primals_308 = add_224 = copy__127 = None
        copy__128: "f32[896]" = torch.ops.aten.copy_.default(primals_309, add_225);  primals_309 = add_225 = copy__128 = None
        copy__129: "i64[]" = torch.ops.aten.copy_.default(primals_313, add_228);  primals_313 = add_228 = copy__129 = None
        copy__130: "f32[896]" = torch.ops.aten.copy_.default(primals_314, add_230);  primals_314 = add_230 = copy__130 = None
        copy__131: "f32[896]" = torch.ops.aten.copy_.default(primals_315, add_231);  primals_315 = add_231 = copy__131 = None
        copy__132: "i64[]" = torch.ops.aten.copy_.default(primals_319, add_233);  primals_319 = add_233 = copy__132 = None
        copy__133: "f32[896]" = torch.ops.aten.copy_.default(primals_320, add_235);  primals_320 = add_235 = copy__133 = None
        copy__134: "f32[896]" = torch.ops.aten.copy_.default(primals_321, add_236);  primals_321 = add_236 = copy__134 = None
        copy__135: "i64[]" = torch.ops.aten.copy_.default(primals_329, add_238);  primals_329 = add_238 = copy__135 = None
        copy__136: "f32[896]" = torch.ops.aten.copy_.default(primals_330, add_240);  primals_330 = add_240 = copy__136 = None
        copy__137: "f32[896]" = torch.ops.aten.copy_.default(primals_331, add_241);  primals_331 = add_241 = copy__137 = None
        copy__138: "i64[]" = torch.ops.aten.copy_.default(primals_335, add_244);  primals_335 = add_244 = copy__138 = None
        copy__139: "f32[896]" = torch.ops.aten.copy_.default(primals_336, add_246);  primals_336 = add_246 = copy__139 = None
        copy__140: "f32[896]" = torch.ops.aten.copy_.default(primals_337, add_247);  primals_337 = add_247 = copy__140 = None
        copy__141: "i64[]" = torch.ops.aten.copy_.default(primals_341, add_249);  primals_341 = add_249 = copy__141 = None
        copy__142: "f32[896]" = torch.ops.aten.copy_.default(primals_342, add_251);  primals_342 = add_251 = copy__142 = None
        copy__143: "f32[896]" = torch.ops.aten.copy_.default(primals_343, add_252);  primals_343 = add_252 = copy__143 = None
        copy__144: "i64[]" = torch.ops.aten.copy_.default(primals_351, add_254);  primals_351 = add_254 = copy__144 = None
        copy__145: "f32[896]" = torch.ops.aten.copy_.default(primals_352, add_256);  primals_352 = add_256 = copy__145 = None
        copy__146: "f32[896]" = torch.ops.aten.copy_.default(primals_353, add_257);  primals_353 = add_257 = copy__146 = None
        copy__147: "i64[]" = torch.ops.aten.copy_.default(primals_357, add_260);  primals_357 = add_260 = copy__147 = None
        copy__148: "f32[896]" = torch.ops.aten.copy_.default(primals_358, add_262);  primals_358 = add_262 = copy__148 = None
        copy__149: "f32[896]" = torch.ops.aten.copy_.default(primals_359, add_263);  primals_359 = add_263 = copy__149 = None
        copy__150: "i64[]" = torch.ops.aten.copy_.default(primals_363, add_265);  primals_363 = add_265 = copy__150 = None
        copy__151: "f32[896]" = torch.ops.aten.copy_.default(primals_364, add_267);  primals_364 = add_267 = copy__151 = None
        copy__152: "f32[896]" = torch.ops.aten.copy_.default(primals_365, add_268);  primals_365 = add_268 = copy__152 = None
        copy__153: "i64[]" = torch.ops.aten.copy_.default(primals_373, add_270);  primals_373 = add_270 = copy__153 = None
        copy__154: "f32[896]" = torch.ops.aten.copy_.default(primals_374, add_272);  primals_374 = add_272 = copy__154 = None
        copy__155: "f32[896]" = torch.ops.aten.copy_.default(primals_375, add_273);  primals_375 = add_273 = copy__155 = None
        copy__156: "i64[]" = torch.ops.aten.copy_.default(primals_379, add_276);  primals_379 = add_276 = copy__156 = None
        copy__157: "f32[896]" = torch.ops.aten.copy_.default(primals_380, add_278);  primals_380 = add_278 = copy__157 = None
        copy__158: "f32[896]" = torch.ops.aten.copy_.default(primals_381, add_279);  primals_381 = add_279 = copy__158 = None
        copy__159: "i64[]" = torch.ops.aten.copy_.default(primals_385, add_281);  primals_385 = add_281 = copy__159 = None
        copy__160: "f32[896]" = torch.ops.aten.copy_.default(primals_386, add_283);  primals_386 = add_283 = copy__160 = None
        copy__161: "f32[896]" = torch.ops.aten.copy_.default(primals_387, add_284);  primals_387 = add_284 = copy__161 = None
        copy__162: "i64[]" = torch.ops.aten.copy_.default(primals_395, add_286);  primals_395 = add_286 = copy__162 = None
        copy__163: "f32[896]" = torch.ops.aten.copy_.default(primals_396, add_288);  primals_396 = add_288 = copy__163 = None
        copy__164: "f32[896]" = torch.ops.aten.copy_.default(primals_397, add_289);  primals_397 = add_289 = copy__164 = None
        copy__165: "i64[]" = torch.ops.aten.copy_.default(primals_401, add_292);  primals_401 = add_292 = copy__165 = None
        copy__166: "f32[896]" = torch.ops.aten.copy_.default(primals_402, add_294);  primals_402 = add_294 = copy__166 = None
        copy__167: "f32[896]" = torch.ops.aten.copy_.default(primals_403, add_295);  primals_403 = add_295 = copy__167 = None
        copy__168: "i64[]" = torch.ops.aten.copy_.default(primals_407, add_297);  primals_407 = add_297 = copy__168 = None
        copy__169: "f32[896]" = torch.ops.aten.copy_.default(primals_408, add_299);  primals_408 = add_299 = copy__169 = None
        copy__170: "f32[896]" = torch.ops.aten.copy_.default(primals_409, add_300);  primals_409 = add_300 = copy__170 = None
        copy__171: "i64[]" = torch.ops.aten.copy_.default(primals_417, add_302);  primals_417 = add_302 = copy__171 = None
        copy__172: "f32[896]" = torch.ops.aten.copy_.default(primals_418, add_304);  primals_418 = add_304 = copy__172 = None
        copy__173: "f32[896]" = torch.ops.aten.copy_.default(primals_419, add_305);  primals_419 = add_305 = copy__173 = None
        copy__174: "i64[]" = torch.ops.aten.copy_.default(primals_423, add_308);  primals_423 = add_308 = copy__174 = None
        copy__175: "f32[2240]" = torch.ops.aten.copy_.default(primals_424, add_310);  primals_424 = add_310 = copy__175 = None
        copy__176: "f32[2240]" = torch.ops.aten.copy_.default(primals_425, add_311);  primals_425 = add_311 = copy__176 = None
        copy__177: "i64[]" = torch.ops.aten.copy_.default(primals_429, add_313);  primals_429 = add_313 = copy__177 = None
        copy__178: "f32[2240]" = torch.ops.aten.copy_.default(primals_430, add_315);  primals_430 = add_315 = copy__178 = None
        copy__179: "f32[2240]" = torch.ops.aten.copy_.default(primals_431, add_316);  primals_431 = add_316 = copy__179 = None
        copy__180: "i64[]" = torch.ops.aten.copy_.default(primals_439, add_318);  primals_439 = add_318 = copy__180 = None
        copy__181: "f32[2240]" = torch.ops.aten.copy_.default(primals_440, add_320);  primals_440 = add_320 = copy__181 = None
        copy__182: "f32[2240]" = torch.ops.aten.copy_.default(primals_441, add_321);  primals_441 = add_321 = copy__182 = None
        copy__183: "i64[]" = torch.ops.aten.copy_.default(primals_445, add_323);  primals_445 = add_323 = copy__183 = None
        copy__184: "f32[2240]" = torch.ops.aten.copy_.default(primals_446, add_325);  primals_446 = add_325 = copy__184 = None
        copy__185: "f32[2240]" = torch.ops.aten.copy_.default(primals_447, add_326);  primals_447 = add_326 = copy__185 = None
        return (addmm, primals_1, primals_2, primals_6, primals_8, primals_12, primals_14, primals_18, primals_19, primals_20, primals_22, primals_24, primals_28, primals_30, primals_34, primals_36, primals_40, primals_42, primals_46, primals_47, primals_48, primals_50, primals_52, primals_56, primals_58, primals_62, primals_64, primals_68, primals_69, primals_70, primals_72, primals_74, primals_78, primals_80, primals_84, primals_86, primals_90, primals_92, primals_96, primals_97, primals_98, primals_100, primals_102, primals_106, primals_108, primals_112, primals_114, primals_118, primals_119, primals_120, primals_122, primals_124, primals_128, primals_130, primals_134, primals_136, primals_140, primals_141, primals_142, primals_144, primals_146, primals_150, primals_152, primals_156, primals_158, primals_162, primals_163, primals_164, primals_166, primals_168, primals_172, primals_174, primals_178, primals_180, primals_184, primals_185, primals_186, primals_188, primals_190, primals_194, primals_196, primals_200, primals_202, primals_206, primals_208, primals_212, primals_213, primals_214, primals_216, primals_218, primals_222, primals_224, primals_228, primals_230, primals_234, primals_235, primals_236, primals_238, primals_240, primals_244, primals_246, primals_250, primals_252, primals_256, primals_257, primals_258, primals_260, primals_262, primals_266, primals_268, primals_272, primals_274, primals_278, primals_279, primals_280, primals_282, primals_284, primals_288, primals_290, primals_294, primals_296, primals_300, primals_301, primals_302, primals_304, primals_306, primals_310, primals_312, primals_316, primals_318, primals_322, primals_323, primals_324, primals_326, primals_328, primals_332, primals_334, primals_338, primals_340, primals_344, primals_345, primals_346, primals_348, primals_350, primals_354, primals_356, primals_360, primals_362, primals_366, primals_367, primals_368, primals_370, primals_372, primals_376, primals_378, primals_382, primals_384, primals_388, primals_389, primals_390, primals_392, primals_394, primals_398, primals_400, primals_404, primals_406, primals_410, primals_411, primals_412, primals_414, primals_416, primals_420, primals_422, primals_426, primals_428, primals_432, primals_433, primals_434, primals_436, primals_438, primals_442, primals_443, primals_444, primals_448, primals_449, primals_450, convolution, squeeze_1, relu, convolution_1, squeeze_4, relu_1, convolution_2, getitem_5, rsqrt_2, mean, relu_3, convolution_4, mul_21, convolution_5, squeeze_10, convolution_6, squeeze_13, relu_4, convolution_7, squeeze_16, relu_5, convolution_8, getitem_13, rsqrt_6, mean_1, relu_7, convolution_10, mul_50, convolution_11, squeeze_22, relu_8, convolution_12, squeeze_25, relu_9, convolution_13, getitem_19, rsqrt_9, mean_2, relu_11, convolution_15, mul_72, convolution_16, squeeze_31, convolution_17, squeeze_34, relu_12, convolution_18, squeeze_37, relu_13, convolution_19, getitem_27, rsqrt_13, mean_3, relu_15, convolution_21, mul_101, convolution_22, squeeze_43, relu_16, convolution_23, squeeze_46, relu_17, convolution_24, getitem_33, rsqrt_16, mean_4, relu_19, convolution_26, mul_123, convolution_27, squeeze_52, relu_20, convolution_28, squeeze_55, relu_21, convolution_29, getitem_39, rsqrt_19, mean_5, relu_23, convolution_31, mul_145, convolution_32, squeeze_61, relu_24, convolution_33, squeeze_64, relu_25, convolution_34, getitem_45, rsqrt_22, mean_6, relu_27, convolution_36, mul_167, convolution_37, squeeze_70, relu_28, convolution_38, squeeze_73, relu_29, convolution_39, getitem_51, rsqrt_25, mean_7, relu_31, convolution_41, mul_189, convolution_42, squeeze_79, convolution_43, squeeze_82, relu_32, convolution_44, squeeze_85, relu_33, convolution_45, getitem_59, rsqrt_29, mean_8, relu_35, convolution_47, mul_218, convolution_48, squeeze_91, relu_36, convolution_49, squeeze_94, relu_37, convolution_50, getitem_65, rsqrt_32, mean_9, relu_39, convolution_52, mul_240, convolution_53, squeeze_100, relu_40, convolution_54, squeeze_103, relu_41, convolution_55, getitem_71, rsqrt_35, mean_10, relu_43, convolution_57, mul_262, convolution_58, squeeze_109, relu_44, convolution_59, squeeze_112, relu_45, convolution_60, getitem_77, rsqrt_38, mean_11, relu_47, convolution_62, mul_284, convolution_63, squeeze_118, relu_48, convolution_64, squeeze_121, relu_49, convolution_65, getitem_83, rsqrt_41, mean_12, relu_51, convolution_67, mul_306, convolution_68, squeeze_127, relu_52, convolution_69, squeeze_130, relu_53, convolution_70, getitem_89, rsqrt_44, mean_13, relu_55, convolution_72, mul_328, convolution_73, squeeze_136, relu_56, convolution_74, squeeze_139, relu_57, convolution_75, getitem_95, rsqrt_47, mean_14, relu_59, convolution_77, mul_350, convolution_78, squeeze_145, relu_60, convolution_79, squeeze_148, relu_61, convolution_80, getitem_101, rsqrt_50, mean_15, relu_63, convolution_82, mul_372, convolution_83, squeeze_154, relu_64, convolution_84, squeeze_157, relu_65, convolution_85, getitem_107, rsqrt_53, mean_16, relu_67, convolution_87, mul_394, convolution_88, squeeze_163, relu_68, convolution_89, squeeze_166, relu_69, convolution_90, getitem_113, rsqrt_56, mean_17, relu_71, convolution_92, mul_416, convolution_93, squeeze_172, relu_72, convolution_94, squeeze_175, relu_73, convolution_95, getitem_119, rsqrt_59, mean_18, relu_75, convolution_97, mul_438, convolution_98, getitem_121, rsqrt_60, convolution_99, getitem_123, rsqrt_61, view, unsqueeze_286, unsqueeze_298, unsqueeze_322, unsqueeze_334, unsqueeze_358, unsqueeze_370, unsqueeze_394, unsqueeze_406, unsqueeze_430, unsqueeze_442, unsqueeze_466, unsqueeze_478, unsqueeze_502, unsqueeze_514, unsqueeze_538, unsqueeze_550, unsqueeze_574, unsqueeze_586, unsqueeze_610, unsqueeze_622, unsqueeze_646, unsqueeze_658, unsqueeze_670, unsqueeze_694, unsqueeze_706, unsqueeze_730, unsqueeze_742, unsqueeze_766, unsqueeze_778, unsqueeze_802, unsqueeze_814, unsqueeze_838, unsqueeze_850, unsqueeze_862, unsqueeze_886, unsqueeze_898, unsqueeze_922, unsqueeze_934, unsqueeze_946, unsqueeze_970, unsqueeze_982)
