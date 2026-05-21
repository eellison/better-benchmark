class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_6: "f32[32]", primals_7: "f32[32]", primals_8: "f32[32, 1, 3, 3]", primals_12: "f32[32]", primals_13: "f32[32]", primals_14: "f32[8, 32, 1, 1]", primals_16: "f32[32, 8, 1, 1]", primals_18: "f32[16, 32, 1, 1]", primals_22: "f32[16]", primals_24: "f32[96, 16, 1, 1]", primals_28: "f32[96]", primals_29: "f32[96]", primals_30: "f32[96, 1, 3, 3]", primals_34: "f32[96]", primals_35: "f32[96]", primals_36: "f32[4, 96, 1, 1]", primals_38: "f32[96, 4, 1, 1]", primals_40: "f32[24, 96, 1, 1]", primals_44: "f32[24]", primals_46: "f32[144, 24, 1, 1]", primals_50: "f32[144]", primals_51: "f32[144]", primals_52: "f32[144, 1, 3, 3]", primals_56: "f32[144]", primals_57: "f32[144]", primals_58: "f32[6, 144, 1, 1]", primals_60: "f32[144, 6, 1, 1]", primals_62: "f32[24, 144, 1, 1]", primals_66: "f32[24]", primals_68: "f32[144, 24, 1, 1]", primals_72: "f32[144]", primals_73: "f32[144]", primals_74: "f32[144, 1, 5, 5]", primals_78: "f32[144]", primals_79: "f32[144]", primals_80: "f32[6, 144, 1, 1]", primals_82: "f32[144, 6, 1, 1]", primals_84: "f32[40, 144, 1, 1]", primals_88: "f32[40]", primals_90: "f32[240, 40, 1, 1]", primals_94: "f32[240]", primals_95: "f32[240]", primals_96: "f32[240, 1, 5, 5]", primals_100: "f32[240]", primals_101: "f32[240]", primals_102: "f32[10, 240, 1, 1]", primals_104: "f32[240, 10, 1, 1]", primals_106: "f32[40, 240, 1, 1]", primals_110: "f32[40]", primals_112: "f32[240, 40, 1, 1]", primals_116: "f32[240]", primals_117: "f32[240]", primals_118: "f32[240, 1, 3, 3]", primals_122: "f32[240]", primals_123: "f32[240]", primals_124: "f32[10, 240, 1, 1]", primals_126: "f32[240, 10, 1, 1]", primals_128: "f32[80, 240, 1, 1]", primals_132: "f32[80]", primals_134: "f32[480, 80, 1, 1]", primals_138: "f32[480]", primals_139: "f32[480]", primals_140: "f32[480, 1, 3, 3]", primals_144: "f32[480]", primals_145: "f32[480]", primals_146: "f32[20, 480, 1, 1]", primals_148: "f32[480, 20, 1, 1]", primals_150: "f32[80, 480, 1, 1]", primals_154: "f32[80]", primals_156: "f32[480, 80, 1, 1]", primals_160: "f32[480]", primals_161: "f32[480]", primals_162: "f32[480, 1, 3, 3]", primals_166: "f32[480]", primals_167: "f32[480]", primals_168: "f32[20, 480, 1, 1]", primals_170: "f32[480, 20, 1, 1]", primals_172: "f32[80, 480, 1, 1]", primals_176: "f32[80]", primals_178: "f32[480, 80, 1, 1]", primals_182: "f32[480]", primals_183: "f32[480]", primals_184: "f32[480, 1, 5, 5]", primals_188: "f32[480]", primals_189: "f32[480]", primals_190: "f32[20, 480, 1, 1]", primals_192: "f32[480, 20, 1, 1]", primals_194: "f32[112, 480, 1, 1]", primals_198: "f32[112]", primals_200: "f32[672, 112, 1, 1]", primals_204: "f32[672]", primals_205: "f32[672]", primals_206: "f32[672, 1, 5, 5]", primals_210: "f32[672]", primals_211: "f32[672]", primals_212: "f32[28, 672, 1, 1]", primals_214: "f32[672, 28, 1, 1]", primals_216: "f32[112, 672, 1, 1]", primals_220: "f32[112]", primals_222: "f32[672, 112, 1, 1]", primals_226: "f32[672]", primals_227: "f32[672]", primals_228: "f32[672, 1, 5, 5]", primals_232: "f32[672]", primals_233: "f32[672]", primals_234: "f32[28, 672, 1, 1]", primals_236: "f32[672, 28, 1, 1]", primals_238: "f32[112, 672, 1, 1]", primals_242: "f32[112]", primals_244: "f32[672, 112, 1, 1]", primals_248: "f32[672]", primals_249: "f32[672]", primals_250: "f32[672, 1, 5, 5]", primals_254: "f32[672]", primals_255: "f32[672]", primals_256: "f32[28, 672, 1, 1]", primals_258: "f32[672, 28, 1, 1]", primals_260: "f32[192, 672, 1, 1]", primals_264: "f32[192]", primals_266: "f32[1152, 192, 1, 1]", primals_270: "f32[1152]", primals_271: "f32[1152]", primals_272: "f32[1152, 1, 5, 5]", primals_276: "f32[1152]", primals_277: "f32[1152]", primals_278: "f32[48, 1152, 1, 1]", primals_280: "f32[1152, 48, 1, 1]", primals_282: "f32[192, 1152, 1, 1]", primals_286: "f32[192]", primals_288: "f32[1152, 192, 1, 1]", primals_292: "f32[1152]", primals_293: "f32[1152]", primals_294: "f32[1152, 1, 5, 5]", primals_298: "f32[1152]", primals_299: "f32[1152]", primals_300: "f32[48, 1152, 1, 1]", primals_302: "f32[1152, 48, 1, 1]", primals_304: "f32[192, 1152, 1, 1]", primals_308: "f32[192]", primals_310: "f32[1152, 192, 1, 1]", primals_314: "f32[1152]", primals_315: "f32[1152]", primals_316: "f32[1152, 1, 5, 5]", primals_320: "f32[1152]", primals_321: "f32[1152]", primals_322: "f32[48, 1152, 1, 1]", primals_324: "f32[1152, 48, 1, 1]", primals_326: "f32[192, 1152, 1, 1]", primals_330: "f32[192]", primals_332: "f32[1152, 192, 1, 1]", primals_336: "f32[1152]", primals_337: "f32[1152]", primals_338: "f32[1152, 1, 3, 3]", primals_342: "f32[1152]", primals_343: "f32[1152]", primals_344: "f32[48, 1152, 1, 1]", primals_346: "f32[1152, 48, 1, 1]", primals_348: "f32[320, 1152, 1, 1]", primals_352: "f32[320]", primals_354: "f32[1280, 320, 1, 1]", primals_358: "f32[1280]", primals_359: "f32[1280]", primals_360: "f32[1000, 1280]", constant_pad_nd: "f32[128, 3, 225, 225]", convolution: "f32[128, 32, 112, 112]", getitem_1: "f32[1, 32, 1, 1]", rsqrt: "f32[1, 32, 1, 1]", div: "f32[128, 32, 112, 112]", convolution_1: "f32[128, 32, 112, 112]", getitem_3: "f32[1, 32, 1, 1]", rsqrt_1: "f32[1, 32, 1, 1]", mean: "f32[128, 32, 1, 1]", convolution_2: "f32[128, 8, 1, 1]", div_2: "f32[128, 8, 1, 1]", convolution_3: "f32[128, 32, 1, 1]", mul_14: "f32[128, 32, 112, 112]", convolution_4: "f32[128, 16, 112, 112]", squeeze_7: "f32[16]", add_17: "f32[128, 16, 112, 112]", convolution_5: "f32[128, 96, 112, 112]", getitem_7: "f32[1, 96, 1, 1]", rsqrt_3: "f32[1, 96, 1, 1]", constant_pad_nd_1: "f32[128, 96, 113, 113]", convolution_6: "f32[128, 96, 56, 56]", getitem_9: "f32[1, 96, 1, 1]", rsqrt_4: "f32[1, 96, 1, 1]", mean_1: "f32[128, 96, 1, 1]", convolution_7: "f32[128, 4, 1, 1]", div_5: "f32[128, 4, 1, 1]", convolution_8: "f32[128, 96, 1, 1]", mul_36: "f32[128, 96, 56, 56]", convolution_9: "f32[128, 24, 56, 56]", squeeze_16: "f32[24]", add_35: "f32[128, 24, 56, 56]", convolution_10: "f32[128, 144, 56, 56]", getitem_13: "f32[1, 144, 1, 1]", rsqrt_6: "f32[1, 144, 1, 1]", div_6: "f32[128, 144, 56, 56]", convolution_11: "f32[128, 144, 56, 56]", getitem_15: "f32[1, 144, 1, 1]", rsqrt_7: "f32[1, 144, 1, 1]", mean_2: "f32[128, 144, 1, 1]", convolution_12: "f32[128, 6, 1, 1]", div_8: "f32[128, 6, 1, 1]", convolution_13: "f32[128, 144, 1, 1]", mul_58: "f32[128, 144, 56, 56]", convolution_14: "f32[128, 24, 56, 56]", squeeze_25: "f32[24]", add_54: "f32[128, 24, 56, 56]", convolution_15: "f32[128, 144, 56, 56]", getitem_19: "f32[1, 144, 1, 1]", rsqrt_9: "f32[1, 144, 1, 1]", constant_pad_nd_2: "f32[128, 144, 59, 59]", convolution_16: "f32[128, 144, 28, 28]", getitem_21: "f32[1, 144, 1, 1]", rsqrt_10: "f32[1, 144, 1, 1]", mean_3: "f32[128, 144, 1, 1]", convolution_17: "f32[128, 6, 1, 1]", div_11: "f32[128, 6, 1, 1]", convolution_18: "f32[128, 144, 1, 1]", mul_80: "f32[128, 144, 28, 28]", convolution_19: "f32[128, 40, 28, 28]", squeeze_34: "f32[40]", add_72: "f32[128, 40, 28, 28]", convolution_20: "f32[128, 240, 28, 28]", getitem_25: "f32[1, 240, 1, 1]", rsqrt_12: "f32[1, 240, 1, 1]", div_12: "f32[128, 240, 28, 28]", convolution_21: "f32[128, 240, 28, 28]", getitem_27: "f32[1, 240, 1, 1]", rsqrt_13: "f32[1, 240, 1, 1]", mean_4: "f32[128, 240, 1, 1]", convolution_22: "f32[128, 10, 1, 1]", div_14: "f32[128, 10, 1, 1]", convolution_23: "f32[128, 240, 1, 1]", mul_102: "f32[128, 240, 28, 28]", convolution_24: "f32[128, 40, 28, 28]", squeeze_43: "f32[40]", add_91: "f32[128, 40, 28, 28]", convolution_25: "f32[128, 240, 28, 28]", getitem_31: "f32[1, 240, 1, 1]", rsqrt_15: "f32[1, 240, 1, 1]", constant_pad_nd_3: "f32[128, 240, 29, 29]", convolution_26: "f32[128, 240, 14, 14]", getitem_33: "f32[1, 240, 1, 1]", rsqrt_16: "f32[1, 240, 1, 1]", mean_5: "f32[128, 240, 1, 1]", convolution_27: "f32[128, 10, 1, 1]", div_17: "f32[128, 10, 1, 1]", convolution_28: "f32[128, 240, 1, 1]", mul_124: "f32[128, 240, 14, 14]", convolution_29: "f32[128, 80, 14, 14]", squeeze_52: "f32[80]", add_109: "f32[128, 80, 14, 14]", convolution_30: "f32[128, 480, 14, 14]", getitem_37: "f32[1, 480, 1, 1]", rsqrt_18: "f32[1, 480, 1, 1]", div_18: "f32[128, 480, 14, 14]", convolution_31: "f32[128, 480, 14, 14]", getitem_39: "f32[1, 480, 1, 1]", rsqrt_19: "f32[1, 480, 1, 1]", mean_6: "f32[128, 480, 1, 1]", convolution_32: "f32[128, 20, 1, 1]", div_20: "f32[128, 20, 1, 1]", convolution_33: "f32[128, 480, 1, 1]", mul_146: "f32[128, 480, 14, 14]", convolution_34: "f32[128, 80, 14, 14]", squeeze_61: "f32[80]", add_128: "f32[128, 80, 14, 14]", convolution_35: "f32[128, 480, 14, 14]", getitem_43: "f32[1, 480, 1, 1]", rsqrt_21: "f32[1, 480, 1, 1]", div_21: "f32[128, 480, 14, 14]", convolution_36: "f32[128, 480, 14, 14]", getitem_45: "f32[1, 480, 1, 1]", rsqrt_22: "f32[1, 480, 1, 1]", mean_7: "f32[128, 480, 1, 1]", convolution_37: "f32[128, 20, 1, 1]", div_23: "f32[128, 20, 1, 1]", convolution_38: "f32[128, 480, 1, 1]", mul_168: "f32[128, 480, 14, 14]", convolution_39: "f32[128, 80, 14, 14]", squeeze_70: "f32[80]", add_147: "f32[128, 80, 14, 14]", convolution_40: "f32[128, 480, 14, 14]", getitem_49: "f32[1, 480, 1, 1]", rsqrt_24: "f32[1, 480, 1, 1]", div_24: "f32[128, 480, 14, 14]", convolution_41: "f32[128, 480, 14, 14]", getitem_51: "f32[1, 480, 1, 1]", rsqrt_25: "f32[1, 480, 1, 1]", mean_8: "f32[128, 480, 1, 1]", convolution_42: "f32[128, 20, 1, 1]", div_26: "f32[128, 20, 1, 1]", convolution_43: "f32[128, 480, 1, 1]", mul_190: "f32[128, 480, 14, 14]", convolution_44: "f32[128, 112, 14, 14]", squeeze_79: "f32[112]", add_165: "f32[128, 112, 14, 14]", convolution_45: "f32[128, 672, 14, 14]", getitem_55: "f32[1, 672, 1, 1]", rsqrt_27: "f32[1, 672, 1, 1]", div_27: "f32[128, 672, 14, 14]", convolution_46: "f32[128, 672, 14, 14]", getitem_57: "f32[1, 672, 1, 1]", rsqrt_28: "f32[1, 672, 1, 1]", mean_9: "f32[128, 672, 1, 1]", convolution_47: "f32[128, 28, 1, 1]", div_29: "f32[128, 28, 1, 1]", convolution_48: "f32[128, 672, 1, 1]", mul_212: "f32[128, 672, 14, 14]", convolution_49: "f32[128, 112, 14, 14]", squeeze_88: "f32[112]", add_184: "f32[128, 112, 14, 14]", convolution_50: "f32[128, 672, 14, 14]", getitem_61: "f32[1, 672, 1, 1]", rsqrt_30: "f32[1, 672, 1, 1]", div_30: "f32[128, 672, 14, 14]", convolution_51: "f32[128, 672, 14, 14]", getitem_63: "f32[1, 672, 1, 1]", rsqrt_31: "f32[1, 672, 1, 1]", mean_10: "f32[128, 672, 1, 1]", convolution_52: "f32[128, 28, 1, 1]", div_32: "f32[128, 28, 1, 1]", convolution_53: "f32[128, 672, 1, 1]", mul_234: "f32[128, 672, 14, 14]", convolution_54: "f32[128, 112, 14, 14]", squeeze_97: "f32[112]", add_203: "f32[128, 112, 14, 14]", convolution_55: "f32[128, 672, 14, 14]", getitem_67: "f32[1, 672, 1, 1]", rsqrt_33: "f32[1, 672, 1, 1]", constant_pad_nd_4: "f32[128, 672, 17, 17]", convolution_56: "f32[128, 672, 7, 7]", getitem_69: "f32[1, 672, 1, 1]", rsqrt_34: "f32[1, 672, 1, 1]", mean_11: "f32[128, 672, 1, 1]", convolution_57: "f32[128, 28, 1, 1]", div_35: "f32[128, 28, 1, 1]", convolution_58: "f32[128, 672, 1, 1]", mul_256: "f32[128, 672, 7, 7]", convolution_59: "f32[128, 192, 7, 7]", squeeze_106: "f32[192]", add_221: "f32[128, 192, 7, 7]", convolution_60: "f32[128, 1152, 7, 7]", getitem_73: "f32[1, 1152, 1, 1]", rsqrt_36: "f32[1, 1152, 1, 1]", div_36: "f32[128, 1152, 7, 7]", convolution_61: "f32[128, 1152, 7, 7]", getitem_75: "f32[1, 1152, 1, 1]", rsqrt_37: "f32[1, 1152, 1, 1]", mean_12: "f32[128, 1152, 1, 1]", convolution_62: "f32[128, 48, 1, 1]", div_38: "f32[128, 48, 1, 1]", convolution_63: "f32[128, 1152, 1, 1]", mul_278: "f32[128, 1152, 7, 7]", convolution_64: "f32[128, 192, 7, 7]", squeeze_115: "f32[192]", add_240: "f32[128, 192, 7, 7]", convolution_65: "f32[128, 1152, 7, 7]", getitem_79: "f32[1, 1152, 1, 1]", rsqrt_39: "f32[1, 1152, 1, 1]", div_39: "f32[128, 1152, 7, 7]", convolution_66: "f32[128, 1152, 7, 7]", getitem_81: "f32[1, 1152, 1, 1]", rsqrt_40: "f32[1, 1152, 1, 1]", mean_13: "f32[128, 1152, 1, 1]", convolution_67: "f32[128, 48, 1, 1]", div_41: "f32[128, 48, 1, 1]", convolution_68: "f32[128, 1152, 1, 1]", mul_300: "f32[128, 1152, 7, 7]", convolution_69: "f32[128, 192, 7, 7]", squeeze_124: "f32[192]", add_259: "f32[128, 192, 7, 7]", convolution_70: "f32[128, 1152, 7, 7]", getitem_85: "f32[1, 1152, 1, 1]", rsqrt_42: "f32[1, 1152, 1, 1]", div_42: "f32[128, 1152, 7, 7]", convolution_71: "f32[128, 1152, 7, 7]", getitem_87: "f32[1, 1152, 1, 1]", rsqrt_43: "f32[1, 1152, 1, 1]", mean_14: "f32[128, 1152, 1, 1]", convolution_72: "f32[128, 48, 1, 1]", div_44: "f32[128, 48, 1, 1]", convolution_73: "f32[128, 1152, 1, 1]", mul_322: "f32[128, 1152, 7, 7]", convolution_74: "f32[128, 192, 7, 7]", squeeze_133: "f32[192]", add_278: "f32[128, 192, 7, 7]", convolution_75: "f32[128, 1152, 7, 7]", getitem_91: "f32[1, 1152, 1, 1]", rsqrt_45: "f32[1, 1152, 1, 1]", div_45: "f32[128, 1152, 7, 7]", convolution_76: "f32[128, 1152, 7, 7]", getitem_93: "f32[1, 1152, 1, 1]", rsqrt_46: "f32[1, 1152, 1, 1]", mean_15: "f32[128, 1152, 1, 1]", convolution_77: "f32[128, 48, 1, 1]", div_47: "f32[128, 48, 1, 1]", convolution_78: "f32[128, 1152, 1, 1]", mul_344: "f32[128, 1152, 7, 7]", convolution_79: "f32[128, 320, 7, 7]", squeeze_142: "f32[320]", add_296: "f32[128, 320, 7, 7]", convolution_80: "f32[128, 1280, 7, 7]", getitem_97: "f32[1, 1280, 1, 1]", rsqrt_48: "f32[1, 1280, 1, 1]", view: "f32[128, 1280]", unsqueeze_210: "f32[1, 320, 1, 1]", unsqueeze_246: "f32[1, 192, 1, 1]", unsqueeze_282: "f32[1, 192, 1, 1]", unsqueeze_318: "f32[1, 192, 1, 1]", unsqueeze_354: "f32[1, 192, 1, 1]", unsqueeze_390: "f32[1, 112, 1, 1]", unsqueeze_426: "f32[1, 112, 1, 1]", unsqueeze_462: "f32[1, 112, 1, 1]", unsqueeze_498: "f32[1, 80, 1, 1]", unsqueeze_534: "f32[1, 80, 1, 1]", unsqueeze_570: "f32[1, 80, 1, 1]", unsqueeze_606: "f32[1, 40, 1, 1]", unsqueeze_642: "f32[1, 40, 1, 1]", unsqueeze_678: "f32[1, 24, 1, 1]", unsqueeze_714: "f32[1, 24, 1, 1]", unsqueeze_750: "f32[1, 16, 1, 1]", tangents_1: "f32[128, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/efficientnet.py:344 in forward_head, code: return x if pre_logits else self.classifier(x)
        permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_360, [1, 0]);  primals_360 = None
        permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[128, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_2: "f32[128, 1280, 1, 1]" = torch.ops.aten.reshape.default(mm, [128, 1280, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_147: "f32[128, 1280, 1]" = torch.ops.aten.squeeze.dim(view_2, 3);  view_2 = None
        squeeze_148: "f32[128, 1280]" = torch.ops.aten.squeeze.dim(squeeze_147, 2);  squeeze_147 = None
        full_49: "f32[163840]" = torch.ops.aten.full.default([163840], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "f32[163840]" = torch.ops.aten.as_strided_scatter.default(full_49, squeeze_148, [128, 1280], [1280, 1], 0);  full_49 = squeeze_148 = None
        as_strided_5: "f32[128, 1280, 1, 1]" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 1280, 1, 1], [1280, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "f32[128, 1280, 7, 7]" = torch.ops.aten.expand.default(as_strided_5, [128, 1280, 7, 7]);  as_strided_5 = None
        div_49: "f32[128, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_48: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, getitem_97)
        mul_352: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        unsqueeze_192: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_358, -1)
        unsqueeze_193: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_358: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_193);  mul_352 = unsqueeze_193 = None
        unsqueeze_194: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_359, -1);  primals_359 = None
        unsqueeze_195: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_301: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_358, unsqueeze_195);  mul_358 = unsqueeze_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_48: "f32[128, 1280, 7, 7]" = torch.ops.aten.neg.default(add_301)
        exp_48: "f32[128, 1280, 7, 7]" = torch.ops.aten.exp.default(neg_48);  neg_48 = None
        add_302: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(exp_48, 1);  exp_48 = None
        reciprocal: "f32[128, 1280, 7, 7]" = torch.ops.aten.reciprocal.default(add_302);  add_302 = None
        mul_359: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_360: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(div_49, mul_359);  div_49 = None
        sub_49: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_359);  mul_359 = None
        mul_361: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(add_301, sub_49);  add_301 = sub_49 = None
        add_304: "f32[128, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_361, 1);  mul_361 = None
        mul_362: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_360, add_304);  mul_360 = add_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_144: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_97, [0, 2, 3]);  getitem_97 = None
        unsqueeze_196: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_197: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 2);  unsqueeze_196 = None
        unsqueeze_198: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 3);  unsqueeze_197 = None
        sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 2, 3])
        sub_50: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_198);  convolution_80 = unsqueeze_198 = None
        mul_363: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_362, sub_50)
        sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 2, 3]);  mul_363 = None
        mul_364: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, 0.00015943877551020407)
        unsqueeze_199: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_364, 0);  mul_364 = None
        unsqueeze_200: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
        unsqueeze_201: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
        mul_365: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, 0.00015943877551020407)
        squeeze_145: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_366: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_367: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        unsqueeze_202: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
        unsqueeze_203: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
        unsqueeze_204: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
        mul_368: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_358);  primals_358 = None
        unsqueeze_205: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
        unsqueeze_206: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
        unsqueeze_207: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
        mul_369: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_204);  sub_50 = unsqueeze_204 = None
        sub_52: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(mul_362, mul_369);  mul_362 = mul_369 = None
        sub_53: "f32[128, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_52, unsqueeze_201);  sub_52 = unsqueeze_201 = None
        mul_370: "f32[128, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_207);  sub_53 = unsqueeze_207 = None
        mul_371: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_145);  sum_3 = squeeze_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/efficientnet.py:327 in forward_features, code: x = self.conv_head(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_370, add_296, primals_354, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_370 = add_296 = primals_354 = None
        getitem_98: "f32[128, 320, 7, 7]" = convolution_backward[0]
        getitem_99: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem_98, [0, 2, 3])
        sub_54: "f32[128, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_210);  convolution_79 = unsqueeze_210 = None
        mul_372: "f32[128, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_98, sub_54)
        sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 2, 3]);  mul_372 = None
        mul_373: "f32[320]" = torch.ops.aten.mul.Tensor(sum_4, 0.00015943877551020407)
        unsqueeze_211: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
        unsqueeze_212: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
        unsqueeze_213: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
        mul_374: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, 0.00015943877551020407)
        mul_375: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_376: "f32[320]" = torch.ops.aten.mul.Tensor(mul_374, mul_375);  mul_374 = mul_375 = None
        unsqueeze_214: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
        unsqueeze_215: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
        unsqueeze_216: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
        mul_377: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_352);  primals_352 = None
        unsqueeze_217: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
        unsqueeze_218: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
        unsqueeze_219: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
        mul_378: "f32[128, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_216);  sub_54 = unsqueeze_216 = None
        sub_56: "f32[128, 320, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_98, mul_378);  getitem_98 = mul_378 = None
        sub_57: "f32[128, 320, 7, 7]" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_213);  sub_56 = unsqueeze_213 = None
        mul_379: "f32[128, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_219);  sub_57 = unsqueeze_219 = None
        mul_380: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_142);  sum_5 = squeeze_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_379, mul_344, primals_348, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_379 = mul_344 = primals_348 = None
        getitem_101: "f32[128, 1152, 7, 7]" = convolution_backward_1[0]
        getitem_102: "f32[320, 1152, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_46: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, getitem_93)
        mul_337: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = None
        unsqueeze_184: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_342, -1)
        unsqueeze_185: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        mul_343: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_185);  mul_337 = unsqueeze_185 = None
        unsqueeze_186: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_343, -1);  primals_343 = None
        unsqueeze_187: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        add_289: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_343, unsqueeze_187);  mul_343 = unsqueeze_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_46: "f32[128, 1152, 7, 7]" = torch.ops.aten.neg.default(add_289)
        exp_46: "f32[128, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_46);  neg_46 = None
        add_290: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_46, 1);  exp_46 = None
        div_46: "f32[128, 1152, 7, 7]" = torch.ops.aten.div.Tensor(add_289, add_290)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_381: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_101, div_46);  div_46 = None
        sigmoid_15: "f32[128, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_78);  convolution_78 = None
        mul_382: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_101, sigmoid_15);  getitem_101 = None
        sum_6: "f32[128, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [2, 3], True);  mul_381 = None
        sub_58: "f32[128, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_15)
        mul_383: "f32[128, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_15, sub_58);  sigmoid_15 = sub_58 = None
        mul_384: "f32[128, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_6, mul_383);  sum_6 = mul_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_384, div_47, primals_346, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_384 = div_47 = primals_346 = None
        getitem_104: "f32[128, 48, 1, 1]" = convolution_backward_2[0]
        getitem_105: "f32[1152, 48, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_47: "f32[128, 48, 1, 1]" = torch.ops.aten.neg.default(convolution_77)
        exp_47: "f32[128, 48, 1, 1]" = torch.ops.aten.exp.default(neg_47);  neg_47 = None
        add_291: "f32[128, 48, 1, 1]" = torch.ops.aten.add.Tensor(exp_47, 1);  exp_47 = None
        reciprocal_1: "f32[128, 48, 1, 1]" = torch.ops.aten.reciprocal.default(add_291);  add_291 = None
        mul_385: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        mul_386: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_104, mul_385);  getitem_104 = None
        sub_59: "f32[128, 48, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_385);  mul_385 = None
        mul_387: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_77, sub_59);  convolution_77 = sub_59 = None
        add_306: "f32[128, 48, 1, 1]" = torch.ops.aten.add.Tensor(mul_387, 1);  mul_387 = None
        mul_388: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(mul_386, add_306);  mul_386 = add_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_8: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_388, mean_15, primals_344, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_388 = mean_15 = primals_344 = None
        getitem_107: "f32[128, 1152, 1, 1]" = convolution_backward_3[0]
        getitem_108: "f32[48, 1152, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_2: "f32[128, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_107, [128, 1152, 7, 7]);  getitem_107 = None
        div_50: "f32[128, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
        add_307: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_382, div_50);  mul_382 = div_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_2: "f32[128, 1152, 7, 7]" = torch.ops.aten.reciprocal.default(add_290);  add_290 = None
        mul_389: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        mul_390: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_307, mul_389);  add_307 = None
        sub_60: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_389);  mul_389 = None
        mul_391: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_289, sub_60);  add_289 = sub_60 = None
        add_309: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_391, 1);  mul_391 = None
        mul_392: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_390, add_309);  mul_390 = add_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_138: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_93, [0, 2, 3]);  getitem_93 = None
        unsqueeze_220: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_138, 0);  squeeze_138 = None
        unsqueeze_221: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 2);  unsqueeze_220 = None
        unsqueeze_222: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 3);  unsqueeze_221 = None
        sum_9: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 2, 3])
        sub_61: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_222);  convolution_76 = unsqueeze_222 = None
        mul_393: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_392, sub_61)
        sum_10: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 2, 3]);  mul_393 = None
        mul_394: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, 0.00015943877551020407)
        unsqueeze_223: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
        unsqueeze_224: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
        unsqueeze_225: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        mul_395: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_10, 0.00015943877551020407)
        squeeze_139: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_46, [0, 2, 3]);  rsqrt_46 = None
        mul_396: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_397: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_395, mul_396);  mul_395 = mul_396 = None
        unsqueeze_226: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
        unsqueeze_227: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        mul_398: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_342);  primals_342 = None
        unsqueeze_229: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_398, 0);  mul_398 = None
        unsqueeze_230: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_399: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_228);  sub_61 = unsqueeze_228 = None
        sub_63: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_392, mul_399);  mul_392 = mul_399 = None
        sub_64: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_225);  sub_63 = unsqueeze_225 = None
        mul_400: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_231);  sub_64 = unsqueeze_231 = None
        mul_401: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_139);  sum_10 = squeeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_400, div_45, primals_338, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_400 = div_45 = primals_338 = None
        getitem_110: "f32[128, 1152, 7, 7]" = convolution_backward_4[0]
        getitem_111: "f32[1152, 1, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_45: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, getitem_91)
        mul_330: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        unsqueeze_180: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_181: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_336: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_330, unsqueeze_181);  mul_330 = unsqueeze_181 = None
        unsqueeze_182: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_337, -1);  primals_337 = None
        unsqueeze_183: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_283: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_336, unsqueeze_183);  mul_336 = unsqueeze_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_45: "f32[128, 1152, 7, 7]" = torch.ops.aten.neg.default(add_283)
        exp_45: "f32[128, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_45);  neg_45 = None
        add_284: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_45, 1);  exp_45 = None
        reciprocal_3: "f32[128, 1152, 7, 7]" = torch.ops.aten.reciprocal.default(add_284);  add_284 = None
        mul_402: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        mul_403: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_110, mul_402);  getitem_110 = None
        sub_65: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_402);  mul_402 = None
        mul_404: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_283, sub_65);  add_283 = sub_65 = None
        add_311: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_404, 1);  mul_404 = None
        mul_405: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_403, add_311);  mul_403 = add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_135: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_91, [0, 2, 3]);  getitem_91 = None
        unsqueeze_232: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_233: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
        unsqueeze_234: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
        sum_11: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 2, 3])
        sub_66: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_234);  convolution_75 = unsqueeze_234 = None
        mul_406: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_405, sub_66)
        sum_12: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_406, [0, 2, 3]);  mul_406 = None
        mul_407: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_11, 0.00015943877551020407)
        unsqueeze_235: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_407, 0);  mul_407 = None
        unsqueeze_236: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_408: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_12, 0.00015943877551020407)
        squeeze_136: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_409: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_410: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_408, mul_409);  mul_408 = mul_409 = None
        unsqueeze_238: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
        unsqueeze_239: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        mul_411: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_336);  primals_336 = None
        unsqueeze_241: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
        unsqueeze_242: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_412: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_240);  sub_66 = unsqueeze_240 = None
        sub_68: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_405, mul_412);  mul_405 = mul_412 = None
        sub_69: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_237);  sub_68 = unsqueeze_237 = None
        mul_413: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_243);  sub_69 = unsqueeze_243 = None
        mul_414: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_136);  sum_12 = squeeze_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_413, add_278, primals_332, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_413 = add_278 = primals_332 = None
        getitem_113: "f32[128, 192, 7, 7]" = convolution_backward_5[0]
        getitem_114: "f32[1152, 192, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_13: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3])
        sub_70: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_246);  convolution_74 = unsqueeze_246 = None
        mul_415: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, sub_70)
        sum_14: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_415, [0, 2, 3]);  mul_415 = None
        mul_416: "f32[192]" = torch.ops.aten.mul.Tensor(sum_13, 0.00015943877551020407)
        unsqueeze_247: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_416, 0);  mul_416 = None
        unsqueeze_248: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_417: "f32[192]" = torch.ops.aten.mul.Tensor(sum_14, 0.00015943877551020407)
        mul_418: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_419: "f32[192]" = torch.ops.aten.mul.Tensor(mul_417, mul_418);  mul_417 = mul_418 = None
        unsqueeze_250: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
        unsqueeze_251: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
        unsqueeze_252: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
        mul_420: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_330);  primals_330 = None
        unsqueeze_253: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_420, 0);  mul_420 = None
        unsqueeze_254: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_421: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_252);  sub_70 = unsqueeze_252 = None
        sub_72: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_113, mul_421);  mul_421 = None
        sub_73: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_249);  sub_72 = unsqueeze_249 = None
        mul_422: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_255);  sub_73 = unsqueeze_255 = None
        mul_423: "f32[192]" = torch.ops.aten.mul.Tensor(sum_14, squeeze_133);  sum_14 = squeeze_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_422, mul_322, primals_326, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_422 = mul_322 = primals_326 = None
        getitem_116: "f32[128, 1152, 7, 7]" = convolution_backward_6[0]
        getitem_117: "f32[192, 1152, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_43: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, getitem_87)
        mul_315: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = None
        unsqueeze_172: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_320, -1)
        unsqueeze_173: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_321: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_173);  mul_315 = unsqueeze_173 = None
        unsqueeze_174: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_321, -1);  primals_321 = None
        unsqueeze_175: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_270: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_175);  mul_321 = unsqueeze_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_43: "f32[128, 1152, 7, 7]" = torch.ops.aten.neg.default(add_270)
        exp_43: "f32[128, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_43);  neg_43 = None
        add_271: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_43, 1);  exp_43 = None
        div_43: "f32[128, 1152, 7, 7]" = torch.ops.aten.div.Tensor(add_270, add_271)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_424: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_116, div_43);  div_43 = None
        sigmoid_14: "f32[128, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_73);  convolution_73 = None
        mul_425: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_116, sigmoid_14);  getitem_116 = None
        sum_15: "f32[128, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_424, [2, 3], True);  mul_424 = None
        sub_74: "f32[128, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_14)
        mul_426: "f32[128, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_14, sub_74);  sigmoid_14 = sub_74 = None
        mul_427: "f32[128, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_15, mul_426);  sum_15 = mul_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_16: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_427, div_44, primals_324, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_427 = div_44 = primals_324 = None
        getitem_119: "f32[128, 48, 1, 1]" = convolution_backward_7[0]
        getitem_120: "f32[1152, 48, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_44: "f32[128, 48, 1, 1]" = torch.ops.aten.neg.default(convolution_72)
        exp_44: "f32[128, 48, 1, 1]" = torch.ops.aten.exp.default(neg_44);  neg_44 = None
        add_272: "f32[128, 48, 1, 1]" = torch.ops.aten.add.Tensor(exp_44, 1);  exp_44 = None
        reciprocal_4: "f32[128, 48, 1, 1]" = torch.ops.aten.reciprocal.default(add_272);  add_272 = None
        mul_428: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        mul_429: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_119, mul_428);  getitem_119 = None
        sub_75: "f32[128, 48, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_428);  mul_428 = None
        mul_430: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_72, sub_75);  convolution_72 = sub_75 = None
        add_313: "f32[128, 48, 1, 1]" = torch.ops.aten.add.Tensor(mul_430, 1);  mul_430 = None
        mul_431: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(mul_429, add_313);  mul_429 = add_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_17: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_431, mean_14, primals_322, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_431 = mean_14 = primals_322 = None
        getitem_122: "f32[128, 1152, 1, 1]" = convolution_backward_8[0]
        getitem_123: "f32[48, 1152, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_3: "f32[128, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_122, [128, 1152, 7, 7]);  getitem_122 = None
        div_51: "f32[128, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
        add_314: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_425, div_51);  mul_425 = div_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_5: "f32[128, 1152, 7, 7]" = torch.ops.aten.reciprocal.default(add_271);  add_271 = None
        mul_432: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        mul_433: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_314, mul_432);  add_314 = None
        sub_76: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_432);  mul_432 = None
        mul_434: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_270, sub_76);  add_270 = sub_76 = None
        add_316: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_434, 1);  mul_434 = None
        mul_435: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_433, add_316);  mul_433 = add_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_129: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        unsqueeze_256: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_129, 0);  squeeze_129 = None
        unsqueeze_257: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
        unsqueeze_258: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
        sum_18: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 2, 3])
        sub_77: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_258);  convolution_71 = unsqueeze_258 = None
        mul_436: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_435, sub_77)
        sum_19: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
        mul_437: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_18, 0.00015943877551020407)
        unsqueeze_259: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
        unsqueeze_260: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_438: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, 0.00015943877551020407)
        squeeze_130: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_43, [0, 2, 3]);  rsqrt_43 = None
        mul_439: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_440: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
        unsqueeze_262: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
        unsqueeze_263: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
        unsqueeze_264: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
        mul_441: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_320);  primals_320 = None
        unsqueeze_265: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
        unsqueeze_266: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_442: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_264);  sub_77 = unsqueeze_264 = None
        sub_79: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_435, mul_442);  mul_435 = mul_442 = None
        sub_80: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_261);  sub_79 = unsqueeze_261 = None
        mul_443: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_267);  sub_80 = unsqueeze_267 = None
        mul_444: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_130);  sum_19 = squeeze_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_443, div_42, primals_316, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_443 = div_42 = primals_316 = None
        getitem_125: "f32[128, 1152, 7, 7]" = convolution_backward_9[0]
        getitem_126: "f32[1152, 1, 5, 5]" = convolution_backward_9[1];  convolution_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_42: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, getitem_85)
        mul_308: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        unsqueeze_168: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_314, -1)
        unsqueeze_169: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_314: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_308, unsqueeze_169);  mul_308 = unsqueeze_169 = None
        unsqueeze_170: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_315, -1);  primals_315 = None
        unsqueeze_171: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_264: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_171);  mul_314 = unsqueeze_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_42: "f32[128, 1152, 7, 7]" = torch.ops.aten.neg.default(add_264)
        exp_42: "f32[128, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_42);  neg_42 = None
        add_265: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_42, 1);  exp_42 = None
        reciprocal_6: "f32[128, 1152, 7, 7]" = torch.ops.aten.reciprocal.default(add_265);  add_265 = None
        mul_445: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        mul_446: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_125, mul_445);  getitem_125 = None
        sub_81: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_445);  mul_445 = None
        mul_447: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_264, sub_81);  add_264 = sub_81 = None
        add_318: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_447, 1);  mul_447 = None
        mul_448: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_446, add_318);  mul_446 = add_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_126: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_85, [0, 2, 3]);  getitem_85 = None
        unsqueeze_268: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_269: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
        unsqueeze_270: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
        sum_20: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 2, 3])
        sub_82: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_270);  convolution_70 = unsqueeze_270 = None
        mul_449: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_448, sub_82)
        sum_21: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 2, 3]);  mul_449 = None
        mul_450: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_20, 0.00015943877551020407)
        unsqueeze_271: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
        unsqueeze_272: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_451: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, 0.00015943877551020407)
        squeeze_127: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_452: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_453: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_451, mul_452);  mul_451 = mul_452 = None
        unsqueeze_274: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_453, 0);  mul_453 = None
        unsqueeze_275: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        mul_454: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_314);  primals_314 = None
        unsqueeze_277: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
        unsqueeze_278: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_455: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_276);  sub_82 = unsqueeze_276 = None
        sub_84: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_448, mul_455);  mul_448 = mul_455 = None
        sub_85: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_84, unsqueeze_273);  sub_84 = unsqueeze_273 = None
        mul_456: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_279);  sub_85 = unsqueeze_279 = None
        mul_457: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_127);  sum_21 = squeeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_456, add_259, primals_310, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_456 = add_259 = primals_310 = None
        getitem_128: "f32[128, 192, 7, 7]" = convolution_backward_10[0]
        getitem_129: "f32[1152, 192, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
        add_319: "f32[128, 192, 7, 7]" = torch.ops.aten.add.Tensor(getitem_113, getitem_128);  getitem_113 = getitem_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_319, [0, 2, 3])
        sub_86: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_282);  convolution_69 = unsqueeze_282 = None
        mul_458: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_319, sub_86)
        sum_23: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
        mul_459: "f32[192]" = torch.ops.aten.mul.Tensor(sum_22, 0.00015943877551020407)
        unsqueeze_283: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
        unsqueeze_284: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_460: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, 0.00015943877551020407)
        mul_461: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_462: "f32[192]" = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
        unsqueeze_286: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
        unsqueeze_287: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
        unsqueeze_288: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
        mul_463: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_308);  primals_308 = None
        unsqueeze_289: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
        unsqueeze_290: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_464: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_288);  sub_86 = unsqueeze_288 = None
        sub_88: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_319, mul_464);  mul_464 = None
        sub_89: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_88, unsqueeze_285);  sub_88 = unsqueeze_285 = None
        mul_465: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_291);  sub_89 = unsqueeze_291 = None
        mul_466: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_124);  sum_23 = squeeze_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_465, mul_300, primals_304, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_465 = mul_300 = primals_304 = None
        getitem_131: "f32[128, 1152, 7, 7]" = convolution_backward_11[0]
        getitem_132: "f32[192, 1152, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_40: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, getitem_81)
        mul_293: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        unsqueeze_160: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_298, -1)
        unsqueeze_161: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_299: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_293, unsqueeze_161);  mul_293 = unsqueeze_161 = None
        unsqueeze_162: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_299, -1);  primals_299 = None
        unsqueeze_163: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_251: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_163);  mul_299 = unsqueeze_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_40: "f32[128, 1152, 7, 7]" = torch.ops.aten.neg.default(add_251)
        exp_40: "f32[128, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_40);  neg_40 = None
        add_252: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_40, 1);  exp_40 = None
        div_40: "f32[128, 1152, 7, 7]" = torch.ops.aten.div.Tensor(add_251, add_252)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_467: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_131, div_40);  div_40 = None
        sigmoid_13: "f32[128, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_68);  convolution_68 = None
        mul_468: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_131, sigmoid_13);  getitem_131 = None
        sum_24: "f32[128, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2, 3], True);  mul_467 = None
        sub_90: "f32[128, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_13)
        mul_469: "f32[128, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_13, sub_90);  sigmoid_13 = sub_90 = None
        mul_470: "f32[128, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_24, mul_469);  sum_24 = mul_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_25: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_470, div_41, primals_302, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_470 = div_41 = primals_302 = None
        getitem_134: "f32[128, 48, 1, 1]" = convolution_backward_12[0]
        getitem_135: "f32[1152, 48, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_41: "f32[128, 48, 1, 1]" = torch.ops.aten.neg.default(convolution_67)
        exp_41: "f32[128, 48, 1, 1]" = torch.ops.aten.exp.default(neg_41);  neg_41 = None
        add_253: "f32[128, 48, 1, 1]" = torch.ops.aten.add.Tensor(exp_41, 1);  exp_41 = None
        reciprocal_7: "f32[128, 48, 1, 1]" = torch.ops.aten.reciprocal.default(add_253);  add_253 = None
        mul_471: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        mul_472: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_134, mul_471);  getitem_134 = None
        sub_91: "f32[128, 48, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_471);  mul_471 = None
        mul_473: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_67, sub_91);  convolution_67 = sub_91 = None
        add_321: "f32[128, 48, 1, 1]" = torch.ops.aten.add.Tensor(mul_473, 1);  mul_473 = None
        mul_474: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(mul_472, add_321);  mul_472 = add_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_26: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_474, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_474, mean_13, primals_300, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = mean_13 = primals_300 = None
        getitem_137: "f32[128, 1152, 1, 1]" = convolution_backward_13[0]
        getitem_138: "f32[48, 1152, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_4: "f32[128, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_137, [128, 1152, 7, 7]);  getitem_137 = None
        div_52: "f32[128, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
        add_322: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_468, div_52);  mul_468 = div_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_8: "f32[128, 1152, 7, 7]" = torch.ops.aten.reciprocal.default(add_252);  add_252 = None
        mul_475: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        mul_476: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_322, mul_475);  add_322 = None
        sub_92: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_475);  mul_475 = None
        mul_477: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_251, sub_92);  add_251 = sub_92 = None
        add_324: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_477, 1);  mul_477 = None
        mul_478: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_476, add_324);  mul_476 = add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_120: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        unsqueeze_292: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_293: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
        unsqueeze_294: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
        sum_27: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_478, [0, 2, 3])
        sub_93: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_294);  convolution_66 = unsqueeze_294 = None
        mul_479: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_478, sub_93)
        sum_28: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 2, 3]);  mul_479 = None
        mul_480: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, 0.00015943877551020407)
        unsqueeze_295: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
        unsqueeze_296: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_481: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_28, 0.00015943877551020407)
        squeeze_121: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_482: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_483: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_481, mul_482);  mul_481 = mul_482 = None
        unsqueeze_298: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
        unsqueeze_299: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
        unsqueeze_300: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
        mul_484: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_298);  primals_298 = None
        unsqueeze_301: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
        unsqueeze_302: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_485: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_300);  sub_93 = unsqueeze_300 = None
        sub_95: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_478, mul_485);  mul_478 = mul_485 = None
        sub_96: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_297);  sub_95 = unsqueeze_297 = None
        mul_486: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_303);  sub_96 = unsqueeze_303 = None
        mul_487: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_121);  sum_28 = squeeze_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_486, div_39, primals_294, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_486 = div_39 = primals_294 = None
        getitem_140: "f32[128, 1152, 7, 7]" = convolution_backward_14[0]
        getitem_141: "f32[1152, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_39: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, getitem_79)
        mul_286: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        unsqueeze_156: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_292, -1)
        unsqueeze_157: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_292: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_157);  mul_286 = unsqueeze_157 = None
        unsqueeze_158: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_293, -1);  primals_293 = None
        unsqueeze_159: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_245: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_292, unsqueeze_159);  mul_292 = unsqueeze_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_39: "f32[128, 1152, 7, 7]" = torch.ops.aten.neg.default(add_245)
        exp_39: "f32[128, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_39);  neg_39 = None
        add_246: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_39, 1);  exp_39 = None
        reciprocal_9: "f32[128, 1152, 7, 7]" = torch.ops.aten.reciprocal.default(add_246);  add_246 = None
        mul_488: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        mul_489: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_140, mul_488);  getitem_140 = None
        sub_97: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_488);  mul_488 = None
        mul_490: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_245, sub_97);  add_245 = sub_97 = None
        add_326: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_490, 1);  mul_490 = None
        mul_491: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_489, add_326);  mul_489 = add_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_117: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        unsqueeze_304: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_117, 0);  squeeze_117 = None
        unsqueeze_305: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
        unsqueeze_306: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
        sum_29: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3])
        sub_98: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_306);  convolution_65 = unsqueeze_306 = None
        mul_492: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_491, sub_98)
        sum_30: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_492, [0, 2, 3]);  mul_492 = None
        mul_493: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_29, 0.00015943877551020407)
        unsqueeze_307: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
        unsqueeze_308: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        unsqueeze_309: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
        mul_494: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
        squeeze_118: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_495: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_496: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
        unsqueeze_310: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
        unsqueeze_311: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
        unsqueeze_312: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
        mul_497: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_292);  primals_292 = None
        unsqueeze_313: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_497, 0);  mul_497 = None
        unsqueeze_314: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_315: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
        mul_498: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_312);  sub_98 = unsqueeze_312 = None
        sub_100: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_491, mul_498);  mul_491 = mul_498 = None
        sub_101: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_100, unsqueeze_309);  sub_100 = unsqueeze_309 = None
        mul_499: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_315);  sub_101 = unsqueeze_315 = None
        mul_500: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_118);  sum_30 = squeeze_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_499, add_240, primals_288, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_499 = add_240 = primals_288 = None
        getitem_143: "f32[128, 192, 7, 7]" = convolution_backward_15[0]
        getitem_144: "f32[1152, 192, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
        add_327: "f32[128, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_319, getitem_143);  add_319 = getitem_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_31: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_327, [0, 2, 3])
        sub_102: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_318);  convolution_64 = unsqueeze_318 = None
        mul_501: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_327, sub_102)
        sum_32: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 2, 3]);  mul_501 = None
        mul_502: "f32[192]" = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
        unsqueeze_319: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
        unsqueeze_320: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_503: "f32[192]" = torch.ops.aten.mul.Tensor(sum_32, 0.00015943877551020407)
        mul_504: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_505: "f32[192]" = torch.ops.aten.mul.Tensor(mul_503, mul_504);  mul_503 = mul_504 = None
        unsqueeze_322: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
        unsqueeze_323: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
        unsqueeze_324: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
        mul_506: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_286);  primals_286 = None
        unsqueeze_325: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_506, 0);  mul_506 = None
        unsqueeze_326: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_507: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_324);  sub_102 = unsqueeze_324 = None
        sub_104: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_327, mul_507);  mul_507 = None
        sub_105: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_321);  sub_104 = unsqueeze_321 = None
        mul_508: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_327);  sub_105 = unsqueeze_327 = None
        mul_509: "f32[192]" = torch.ops.aten.mul.Tensor(sum_32, squeeze_115);  sum_32 = squeeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_508, mul_278, primals_282, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_508 = mul_278 = primals_282 = None
        getitem_146: "f32[128, 1152, 7, 7]" = convolution_backward_16[0]
        getitem_147: "f32[192, 1152, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_37: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, getitem_75)
        mul_271: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        unsqueeze_148: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_149: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_277: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_149);  mul_271 = unsqueeze_149 = None
        unsqueeze_150: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_151: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_232: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_277, unsqueeze_151);  mul_277 = unsqueeze_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_37: "f32[128, 1152, 7, 7]" = torch.ops.aten.neg.default(add_232)
        exp_37: "f32[128, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_37);  neg_37 = None
        add_233: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_37, 1);  exp_37 = None
        div_37: "f32[128, 1152, 7, 7]" = torch.ops.aten.div.Tensor(add_232, add_233)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_510: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_146, div_37);  div_37 = None
        sigmoid_12: "f32[128, 1152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63);  convolution_63 = None
        mul_511: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_146, sigmoid_12);  getitem_146 = None
        sum_33: "f32[128, 1152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_510, [2, 3], True);  mul_510 = None
        sub_106: "f32[128, 1152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_12)
        mul_512: "f32[128, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_12, sub_106);  sigmoid_12 = sub_106 = None
        mul_513: "f32[128, 1152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_33, mul_512);  sum_33 = mul_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_34: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_513, div_38, primals_280, [1152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_513 = div_38 = primals_280 = None
        getitem_149: "f32[128, 48, 1, 1]" = convolution_backward_17[0]
        getitem_150: "f32[1152, 48, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_38: "f32[128, 48, 1, 1]" = torch.ops.aten.neg.default(convolution_62)
        exp_38: "f32[128, 48, 1, 1]" = torch.ops.aten.exp.default(neg_38);  neg_38 = None
        add_234: "f32[128, 48, 1, 1]" = torch.ops.aten.add.Tensor(exp_38, 1);  exp_38 = None
        reciprocal_10: "f32[128, 48, 1, 1]" = torch.ops.aten.reciprocal.default(add_234);  add_234 = None
        mul_514: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        mul_515: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_149, mul_514);  getitem_149 = None
        sub_107: "f32[128, 48, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_514);  mul_514 = None
        mul_516: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_62, sub_107);  convolution_62 = sub_107 = None
        add_329: "f32[128, 48, 1, 1]" = torch.ops.aten.add.Tensor(mul_516, 1);  mul_516 = None
        mul_517: "f32[128, 48, 1, 1]" = torch.ops.aten.mul.Tensor(mul_515, add_329);  mul_515 = add_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_35: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_517, mean_12, primals_278, [48], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_517 = mean_12 = primals_278 = None
        getitem_152: "f32[128, 1152, 1, 1]" = convolution_backward_18[0]
        getitem_153: "f32[48, 1152, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_5: "f32[128, 1152, 7, 7]" = torch.ops.aten.expand.default(getitem_152, [128, 1152, 7, 7]);  getitem_152 = None
        div_53: "f32[128, 1152, 7, 7]" = torch.ops.aten.div.Scalar(expand_5, 49);  expand_5 = None
        add_330: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_511, div_53);  mul_511 = div_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_11: "f32[128, 1152, 7, 7]" = torch.ops.aten.reciprocal.default(add_233);  add_233 = None
        mul_518: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        mul_519: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_330, mul_518);  add_330 = None
        sub_108: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_518);  mul_518 = None
        mul_520: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_232, sub_108);  add_232 = sub_108 = None
        add_332: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_520, 1);  mul_520 = None
        mul_521: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_519, add_332);  mul_519 = add_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_111: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_75, [0, 2, 3]);  getitem_75 = None
        unsqueeze_328: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_329: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
        unsqueeze_330: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
        sum_36: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 2, 3])
        sub_109: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_330);  convolution_61 = unsqueeze_330 = None
        mul_522: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_521, sub_109)
        sum_37: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_522, [0, 2, 3]);  mul_522 = None
        mul_523: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_36, 0.00015943877551020407)
        unsqueeze_331: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
        unsqueeze_332: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_524: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_37, 0.00015943877551020407)
        squeeze_112: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_525: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_526: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_524, mul_525);  mul_524 = mul_525 = None
        unsqueeze_334: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
        unsqueeze_335: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
        unsqueeze_336: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
        mul_527: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_276);  primals_276 = None
        unsqueeze_337: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
        unsqueeze_338: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_528: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_336);  sub_109 = unsqueeze_336 = None
        sub_111: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_521, mul_528);  mul_521 = mul_528 = None
        sub_112: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_333);  sub_111 = unsqueeze_333 = None
        mul_529: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_339);  sub_112 = unsqueeze_339 = None
        mul_530: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_112);  sum_37 = squeeze_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_529, div_36, primals_272, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_529 = div_36 = primals_272 = None
        getitem_155: "f32[128, 1152, 7, 7]" = convolution_backward_19[0]
        getitem_156: "f32[1152, 1, 5, 5]" = convolution_backward_19[1];  convolution_backward_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_36: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, getitem_73)
        mul_264: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = None
        unsqueeze_144: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_270, -1)
        unsqueeze_145: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        mul_270: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_264, unsqueeze_145);  mul_264 = unsqueeze_145 = None
        unsqueeze_146: "f32[1152, 1]" = torch.ops.aten.unsqueeze.default(primals_271, -1);  primals_271 = None
        unsqueeze_147: "f32[1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        add_226: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_270, unsqueeze_147);  mul_270 = unsqueeze_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_36: "f32[128, 1152, 7, 7]" = torch.ops.aten.neg.default(add_226)
        exp_36: "f32[128, 1152, 7, 7]" = torch.ops.aten.exp.default(neg_36);  neg_36 = None
        add_227: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(exp_36, 1);  exp_36 = None
        reciprocal_12: "f32[128, 1152, 7, 7]" = torch.ops.aten.reciprocal.default(add_227);  add_227 = None
        mul_531: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        mul_532: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_155, mul_531);  getitem_155 = None
        sub_113: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_531);  mul_531 = None
        mul_533: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(add_226, sub_113);  add_226 = sub_113 = None
        add_334: "f32[128, 1152, 7, 7]" = torch.ops.aten.add.Tensor(mul_533, 1);  mul_533 = None
        mul_534: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_532, add_334);  mul_532 = add_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_108: "f32[1152]" = torch.ops.aten.squeeze.dims(getitem_73, [0, 2, 3]);  getitem_73 = None
        unsqueeze_340: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(squeeze_108, 0);  squeeze_108 = None
        unsqueeze_341: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
        unsqueeze_342: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
        sum_38: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_534, [0, 2, 3])
        sub_114: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_342);  convolution_60 = unsqueeze_342 = None
        mul_535: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(mul_534, sub_114)
        sum_39: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
        mul_536: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_38, 0.00015943877551020407)
        unsqueeze_343: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
        unsqueeze_344: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_537: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_39, 0.00015943877551020407)
        squeeze_109: "f32[1152]" = torch.ops.aten.squeeze.dims(rsqrt_36, [0, 2, 3]);  rsqrt_36 = None
        mul_538: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_539: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
        unsqueeze_346: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
        unsqueeze_347: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        mul_540: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_270);  primals_270 = None
        unsqueeze_349: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
        unsqueeze_350: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_541: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_348);  sub_114 = unsqueeze_348 = None
        sub_116: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(mul_534, mul_541);  mul_534 = mul_541 = None
        sub_117: "f32[128, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_345);  sub_116 = unsqueeze_345 = None
        mul_542: "f32[128, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_351);  sub_117 = unsqueeze_351 = None
        mul_543: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_109);  sum_39 = squeeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_542, add_221, primals_266, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_542 = add_221 = primals_266 = None
        getitem_158: "f32[128, 192, 7, 7]" = convolution_backward_20[0]
        getitem_159: "f32[1152, 192, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
        add_335: "f32[128, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_327, getitem_158);  add_327 = getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_40: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_335, [0, 2, 3])
        sub_118: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_354);  convolution_59 = unsqueeze_354 = None
        mul_544: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_335, sub_118)
        sum_41: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3]);  mul_544 = None
        mul_545: "f32[192]" = torch.ops.aten.mul.Tensor(sum_40, 0.00015943877551020407)
        unsqueeze_355: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
        unsqueeze_356: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_546: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, 0.00015943877551020407)
        mul_547: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_548: "f32[192]" = torch.ops.aten.mul.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
        unsqueeze_358: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
        unsqueeze_359: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
        unsqueeze_360: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
        mul_549: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_264);  primals_264 = None
        unsqueeze_361: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
        unsqueeze_362: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_550: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_360);  sub_118 = unsqueeze_360 = None
        sub_120: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_335, mul_550);  add_335 = mul_550 = None
        sub_121: "f32[128, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_357);  sub_120 = unsqueeze_357 = None
        mul_551: "f32[128, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_363);  sub_121 = unsqueeze_363 = None
        mul_552: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_106);  sum_41 = squeeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_551, mul_256, primals_260, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_551 = mul_256 = primals_260 = None
        getitem_161: "f32[128, 672, 7, 7]" = convolution_backward_21[0]
        getitem_162: "f32[192, 672, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_34: "f32[128, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, getitem_69)
        mul_249: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        unsqueeze_136: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_254, -1)
        unsqueeze_137: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_255: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_249, unsqueeze_137);  mul_249 = unsqueeze_137 = None
        unsqueeze_138: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_255, -1);  primals_255 = None
        unsqueeze_139: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_214: "f32[128, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_255, unsqueeze_139);  mul_255 = unsqueeze_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_34: "f32[128, 672, 7, 7]" = torch.ops.aten.neg.default(add_214)
        exp_34: "f32[128, 672, 7, 7]" = torch.ops.aten.exp.default(neg_34);  neg_34 = None
        add_215: "f32[128, 672, 7, 7]" = torch.ops.aten.add.Tensor(exp_34, 1);  exp_34 = None
        div_34: "f32[128, 672, 7, 7]" = torch.ops.aten.div.Tensor(add_214, add_215)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_553: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_161, div_34);  div_34 = None
        sigmoid_11: "f32[128, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
        mul_554: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_161, sigmoid_11);  getitem_161 = None
        sum_42: "f32[128, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2, 3], True);  mul_553 = None
        sub_122: "f32[128, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_11)
        mul_555: "f32[128, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_11, sub_122);  sigmoid_11 = sub_122 = None
        mul_556: "f32[128, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_42, mul_555);  sum_42 = mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_43: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_556, div_35, primals_258, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_556 = div_35 = primals_258 = None
        getitem_164: "f32[128, 28, 1, 1]" = convolution_backward_22[0]
        getitem_165: "f32[672, 28, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_35: "f32[128, 28, 1, 1]" = torch.ops.aten.neg.default(convolution_57)
        exp_35: "f32[128, 28, 1, 1]" = torch.ops.aten.exp.default(neg_35);  neg_35 = None
        add_216: "f32[128, 28, 1, 1]" = torch.ops.aten.add.Tensor(exp_35, 1);  exp_35 = None
        reciprocal_13: "f32[128, 28, 1, 1]" = torch.ops.aten.reciprocal.default(add_216);  add_216 = None
        mul_557: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        mul_558: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_164, mul_557);  getitem_164 = None
        sub_123: "f32[128, 28, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_557);  mul_557 = None
        mul_559: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_57, sub_123);  convolution_57 = sub_123 = None
        add_337: "f32[128, 28, 1, 1]" = torch.ops.aten.add.Tensor(mul_559, 1);  mul_559 = None
        mul_560: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(mul_558, add_337);  mul_558 = add_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_44: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_560, mean_11, primals_256, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_560 = mean_11 = primals_256 = None
        getitem_167: "f32[128, 672, 1, 1]" = convolution_backward_23[0]
        getitem_168: "f32[28, 672, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_6: "f32[128, 672, 7, 7]" = torch.ops.aten.expand.default(getitem_167, [128, 672, 7, 7]);  getitem_167 = None
        div_54: "f32[128, 672, 7, 7]" = torch.ops.aten.div.Scalar(expand_6, 49);  expand_6 = None
        add_338: "f32[128, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_554, div_54);  mul_554 = div_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_14: "f32[128, 672, 7, 7]" = torch.ops.aten.reciprocal.default(add_215);  add_215 = None
        mul_561: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        mul_562: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_338, mul_561);  add_338 = None
        sub_124: "f32[128, 672, 7, 7]" = torch.ops.aten.sub.Tensor(1, mul_561);  mul_561 = None
        mul_563: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_214, sub_124);  add_214 = sub_124 = None
        add_340: "f32[128, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_563, 1);  mul_563 = None
        mul_564: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_562, add_340);  mul_562 = add_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_102: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        unsqueeze_364: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_365: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
        unsqueeze_366: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
        sum_45: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_564, [0, 2, 3])
        sub_125: "f32[128, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_366);  convolution_56 = unsqueeze_366 = None
        mul_565: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(mul_564, sub_125)
        sum_46: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_565, [0, 2, 3]);  mul_565 = None
        mul_566: "f32[672]" = torch.ops.aten.mul.Tensor(sum_45, 0.00015943877551020407)
        unsqueeze_367: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
        unsqueeze_368: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_567: "f32[672]" = torch.ops.aten.mul.Tensor(sum_46, 0.00015943877551020407)
        squeeze_103: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_568: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_569: "f32[672]" = torch.ops.aten.mul.Tensor(mul_567, mul_568);  mul_567 = mul_568 = None
        unsqueeze_370: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
        unsqueeze_371: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
        unsqueeze_372: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
        mul_570: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_254);  primals_254 = None
        unsqueeze_373: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
        unsqueeze_374: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_571: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_372);  sub_125 = unsqueeze_372 = None
        sub_127: "f32[128, 672, 7, 7]" = torch.ops.aten.sub.Tensor(mul_564, mul_571);  mul_564 = mul_571 = None
        sub_128: "f32[128, 672, 7, 7]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_369);  sub_127 = unsqueeze_369 = None
        mul_572: "f32[128, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_375);  sub_128 = unsqueeze_375 = None
        mul_573: "f32[672]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_103);  sum_46 = squeeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_572, constant_pad_nd_4, primals_250, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_572 = constant_pad_nd_4 = primals_250 = None
        getitem_170: "f32[128, 672, 17, 17]" = convolution_backward_24[0]
        getitem_171: "f32[672, 1, 5, 5]" = convolution_backward_24[1];  convolution_backward_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_5: "f32[128, 672, 14, 14]" = torch.ops.aten.constant_pad_nd.default(getitem_170, [-1, -2, -1, -2]);  getitem_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_33: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_67)
        mul_242: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = None
        unsqueeze_132: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_248, -1)
        unsqueeze_133: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_248: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_242, unsqueeze_133);  mul_242 = unsqueeze_133 = None
        unsqueeze_134: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_249, -1);  primals_249 = None
        unsqueeze_135: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_208: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_248, unsqueeze_135);  mul_248 = unsqueeze_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_33: "f32[128, 672, 14, 14]" = torch.ops.aten.neg.default(add_208)
        exp_33: "f32[128, 672, 14, 14]" = torch.ops.aten.exp.default(neg_33);  neg_33 = None
        add_209: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(exp_33, 1);  exp_33 = None
        reciprocal_15: "f32[128, 672, 14, 14]" = torch.ops.aten.reciprocal.default(add_209);  add_209 = None
        mul_574: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        mul_575: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(constant_pad_nd_5, mul_574);  constant_pad_nd_5 = None
        sub_129: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_574);  mul_574 = None
        mul_576: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_208, sub_129);  add_208 = sub_129 = None
        add_342: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_576, 1);  mul_576 = None
        mul_577: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_575, add_342);  mul_575 = add_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_99: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_67, [0, 2, 3]);  getitem_67 = None
        unsqueeze_376: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_99, 0);  squeeze_99 = None
        unsqueeze_377: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        sum_47: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 2, 3])
        sub_130: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_378);  convolution_55 = unsqueeze_378 = None
        mul_578: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_577, sub_130)
        sum_48: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 2, 3]);  mul_578 = None
        mul_579: "f32[672]" = torch.ops.aten.mul.Tensor(sum_47, 3.985969387755102e-05)
        unsqueeze_379: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
        unsqueeze_380: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_580: "f32[672]" = torch.ops.aten.mul.Tensor(sum_48, 3.985969387755102e-05)
        squeeze_100: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_33, [0, 2, 3]);  rsqrt_33 = None
        mul_581: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_582: "f32[672]" = torch.ops.aten.mul.Tensor(mul_580, mul_581);  mul_580 = mul_581 = None
        unsqueeze_382: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
        unsqueeze_383: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        mul_583: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_248);  primals_248 = None
        unsqueeze_385: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
        unsqueeze_386: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_584: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_384);  sub_130 = unsqueeze_384 = None
        sub_132: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_577, mul_584);  mul_577 = mul_584 = None
        sub_133: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_132, unsqueeze_381);  sub_132 = unsqueeze_381 = None
        mul_585: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_387);  sub_133 = unsqueeze_387 = None
        mul_586: "f32[672]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_100);  sum_48 = squeeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_585, add_203, primals_244, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_585 = add_203 = primals_244 = None
        getitem_173: "f32[128, 112, 14, 14]" = convolution_backward_25[0]
        getitem_174: "f32[672, 112, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_49: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_173, [0, 2, 3])
        sub_134: "f32[128, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_390);  convolution_54 = unsqueeze_390 = None
        mul_587: "f32[128, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_173, sub_134)
        sum_50: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_587, [0, 2, 3]);  mul_587 = None
        mul_588: "f32[112]" = torch.ops.aten.mul.Tensor(sum_49, 3.985969387755102e-05)
        unsqueeze_391: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
        unsqueeze_392: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_589: "f32[112]" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
        mul_590: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_591: "f32[112]" = torch.ops.aten.mul.Tensor(mul_589, mul_590);  mul_589 = mul_590 = None
        unsqueeze_394: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
        unsqueeze_395: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
        unsqueeze_396: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
        mul_592: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_242);  primals_242 = None
        unsqueeze_397: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
        unsqueeze_398: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_593: "f32[128, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_396);  sub_134 = unsqueeze_396 = None
        sub_136: "f32[128, 112, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_173, mul_593);  mul_593 = None
        sub_137: "f32[128, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_136, unsqueeze_393);  sub_136 = unsqueeze_393 = None
        mul_594: "f32[128, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_399);  sub_137 = unsqueeze_399 = None
        mul_595: "f32[112]" = torch.ops.aten.mul.Tensor(sum_50, squeeze_97);  sum_50 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_594, mul_234, primals_238, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_594 = mul_234 = primals_238 = None
        getitem_176: "f32[128, 672, 14, 14]" = convolution_backward_26[0]
        getitem_177: "f32[112, 672, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_31: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_63)
        mul_227: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        unsqueeze_124: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_232, -1)
        unsqueeze_125: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_233: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_227, unsqueeze_125);  mul_227 = unsqueeze_125 = None
        unsqueeze_126: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_233, -1);  primals_233 = None
        unsqueeze_127: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_195: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_127);  mul_233 = unsqueeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_31: "f32[128, 672, 14, 14]" = torch.ops.aten.neg.default(add_195)
        exp_31: "f32[128, 672, 14, 14]" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_196: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        div_31: "f32[128, 672, 14, 14]" = torch.ops.aten.div.Tensor(add_195, add_196)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_596: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_176, div_31);  div_31 = None
        sigmoid_10: "f32[128, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
        mul_597: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_176, sigmoid_10);  getitem_176 = None
        sum_51: "f32[128, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [2, 3], True);  mul_596 = None
        sub_138: "f32[128, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_10)
        mul_598: "f32[128, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_10, sub_138);  sigmoid_10 = sub_138 = None
        mul_599: "f32[128, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_51, mul_598);  sum_51 = mul_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_52: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3])
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_599, div_32, primals_236, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_599 = div_32 = primals_236 = None
        getitem_179: "f32[128, 28, 1, 1]" = convolution_backward_27[0]
        getitem_180: "f32[672, 28, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_32: "f32[128, 28, 1, 1]" = torch.ops.aten.neg.default(convolution_52)
        exp_32: "f32[128, 28, 1, 1]" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_197: "f32[128, 28, 1, 1]" = torch.ops.aten.add.Tensor(exp_32, 1);  exp_32 = None
        reciprocal_16: "f32[128, 28, 1, 1]" = torch.ops.aten.reciprocal.default(add_197);  add_197 = None
        mul_600: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        mul_601: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_179, mul_600);  getitem_179 = None
        sub_139: "f32[128, 28, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_600);  mul_600 = None
        mul_602: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_52, sub_139);  convolution_52 = sub_139 = None
        add_344: "f32[128, 28, 1, 1]" = torch.ops.aten.add.Tensor(mul_602, 1);  mul_602 = None
        mul_603: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(mul_601, add_344);  mul_601 = add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_53: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_603, [0, 2, 3])
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_603, mean_10, primals_234, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_603 = mean_10 = primals_234 = None
        getitem_182: "f32[128, 672, 1, 1]" = convolution_backward_28[0]
        getitem_183: "f32[28, 672, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_7: "f32[128, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_182, [128, 672, 14, 14]);  getitem_182 = None
        div_55: "f32[128, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_7, 196);  expand_7 = None
        add_345: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_597, div_55);  mul_597 = div_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_17: "f32[128, 672, 14, 14]" = torch.ops.aten.reciprocal.default(add_196);  add_196 = None
        mul_604: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        mul_605: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_345, mul_604);  add_345 = None
        sub_140: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_604);  mul_604 = None
        mul_606: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_195, sub_140);  add_195 = sub_140 = None
        add_347: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_606, 1);  mul_606 = None
        mul_607: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_605, add_347);  mul_605 = add_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_93: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        unsqueeze_400: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_401: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
        unsqueeze_402: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
        sum_54: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3])
        sub_141: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_402);  convolution_51 = unsqueeze_402 = None
        mul_608: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_607, sub_141)
        sum_55: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
        mul_609: "f32[672]" = torch.ops.aten.mul.Tensor(sum_54, 3.985969387755102e-05)
        unsqueeze_403: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
        unsqueeze_404: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_610: "f32[672]" = torch.ops.aten.mul.Tensor(sum_55, 3.985969387755102e-05)
        squeeze_94: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_611: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_612: "f32[672]" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
        unsqueeze_406: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_407: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        mul_613: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_232);  primals_232 = None
        unsqueeze_409: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
        unsqueeze_410: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_614: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_408);  sub_141 = unsqueeze_408 = None
        sub_143: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_607, mul_614);  mul_607 = mul_614 = None
        sub_144: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_405);  sub_143 = unsqueeze_405 = None
        mul_615: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_411);  sub_144 = unsqueeze_411 = None
        mul_616: "f32[672]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_94);  sum_55 = squeeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_615, div_30, primals_228, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_615 = div_30 = primals_228 = None
        getitem_185: "f32[128, 672, 14, 14]" = convolution_backward_29[0]
        getitem_186: "f32[672, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_30: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, getitem_61)
        mul_220: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = None
        unsqueeze_120: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_226, -1)
        unsqueeze_121: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_226: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_121);  mul_220 = unsqueeze_121 = None
        unsqueeze_122: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_227, -1);  primals_227 = None
        unsqueeze_123: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_189: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_226, unsqueeze_123);  mul_226 = unsqueeze_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_30: "f32[128, 672, 14, 14]" = torch.ops.aten.neg.default(add_189)
        exp_30: "f32[128, 672, 14, 14]" = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_190: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        reciprocal_18: "f32[128, 672, 14, 14]" = torch.ops.aten.reciprocal.default(add_190);  add_190 = None
        mul_617: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        mul_618: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_185, mul_617);  getitem_185 = None
        sub_145: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_617);  mul_617 = None
        mul_619: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_189, sub_145);  add_189 = sub_145 = None
        add_349: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_619, 1);  mul_619 = None
        mul_620: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_618, add_349);  mul_618 = add_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_90: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        unsqueeze_412: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_413: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
        unsqueeze_414: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
        sum_56: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3])
        sub_146: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_414);  convolution_50 = unsqueeze_414 = None
        mul_621: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_620, sub_146)
        sum_57: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 2, 3]);  mul_621 = None
        mul_622: "f32[672]" = torch.ops.aten.mul.Tensor(sum_56, 3.985969387755102e-05)
        unsqueeze_415: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
        unsqueeze_416: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_623: "f32[672]" = torch.ops.aten.mul.Tensor(sum_57, 3.985969387755102e-05)
        squeeze_91: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_30, [0, 2, 3]);  rsqrt_30 = None
        mul_624: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_625: "f32[672]" = torch.ops.aten.mul.Tensor(mul_623, mul_624);  mul_623 = mul_624 = None
        unsqueeze_418: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
        unsqueeze_419: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
        unsqueeze_420: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
        mul_626: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_226);  primals_226 = None
        unsqueeze_421: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
        unsqueeze_422: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_627: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_420);  sub_146 = unsqueeze_420 = None
        sub_148: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_620, mul_627);  mul_620 = mul_627 = None
        sub_149: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_148, unsqueeze_417);  sub_148 = unsqueeze_417 = None
        mul_628: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_423);  sub_149 = unsqueeze_423 = None
        mul_629: "f32[672]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_91);  sum_57 = squeeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_628, add_184, primals_222, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_628 = add_184 = primals_222 = None
        getitem_188: "f32[128, 112, 14, 14]" = convolution_backward_30[0]
        getitem_189: "f32[672, 112, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
        add_350: "f32[128, 112, 14, 14]" = torch.ops.aten.add.Tensor(getitem_173, getitem_188);  getitem_173 = getitem_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_58: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_350, [0, 2, 3])
        sub_150: "f32[128, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_426);  convolution_49 = unsqueeze_426 = None
        mul_630: "f32[128, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_350, sub_150)
        sum_59: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 2, 3]);  mul_630 = None
        mul_631: "f32[112]" = torch.ops.aten.mul.Tensor(sum_58, 3.985969387755102e-05)
        unsqueeze_427: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
        unsqueeze_428: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_632: "f32[112]" = torch.ops.aten.mul.Tensor(sum_59, 3.985969387755102e-05)
        mul_633: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_634: "f32[112]" = torch.ops.aten.mul.Tensor(mul_632, mul_633);  mul_632 = mul_633 = None
        unsqueeze_430: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_634, 0);  mul_634 = None
        unsqueeze_431: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
        unsqueeze_432: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
        mul_635: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_220);  primals_220 = None
        unsqueeze_433: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
        unsqueeze_434: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_636: "f32[128, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_432);  sub_150 = unsqueeze_432 = None
        sub_152: "f32[128, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_350, mul_636);  mul_636 = None
        sub_153: "f32[128, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_152, unsqueeze_429);  sub_152 = unsqueeze_429 = None
        mul_637: "f32[128, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_435);  sub_153 = unsqueeze_435 = None
        mul_638: "f32[112]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_88);  sum_59 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_637, mul_212, primals_216, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_637 = mul_212 = primals_216 = None
        getitem_191: "f32[128, 672, 14, 14]" = convolution_backward_31[0]
        getitem_192: "f32[112, 672, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_28: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, getitem_57)
        mul_205: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        unsqueeze_112: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_113: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_211: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_113);  mul_205 = unsqueeze_113 = None
        unsqueeze_114: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
        unsqueeze_115: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_176: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_211, unsqueeze_115);  mul_211 = unsqueeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_28: "f32[128, 672, 14, 14]" = torch.ops.aten.neg.default(add_176)
        exp_28: "f32[128, 672, 14, 14]" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_177: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        div_28: "f32[128, 672, 14, 14]" = torch.ops.aten.div.Tensor(add_176, add_177)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_639: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_191, div_28);  div_28 = None
        sigmoid_9: "f32[128, 672, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
        mul_640: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_191, sigmoid_9);  getitem_191 = None
        sum_60: "f32[128, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_639, [2, 3], True);  mul_639 = None
        sub_154: "f32[128, 672, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_9)
        mul_641: "f32[128, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_9, sub_154);  sigmoid_9 = sub_154 = None
        mul_642: "f32[128, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_60, mul_641);  sum_60 = mul_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_61: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_642, [0, 2, 3])
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_642, div_29, primals_214, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_642 = div_29 = primals_214 = None
        getitem_194: "f32[128, 28, 1, 1]" = convolution_backward_32[0]
        getitem_195: "f32[672, 28, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_29: "f32[128, 28, 1, 1]" = torch.ops.aten.neg.default(convolution_47)
        exp_29: "f32[128, 28, 1, 1]" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_178: "f32[128, 28, 1, 1]" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        reciprocal_19: "f32[128, 28, 1, 1]" = torch.ops.aten.reciprocal.default(add_178);  add_178 = None
        mul_643: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        mul_644: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_194, mul_643);  getitem_194 = None
        sub_155: "f32[128, 28, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_643);  mul_643 = None
        mul_645: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_47, sub_155);  convolution_47 = sub_155 = None
        add_352: "f32[128, 28, 1, 1]" = torch.ops.aten.add.Tensor(mul_645, 1);  mul_645 = None
        mul_646: "f32[128, 28, 1, 1]" = torch.ops.aten.mul.Tensor(mul_644, add_352);  mul_644 = add_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_62: "f32[28]" = torch.ops.aten.sum.dim_IntList(mul_646, [0, 2, 3])
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_646, mean_9, primals_212, [28], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_646 = mean_9 = primals_212 = None
        getitem_197: "f32[128, 672, 1, 1]" = convolution_backward_33[0]
        getitem_198: "f32[28, 672, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_8: "f32[128, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_197, [128, 672, 14, 14]);  getitem_197 = None
        div_56: "f32[128, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
        add_353: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_640, div_56);  mul_640 = div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_20: "f32[128, 672, 14, 14]" = torch.ops.aten.reciprocal.default(add_177);  add_177 = None
        mul_647: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        mul_648: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_353, mul_647);  add_353 = None
        sub_156: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_647);  mul_647 = None
        mul_649: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_176, sub_156);  add_176 = sub_156 = None
        add_355: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_649, 1);  mul_649 = None
        mul_650: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_648, add_355);  mul_648 = add_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_84: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        unsqueeze_436: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_437: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
        unsqueeze_438: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
        sum_63: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 2, 3])
        sub_157: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_438);  convolution_46 = unsqueeze_438 = None
        mul_651: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_650, sub_157)
        sum_64: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 2, 3]);  mul_651 = None
        mul_652: "f32[672]" = torch.ops.aten.mul.Tensor(sum_63, 3.985969387755102e-05)
        unsqueeze_439: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
        unsqueeze_440: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_653: "f32[672]" = torch.ops.aten.mul.Tensor(sum_64, 3.985969387755102e-05)
        squeeze_85: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_654: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_655: "f32[672]" = torch.ops.aten.mul.Tensor(mul_653, mul_654);  mul_653 = mul_654 = None
        unsqueeze_442: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
        unsqueeze_443: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        mul_656: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_210);  primals_210 = None
        unsqueeze_445: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
        unsqueeze_446: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_657: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_444);  sub_157 = unsqueeze_444 = None
        sub_159: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_650, mul_657);  mul_650 = mul_657 = None
        sub_160: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_441);  sub_159 = unsqueeze_441 = None
        mul_658: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_447);  sub_160 = unsqueeze_447 = None
        mul_659: "f32[672]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_85);  sum_64 = squeeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_658, div_27, primals_206, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_658 = div_27 = primals_206 = None
        getitem_200: "f32[128, 672, 14, 14]" = convolution_backward_34[0]
        getitem_201: "f32[672, 1, 5, 5]" = convolution_backward_34[1];  convolution_backward_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_27: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_55)
        mul_198: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        unsqueeze_108: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_204, -1)
        unsqueeze_109: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_204: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_198, unsqueeze_109);  mul_198 = unsqueeze_109 = None
        unsqueeze_110: "f32[672, 1]" = torch.ops.aten.unsqueeze.default(primals_205, -1);  primals_205 = None
        unsqueeze_111: "f32[672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_170: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_204, unsqueeze_111);  mul_204 = unsqueeze_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_27: "f32[128, 672, 14, 14]" = torch.ops.aten.neg.default(add_170)
        exp_27: "f32[128, 672, 14, 14]" = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_171: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        reciprocal_21: "f32[128, 672, 14, 14]" = torch.ops.aten.reciprocal.default(add_171);  add_171 = None
        mul_660: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        mul_661: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_200, mul_660);  getitem_200 = None
        sub_161: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_660);  mul_660 = None
        mul_662: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_170, sub_161);  add_170 = sub_161 = None
        add_357: "f32[128, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_662, 1);  mul_662 = None
        mul_663: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_661, add_357);  mul_661 = add_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_81: "f32[672]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        unsqueeze_448: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(squeeze_81, 0);  squeeze_81 = None
        unsqueeze_449: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
        unsqueeze_450: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
        sum_65: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_663, [0, 2, 3])
        sub_162: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_450);  convolution_45 = unsqueeze_450 = None
        mul_664: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(mul_663, sub_162)
        sum_66: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_664, [0, 2, 3]);  mul_664 = None
        mul_665: "f32[672]" = torch.ops.aten.mul.Tensor(sum_65, 3.985969387755102e-05)
        unsqueeze_451: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
        unsqueeze_452: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        unsqueeze_453: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
        mul_666: "f32[672]" = torch.ops.aten.mul.Tensor(sum_66, 3.985969387755102e-05)
        squeeze_82: "f32[672]" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_667: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_668: "f32[672]" = torch.ops.aten.mul.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
        unsqueeze_454: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
        unsqueeze_455: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
        unsqueeze_456: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
        mul_669: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_204);  primals_204 = None
        unsqueeze_457: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_669, 0);  mul_669 = None
        unsqueeze_458: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_459: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
        mul_670: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_456);  sub_162 = unsqueeze_456 = None
        sub_164: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(mul_663, mul_670);  mul_663 = mul_670 = None
        sub_165: "f32[128, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_453);  sub_164 = unsqueeze_453 = None
        mul_671: "f32[128, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_459);  sub_165 = unsqueeze_459 = None
        mul_672: "f32[672]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_82);  sum_66 = squeeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_671, add_165, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_671 = add_165 = primals_200 = None
        getitem_203: "f32[128, 112, 14, 14]" = convolution_backward_35[0]
        getitem_204: "f32[672, 112, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
        add_358: "f32[128, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_350, getitem_203);  add_350 = getitem_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_67: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_358, [0, 2, 3])
        sub_166: "f32[128, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_462);  convolution_44 = unsqueeze_462 = None
        mul_673: "f32[128, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_358, sub_166)
        sum_68: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_673, [0, 2, 3]);  mul_673 = None
        mul_674: "f32[112]" = torch.ops.aten.mul.Tensor(sum_67, 3.985969387755102e-05)
        unsqueeze_463: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
        unsqueeze_464: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        unsqueeze_465: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        mul_675: "f32[112]" = torch.ops.aten.mul.Tensor(sum_68, 3.985969387755102e-05)
        mul_676: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_677: "f32[112]" = torch.ops.aten.mul.Tensor(mul_675, mul_676);  mul_675 = mul_676 = None
        unsqueeze_466: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
        unsqueeze_467: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
        unsqueeze_468: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
        mul_678: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_198);  primals_198 = None
        unsqueeze_469: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
        unsqueeze_470: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_679: "f32[128, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_468);  sub_166 = unsqueeze_468 = None
        sub_168: "f32[128, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_358, mul_679);  add_358 = mul_679 = None
        sub_169: "f32[128, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_168, unsqueeze_465);  sub_168 = unsqueeze_465 = None
        mul_680: "f32[128, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_471);  sub_169 = unsqueeze_471 = None
        mul_681: "f32[112]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_79);  sum_68 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_680, mul_190, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_680 = mul_190 = primals_194 = None
        getitem_206: "f32[128, 480, 14, 14]" = convolution_backward_36[0]
        getitem_207: "f32[112, 480, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_25: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, getitem_51)
        mul_183: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_100: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_188, -1)
        unsqueeze_101: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_189: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_183, unsqueeze_101);  mul_183 = unsqueeze_101 = None
        unsqueeze_102: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_189, -1);  primals_189 = None
        unsqueeze_103: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_158: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_189, unsqueeze_103);  mul_189 = unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_25: "f32[128, 480, 14, 14]" = torch.ops.aten.neg.default(add_158)
        exp_25: "f32[128, 480, 14, 14]" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_159: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        div_25: "f32[128, 480, 14, 14]" = torch.ops.aten.div.Tensor(add_158, add_159)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_682: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_206, div_25);  div_25 = None
        sigmoid_8: "f32[128, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None
        mul_683: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_206, sigmoid_8);  getitem_206 = None
        sum_69: "f32[128, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_682, [2, 3], True);  mul_682 = None
        sub_170: "f32[128, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_8)
        mul_684: "f32[128, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_8, sub_170);  sigmoid_8 = sub_170 = None
        mul_685: "f32[128, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_684);  sum_69 = mul_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_70: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 2, 3])
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_685, div_26, primals_192, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_685 = div_26 = primals_192 = None
        getitem_209: "f32[128, 20, 1, 1]" = convolution_backward_37[0]
        getitem_210: "f32[480, 20, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_26: "f32[128, 20, 1, 1]" = torch.ops.aten.neg.default(convolution_42)
        exp_26: "f32[128, 20, 1, 1]" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_160: "f32[128, 20, 1, 1]" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        reciprocal_22: "f32[128, 20, 1, 1]" = torch.ops.aten.reciprocal.default(add_160);  add_160 = None
        mul_686: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        mul_687: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_209, mul_686);  getitem_209 = None
        sub_171: "f32[128, 20, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_686);  mul_686 = None
        mul_688: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_42, sub_171);  convolution_42 = sub_171 = None
        add_360: "f32[128, 20, 1, 1]" = torch.ops.aten.add.Tensor(mul_688, 1);  mul_688 = None
        mul_689: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(mul_687, add_360);  mul_687 = add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_71: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3])
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_689, mean_8, primals_190, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_689 = mean_8 = primals_190 = None
        getitem_212: "f32[128, 480, 1, 1]" = convolution_backward_38[0]
        getitem_213: "f32[20, 480, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_9: "f32[128, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_212, [128, 480, 14, 14]);  getitem_212 = None
        div_57: "f32[128, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
        add_361: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_683, div_57);  mul_683 = div_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_23: "f32[128, 480, 14, 14]" = torch.ops.aten.reciprocal.default(add_159);  add_159 = None
        mul_690: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        mul_691: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_361, mul_690);  add_361 = None
        sub_172: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_690);  mul_690 = None
        mul_692: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_158, sub_172);  add_158 = sub_172 = None
        add_363: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_692, 1);  mul_692 = None
        mul_693: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_691, add_363);  mul_691 = add_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_75: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_51, [0, 2, 3]);  getitem_51 = None
        unsqueeze_472: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_473: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
        unsqueeze_474: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
        sum_72: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_693, [0, 2, 3])
        sub_173: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_474);  convolution_41 = unsqueeze_474 = None
        mul_694: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_693, sub_173)
        sum_73: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_694, [0, 2, 3]);  mul_694 = None
        mul_695: "f32[480]" = torch.ops.aten.mul.Tensor(sum_72, 3.985969387755102e-05)
        unsqueeze_475: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
        unsqueeze_476: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_696: "f32[480]" = torch.ops.aten.mul.Tensor(sum_73, 3.985969387755102e-05)
        squeeze_76: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_697: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_698: "f32[480]" = torch.ops.aten.mul.Tensor(mul_696, mul_697);  mul_696 = mul_697 = None
        unsqueeze_478: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
        unsqueeze_479: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
        unsqueeze_480: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
        mul_699: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_188);  primals_188 = None
        unsqueeze_481: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
        unsqueeze_482: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_483: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
        mul_700: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_480);  sub_173 = unsqueeze_480 = None
        sub_175: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_693, mul_700);  mul_693 = mul_700 = None
        sub_176: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_477);  sub_175 = unsqueeze_477 = None
        mul_701: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_483);  sub_176 = unsqueeze_483 = None
        mul_702: "f32[480]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_76);  sum_73 = squeeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_701, div_24, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_701 = div_24 = primals_184 = None
        getitem_215: "f32[128, 480, 14, 14]" = convolution_backward_39[0]
        getitem_216: "f32[480, 1, 5, 5]" = convolution_backward_39[1];  convolution_backward_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_24: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_49)
        mul_176: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        unsqueeze_96: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_182, -1)
        unsqueeze_97: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_182: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_176, unsqueeze_97);  mul_176 = unsqueeze_97 = None
        unsqueeze_98: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_183, -1);  primals_183 = None
        unsqueeze_99: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_152: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_99);  mul_182 = unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_24: "f32[128, 480, 14, 14]" = torch.ops.aten.neg.default(add_152)
        exp_24: "f32[128, 480, 14, 14]" = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_153: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        reciprocal_24: "f32[128, 480, 14, 14]" = torch.ops.aten.reciprocal.default(add_153);  add_153 = None
        mul_703: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        mul_704: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_215, mul_703);  getitem_215 = None
        sub_177: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_703);  mul_703 = None
        mul_705: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_152, sub_177);  add_152 = sub_177 = None
        add_365: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_705, 1);  mul_705 = None
        mul_706: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_704, add_365);  mul_704 = add_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_72: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_49, [0, 2, 3]);  getitem_49 = None
        unsqueeze_484: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_485: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
        unsqueeze_486: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        sum_74: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 2, 3])
        sub_178: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_486);  convolution_40 = unsqueeze_486 = None
        mul_707: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_706, sub_178)
        sum_75: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3]);  mul_707 = None
        mul_708: "f32[480]" = torch.ops.aten.mul.Tensor(sum_74, 3.985969387755102e-05)
        unsqueeze_487: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
        unsqueeze_488: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_709: "f32[480]" = torch.ops.aten.mul.Tensor(sum_75, 3.985969387755102e-05)
        squeeze_73: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_710: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_711: "f32[480]" = torch.ops.aten.mul.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
        unsqueeze_490: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_491: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
        unsqueeze_492: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
        mul_712: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_182);  primals_182 = None
        unsqueeze_493: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
        unsqueeze_494: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_713: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_492);  sub_178 = unsqueeze_492 = None
        sub_180: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_706, mul_713);  mul_706 = mul_713 = None
        sub_181: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_489);  sub_180 = unsqueeze_489 = None
        mul_714: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_495);  sub_181 = unsqueeze_495 = None
        mul_715: "f32[480]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_73);  sum_75 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_714, add_147, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_714 = add_147 = primals_178 = None
        getitem_218: "f32[128, 80, 14, 14]" = convolution_backward_40[0]
        getitem_219: "f32[480, 80, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_76: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_218, [0, 2, 3])
        sub_182: "f32[128, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_498);  convolution_39 = unsqueeze_498 = None
        mul_716: "f32[128, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_218, sub_182)
        sum_77: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 2, 3]);  mul_716 = None
        mul_717: "f32[80]" = torch.ops.aten.mul.Tensor(sum_76, 3.985969387755102e-05)
        unsqueeze_499: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
        unsqueeze_500: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_718: "f32[80]" = torch.ops.aten.mul.Tensor(sum_77, 3.985969387755102e-05)
        mul_719: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_720: "f32[80]" = torch.ops.aten.mul.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
        unsqueeze_502: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
        unsqueeze_503: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
        unsqueeze_504: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
        mul_721: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_176);  primals_176 = None
        unsqueeze_505: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
        unsqueeze_506: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_722: "f32[128, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_504);  sub_182 = unsqueeze_504 = None
        sub_184: "f32[128, 80, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_218, mul_722);  mul_722 = None
        sub_185: "f32[128, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_501);  sub_184 = unsqueeze_501 = None
        mul_723: "f32[128, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_507);  sub_185 = unsqueeze_507 = None
        mul_724: "f32[80]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_70);  sum_77 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_723, mul_168, primals_172, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_723 = mul_168 = primals_172 = None
        getitem_221: "f32[128, 480, 14, 14]" = convolution_backward_41[0]
        getitem_222: "f32[80, 480, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_22: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, getitem_45)
        mul_161: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        unsqueeze_88: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_166, -1)
        unsqueeze_89: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_167: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_161, unsqueeze_89);  mul_161 = unsqueeze_89 = None
        unsqueeze_90: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_167, -1);  primals_167 = None
        unsqueeze_91: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_139: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_91);  mul_167 = unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_22: "f32[128, 480, 14, 14]" = torch.ops.aten.neg.default(add_139)
        exp_22: "f32[128, 480, 14, 14]" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_140: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        div_22: "f32[128, 480, 14, 14]" = torch.ops.aten.div.Tensor(add_139, add_140)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_725: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_221, div_22);  div_22 = None
        sigmoid_7: "f32[128, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_38);  convolution_38 = None
        mul_726: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_221, sigmoid_7);  getitem_221 = None
        sum_78: "f32[128, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_725, [2, 3], True);  mul_725 = None
        sub_186: "f32[128, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_7)
        mul_727: "f32[128, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_7, sub_186);  sigmoid_7 = sub_186 = None
        mul_728: "f32[128, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_78, mul_727);  sum_78 = mul_727 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_79: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_728, [0, 2, 3])
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_728, div_23, primals_170, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_728 = div_23 = primals_170 = None
        getitem_224: "f32[128, 20, 1, 1]" = convolution_backward_42[0]
        getitem_225: "f32[480, 20, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_23: "f32[128, 20, 1, 1]" = torch.ops.aten.neg.default(convolution_37)
        exp_23: "f32[128, 20, 1, 1]" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_141: "f32[128, 20, 1, 1]" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        reciprocal_25: "f32[128, 20, 1, 1]" = torch.ops.aten.reciprocal.default(add_141);  add_141 = None
        mul_729: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        mul_730: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_224, mul_729);  getitem_224 = None
        sub_187: "f32[128, 20, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_729);  mul_729 = None
        mul_731: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_37, sub_187);  convolution_37 = sub_187 = None
        add_367: "f32[128, 20, 1, 1]" = torch.ops.aten.add.Tensor(mul_731, 1);  mul_731 = None
        mul_732: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(mul_730, add_367);  mul_730 = add_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_80: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2, 3])
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_732, mean_7, primals_168, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_732 = mean_7 = primals_168 = None
        getitem_227: "f32[128, 480, 1, 1]" = convolution_backward_43[0]
        getitem_228: "f32[20, 480, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_10: "f32[128, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_227, [128, 480, 14, 14]);  getitem_227 = None
        div_58: "f32[128, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
        add_368: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_726, div_58);  mul_726 = div_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_26: "f32[128, 480, 14, 14]" = torch.ops.aten.reciprocal.default(add_140);  add_140 = None
        mul_733: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        mul_734: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_368, mul_733);  add_368 = None
        sub_188: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_733);  mul_733 = None
        mul_735: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_139, sub_188);  add_139 = sub_188 = None
        add_370: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_735, 1);  mul_735 = None
        mul_736: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_734, add_370);  mul_734 = add_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_66: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_45, [0, 2, 3]);  getitem_45 = None
        unsqueeze_508: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_509: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
        unsqueeze_510: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
        sum_81: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 2, 3])
        sub_189: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_510);  convolution_36 = unsqueeze_510 = None
        mul_737: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_736, sub_189)
        sum_82: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_737, [0, 2, 3]);  mul_737 = None
        mul_738: "f32[480]" = torch.ops.aten.mul.Tensor(sum_81, 3.985969387755102e-05)
        unsqueeze_511: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
        unsqueeze_512: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_739: "f32[480]" = torch.ops.aten.mul.Tensor(sum_82, 3.985969387755102e-05)
        squeeze_67: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_740: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_741: "f32[480]" = torch.ops.aten.mul.Tensor(mul_739, mul_740);  mul_739 = mul_740 = None
        unsqueeze_514: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
        unsqueeze_515: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        mul_742: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_166);  primals_166 = None
        unsqueeze_517: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
        unsqueeze_518: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_743: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_516);  sub_189 = unsqueeze_516 = None
        sub_191: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_736, mul_743);  mul_736 = mul_743 = None
        sub_192: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_513);  sub_191 = unsqueeze_513 = None
        mul_744: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_519);  sub_192 = unsqueeze_519 = None
        mul_745: "f32[480]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_67);  sum_82 = squeeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_744, div_21, primals_162, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_744 = div_21 = primals_162 = None
        getitem_230: "f32[128, 480, 14, 14]" = convolution_backward_44[0]
        getitem_231: "f32[480, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_21: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, getitem_43)
        mul_154: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
        unsqueeze_84: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_160, -1)
        unsqueeze_85: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_160: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_85);  mul_154 = unsqueeze_85 = None
        unsqueeze_86: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_161, -1);  primals_161 = None
        unsqueeze_87: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_133: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_87);  mul_160 = unsqueeze_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_21: "f32[128, 480, 14, 14]" = torch.ops.aten.neg.default(add_133)
        exp_21: "f32[128, 480, 14, 14]" = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_134: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        reciprocal_27: "f32[128, 480, 14, 14]" = torch.ops.aten.reciprocal.default(add_134);  add_134 = None
        mul_746: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        mul_747: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_230, mul_746);  getitem_230 = None
        sub_193: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_746);  mul_746 = None
        mul_748: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_133, sub_193);  add_133 = sub_193 = None
        add_372: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_748, 1);  mul_748 = None
        mul_749: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_747, add_372);  mul_747 = add_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_63: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        unsqueeze_520: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_63, 0);  squeeze_63 = None
        unsqueeze_521: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
        sum_83: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 2, 3])
        sub_194: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_522);  convolution_35 = unsqueeze_522 = None
        mul_750: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_749, sub_194)
        sum_84: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_750, [0, 2, 3]);  mul_750 = None
        mul_751: "f32[480]" = torch.ops.aten.mul.Tensor(sum_83, 3.985969387755102e-05)
        unsqueeze_523: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
        unsqueeze_524: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_752: "f32[480]" = torch.ops.aten.mul.Tensor(sum_84, 3.985969387755102e-05)
        squeeze_64: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_21, [0, 2, 3]);  rsqrt_21 = None
        mul_753: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_754: "f32[480]" = torch.ops.aten.mul.Tensor(mul_752, mul_753);  mul_752 = mul_753 = None
        unsqueeze_526: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
        unsqueeze_527: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
        unsqueeze_528: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
        mul_755: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_160);  primals_160 = None
        unsqueeze_529: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
        unsqueeze_530: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_756: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_528);  sub_194 = unsqueeze_528 = None
        sub_196: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_749, mul_756);  mul_749 = mul_756 = None
        sub_197: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_196, unsqueeze_525);  sub_196 = unsqueeze_525 = None
        mul_757: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_531);  sub_197 = unsqueeze_531 = None
        mul_758: "f32[480]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_64);  sum_84 = squeeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_757, add_128, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_757 = add_128 = primals_156 = None
        getitem_233: "f32[128, 80, 14, 14]" = convolution_backward_45[0]
        getitem_234: "f32[480, 80, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
        add_373: "f32[128, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_218, getitem_233);  getitem_218 = getitem_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_85: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_373, [0, 2, 3])
        sub_198: "f32[128, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_534);  convolution_34 = unsqueeze_534 = None
        mul_759: "f32[128, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_373, sub_198)
        sum_86: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3]);  mul_759 = None
        mul_760: "f32[80]" = torch.ops.aten.mul.Tensor(sum_85, 3.985969387755102e-05)
        unsqueeze_535: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
        unsqueeze_536: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_761: "f32[80]" = torch.ops.aten.mul.Tensor(sum_86, 3.985969387755102e-05)
        mul_762: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_763: "f32[80]" = torch.ops.aten.mul.Tensor(mul_761, mul_762);  mul_761 = mul_762 = None
        unsqueeze_538: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
        unsqueeze_539: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
        unsqueeze_540: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
        mul_764: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_154);  primals_154 = None
        unsqueeze_541: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
        unsqueeze_542: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_765: "f32[128, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_540);  sub_198 = unsqueeze_540 = None
        sub_200: "f32[128, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_373, mul_765);  mul_765 = None
        sub_201: "f32[128, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_200, unsqueeze_537);  sub_200 = unsqueeze_537 = None
        mul_766: "f32[128, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_543);  sub_201 = unsqueeze_543 = None
        mul_767: "f32[80]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_61);  sum_86 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_766, mul_146, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_766 = mul_146 = primals_150 = None
        getitem_236: "f32[128, 480, 14, 14]" = convolution_backward_46[0]
        getitem_237: "f32[80, 480, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_19: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_39)
        mul_139: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        unsqueeze_76: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_144, -1)
        unsqueeze_77: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_145: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_77);  mul_139 = unsqueeze_77 = None
        unsqueeze_78: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_145, -1);  primals_145 = None
        unsqueeze_79: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_120: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_145, unsqueeze_79);  mul_145 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_19: "f32[128, 480, 14, 14]" = torch.ops.aten.neg.default(add_120)
        exp_19: "f32[128, 480, 14, 14]" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_121: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        div_19: "f32[128, 480, 14, 14]" = torch.ops.aten.div.Tensor(add_120, add_121)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_768: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_236, div_19);  div_19 = None
        sigmoid_6: "f32[128, 480, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_33);  convolution_33 = None
        mul_769: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_236, sigmoid_6);  getitem_236 = None
        sum_87: "f32[128, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_768, [2, 3], True);  mul_768 = None
        sub_202: "f32[128, 480, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_6)
        mul_770: "f32[128, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_6, sub_202);  sigmoid_6 = sub_202 = None
        mul_771: "f32[128, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_87, mul_770);  sum_87 = mul_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_88: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3])
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_771, div_20, primals_148, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_771 = div_20 = primals_148 = None
        getitem_239: "f32[128, 20, 1, 1]" = convolution_backward_47[0]
        getitem_240: "f32[480, 20, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_20: "f32[128, 20, 1, 1]" = torch.ops.aten.neg.default(convolution_32)
        exp_20: "f32[128, 20, 1, 1]" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_122: "f32[128, 20, 1, 1]" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        reciprocal_28: "f32[128, 20, 1, 1]" = torch.ops.aten.reciprocal.default(add_122);  add_122 = None
        mul_772: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        mul_773: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_239, mul_772);  getitem_239 = None
        sub_203: "f32[128, 20, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_772);  mul_772 = None
        mul_774: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_32, sub_203);  convolution_32 = sub_203 = None
        add_375: "f32[128, 20, 1, 1]" = torch.ops.aten.add.Tensor(mul_774, 1);  mul_774 = None
        mul_775: "f32[128, 20, 1, 1]" = torch.ops.aten.mul.Tensor(mul_773, add_375);  mul_773 = add_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_89: "f32[20]" = torch.ops.aten.sum.dim_IntList(mul_775, [0, 2, 3])
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_775, mean_6, primals_146, [20], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_775 = mean_6 = primals_146 = None
        getitem_242: "f32[128, 480, 1, 1]" = convolution_backward_48[0]
        getitem_243: "f32[20, 480, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_11: "f32[128, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_242, [128, 480, 14, 14]);  getitem_242 = None
        div_59: "f32[128, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
        add_376: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_769, div_59);  mul_769 = div_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_29: "f32[128, 480, 14, 14]" = torch.ops.aten.reciprocal.default(add_121);  add_121 = None
        mul_776: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        mul_777: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_376, mul_776);  add_376 = None
        sub_204: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_776);  mul_776 = None
        mul_778: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_120, sub_204);  add_120 = sub_204 = None
        add_378: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_778, 1);  mul_778 = None
        mul_779: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_777, add_378);  mul_777 = add_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_57: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_39, [0, 2, 3]);  getitem_39 = None
        unsqueeze_544: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_545: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
        unsqueeze_546: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
        sum_90: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_779, [0, 2, 3])
        sub_205: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_546);  convolution_31 = unsqueeze_546 = None
        mul_780: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_779, sub_205)
        sum_91: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 2, 3]);  mul_780 = None
        mul_781: "f32[480]" = torch.ops.aten.mul.Tensor(sum_90, 3.985969387755102e-05)
        unsqueeze_547: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
        unsqueeze_548: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_782: "f32[480]" = torch.ops.aten.mul.Tensor(sum_91, 3.985969387755102e-05)
        squeeze_58: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_783: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_784: "f32[480]" = torch.ops.aten.mul.Tensor(mul_782, mul_783);  mul_782 = mul_783 = None
        unsqueeze_550: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
        unsqueeze_551: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        mul_785: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_144);  primals_144 = None
        unsqueeze_553: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
        unsqueeze_554: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_786: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_552);  sub_205 = unsqueeze_552 = None
        sub_207: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_779, mul_786);  mul_779 = mul_786 = None
        sub_208: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_549);  sub_207 = unsqueeze_549 = None
        mul_787: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_555);  sub_208 = unsqueeze_555 = None
        mul_788: "f32[480]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_58);  sum_91 = squeeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_787, div_18, primals_140, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_787 = div_18 = primals_140 = None
        getitem_245: "f32[128, 480, 14, 14]" = convolution_backward_49[0]
        getitem_246: "f32[480, 1, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_18: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, getitem_37)
        mul_132: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        unsqueeze_72: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_73: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_138: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_132, unsqueeze_73);  mul_132 = unsqueeze_73 = None
        unsqueeze_74: "f32[480, 1]" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_75: "f32[480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_114: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_138, unsqueeze_75);  mul_138 = unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_18: "f32[128, 480, 14, 14]" = torch.ops.aten.neg.default(add_114)
        exp_18: "f32[128, 480, 14, 14]" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_115: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        reciprocal_30: "f32[128, 480, 14, 14]" = torch.ops.aten.reciprocal.default(add_115);  add_115 = None
        mul_789: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        mul_790: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_245, mul_789);  getitem_245 = None
        sub_209: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_789);  mul_789 = None
        mul_791: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_114, sub_209);  add_114 = sub_209 = None
        add_380: "f32[128, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_791, 1);  mul_791 = None
        mul_792: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_790, add_380);  mul_790 = add_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_54: "f32[480]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        unsqueeze_556: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_557: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
        unsqueeze_558: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
        sum_92: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_792, [0, 2, 3])
        sub_210: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_558);  convolution_30 = unsqueeze_558 = None
        mul_793: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(mul_792, sub_210)
        sum_93: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 2, 3]);  mul_793 = None
        mul_794: "f32[480]" = torch.ops.aten.mul.Tensor(sum_92, 3.985969387755102e-05)
        unsqueeze_559: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
        unsqueeze_560: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_795: "f32[480]" = torch.ops.aten.mul.Tensor(sum_93, 3.985969387755102e-05)
        squeeze_55: "f32[480]" = torch.ops.aten.squeeze.dims(rsqrt_18, [0, 2, 3]);  rsqrt_18 = None
        mul_796: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_797: "f32[480]" = torch.ops.aten.mul.Tensor(mul_795, mul_796);  mul_795 = mul_796 = None
        unsqueeze_562: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
        unsqueeze_563: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
        unsqueeze_564: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
        mul_798: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_138);  primals_138 = None
        unsqueeze_565: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
        unsqueeze_566: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_799: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_210, unsqueeze_564);  sub_210 = unsqueeze_564 = None
        sub_212: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(mul_792, mul_799);  mul_792 = mul_799 = None
        sub_213: "f32[128, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_212, unsqueeze_561);  sub_212 = unsqueeze_561 = None
        mul_800: "f32[128, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_567);  sub_213 = unsqueeze_567 = None
        mul_801: "f32[480]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_55);  sum_93 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_800, add_109, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_800 = add_109 = primals_134 = None
        getitem_248: "f32[128, 80, 14, 14]" = convolution_backward_50[0]
        getitem_249: "f32[480, 80, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
        add_381: "f32[128, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_373, getitem_248);  add_373 = getitem_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_94: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_381, [0, 2, 3])
        sub_214: "f32[128, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_570);  convolution_29 = unsqueeze_570 = None
        mul_802: "f32[128, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_381, sub_214)
        sum_95: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 2, 3]);  mul_802 = None
        mul_803: "f32[80]" = torch.ops.aten.mul.Tensor(sum_94, 3.985969387755102e-05)
        unsqueeze_571: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
        unsqueeze_572: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_804: "f32[80]" = torch.ops.aten.mul.Tensor(sum_95, 3.985969387755102e-05)
        mul_805: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_806: "f32[80]" = torch.ops.aten.mul.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
        unsqueeze_574: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_575: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
        unsqueeze_576: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
        mul_807: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_132);  primals_132 = None
        unsqueeze_577: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_578: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_808: "f32[128, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_214, unsqueeze_576);  sub_214 = unsqueeze_576 = None
        sub_216: "f32[128, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_381, mul_808);  add_381 = mul_808 = None
        sub_217: "f32[128, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_216, unsqueeze_573);  sub_216 = unsqueeze_573 = None
        mul_809: "f32[128, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_579);  sub_217 = unsqueeze_579 = None
        mul_810: "f32[80]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_52);  sum_95 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_809, mul_124, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_809 = mul_124 = primals_128 = None
        getitem_251: "f32[128, 240, 14, 14]" = convolution_backward_51[0]
        getitem_252: "f32[80, 240, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_16: "f32[128, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_33)
        mul_117: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        unsqueeze_64: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_122, -1)
        unsqueeze_65: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_123: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_117, unsqueeze_65);  mul_117 = unsqueeze_65 = None
        unsqueeze_66: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_123, -1);  primals_123 = None
        unsqueeze_67: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_102: "f32[128, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_123, unsqueeze_67);  mul_123 = unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_16: "f32[128, 240, 14, 14]" = torch.ops.aten.neg.default(add_102)
        exp_16: "f32[128, 240, 14, 14]" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_103: "f32[128, 240, 14, 14]" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        div_16: "f32[128, 240, 14, 14]" = torch.ops.aten.div.Tensor(add_102, add_103)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_811: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_251, div_16);  div_16 = None
        sigmoid_5: "f32[128, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_28);  convolution_28 = None
        mul_812: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_251, sigmoid_5);  getitem_251 = None
        sum_96: "f32[128, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_811, [2, 3], True);  mul_811 = None
        sub_218: "f32[128, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_5)
        mul_813: "f32[128, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_5, sub_218);  sigmoid_5 = sub_218 = None
        mul_814: "f32[128, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_96, mul_813);  sum_96 = mul_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_97: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3])
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_814, div_17, primals_126, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_814 = div_17 = primals_126 = None
        getitem_254: "f32[128, 10, 1, 1]" = convolution_backward_52[0]
        getitem_255: "f32[240, 10, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_17: "f32[128, 10, 1, 1]" = torch.ops.aten.neg.default(convolution_27)
        exp_17: "f32[128, 10, 1, 1]" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_104: "f32[128, 10, 1, 1]" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        reciprocal_31: "f32[128, 10, 1, 1]" = torch.ops.aten.reciprocal.default(add_104);  add_104 = None
        mul_815: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        mul_816: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_254, mul_815);  getitem_254 = None
        sub_219: "f32[128, 10, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_815);  mul_815 = None
        mul_817: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_27, sub_219);  convolution_27 = sub_219 = None
        add_383: "f32[128, 10, 1, 1]" = torch.ops.aten.add.Tensor(mul_817, 1);  mul_817 = None
        mul_818: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(mul_816, add_383);  mul_816 = add_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_98: "f32[10]" = torch.ops.aten.sum.dim_IntList(mul_818, [0, 2, 3])
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_818, mean_5, primals_124, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_818 = mean_5 = primals_124 = None
        getitem_257: "f32[128, 240, 1, 1]" = convolution_backward_53[0]
        getitem_258: "f32[10, 240, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_12: "f32[128, 240, 14, 14]" = torch.ops.aten.expand.default(getitem_257, [128, 240, 14, 14]);  getitem_257 = None
        div_60: "f32[128, 240, 14, 14]" = torch.ops.aten.div.Scalar(expand_12, 196);  expand_12 = None
        add_384: "f32[128, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_812, div_60);  mul_812 = div_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_32: "f32[128, 240, 14, 14]" = torch.ops.aten.reciprocal.default(add_103);  add_103 = None
        mul_819: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        mul_820: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_384, mul_819);  add_384 = None
        sub_220: "f32[128, 240, 14, 14]" = torch.ops.aten.sub.Tensor(1, mul_819);  mul_819 = None
        mul_821: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(add_102, sub_220);  add_102 = sub_220 = None
        add_386: "f32[128, 240, 14, 14]" = torch.ops.aten.add.Tensor(mul_821, 1);  mul_821 = None
        mul_822: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_820, add_386);  mul_820 = add_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_48: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        unsqueeze_580: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_581: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
        unsqueeze_582: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
        sum_99: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_822, [0, 2, 3])
        sub_221: "f32[128, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_582);  convolution_26 = unsqueeze_582 = None
        mul_823: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(mul_822, sub_221)
        sum_100: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 2, 3]);  mul_823 = None
        mul_824: "f32[240]" = torch.ops.aten.mul.Tensor(sum_99, 3.985969387755102e-05)
        unsqueeze_583: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
        unsqueeze_584: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_825: "f32[240]" = torch.ops.aten.mul.Tensor(sum_100, 3.985969387755102e-05)
        squeeze_49: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_826: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_827: "f32[240]" = torch.ops.aten.mul.Tensor(mul_825, mul_826);  mul_825 = mul_826 = None
        unsqueeze_586: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
        unsqueeze_587: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        mul_828: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_122);  primals_122 = None
        unsqueeze_589: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
        unsqueeze_590: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_829: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_588);  sub_221 = unsqueeze_588 = None
        sub_223: "f32[128, 240, 14, 14]" = torch.ops.aten.sub.Tensor(mul_822, mul_829);  mul_822 = mul_829 = None
        sub_224: "f32[128, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_585);  sub_223 = unsqueeze_585 = None
        mul_830: "f32[128, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_591);  sub_224 = unsqueeze_591 = None
        mul_831: "f32[240]" = torch.ops.aten.mul.Tensor(sum_100, squeeze_49);  sum_100 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_830, constant_pad_nd_3, primals_118, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_830 = constant_pad_nd_3 = primals_118 = None
        getitem_260: "f32[128, 240, 29, 29]" = convolution_backward_54[0]
        getitem_261: "f32[240, 1, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_6: "f32[128, 240, 28, 28]" = torch.ops.aten.constant_pad_nd.default(getitem_260, [0, -1, 0, -1]);  getitem_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_15: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_31)
        mul_110: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        unsqueeze_60: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_116, -1)
        unsqueeze_61: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_116: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_110, unsqueeze_61);  mul_110 = unsqueeze_61 = None
        unsqueeze_62: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_117, -1);  primals_117 = None
        unsqueeze_63: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_96: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_63);  mul_116 = unsqueeze_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_15: "f32[128, 240, 28, 28]" = torch.ops.aten.neg.default(add_96)
        exp_15: "f32[128, 240, 28, 28]" = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_97: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        reciprocal_33: "f32[128, 240, 28, 28]" = torch.ops.aten.reciprocal.default(add_97);  add_97 = None
        mul_832: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        mul_833: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(constant_pad_nd_6, mul_832);  constant_pad_nd_6 = None
        sub_225: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(1, mul_832);  mul_832 = None
        mul_834: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_96, sub_225);  add_96 = sub_225 = None
        add_388: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_834, 1);  mul_834 = None
        mul_835: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_833, add_388);  mul_833 = add_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_45: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        unsqueeze_592: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_45, 0);  squeeze_45 = None
        unsqueeze_593: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
        unsqueeze_594: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
        sum_101: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_835, [0, 2, 3])
        sub_226: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_594);  convolution_25 = unsqueeze_594 = None
        mul_836: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_835, sub_226)
        sum_102: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 2, 3]);  mul_836 = None
        mul_837: "f32[240]" = torch.ops.aten.mul.Tensor(sum_101, 9.964923469387754e-06)
        unsqueeze_595: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
        unsqueeze_596: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_838: "f32[240]" = torch.ops.aten.mul.Tensor(sum_102, 9.964923469387754e-06)
        squeeze_46: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_15, [0, 2, 3]);  rsqrt_15 = None
        mul_839: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_840: "f32[240]" = torch.ops.aten.mul.Tensor(mul_838, mul_839);  mul_838 = mul_839 = None
        unsqueeze_598: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
        unsqueeze_599: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
        unsqueeze_600: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
        mul_841: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_116);  primals_116 = None
        unsqueeze_601: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_841, 0);  mul_841 = None
        unsqueeze_602: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_842: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_226, unsqueeze_600);  sub_226 = unsqueeze_600 = None
        sub_228: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(mul_835, mul_842);  mul_835 = mul_842 = None
        sub_229: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_228, unsqueeze_597);  sub_228 = unsqueeze_597 = None
        mul_843: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_603);  sub_229 = unsqueeze_603 = None
        mul_844: "f32[240]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_46);  sum_102 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_843, add_91, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_843 = add_91 = primals_112 = None
        getitem_263: "f32[128, 40, 28, 28]" = convolution_backward_55[0]
        getitem_264: "f32[240, 40, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_103: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_263, [0, 2, 3])
        sub_230: "f32[128, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_606);  convolution_24 = unsqueeze_606 = None
        mul_845: "f32[128, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_263, sub_230)
        sum_104: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_845, [0, 2, 3]);  mul_845 = None
        mul_846: "f32[40]" = torch.ops.aten.mul.Tensor(sum_103, 9.964923469387754e-06)
        unsqueeze_607: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
        unsqueeze_608: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_847: "f32[40]" = torch.ops.aten.mul.Tensor(sum_104, 9.964923469387754e-06)
        mul_848: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_849: "f32[40]" = torch.ops.aten.mul.Tensor(mul_847, mul_848);  mul_847 = mul_848 = None
        unsqueeze_610: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_849, 0);  mul_849 = None
        unsqueeze_611: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
        unsqueeze_612: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
        mul_850: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_110);  primals_110 = None
        unsqueeze_613: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_850, 0);  mul_850 = None
        unsqueeze_614: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_851: "f32[128, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_612);  sub_230 = unsqueeze_612 = None
        sub_232: "f32[128, 40, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_263, mul_851);  mul_851 = None
        sub_233: "f32[128, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_232, unsqueeze_609);  sub_232 = unsqueeze_609 = None
        mul_852: "f32[128, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_615);  sub_233 = unsqueeze_615 = None
        mul_853: "f32[40]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_43);  sum_104 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_852, mul_102, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_852 = mul_102 = primals_106 = None
        getitem_266: "f32[128, 240, 28, 28]" = convolution_backward_56[0]
        getitem_267: "f32[40, 240, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_13: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, getitem_27)
        mul_95: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        unsqueeze_52: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_100, -1)
        unsqueeze_53: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_101: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_95, unsqueeze_53);  mul_95 = unsqueeze_53 = None
        unsqueeze_54: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_101, -1);  primals_101 = None
        unsqueeze_55: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_83: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_55);  mul_101 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_13: "f32[128, 240, 28, 28]" = torch.ops.aten.neg.default(add_83)
        exp_13: "f32[128, 240, 28, 28]" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_84: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        div_13: "f32[128, 240, 28, 28]" = torch.ops.aten.div.Tensor(add_83, add_84)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_854: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_266, div_13);  div_13 = None
        sigmoid_4: "f32[128, 240, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_23);  convolution_23 = None
        mul_855: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_266, sigmoid_4);  getitem_266 = None
        sum_105: "f32[128, 240, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_854, [2, 3], True);  mul_854 = None
        sub_234: "f32[128, 240, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_4)
        mul_856: "f32[128, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_4, sub_234);  sigmoid_4 = sub_234 = None
        mul_857: "f32[128, 240, 1, 1]" = torch.ops.aten.mul.Tensor(sum_105, mul_856);  sum_105 = mul_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_106: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_857, [0, 2, 3])
        convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_857, div_14, primals_104, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_857 = div_14 = primals_104 = None
        getitem_269: "f32[128, 10, 1, 1]" = convolution_backward_57[0]
        getitem_270: "f32[240, 10, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_14: "f32[128, 10, 1, 1]" = torch.ops.aten.neg.default(convolution_22)
        exp_14: "f32[128, 10, 1, 1]" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_85: "f32[128, 10, 1, 1]" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        reciprocal_34: "f32[128, 10, 1, 1]" = torch.ops.aten.reciprocal.default(add_85);  add_85 = None
        mul_858: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        mul_859: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_269, mul_858);  getitem_269 = None
        sub_235: "f32[128, 10, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_858);  mul_858 = None
        mul_860: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_22, sub_235);  convolution_22 = sub_235 = None
        add_390: "f32[128, 10, 1, 1]" = torch.ops.aten.add.Tensor(mul_860, 1);  mul_860 = None
        mul_861: "f32[128, 10, 1, 1]" = torch.ops.aten.mul.Tensor(mul_859, add_390);  mul_859 = add_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_107: "f32[10]" = torch.ops.aten.sum.dim_IntList(mul_861, [0, 2, 3])
        convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_861, mean_4, primals_102, [10], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_861 = mean_4 = primals_102 = None
        getitem_272: "f32[128, 240, 1, 1]" = convolution_backward_58[0]
        getitem_273: "f32[10, 240, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_13: "f32[128, 240, 28, 28]" = torch.ops.aten.expand.default(getitem_272, [128, 240, 28, 28]);  getitem_272 = None
        div_61: "f32[128, 240, 28, 28]" = torch.ops.aten.div.Scalar(expand_13, 784);  expand_13 = None
        add_391: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_855, div_61);  mul_855 = div_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_35: "f32[128, 240, 28, 28]" = torch.ops.aten.reciprocal.default(add_84);  add_84 = None
        mul_862: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        mul_863: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_391, mul_862);  add_391 = None
        sub_236: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(1, mul_862);  mul_862 = None
        mul_864: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_83, sub_236);  add_83 = sub_236 = None
        add_393: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_864, 1);  mul_864 = None
        mul_865: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_863, add_393);  mul_863 = add_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_39: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_616: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_617: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
        unsqueeze_618: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
        sum_108: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_865, [0, 2, 3])
        sub_237: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_618);  convolution_21 = unsqueeze_618 = None
        mul_866: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_865, sub_237)
        sum_109: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_866, [0, 2, 3]);  mul_866 = None
        mul_867: "f32[240]" = torch.ops.aten.mul.Tensor(sum_108, 9.964923469387754e-06)
        unsqueeze_619: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_867, 0);  mul_867 = None
        unsqueeze_620: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_868: "f32[240]" = torch.ops.aten.mul.Tensor(sum_109, 9.964923469387754e-06)
        squeeze_40: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_869: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_870: "f32[240]" = torch.ops.aten.mul.Tensor(mul_868, mul_869);  mul_868 = mul_869 = None
        unsqueeze_622: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
        unsqueeze_623: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        mul_871: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_100);  primals_100 = None
        unsqueeze_625: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
        unsqueeze_626: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_872: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_624);  sub_237 = unsqueeze_624 = None
        sub_239: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(mul_865, mul_872);  mul_865 = mul_872 = None
        sub_240: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_621);  sub_239 = unsqueeze_621 = None
        mul_873: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_627);  sub_240 = unsqueeze_627 = None
        mul_874: "f32[240]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_40);  sum_109 = squeeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_873, div_12, primals_96, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_873 = div_12 = primals_96 = None
        getitem_275: "f32[128, 240, 28, 28]" = convolution_backward_59[0]
        getitem_276: "f32[240, 1, 5, 5]" = convolution_backward_59[1];  convolution_backward_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_12: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, getitem_25)
        mul_88: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        unsqueeze_48: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_94, -1)
        unsqueeze_49: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        mul_94: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_49);  mul_88 = unsqueeze_49 = None
        unsqueeze_50: "f32[240, 1]" = torch.ops.aten.unsqueeze.default(primals_95, -1);  primals_95 = None
        unsqueeze_51: "f32[240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        add_77: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_94, unsqueeze_51);  mul_94 = unsqueeze_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_12: "f32[128, 240, 28, 28]" = torch.ops.aten.neg.default(add_77)
        exp_12: "f32[128, 240, 28, 28]" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_78: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        reciprocal_36: "f32[128, 240, 28, 28]" = torch.ops.aten.reciprocal.default(add_78);  add_78 = None
        mul_875: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        mul_876: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_275, mul_875);  getitem_275 = None
        sub_241: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(1, mul_875);  mul_875 = None
        mul_877: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(add_77, sub_241);  add_77 = sub_241 = None
        add_395: "f32[128, 240, 28, 28]" = torch.ops.aten.add.Tensor(mul_877, 1);  mul_877 = None
        mul_878: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_876, add_395);  mul_876 = add_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_36: "f32[240]" = torch.ops.aten.squeeze.dims(getitem_25, [0, 2, 3]);  getitem_25 = None
        unsqueeze_628: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(squeeze_36, 0);  squeeze_36 = None
        unsqueeze_629: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
        unsqueeze_630: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
        sum_110: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 2, 3])
        sub_242: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_630);  convolution_20 = unsqueeze_630 = None
        mul_879: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(mul_878, sub_242)
        sum_111: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_879, [0, 2, 3]);  mul_879 = None
        mul_880: "f32[240]" = torch.ops.aten.mul.Tensor(sum_110, 9.964923469387754e-06)
        unsqueeze_631: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
        unsqueeze_632: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_881: "f32[240]" = torch.ops.aten.mul.Tensor(sum_111, 9.964923469387754e-06)
        squeeze_37: "f32[240]" = torch.ops.aten.squeeze.dims(rsqrt_12, [0, 2, 3]);  rsqrt_12 = None
        mul_882: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_883: "f32[240]" = torch.ops.aten.mul.Tensor(mul_881, mul_882);  mul_881 = mul_882 = None
        unsqueeze_634: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
        unsqueeze_635: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
        unsqueeze_636: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
        mul_884: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_94);  primals_94 = None
        unsqueeze_637: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
        unsqueeze_638: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_885: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_242, unsqueeze_636);  sub_242 = unsqueeze_636 = None
        sub_244: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(mul_878, mul_885);  mul_878 = mul_885 = None
        sub_245: "f32[128, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_244, unsqueeze_633);  sub_244 = unsqueeze_633 = None
        mul_886: "f32[128, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_639);  sub_245 = unsqueeze_639 = None
        mul_887: "f32[240]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_37);  sum_111 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_886, add_72, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_886 = add_72 = primals_90 = None
        getitem_278: "f32[128, 40, 28, 28]" = convolution_backward_60[0]
        getitem_279: "f32[240, 40, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
        add_396: "f32[128, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_263, getitem_278);  getitem_263 = getitem_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_112: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_396, [0, 2, 3])
        sub_246: "f32[128, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_642);  convolution_19 = unsqueeze_642 = None
        mul_888: "f32[128, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_396, sub_246)
        sum_113: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_888, [0, 2, 3]);  mul_888 = None
        mul_889: "f32[40]" = torch.ops.aten.mul.Tensor(sum_112, 9.964923469387754e-06)
        unsqueeze_643: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_889, 0);  mul_889 = None
        unsqueeze_644: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_890: "f32[40]" = torch.ops.aten.mul.Tensor(sum_113, 9.964923469387754e-06)
        mul_891: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_892: "f32[40]" = torch.ops.aten.mul.Tensor(mul_890, mul_891);  mul_890 = mul_891 = None
        unsqueeze_646: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
        unsqueeze_647: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
        unsqueeze_648: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
        mul_893: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_88);  primals_88 = None
        unsqueeze_649: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
        unsqueeze_650: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_894: "f32[128, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_246, unsqueeze_648);  sub_246 = unsqueeze_648 = None
        sub_248: "f32[128, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_396, mul_894);  add_396 = mul_894 = None
        sub_249: "f32[128, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_248, unsqueeze_645);  sub_248 = unsqueeze_645 = None
        mul_895: "f32[128, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_651);  sub_249 = unsqueeze_651 = None
        mul_896: "f32[40]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_34);  sum_113 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_895, mul_80, primals_84, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_895 = mul_80 = primals_84 = None
        getitem_281: "f32[128, 144, 28, 28]" = convolution_backward_61[0]
        getitem_282: "f32[40, 144, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_10: "f32[128, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_21)
        mul_73: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        unsqueeze_40: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_78, -1)
        unsqueeze_41: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_79: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_41);  mul_73 = unsqueeze_41 = None
        unsqueeze_42: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_79, -1);  primals_79 = None
        unsqueeze_43: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_65: "f32[128, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_79, unsqueeze_43);  mul_79 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_10: "f32[128, 144, 28, 28]" = torch.ops.aten.neg.default(add_65)
        exp_10: "f32[128, 144, 28, 28]" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_66: "f32[128, 144, 28, 28]" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        div_10: "f32[128, 144, 28, 28]" = torch.ops.aten.div.Tensor(add_65, add_66)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_897: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_281, div_10);  div_10 = None
        sigmoid_3: "f32[128, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_18);  convolution_18 = None
        mul_898: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_281, sigmoid_3);  getitem_281 = None
        sum_114: "f32[128, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_897, [2, 3], True);  mul_897 = None
        sub_250: "f32[128, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_3)
        mul_899: "f32[128, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_3, sub_250);  sigmoid_3 = sub_250 = None
        mul_900: "f32[128, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_114, mul_899);  sum_114 = mul_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_115: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_900, [0, 2, 3])
        convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_900, div_11, primals_82, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_900 = div_11 = primals_82 = None
        getitem_284: "f32[128, 6, 1, 1]" = convolution_backward_62[0]
        getitem_285: "f32[144, 6, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_11: "f32[128, 6, 1, 1]" = torch.ops.aten.neg.default(convolution_17)
        exp_11: "f32[128, 6, 1, 1]" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_67: "f32[128, 6, 1, 1]" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        reciprocal_37: "f32[128, 6, 1, 1]" = torch.ops.aten.reciprocal.default(add_67);  add_67 = None
        mul_901: "f32[128, 6, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        mul_902: "f32[128, 6, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_284, mul_901);  getitem_284 = None
        sub_251: "f32[128, 6, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_901);  mul_901 = None
        mul_903: "f32[128, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_17, sub_251);  convolution_17 = sub_251 = None
        add_398: "f32[128, 6, 1, 1]" = torch.ops.aten.add.Tensor(mul_903, 1);  mul_903 = None
        mul_904: "f32[128, 6, 1, 1]" = torch.ops.aten.mul.Tensor(mul_902, add_398);  mul_902 = add_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_116: "f32[6]" = torch.ops.aten.sum.dim_IntList(mul_904, [0, 2, 3])
        convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_904, mean_3, primals_80, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_904 = mean_3 = primals_80 = None
        getitem_287: "f32[128, 144, 1, 1]" = convolution_backward_63[0]
        getitem_288: "f32[6, 144, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_14: "f32[128, 144, 28, 28]" = torch.ops.aten.expand.default(getitem_287, [128, 144, 28, 28]);  getitem_287 = None
        div_62: "f32[128, 144, 28, 28]" = torch.ops.aten.div.Scalar(expand_14, 784);  expand_14 = None
        add_399: "f32[128, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_898, div_62);  mul_898 = div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_38: "f32[128, 144, 28, 28]" = torch.ops.aten.reciprocal.default(add_66);  add_66 = None
        mul_905: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        mul_906: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_399, mul_905);  add_399 = None
        sub_252: "f32[128, 144, 28, 28]" = torch.ops.aten.sub.Tensor(1, mul_905);  mul_905 = None
        mul_907: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(add_65, sub_252);  add_65 = sub_252 = None
        add_401: "f32[128, 144, 28, 28]" = torch.ops.aten.add.Tensor(mul_907, 1);  mul_907 = None
        mul_908: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_906, add_401);  mul_906 = add_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_30: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_652: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_653: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
        unsqueeze_654: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
        sum_117: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_908, [0, 2, 3])
        sub_253: "f32[128, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_654);  convolution_16 = unsqueeze_654 = None
        mul_909: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(mul_908, sub_253)
        sum_118: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_909, [0, 2, 3]);  mul_909 = None
        mul_910: "f32[144]" = torch.ops.aten.mul.Tensor(sum_117, 9.964923469387754e-06)
        unsqueeze_655: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_910, 0);  mul_910 = None
        unsqueeze_656: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_911: "f32[144]" = torch.ops.aten.mul.Tensor(sum_118, 9.964923469387754e-06)
        squeeze_31: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_912: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_913: "f32[144]" = torch.ops.aten.mul.Tensor(mul_911, mul_912);  mul_911 = mul_912 = None
        unsqueeze_658: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_913, 0);  mul_913 = None
        unsqueeze_659: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        mul_914: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_78);  primals_78 = None
        unsqueeze_661: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
        unsqueeze_662: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_915: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_660);  sub_253 = unsqueeze_660 = None
        sub_255: "f32[128, 144, 28, 28]" = torch.ops.aten.sub.Tensor(mul_908, mul_915);  mul_908 = mul_915 = None
        sub_256: "f32[128, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_657);  sub_255 = unsqueeze_657 = None
        mul_916: "f32[128, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_663);  sub_256 = unsqueeze_663 = None
        mul_917: "f32[144]" = torch.ops.aten.mul.Tensor(sum_118, squeeze_31);  sum_118 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_916, constant_pad_nd_2, primals_74, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_916 = constant_pad_nd_2 = primals_74 = None
        getitem_290: "f32[128, 144, 59, 59]" = convolution_backward_64[0]
        getitem_291: "f32[144, 1, 5, 5]" = convolution_backward_64[1];  convolution_backward_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_7: "f32[128, 144, 56, 56]" = torch.ops.aten.constant_pad_nd.default(getitem_290, [-1, -2, -1, -2]);  getitem_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_9: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, getitem_19)
        mul_66: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        unsqueeze_36: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_37: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_72: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_66, unsqueeze_37);  mul_66 = unsqueeze_37 = None
        unsqueeze_38: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_39: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_59: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_72, unsqueeze_39);  mul_72 = unsqueeze_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_9: "f32[128, 144, 56, 56]" = torch.ops.aten.neg.default(add_59)
        exp_9: "f32[128, 144, 56, 56]" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_60: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        reciprocal_39: "f32[128, 144, 56, 56]" = torch.ops.aten.reciprocal.default(add_60);  add_60 = None
        mul_918: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        mul_919: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(constant_pad_nd_7, mul_918);  constant_pad_nd_7 = None
        sub_257: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(1, mul_918);  mul_918 = None
        mul_920: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_59, sub_257);  add_59 = sub_257 = None
        add_403: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_920, 1);  mul_920 = None
        mul_921: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_919, add_403);  mul_919 = add_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_27: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_19, [0, 2, 3]);  getitem_19 = None
        unsqueeze_664: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_27, 0);  squeeze_27 = None
        unsqueeze_665: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
        unsqueeze_666: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
        sum_119: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_921, [0, 2, 3])
        sub_258: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_666);  convolution_15 = unsqueeze_666 = None
        mul_922: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_921, sub_258)
        sum_120: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_922, [0, 2, 3]);  mul_922 = None
        mul_923: "f32[144]" = torch.ops.aten.mul.Tensor(sum_119, 2.4912308673469386e-06)
        unsqueeze_667: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_923, 0);  mul_923 = None
        unsqueeze_668: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_924: "f32[144]" = torch.ops.aten.mul.Tensor(sum_120, 2.4912308673469386e-06)
        squeeze_28: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_9, [0, 2, 3]);  rsqrt_9 = None
        mul_925: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_926: "f32[144]" = torch.ops.aten.mul.Tensor(mul_924, mul_925);  mul_924 = mul_925 = None
        unsqueeze_670: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_926, 0);  mul_926 = None
        unsqueeze_671: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
        unsqueeze_672: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
        mul_927: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_72);  primals_72 = None
        unsqueeze_673: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
        unsqueeze_674: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_928: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_672);  sub_258 = unsqueeze_672 = None
        sub_260: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(mul_921, mul_928);  mul_921 = mul_928 = None
        sub_261: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(sub_260, unsqueeze_669);  sub_260 = unsqueeze_669 = None
        mul_929: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_675);  sub_261 = unsqueeze_675 = None
        mul_930: "f32[144]" = torch.ops.aten.mul.Tensor(sum_120, squeeze_28);  sum_120 = squeeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_929, add_54, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_929 = add_54 = primals_68 = None
        getitem_293: "f32[128, 24, 56, 56]" = convolution_backward_65[0]
        getitem_294: "f32[144, 24, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_121: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_293, [0, 2, 3])
        sub_262: "f32[128, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_678);  convolution_14 = unsqueeze_678 = None
        mul_931: "f32[128, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_293, sub_262)
        sum_122: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_931, [0, 2, 3]);  mul_931 = None
        mul_932: "f32[24]" = torch.ops.aten.mul.Tensor(sum_121, 2.4912308673469386e-06)
        unsqueeze_679: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_932, 0);  mul_932 = None
        unsqueeze_680: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_933: "f32[24]" = torch.ops.aten.mul.Tensor(sum_122, 2.4912308673469386e-06)
        mul_934: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_935: "f32[24]" = torch.ops.aten.mul.Tensor(mul_933, mul_934);  mul_933 = mul_934 = None
        unsqueeze_682: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
        unsqueeze_683: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
        unsqueeze_684: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
        mul_936: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_66);  primals_66 = None
        unsqueeze_685: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
        unsqueeze_686: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_937: "f32[128, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_684);  sub_262 = unsqueeze_684 = None
        sub_264: "f32[128, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_293, mul_937);  mul_937 = None
        sub_265: "f32[128, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_264, unsqueeze_681);  sub_264 = unsqueeze_681 = None
        mul_938: "f32[128, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_687);  sub_265 = unsqueeze_687 = None
        mul_939: "f32[24]" = torch.ops.aten.mul.Tensor(sum_122, squeeze_25);  sum_122 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_66 = torch.ops.aten.convolution_backward.default(mul_938, mul_58, primals_62, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_938 = mul_58 = primals_62 = None
        getitem_296: "f32[128, 144, 56, 56]" = convolution_backward_66[0]
        getitem_297: "f32[24, 144, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_7: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_15)
        mul_51: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        unsqueeze_28: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_56, -1)
        unsqueeze_29: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_57: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_51, unsqueeze_29);  mul_51 = unsqueeze_29 = None
        unsqueeze_30: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_57, -1);  primals_57 = None
        unsqueeze_31: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_46: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_57, unsqueeze_31);  mul_57 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_7: "f32[128, 144, 56, 56]" = torch.ops.aten.neg.default(add_46)
        exp_7: "f32[128, 144, 56, 56]" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_47: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        div_7: "f32[128, 144, 56, 56]" = torch.ops.aten.div.Tensor(add_46, add_47)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_940: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_296, div_7);  div_7 = None
        sigmoid_2: "f32[128, 144, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_13);  convolution_13 = None
        mul_941: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_296, sigmoid_2);  getitem_296 = None
        sum_123: "f32[128, 144, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_940, [2, 3], True);  mul_940 = None
        sub_266: "f32[128, 144, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_2)
        mul_942: "f32[128, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_2, sub_266);  sigmoid_2 = sub_266 = None
        mul_943: "f32[128, 144, 1, 1]" = torch.ops.aten.mul.Tensor(sum_123, mul_942);  sum_123 = mul_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_124: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_943, [0, 2, 3])
        convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_943, div_8, primals_60, [144], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_943 = div_8 = primals_60 = None
        getitem_299: "f32[128, 6, 1, 1]" = convolution_backward_67[0]
        getitem_300: "f32[144, 6, 1, 1]" = convolution_backward_67[1];  convolution_backward_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_8: "f32[128, 6, 1, 1]" = torch.ops.aten.neg.default(convolution_12)
        exp_8: "f32[128, 6, 1, 1]" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_48: "f32[128, 6, 1, 1]" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        reciprocal_40: "f32[128, 6, 1, 1]" = torch.ops.aten.reciprocal.default(add_48);  add_48 = None
        mul_944: "f32[128, 6, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        mul_945: "f32[128, 6, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_299, mul_944);  getitem_299 = None
        sub_267: "f32[128, 6, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_944);  mul_944 = None
        mul_946: "f32[128, 6, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_12, sub_267);  convolution_12 = sub_267 = None
        add_405: "f32[128, 6, 1, 1]" = torch.ops.aten.add.Tensor(mul_946, 1);  mul_946 = None
        mul_947: "f32[128, 6, 1, 1]" = torch.ops.aten.mul.Tensor(mul_945, add_405);  mul_945 = add_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_125: "f32[6]" = torch.ops.aten.sum.dim_IntList(mul_947, [0, 2, 3])
        convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_947, mean_2, primals_58, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_947 = mean_2 = primals_58 = None
        getitem_302: "f32[128, 144, 1, 1]" = convolution_backward_68[0]
        getitem_303: "f32[6, 144, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_15: "f32[128, 144, 56, 56]" = torch.ops.aten.expand.default(getitem_302, [128, 144, 56, 56]);  getitem_302 = None
        div_63: "f32[128, 144, 56, 56]" = torch.ops.aten.div.Scalar(expand_15, 3136);  expand_15 = None
        add_406: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_941, div_63);  mul_941 = div_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_41: "f32[128, 144, 56, 56]" = torch.ops.aten.reciprocal.default(add_47);  add_47 = None
        mul_948: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        mul_949: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_406, mul_948);  add_406 = None
        sub_268: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(1, mul_948);  mul_948 = None
        mul_950: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_46, sub_268);  add_46 = sub_268 = None
        add_408: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_950, 1);  mul_950 = None
        mul_951: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_949, add_408);  mul_949 = add_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_21: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        unsqueeze_688: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_689: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
        unsqueeze_690: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
        sum_126: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_951, [0, 2, 3])
        sub_269: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_690);  convolution_11 = unsqueeze_690 = None
        mul_952: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_951, sub_269)
        sum_127: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_952, [0, 2, 3]);  mul_952 = None
        mul_953: "f32[144]" = torch.ops.aten.mul.Tensor(sum_126, 2.4912308673469386e-06)
        unsqueeze_691: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
        unsqueeze_692: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_954: "f32[144]" = torch.ops.aten.mul.Tensor(sum_127, 2.4912308673469386e-06)
        squeeze_22: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_955: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_956: "f32[144]" = torch.ops.aten.mul.Tensor(mul_954, mul_955);  mul_954 = mul_955 = None
        unsqueeze_694: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_956, 0);  mul_956 = None
        unsqueeze_695: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        mul_957: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_56);  primals_56 = None
        unsqueeze_697: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_957, 0);  mul_957 = None
        unsqueeze_698: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_958: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_696);  sub_269 = unsqueeze_696 = None
        sub_271: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(mul_951, mul_958);  mul_951 = mul_958 = None
        sub_272: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_693);  sub_271 = unsqueeze_693 = None
        mul_959: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_699);  sub_272 = unsqueeze_699 = None
        mul_960: "f32[144]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_22);  sum_127 = squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:331 in forward, code: x = self.conv_dw(x)
        convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_959, div_6, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_959 = div_6 = primals_52 = None
        getitem_305: "f32[128, 144, 56, 56]" = convolution_backward_69[0]
        getitem_306: "f32[144, 1, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_6: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, getitem_13)
        mul_44: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        unsqueeze_24: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_50, -1)
        unsqueeze_25: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        mul_50: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_44, unsqueeze_25);  mul_44 = unsqueeze_25 = None
        unsqueeze_26: "f32[144, 1]" = torch.ops.aten.unsqueeze.default(primals_51, -1);  primals_51 = None
        unsqueeze_27: "f32[144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        add_40: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_27);  mul_50 = unsqueeze_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_6: "f32[128, 144, 56, 56]" = torch.ops.aten.neg.default(add_40)
        exp_6: "f32[128, 144, 56, 56]" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_41: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        reciprocal_42: "f32[128, 144, 56, 56]" = torch.ops.aten.reciprocal.default(add_41);  add_41 = None
        mul_961: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        mul_962: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_305, mul_961);  getitem_305 = None
        sub_273: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(1, mul_961);  mul_961 = None
        mul_963: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(add_40, sub_273);  add_40 = sub_273 = None
        add_410: "f32[128, 144, 56, 56]" = torch.ops.aten.add.Tensor(mul_963, 1);  mul_963 = None
        mul_964: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_962, add_410);  mul_962 = add_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_18: "f32[144]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        unsqueeze_700: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(squeeze_18, 0);  squeeze_18 = None
        unsqueeze_701: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
        unsqueeze_702: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
        sum_128: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_964, [0, 2, 3])
        sub_274: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_702);  convolution_10 = unsqueeze_702 = None
        mul_965: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(mul_964, sub_274)
        sum_129: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_965, [0, 2, 3]);  mul_965 = None
        mul_966: "f32[144]" = torch.ops.aten.mul.Tensor(sum_128, 2.4912308673469386e-06)
        unsqueeze_703: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_966, 0);  mul_966 = None
        unsqueeze_704: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_967: "f32[144]" = torch.ops.aten.mul.Tensor(sum_129, 2.4912308673469386e-06)
        squeeze_19: "f32[144]" = torch.ops.aten.squeeze.dims(rsqrt_6, [0, 2, 3]);  rsqrt_6 = None
        mul_968: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_969: "f32[144]" = torch.ops.aten.mul.Tensor(mul_967, mul_968);  mul_967 = mul_968 = None
        unsqueeze_706: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_969, 0);  mul_969 = None
        unsqueeze_707: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
        unsqueeze_708: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
        mul_970: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_50);  primals_50 = None
        unsqueeze_709: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_970, 0);  mul_970 = None
        unsqueeze_710: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_971: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_708);  sub_274 = unsqueeze_708 = None
        sub_276: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(mul_964, mul_971);  mul_964 = mul_971 = None
        sub_277: "f32[128, 144, 56, 56]" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_705);  sub_276 = unsqueeze_705 = None
        mul_972: "f32[128, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_711);  sub_277 = unsqueeze_711 = None
        mul_973: "f32[144]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_19);  sum_129 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_70 = torch.ops.aten.convolution_backward.default(mul_972, add_35, primals_46, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_972 = add_35 = primals_46 = None
        getitem_308: "f32[128, 24, 56, 56]" = convolution_backward_70[0]
        getitem_309: "f32[144, 24, 1, 1]" = convolution_backward_70[1];  convolution_backward_70 = None
        add_411: "f32[128, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_293, getitem_308);  getitem_293 = getitem_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_130: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_411, [0, 2, 3])
        sub_278: "f32[128, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_714);  convolution_9 = unsqueeze_714 = None
        mul_974: "f32[128, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_411, sub_278)
        sum_131: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_974, [0, 2, 3]);  mul_974 = None
        mul_975: "f32[24]" = torch.ops.aten.mul.Tensor(sum_130, 2.4912308673469386e-06)
        unsqueeze_715: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
        unsqueeze_716: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_976: "f32[24]" = torch.ops.aten.mul.Tensor(sum_131, 2.4912308673469386e-06)
        mul_977: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_978: "f32[24]" = torch.ops.aten.mul.Tensor(mul_976, mul_977);  mul_976 = mul_977 = None
        unsqueeze_718: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
        unsqueeze_719: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        mul_979: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_44);  primals_44 = None
        unsqueeze_721: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_979, 0);  mul_979 = None
        unsqueeze_722: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_980: "f32[128, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_278, unsqueeze_720);  sub_278 = unsqueeze_720 = None
        sub_280: "f32[128, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_411, mul_980);  add_411 = mul_980 = None
        sub_281: "f32[128, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_280, unsqueeze_717);  sub_280 = unsqueeze_717 = None
        mul_981: "f32[128, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_723);  sub_281 = unsqueeze_723 = None
        mul_982: "f32[24]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_16);  sum_131 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:335 in forward, code: x = self.conv_pwl(x)
        convolution_backward_71 = torch.ops.aten.convolution_backward.default(mul_981, mul_36, primals_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_981 = mul_36 = primals_40 = None
        getitem_311: "f32[128, 96, 56, 56]" = convolution_backward_71[0]
        getitem_312: "f32[24, 96, 1, 1]" = convolution_backward_71[1];  convolution_backward_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_4: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, getitem_9)
        mul_29: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        unsqueeze_16: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_34, -1)
        unsqueeze_17: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_35: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_29, unsqueeze_17);  mul_29 = unsqueeze_17 = None
        unsqueeze_18: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_35, -1);  primals_35 = None
        unsqueeze_19: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_28: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_19);  mul_35 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_4: "f32[128, 96, 56, 56]" = torch.ops.aten.neg.default(add_28)
        exp_4: "f32[128, 96, 56, 56]" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_29: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        div_4: "f32[128, 96, 56, 56]" = torch.ops.aten.div.Tensor(add_28, add_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_983: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_311, div_4);  div_4 = None
        sigmoid_1: "f32[128, 96, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_8);  convolution_8 = None
        mul_984: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_311, sigmoid_1);  getitem_311 = None
        sum_132: "f32[128, 96, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_983, [2, 3], True);  mul_983 = None
        sub_282: "f32[128, 96, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_1)
        mul_985: "f32[128, 96, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_1, sub_282);  sigmoid_1 = sub_282 = None
        mul_986: "f32[128, 96, 1, 1]" = torch.ops.aten.mul.Tensor(sum_132, mul_985);  sum_132 = mul_985 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_133: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_986, [0, 2, 3])
        convolution_backward_72 = torch.ops.aten.convolution_backward.default(mul_986, div_5, primals_38, [96], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_986 = div_5 = primals_38 = None
        getitem_314: "f32[128, 4, 1, 1]" = convolution_backward_72[0]
        getitem_315: "f32[96, 4, 1, 1]" = convolution_backward_72[1];  convolution_backward_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_5: "f32[128, 4, 1, 1]" = torch.ops.aten.neg.default(convolution_7)
        exp_5: "f32[128, 4, 1, 1]" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_30: "f32[128, 4, 1, 1]" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        reciprocal_43: "f32[128, 4, 1, 1]" = torch.ops.aten.reciprocal.default(add_30);  add_30 = None
        mul_987: "f32[128, 4, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        mul_988: "f32[128, 4, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_314, mul_987);  getitem_314 = None
        sub_283: "f32[128, 4, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_987);  mul_987 = None
        mul_989: "f32[128, 4, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_7, sub_283);  convolution_7 = sub_283 = None
        add_413: "f32[128, 4, 1, 1]" = torch.ops.aten.add.Tensor(mul_989, 1);  mul_989 = None
        mul_990: "f32[128, 4, 1, 1]" = torch.ops.aten.mul.Tensor(mul_988, add_413);  mul_988 = add_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_134: "f32[4]" = torch.ops.aten.sum.dim_IntList(mul_990, [0, 2, 3])
        convolution_backward_73 = torch.ops.aten.convolution_backward.default(mul_990, mean_1, primals_36, [4], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_990 = mean_1 = primals_36 = None
        getitem_317: "f32[128, 96, 1, 1]" = convolution_backward_73[0]
        getitem_318: "f32[4, 96, 1, 1]" = convolution_backward_73[1];  convolution_backward_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_16: "f32[128, 96, 56, 56]" = torch.ops.aten.expand.default(getitem_317, [128, 96, 56, 56]);  getitem_317 = None
        div_64: "f32[128, 96, 56, 56]" = torch.ops.aten.div.Scalar(expand_16, 3136);  expand_16 = None
        add_414: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_984, div_64);  mul_984 = div_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_44: "f32[128, 96, 56, 56]" = torch.ops.aten.reciprocal.default(add_29);  add_29 = None
        mul_991: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        mul_992: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_414, mul_991);  add_414 = None
        sub_284: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(1, mul_991);  mul_991 = None
        mul_993: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(add_28, sub_284);  add_28 = sub_284 = None
        add_416: "f32[128, 96, 56, 56]" = torch.ops.aten.add.Tensor(mul_993, 1);  mul_993 = None
        mul_994: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_992, add_416);  mul_992 = add_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_12: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        unsqueeze_724: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_725: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
        unsqueeze_726: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
        sum_135: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_994, [0, 2, 3])
        sub_285: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_726);  convolution_6 = unsqueeze_726 = None
        mul_995: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(mul_994, sub_285)
        sum_136: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_995, [0, 2, 3]);  mul_995 = None
        mul_996: "f32[96]" = torch.ops.aten.mul.Tensor(sum_135, 2.4912308673469386e-06)
        unsqueeze_727: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
        unsqueeze_728: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_997: "f32[96]" = torch.ops.aten.mul.Tensor(sum_136, 2.4912308673469386e-06)
        squeeze_13: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_998: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_999: "f32[96]" = torch.ops.aten.mul.Tensor(mul_997, mul_998);  mul_997 = mul_998 = None
        unsqueeze_730: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
        unsqueeze_731: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
        unsqueeze_732: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
        mul_1000: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_34);  primals_34 = None
        unsqueeze_733: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
        unsqueeze_734: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_1001: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_732);  sub_285 = unsqueeze_732 = None
        sub_287: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(mul_994, mul_1001);  mul_994 = mul_1001 = None
        sub_288: "f32[128, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_729);  sub_287 = unsqueeze_729 = None
        mul_1002: "f32[128, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_735);  sub_288 = unsqueeze_735 = None
        mul_1003: "f32[96]" = torch.ops.aten.mul.Tensor(sum_136, squeeze_13);  sum_136 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_backward_74 = torch.ops.aten.convolution_backward.default(mul_1002, constant_pad_nd_1, primals_30, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_1002 = constant_pad_nd_1 = primals_30 = None
        getitem_320: "f32[128, 96, 113, 113]" = convolution_backward_74[0]
        getitem_321: "f32[96, 1, 3, 3]" = convolution_backward_74[1];  convolution_backward_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torch/nn/functional.py:5462 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_8: "f32[128, 96, 112, 112]" = torch.ops.aten.constant_pad_nd.default(getitem_320, [0, -1, 0, -1]);  getitem_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_3: "f32[128, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_7)
        mul_22: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
        unsqueeze_12: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_28, -1)
        unsqueeze_13: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_28: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_13);  mul_22 = unsqueeze_13 = None
        unsqueeze_14: "f32[96, 1]" = torch.ops.aten.unsqueeze.default(primals_29, -1);  primals_29 = None
        unsqueeze_15: "f32[96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_22: "f32[128, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_28, unsqueeze_15);  mul_28 = unsqueeze_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_3: "f32[128, 96, 112, 112]" = torch.ops.aten.neg.default(add_22)
        exp_3: "f32[128, 96, 112, 112]" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_23: "f32[128, 96, 112, 112]" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        reciprocal_45: "f32[128, 96, 112, 112]" = torch.ops.aten.reciprocal.default(add_23);  add_23 = None
        mul_1004: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        mul_1005: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(constant_pad_nd_8, mul_1004);  constant_pad_nd_8 = None
        sub_289: "f32[128, 96, 112, 112]" = torch.ops.aten.sub.Tensor(1, mul_1004);  mul_1004 = None
        mul_1006: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(add_22, sub_289);  add_22 = sub_289 = None
        add_418: "f32[128, 96, 112, 112]" = torch.ops.aten.add.Tensor(mul_1006, 1);  mul_1006 = None
        mul_1007: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1005, add_418);  mul_1005 = add_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_9: "f32[96]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        unsqueeze_736: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(squeeze_9, 0);  squeeze_9 = None
        unsqueeze_737: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
        unsqueeze_738: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
        sum_137: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1007, [0, 2, 3])
        sub_290: "f32[128, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_738);  convolution_5 = unsqueeze_738 = None
        mul_1008: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1007, sub_290)
        sum_138: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_1008, [0, 2, 3]);  mul_1008 = None
        mul_1009: "f32[96]" = torch.ops.aten.mul.Tensor(sum_137, 6.228077168367346e-07)
        unsqueeze_739: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
        unsqueeze_740: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
        unsqueeze_741: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
        mul_1010: "f32[96]" = torch.ops.aten.mul.Tensor(sum_138, 6.228077168367346e-07)
        squeeze_10: "f32[96]" = torch.ops.aten.squeeze.dims(rsqrt_3, [0, 2, 3]);  rsqrt_3 = None
        mul_1011: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_1012: "f32[96]" = torch.ops.aten.mul.Tensor(mul_1010, mul_1011);  mul_1010 = mul_1011 = None
        unsqueeze_742: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1012, 0);  mul_1012 = None
        unsqueeze_743: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
        unsqueeze_744: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
        mul_1013: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_28);  primals_28 = None
        unsqueeze_745: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1013, 0);  mul_1013 = None
        unsqueeze_746: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
        unsqueeze_747: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
        mul_1014: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_290, unsqueeze_744);  sub_290 = unsqueeze_744 = None
        sub_292: "f32[128, 96, 112, 112]" = torch.ops.aten.sub.Tensor(mul_1007, mul_1014);  mul_1007 = mul_1014 = None
        sub_293: "f32[128, 96, 112, 112]" = torch.ops.aten.sub.Tensor(sub_292, unsqueeze_741);  sub_292 = unsqueeze_741 = None
        mul_1015: "f32[128, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_747);  sub_293 = unsqueeze_747 = None
        mul_1016: "f32[96]" = torch.ops.aten.mul.Tensor(sum_138, squeeze_10);  sum_138 = squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:329 in forward, code: x = self.conv_pw(x)
        convolution_backward_75 = torch.ops.aten.convolution_backward.default(mul_1015, add_17, primals_24, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1015 = add_17 = primals_24 = None
        getitem_323: "f32[128, 16, 112, 112]" = convolution_backward_75[0]
        getitem_324: "f32[96, 16, 1, 1]" = convolution_backward_75[1];  convolution_backward_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_139: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_323, [0, 2, 3])
        sub_294: "f32[128, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_750);  convolution_4 = unsqueeze_750 = None
        mul_1017: "f32[128, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_323, sub_294)
        sum_140: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1017, [0, 2, 3]);  mul_1017 = None
        mul_1018: "f32[16]" = torch.ops.aten.mul.Tensor(sum_139, 6.228077168367346e-07)
        unsqueeze_751: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
        unsqueeze_752: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
        unsqueeze_753: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
        mul_1019: "f32[16]" = torch.ops.aten.mul.Tensor(sum_140, 6.228077168367346e-07)
        mul_1020: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_1021: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1019, mul_1020);  mul_1019 = mul_1020 = None
        unsqueeze_754: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1021, 0);  mul_1021 = None
        unsqueeze_755: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
        unsqueeze_756: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
        mul_1022: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_22);  primals_22 = None
        unsqueeze_757: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1022, 0);  mul_1022 = None
        unsqueeze_758: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
        unsqueeze_759: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
        mul_1023: "f32[128, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_294, unsqueeze_756);  sub_294 = unsqueeze_756 = None
        sub_296: "f32[128, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_323, mul_1023);  getitem_323 = mul_1023 = None
        sub_297: "f32[128, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_296, unsqueeze_753);  sub_296 = unsqueeze_753 = None
        mul_1024: "f32[128, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_759);  sub_297 = unsqueeze_759 = None
        mul_1025: "f32[16]" = torch.ops.aten.mul.Tensor(sum_140, squeeze_7);  sum_140 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:227 in forward, code: x = self.conv_pw(x)
        convolution_backward_76 = torch.ops.aten.convolution_backward.default(mul_1024, mul_14, primals_18, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1024 = mul_14 = primals_18 = None
        getitem_326: "f32[128, 32, 112, 112]" = convolution_backward_76[0]
        getitem_327: "f32[16, 32, 1, 1]" = convolution_backward_76[1];  convolution_backward_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_1: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        unsqueeze_4: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_10: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_1: "f32[128, 32, 112, 112]" = torch.ops.aten.neg.default(add_10)
        exp_1: "f32[128, 32, 112, 112]" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_11: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        div_1: "f32[128, 32, 112, 112]" = torch.ops.aten.div.Tensor(add_10, add_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:83 in forward, code: return x * self.gate(x_se)
        mul_1026: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_326, div_1);  div_1 = None
        sigmoid: "f32[128, 32, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_3);  convolution_3 = None
        mul_1027: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_326, sigmoid);  getitem_326 = None
        sum_141: "f32[128, 32, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_1026, [2, 3], True);  mul_1026 = None
        sub_298: "f32[128, 32, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid)
        mul_1028: "f32[128, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid, sub_298);  sigmoid = sub_298 = None
        mul_1029: "f32[128, 32, 1, 1]" = torch.ops.aten.mul.Tensor(sum_141, mul_1028);  sum_141 = mul_1028 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:82 in forward, code: x_se = self.conv_expand(x_se)
        sum_142: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1029, [0, 2, 3])
        convolution_backward_77 = torch.ops.aten.convolution_backward.default(mul_1029, div_2, primals_16, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1029 = div_2 = primals_16 = None
        getitem_329: "f32[128, 8, 1, 1]" = convolution_backward_77[0]
        getitem_330: "f32[32, 8, 1, 1]" = convolution_backward_77[1];  convolution_backward_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:81 in forward, code: x_se = self.act1(x_se)
        neg_2: "f32[128, 8, 1, 1]" = torch.ops.aten.neg.default(convolution_2)
        exp_2: "f32[128, 8, 1, 1]" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_12: "f32[128, 8, 1, 1]" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        reciprocal_46: "f32[128, 8, 1, 1]" = torch.ops.aten.reciprocal.default(add_12);  add_12 = None
        mul_1030: "f32[128, 8, 1, 1]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        mul_1031: "f32[128, 8, 1, 1]" = torch.ops.aten.mul.Tensor(getitem_329, mul_1030);  getitem_329 = None
        sub_299: "f32[128, 8, 1, 1]" = torch.ops.aten.sub.Tensor(1, mul_1030);  mul_1030 = None
        mul_1032: "f32[128, 8, 1, 1]" = torch.ops.aten.mul.Tensor(convolution_2, sub_299);  convolution_2 = sub_299 = None
        add_420: "f32[128, 8, 1, 1]" = torch.ops.aten.add.Tensor(mul_1032, 1);  mul_1032 = None
        mul_1033: "f32[128, 8, 1, 1]" = torch.ops.aten.mul.Tensor(mul_1031, add_420);  mul_1031 = add_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:80 in forward, code: x_se = self.conv_reduce(x_se)
        sum_143: "f32[8]" = torch.ops.aten.sum.dim_IntList(mul_1033, [0, 2, 3])
        convolution_backward_78 = torch.ops.aten.convolution_backward.default(mul_1033, mean, primals_14, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1033 = mean = primals_14 = None
        getitem_332: "f32[128, 32, 1, 1]" = convolution_backward_78[0]
        getitem_333: "f32[8, 32, 1, 1]" = convolution_backward_78[1];  convolution_backward_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:79 in forward, code: x_se = x.mean((2, 3), keepdim=True)
        expand_17: "f32[128, 32, 112, 112]" = torch.ops.aten.expand.default(getitem_332, [128, 32, 112, 112]);  getitem_332 = None
        div_65: "f32[128, 32, 112, 112]" = torch.ops.aten.div.Scalar(expand_17, 12544);  expand_17 = None
        add_421: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_1027, div_65);  mul_1027 = div_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        reciprocal_47: "f32[128, 32, 112, 112]" = torch.ops.aten.reciprocal.default(add_11);  add_11 = None
        mul_1034: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        mul_1035: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_421, mul_1034);  add_421 = None
        sub_300: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(1, mul_1034);  mul_1034 = None
        mul_1036: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_10, sub_300);  add_10 = sub_300 = None
        add_423: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_1036, 1);  mul_1036 = None
        mul_1037: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1035, add_423);  mul_1035 = add_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_3: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        unsqueeze_760: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_761: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
        unsqueeze_762: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
        sum_144: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1037, [0, 2, 3])
        sub_301: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_762);  convolution_1 = unsqueeze_762 = None
        mul_1038: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1037, sub_301)
        sum_145: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1038, [0, 2, 3]);  mul_1038 = None
        mul_1039: "f32[32]" = torch.ops.aten.mul.Tensor(sum_144, 6.228077168367346e-07)
        unsqueeze_763: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1039, 0);  mul_1039 = None
        unsqueeze_764: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
        unsqueeze_765: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
        mul_1040: "f32[32]" = torch.ops.aten.mul.Tensor(sum_145, 6.228077168367346e-07)
        squeeze_4: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_1041: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_1042: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1040, mul_1041);  mul_1040 = mul_1041 = None
        unsqueeze_766: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1042, 0);  mul_1042 = None
        unsqueeze_767: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
        unsqueeze_768: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
        mul_1043: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_769: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1043, 0);  mul_1043 = None
        unsqueeze_770: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
        unsqueeze_771: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
        mul_1044: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_768);  sub_301 = unsqueeze_768 = None
        sub_303: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(mul_1037, mul_1044);  mul_1037 = mul_1044 = None
        sub_304: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_765);  sub_303 = unsqueeze_765 = None
        mul_1045: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_771);  sub_304 = unsqueeze_771 = None
        mul_1046: "f32[32]" = torch.ops.aten.mul.Tensor(sum_145, squeeze_4);  sum_145 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/_efficientnet_blocks.py:223 in forward, code: x = self.conv_dw(x)
        convolution_backward_79 = torch.ops.aten.convolution_backward.default(mul_1045, div, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1045 = div = primals_8 = None
        getitem_335: "f32[128, 32, 112, 112]" = convolution_backward_79[0]
        getitem_336: "f32[32, 1, 3, 3]" = convolution_backward_79[1];  convolution_backward_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[32, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg: "f32[128, 32, 112, 112]" = torch.ops.aten.neg.default(add_4)
        exp: "f32[128, 32, 112, 112]" = torch.ops.aten.exp.default(neg);  neg = None
        add_5: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        reciprocal_48: "f32[128, 32, 112, 112]" = torch.ops.aten.reciprocal.default(add_5);  add_5 = None
        mul_1047: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        mul_1048: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_335, mul_1047);  getitem_335 = None
        sub_305: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(1, mul_1047);  mul_1047 = None
        mul_1049: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(add_4, sub_305);  add_4 = sub_305 = None
        add_425: "f32[128, 32, 112, 112]" = torch.ops.aten.add.Tensor(mul_1049, 1);  mul_1049 = None
        mul_1050: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1048, add_425);  mul_1048 = add_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze: "f32[32]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_772: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_773: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
        unsqueeze_774: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
        sum_146: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1050, [0, 2, 3])
        sub_306: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_774);  convolution = unsqueeze_774 = None
        mul_1051: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1050, sub_306)
        sum_147: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1051, [0, 2, 3]);  mul_1051 = None
        mul_1052: "f32[32]" = torch.ops.aten.mul.Tensor(sum_146, 6.228077168367346e-07)
        unsqueeze_775: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1052, 0);  mul_1052 = None
        unsqueeze_776: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
        unsqueeze_777: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
        mul_1053: "f32[32]" = torch.ops.aten.mul.Tensor(sum_147, 6.228077168367346e-07)
        squeeze_1: "f32[32]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_1054: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_1055: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1053, mul_1054);  mul_1053 = mul_1054 = None
        unsqueeze_778: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1055, 0);  mul_1055 = None
        unsqueeze_779: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
        unsqueeze_780: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
        mul_1056: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_781: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1056, 0);  mul_1056 = None
        unsqueeze_782: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
        unsqueeze_783: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
        mul_1057: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_306, unsqueeze_780);  sub_306 = unsqueeze_780 = None
        sub_308: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(mul_1050, mul_1057);  mul_1050 = mul_1057 = None
        sub_309: "f32[128, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_308, unsqueeze_777);  sub_308 = unsqueeze_777 = None
        mul_1058: "f32[128, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_783);  sub_309 = unsqueeze_783 = None
        mul_1059: "f32[32]" = torch.ops.aten.mul.Tensor(sum_147, squeeze_1);  sum_147 = squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv2d_same.py:28 in conv2d_same, code: return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)
        convolution_backward_80 = torch.ops.aten.convolution_backward.default(mul_1058, constant_pad_nd, primals_1, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1058 = constant_pad_nd = primals_1 = None
        getitem_339: "f32[32, 3, 3, 3]" = convolution_backward_80[1];  convolution_backward_80 = None
        return (getitem_339, None, None, None, None, mul_1059, sum_146, getitem_336, None, None, None, mul_1046, sum_144, getitem_333, sum_143, getitem_330, sum_142, getitem_327, None, None, None, mul_1025, sum_139, getitem_324, None, None, None, mul_1016, sum_137, getitem_321, None, None, None, mul_1003, sum_135, getitem_318, sum_134, getitem_315, sum_133, getitem_312, None, None, None, mul_982, sum_130, getitem_309, None, None, None, mul_973, sum_128, getitem_306, None, None, None, mul_960, sum_126, getitem_303, sum_125, getitem_300, sum_124, getitem_297, None, None, None, mul_939, sum_121, getitem_294, None, None, None, mul_930, sum_119, getitem_291, None, None, None, mul_917, sum_117, getitem_288, sum_116, getitem_285, sum_115, getitem_282, None, None, None, mul_896, sum_112, getitem_279, None, None, None, mul_887, sum_110, getitem_276, None, None, None, mul_874, sum_108, getitem_273, sum_107, getitem_270, sum_106, getitem_267, None, None, None, mul_853, sum_103, getitem_264, None, None, None, mul_844, sum_101, getitem_261, None, None, None, mul_831, sum_99, getitem_258, sum_98, getitem_255, sum_97, getitem_252, None, None, None, mul_810, sum_94, getitem_249, None, None, None, mul_801, sum_92, getitem_246, None, None, None, mul_788, sum_90, getitem_243, sum_89, getitem_240, sum_88, getitem_237, None, None, None, mul_767, sum_85, getitem_234, None, None, None, mul_758, sum_83, getitem_231, None, None, None, mul_745, sum_81, getitem_228, sum_80, getitem_225, sum_79, getitem_222, None, None, None, mul_724, sum_76, getitem_219, None, None, None, mul_715, sum_74, getitem_216, None, None, None, mul_702, sum_72, getitem_213, sum_71, getitem_210, sum_70, getitem_207, None, None, None, mul_681, sum_67, getitem_204, None, None, None, mul_672, sum_65, getitem_201, None, None, None, mul_659, sum_63, getitem_198, sum_62, getitem_195, sum_61, getitem_192, None, None, None, mul_638, sum_58, getitem_189, None, None, None, mul_629, sum_56, getitem_186, None, None, None, mul_616, sum_54, getitem_183, sum_53, getitem_180, sum_52, getitem_177, None, None, None, mul_595, sum_49, getitem_174, None, None, None, mul_586, sum_47, getitem_171, None, None, None, mul_573, sum_45, getitem_168, sum_44, getitem_165, sum_43, getitem_162, None, None, None, mul_552, sum_40, getitem_159, None, None, None, mul_543, sum_38, getitem_156, None, None, None, mul_530, sum_36, getitem_153, sum_35, getitem_150, sum_34, getitem_147, None, None, None, mul_509, sum_31, getitem_144, None, None, None, mul_500, sum_29, getitem_141, None, None, None, mul_487, sum_27, getitem_138, sum_26, getitem_135, sum_25, getitem_132, None, None, None, mul_466, sum_22, getitem_129, None, None, None, mul_457, sum_20, getitem_126, None, None, None, mul_444, sum_18, getitem_123, sum_17, getitem_120, sum_16, getitem_117, None, None, None, mul_423, sum_13, getitem_114, None, None, None, mul_414, sum_11, getitem_111, None, None, None, mul_401, sum_9, getitem_108, sum_8, getitem_105, sum_7, getitem_102, None, None, None, mul_380, sum_4, getitem_99, None, None, None, mul_371, sum_2, mm_1, view_1)
