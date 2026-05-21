class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[24, 3, 3, 3]", primals_2: "f32[512, 3, 224, 224]", primals_6: "f32[24]", primals_7: "f32[24]", primals_8: "f32[24, 1, 3, 3]", primals_12: "f32[24]", primals_14: "f32[58, 24, 1, 1]", primals_18: "f32[58]", primals_19: "f32[58]", primals_20: "f32[58, 24, 1, 1]", primals_24: "f32[58]", primals_26: "f32[58, 1, 3, 3]", primals_30: "f32[58]", primals_32: "f32[58, 58, 1, 1]", primals_36: "f32[58]", primals_37: "f32[58]", primals_38: "f32[58, 58, 1, 1]", primals_42: "f32[58]", primals_44: "f32[58, 1, 3, 3]", primals_48: "f32[58]", primals_50: "f32[58, 58, 1, 1]", primals_54: "f32[58]", primals_55: "f32[58]", primals_56: "f32[58, 58, 1, 1]", primals_60: "f32[58]", primals_62: "f32[58, 1, 3, 3]", primals_66: "f32[58]", primals_68: "f32[58, 58, 1, 1]", primals_72: "f32[58]", primals_73: "f32[58]", primals_74: "f32[58, 58, 1, 1]", primals_78: "f32[58]", primals_80: "f32[58, 1, 3, 3]", primals_84: "f32[58]", primals_86: "f32[58, 58, 1, 1]", primals_90: "f32[58]", primals_91: "f32[58]", primals_92: "f32[116, 1, 3, 3]", primals_96: "f32[116]", primals_98: "f32[116, 116, 1, 1]", primals_102: "f32[116]", primals_103: "f32[116]", primals_104: "f32[116, 116, 1, 1]", primals_108: "f32[116]", primals_110: "f32[116, 1, 3, 3]", primals_114: "f32[116]", primals_116: "f32[116, 116, 1, 1]", primals_120: "f32[116]", primals_121: "f32[116]", primals_122: "f32[116, 116, 1, 1]", primals_126: "f32[116]", primals_128: "f32[116, 1, 3, 3]", primals_132: "f32[116]", primals_134: "f32[116, 116, 1, 1]", primals_138: "f32[116]", primals_139: "f32[116]", primals_140: "f32[116, 116, 1, 1]", primals_144: "f32[116]", primals_146: "f32[116, 1, 3, 3]", primals_150: "f32[116]", primals_152: "f32[116, 116, 1, 1]", primals_156: "f32[116]", primals_157: "f32[116]", primals_158: "f32[116, 116, 1, 1]", primals_162: "f32[116]", primals_164: "f32[116, 1, 3, 3]", primals_168: "f32[116]", primals_170: "f32[116, 116, 1, 1]", primals_174: "f32[116]", primals_175: "f32[116]", primals_176: "f32[116, 116, 1, 1]", primals_180: "f32[116]", primals_182: "f32[116, 1, 3, 3]", primals_186: "f32[116]", primals_188: "f32[116, 116, 1, 1]", primals_192: "f32[116]", primals_193: "f32[116]", primals_194: "f32[116, 116, 1, 1]", primals_198: "f32[116]", primals_200: "f32[116, 1, 3, 3]", primals_204: "f32[116]", primals_206: "f32[116, 116, 1, 1]", primals_210: "f32[116]", primals_211: "f32[116]", primals_212: "f32[116, 116, 1, 1]", primals_216: "f32[116]", primals_218: "f32[116, 1, 3, 3]", primals_222: "f32[116]", primals_224: "f32[116, 116, 1, 1]", primals_228: "f32[116]", primals_229: "f32[116]", primals_230: "f32[116, 116, 1, 1]", primals_234: "f32[116]", primals_236: "f32[116, 1, 3, 3]", primals_240: "f32[116]", primals_242: "f32[116, 116, 1, 1]", primals_246: "f32[116]", primals_247: "f32[116]", primals_248: "f32[232, 1, 3, 3]", primals_252: "f32[232]", primals_254: "f32[232, 232, 1, 1]", primals_258: "f32[232]", primals_259: "f32[232]", primals_260: "f32[232, 232, 1, 1]", primals_264: "f32[232]", primals_266: "f32[232, 1, 3, 3]", primals_270: "f32[232]", primals_272: "f32[232, 232, 1, 1]", primals_276: "f32[232]", primals_277: "f32[232]", primals_278: "f32[232, 232, 1, 1]", primals_282: "f32[232]", primals_284: "f32[232, 1, 3, 3]", primals_288: "f32[232]", primals_290: "f32[232, 232, 1, 1]", primals_294: "f32[232]", primals_295: "f32[232]", primals_296: "f32[232, 232, 1, 1]", primals_300: "f32[232]", primals_302: "f32[232, 1, 3, 3]", primals_306: "f32[232]", primals_308: "f32[232, 232, 1, 1]", primals_312: "f32[232]", primals_313: "f32[232]", primals_314: "f32[232, 232, 1, 1]", primals_318: "f32[232]", primals_320: "f32[232, 1, 3, 3]", primals_324: "f32[232]", primals_326: "f32[232, 232, 1, 1]", primals_330: "f32[232]", primals_331: "f32[232]", primals_332: "f32[1024, 464, 1, 1]", primals_336: "f32[1024]", primals_337: "f32[1024]", primals_338: "f32[1000, 1024]", convolution: "f32[512, 24, 112, 112]", getitem_1: "f32[1, 24, 1, 1]", rsqrt: "f32[1, 24, 1, 1]", getitem_2: "f32[512, 24, 56, 56]", getitem_3: "i8[512, 24, 56, 56]", convolution_1: "f32[512, 24, 28, 28]", squeeze_4: "f32[24]", add_9: "f32[512, 24, 28, 28]", convolution_2: "f32[512, 58, 28, 28]", getitem_7: "f32[1, 58, 1, 1]", rsqrt_2: "f32[1, 58, 1, 1]", convolution_3: "f32[512, 58, 56, 56]", squeeze_10: "f32[58]", relu_2: "f32[512, 58, 56, 56]", convolution_4: "f32[512, 58, 28, 28]", squeeze_13: "f32[58]", add_24: "f32[512, 58, 28, 28]", convolution_5: "f32[512, 58, 28, 28]", getitem_13: "f32[1, 58, 1, 1]", rsqrt_5: "f32[1, 58, 1, 1]", getitem_15: "f32[512, 58, 28, 28]", convolution_6: "f32[512, 58, 28, 28]", squeeze_19: "f32[58]", relu_4: "f32[512, 58, 28, 28]", convolution_7: "f32[512, 58, 28, 28]", squeeze_22: "f32[58]", add_39: "f32[512, 58, 28, 28]", convolution_8: "f32[512, 58, 28, 28]", getitem_21: "f32[1, 58, 1, 1]", rsqrt_8: "f32[1, 58, 1, 1]", getitem_23: "f32[512, 58, 28, 28]", convolution_9: "f32[512, 58, 28, 28]", squeeze_28: "f32[58]", relu_6: "f32[512, 58, 28, 28]", convolution_10: "f32[512, 58, 28, 28]", squeeze_31: "f32[58]", add_54: "f32[512, 58, 28, 28]", convolution_11: "f32[512, 58, 28, 28]", getitem_29: "f32[1, 58, 1, 1]", rsqrt_11: "f32[1, 58, 1, 1]", getitem_31: "f32[512, 58, 28, 28]", convolution_12: "f32[512, 58, 28, 28]", squeeze_37: "f32[58]", relu_8: "f32[512, 58, 28, 28]", convolution_13: "f32[512, 58, 28, 28]", squeeze_40: "f32[58]", add_69: "f32[512, 58, 28, 28]", convolution_14: "f32[512, 58, 28, 28]", getitem_37: "f32[1, 58, 1, 1]", rsqrt_14: "f32[1, 58, 1, 1]", view_7: "f32[512, 116, 28, 28]", convolution_15: "f32[512, 116, 14, 14]", squeeze_46: "f32[116]", add_79: "f32[512, 116, 14, 14]", convolution_16: "f32[512, 116, 14, 14]", getitem_41: "f32[1, 116, 1, 1]", rsqrt_16: "f32[1, 116, 1, 1]", convolution_17: "f32[512, 116, 28, 28]", squeeze_52: "f32[116]", relu_11: "f32[512, 116, 28, 28]", convolution_18: "f32[512, 116, 14, 14]", squeeze_55: "f32[116]", add_94: "f32[512, 116, 14, 14]", convolution_19: "f32[512, 116, 14, 14]", getitem_47: "f32[1, 116, 1, 1]", rsqrt_19: "f32[1, 116, 1, 1]", getitem_49: "f32[512, 116, 14, 14]", convolution_20: "f32[512, 116, 14, 14]", squeeze_61: "f32[116]", relu_13: "f32[512, 116, 14, 14]", convolution_21: "f32[512, 116, 14, 14]", squeeze_64: "f32[116]", add_109: "f32[512, 116, 14, 14]", convolution_22: "f32[512, 116, 14, 14]", getitem_55: "f32[1, 116, 1, 1]", rsqrt_22: "f32[1, 116, 1, 1]", getitem_57: "f32[512, 116, 14, 14]", convolution_23: "f32[512, 116, 14, 14]", squeeze_70: "f32[116]", relu_15: "f32[512, 116, 14, 14]", convolution_24: "f32[512, 116, 14, 14]", squeeze_73: "f32[116]", add_124: "f32[512, 116, 14, 14]", convolution_25: "f32[512, 116, 14, 14]", getitem_63: "f32[1, 116, 1, 1]", rsqrt_25: "f32[1, 116, 1, 1]", getitem_65: "f32[512, 116, 14, 14]", convolution_26: "f32[512, 116, 14, 14]", squeeze_79: "f32[116]", relu_17: "f32[512, 116, 14, 14]", convolution_27: "f32[512, 116, 14, 14]", squeeze_82: "f32[116]", add_139: "f32[512, 116, 14, 14]", convolution_28: "f32[512, 116, 14, 14]", getitem_71: "f32[1, 116, 1, 1]", rsqrt_28: "f32[1, 116, 1, 1]", getitem_73: "f32[512, 116, 14, 14]", convolution_29: "f32[512, 116, 14, 14]", squeeze_88: "f32[116]", relu_19: "f32[512, 116, 14, 14]", convolution_30: "f32[512, 116, 14, 14]", squeeze_91: "f32[116]", add_154: "f32[512, 116, 14, 14]", convolution_31: "f32[512, 116, 14, 14]", getitem_79: "f32[1, 116, 1, 1]", rsqrt_31: "f32[1, 116, 1, 1]", getitem_81: "f32[512, 116, 14, 14]", convolution_32: "f32[512, 116, 14, 14]", squeeze_97: "f32[116]", relu_21: "f32[512, 116, 14, 14]", convolution_33: "f32[512, 116, 14, 14]", squeeze_100: "f32[116]", add_169: "f32[512, 116, 14, 14]", convolution_34: "f32[512, 116, 14, 14]", getitem_87: "f32[1, 116, 1, 1]", rsqrt_34: "f32[1, 116, 1, 1]", getitem_89: "f32[512, 116, 14, 14]", convolution_35: "f32[512, 116, 14, 14]", squeeze_106: "f32[116]", relu_23: "f32[512, 116, 14, 14]", convolution_36: "f32[512, 116, 14, 14]", squeeze_109: "f32[116]", add_184: "f32[512, 116, 14, 14]", convolution_37: "f32[512, 116, 14, 14]", getitem_95: "f32[1, 116, 1, 1]", rsqrt_37: "f32[1, 116, 1, 1]", getitem_97: "f32[512, 116, 14, 14]", convolution_38: "f32[512, 116, 14, 14]", squeeze_115: "f32[116]", relu_25: "f32[512, 116, 14, 14]", convolution_39: "f32[512, 116, 14, 14]", squeeze_118: "f32[116]", add_199: "f32[512, 116, 14, 14]", convolution_40: "f32[512, 116, 14, 14]", getitem_103: "f32[1, 116, 1, 1]", rsqrt_40: "f32[1, 116, 1, 1]", view_23: "f32[512, 232, 14, 14]", convolution_41: "f32[512, 232, 7, 7]", squeeze_124: "f32[232]", add_209: "f32[512, 232, 7, 7]", convolution_42: "f32[512, 232, 7, 7]", getitem_107: "f32[1, 232, 1, 1]", rsqrt_42: "f32[1, 232, 1, 1]", convolution_43: "f32[512, 232, 14, 14]", squeeze_130: "f32[232]", relu_28: "f32[512, 232, 14, 14]", convolution_44: "f32[512, 232, 7, 7]", squeeze_133: "f32[232]", add_224: "f32[512, 232, 7, 7]", convolution_45: "f32[512, 232, 7, 7]", getitem_113: "f32[1, 232, 1, 1]", rsqrt_45: "f32[1, 232, 1, 1]", getitem_115: "f32[512, 232, 7, 7]", convolution_46: "f32[512, 232, 7, 7]", squeeze_139: "f32[232]", relu_30: "f32[512, 232, 7, 7]", convolution_47: "f32[512, 232, 7, 7]", squeeze_142: "f32[232]", add_239: "f32[512, 232, 7, 7]", convolution_48: "f32[512, 232, 7, 7]", getitem_121: "f32[1, 232, 1, 1]", rsqrt_48: "f32[1, 232, 1, 1]", getitem_123: "f32[512, 232, 7, 7]", convolution_49: "f32[512, 232, 7, 7]", squeeze_148: "f32[232]", relu_32: "f32[512, 232, 7, 7]", convolution_50: "f32[512, 232, 7, 7]", squeeze_151: "f32[232]", add_254: "f32[512, 232, 7, 7]", convolution_51: "f32[512, 232, 7, 7]", getitem_129: "f32[1, 232, 1, 1]", rsqrt_51: "f32[1, 232, 1, 1]", getitem_131: "f32[512, 232, 7, 7]", convolution_52: "f32[512, 232, 7, 7]", squeeze_157: "f32[232]", relu_34: "f32[512, 232, 7, 7]", convolution_53: "f32[512, 232, 7, 7]", squeeze_160: "f32[232]", add_269: "f32[512, 232, 7, 7]", convolution_54: "f32[512, 232, 7, 7]", getitem_137: "f32[1, 232, 1, 1]", rsqrt_54: "f32[1, 232, 1, 1]", view_31: "f32[512, 464, 7, 7]", convolution_55: "f32[512, 1024, 7, 7]", getitem_139: "f32[1, 1024, 1, 1]", rsqrt_55: "f32[1, 1024, 1, 1]", mean: "f32[512, 1024]", unsqueeze_252: "f32[1, 232, 1, 1]", unsqueeze_264: "f32[1, 232, 1, 1]", unsqueeze_288: "f32[1, 232, 1, 1]", unsqueeze_300: "f32[1, 232, 1, 1]", unsqueeze_324: "f32[1, 232, 1, 1]", unsqueeze_336: "f32[1, 232, 1, 1]", unsqueeze_360: "f32[1, 232, 1, 1]", unsqueeze_372: "f32[1, 232, 1, 1]", unsqueeze_396: "f32[1, 232, 1, 1]", unsqueeze_420: "f32[1, 116, 1, 1]", unsqueeze_432: "f32[1, 116, 1, 1]", unsqueeze_456: "f32[1, 116, 1, 1]", unsqueeze_468: "f32[1, 116, 1, 1]", unsqueeze_492: "f32[1, 116, 1, 1]", unsqueeze_504: "f32[1, 116, 1, 1]", unsqueeze_528: "f32[1, 116, 1, 1]", unsqueeze_540: "f32[1, 116, 1, 1]", unsqueeze_564: "f32[1, 116, 1, 1]", unsqueeze_576: "f32[1, 116, 1, 1]", unsqueeze_600: "f32[1, 116, 1, 1]", unsqueeze_612: "f32[1, 116, 1, 1]", unsqueeze_636: "f32[1, 116, 1, 1]", unsqueeze_648: "f32[1, 116, 1, 1]", unsqueeze_672: "f32[1, 116, 1, 1]", unsqueeze_684: "f32[1, 116, 1, 1]", unsqueeze_708: "f32[1, 116, 1, 1]", unsqueeze_732: "f32[1, 58, 1, 1]", unsqueeze_744: "f32[1, 58, 1, 1]", unsqueeze_768: "f32[1, 58, 1, 1]", unsqueeze_780: "f32[1, 58, 1, 1]", unsqueeze_804: "f32[1, 58, 1, 1]", unsqueeze_816: "f32[1, 58, 1, 1]", unsqueeze_840: "f32[1, 58, 1, 1]", unsqueeze_852: "f32[1, 58, 1, 1]", unsqueeze_876: "f32[1, 24, 1, 1]", tangents_1: "f32[512, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:162 in _forward_impl, code: x = self.fc(x)
        permute_16: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_338, [1, 0]);  primals_338 = None
        permute_17: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        mm: "f32[512, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_17);  permute_17 = None
        permute_18: "f32[1000, 512]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_18, mean);  permute_18 = mean = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_32: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:161 in _forward_impl, code: x = x.mean([2, 3])  # globalpool
        unsqueeze_224: "f32[512, 1024, 1]" = torch.ops.aten.unsqueeze.default(mm, 2);  mm = None
        unsqueeze_225: "f32[512, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
        expand: "f32[512, 1024, 7, 7]" = torch.ops.aten.expand.default(unsqueeze_225, [512, 1024, 7, 7]);  unsqueeze_225 = None
        div: "f32[512, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:160 in _forward_impl, code: x = self.conv5(x)
        sub_55: "f32[512, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, getitem_139)
        mul_385: "f32[512, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = None
        unsqueeze_220: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_336, -1)
        unsqueeze_221: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_391: "f32[512, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_221);  mul_385 = unsqueeze_221 = None
        unsqueeze_222: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_337, -1);  primals_337 = None
        unsqueeze_223: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_279: "f32[512, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_391, unsqueeze_223);  mul_391 = unsqueeze_223 = None
        relu_36: "f32[512, 1024, 7, 7]" = torch.ops.aten.relu.default(add_279);  add_279 = None
        le: "b8[512, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[512, 1024, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
        squeeze_165: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_139, [0, 2, 3]);  getitem_139 = None
        unsqueeze_226: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_165, 0);  squeeze_165 = None
        unsqueeze_227: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
        unsqueeze_228: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
        sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_56: "f32[512, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_228);  convolution_55 = unsqueeze_228 = None
        mul_392: "f32[512, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_56)
        sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 2, 3]);  mul_392 = None
        mul_393: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_2, 3.985969387755102e-05)
        unsqueeze_229: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_393, 0);  mul_393 = None
        unsqueeze_230: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
        unsqueeze_231: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
        mul_394: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, 3.985969387755102e-05)
        squeeze_166: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_55, [0, 2, 3]);  rsqrt_55 = None
        mul_395: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
        mul_396: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
        unsqueeze_232: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
        unsqueeze_233: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 2);  unsqueeze_232 = None
        unsqueeze_234: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 3);  unsqueeze_233 = None
        mul_397: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_336);  primals_336 = None
        unsqueeze_235: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
        unsqueeze_236: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
        unsqueeze_237: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
        mul_398: "f32[512, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_234);  sub_56 = unsqueeze_234 = None
        sub_58: "f32[512, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_398);  where = mul_398 = None
        sub_59: "f32[512, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_58, unsqueeze_231);  sub_58 = unsqueeze_231 = None
        mul_399: "f32[512, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_237);  sub_59 = unsqueeze_237 = None
        mul_400: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_166);  sum_3 = squeeze_166 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_399, view_31, primals_332, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_399 = view_31 = primals_332 = None
        getitem_140: "f32[512, 464, 7, 7]" = convolution_backward[0]
        getitem_141: "f32[1024, 464, 1, 1]" = convolution_backward[1];  convolution_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_33: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.reshape.default(getitem_140, [512, 232, 2, 7, 7]);  getitem_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_21: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3, 4]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_16: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
        view_34: "f32[512, 464, 7, 7]" = torch.ops.aten.reshape.default(clone_16, [512, 464, 7, 7]);  clone_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_1: "f32[512, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_34, 1, 0, 232)
        slice_2: "f32[512, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_34, 1, 232, 464);  view_34 = None
        sub_54: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, getitem_137)
        mul_378: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = None
        unsqueeze_216: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_330, -1)
        unsqueeze_217: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        mul_384: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_378, unsqueeze_217);  mul_378 = unsqueeze_217 = None
        unsqueeze_218: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_331, -1);  primals_331 = None
        unsqueeze_219: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        add_274: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_384, unsqueeze_219);  mul_384 = unsqueeze_219 = None
        relu_35: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_274);  add_274 = None
        le_1: "b8[512, 232, 7, 7]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_1: "f32[512, 232, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, slice_2);  le_1 = slice_2 = None
        squeeze_162: "f32[232]" = torch.ops.aten.squeeze.dims(getitem_137, [0, 2, 3]);  getitem_137 = None
        unsqueeze_238: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(squeeze_162, 0);  squeeze_162 = None
        unsqueeze_239: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
        unsqueeze_240: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
        sum_4: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_60: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_240);  convolution_54 = unsqueeze_240 = None
        mul_401: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_60)
        sum_5: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 2, 3]);  mul_401 = None
        mul_402: "f32[232]" = torch.ops.aten.mul.Tensor(sum_4, 3.985969387755102e-05)
        unsqueeze_241: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_402, 0);  mul_402 = None
        unsqueeze_242: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
        unsqueeze_243: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
        mul_403: "f32[232]" = torch.ops.aten.mul.Tensor(sum_5, 3.985969387755102e-05)
        squeeze_163: "f32[232]" = torch.ops.aten.squeeze.dims(rsqrt_54, [0, 2, 3]);  rsqrt_54 = None
        mul_404: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
        mul_405: "f32[232]" = torch.ops.aten.mul.Tensor(mul_403, mul_404);  mul_403 = mul_404 = None
        unsqueeze_244: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
        unsqueeze_245: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 2);  unsqueeze_244 = None
        unsqueeze_246: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 3);  unsqueeze_245 = None
        mul_406: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_330);  primals_330 = None
        unsqueeze_247: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_406, 0);  mul_406 = None
        unsqueeze_248: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
        unsqueeze_249: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
        mul_407: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_246);  sub_60 = unsqueeze_246 = None
        sub_62: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_407);  where_1 = mul_407 = None
        sub_63: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_243);  sub_62 = unsqueeze_243 = None
        mul_408: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_249);  sub_63 = unsqueeze_249 = None
        mul_409: "f32[232]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_163);  sum_5 = squeeze_163 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_408, add_269, primals_326, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_408 = add_269 = primals_326 = None
        getitem_143: "f32[512, 232, 7, 7]" = convolution_backward_1[0]
        getitem_144: "f32[232, 232, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
        sum_6: "f32[232]" = torch.ops.aten.sum.dim_IntList(getitem_143, [0, 2, 3])
        sub_64: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_252);  convolution_53 = unsqueeze_252 = None
        mul_410: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_143, sub_64)
        sum_7: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_410, [0, 2, 3]);  mul_410 = None
        mul_411: "f32[232]" = torch.ops.aten.mul.Tensor(sum_6, 3.985969387755102e-05)
        unsqueeze_253: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_411, 0);  mul_411 = None
        unsqueeze_254: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
        unsqueeze_255: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
        mul_412: "f32[232]" = torch.ops.aten.mul.Tensor(sum_7, 3.985969387755102e-05)
        mul_413: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
        mul_414: "f32[232]" = torch.ops.aten.mul.Tensor(mul_412, mul_413);  mul_412 = mul_413 = None
        unsqueeze_256: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
        unsqueeze_257: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 2);  unsqueeze_256 = None
        unsqueeze_258: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 3);  unsqueeze_257 = None
        mul_415: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_324);  primals_324 = None
        unsqueeze_259: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_415, 0);  mul_415 = None
        unsqueeze_260: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
        unsqueeze_261: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
        mul_416: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_258);  sub_64 = unsqueeze_258 = None
        sub_66: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_143, mul_416);  getitem_143 = mul_416 = None
        sub_67: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_255);  sub_66 = unsqueeze_255 = None
        mul_417: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_261);  sub_67 = unsqueeze_261 = None
        mul_418: "f32[232]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_160);  sum_7 = squeeze_160 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_417, relu_34, primals_320, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  mul_417 = primals_320 = None
        getitem_146: "f32[512, 232, 7, 7]" = convolution_backward_2[0]
        getitem_147: "f32[232, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
        le_2: "b8[512, 232, 7, 7]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_2: "f32[512, 232, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_146);  le_2 = getitem_146 = None
        sum_8: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_68: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_264);  convolution_52 = unsqueeze_264 = None
        mul_419: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_68)
        sum_9: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 2, 3]);  mul_419 = None
        mul_420: "f32[232]" = torch.ops.aten.mul.Tensor(sum_8, 3.985969387755102e-05)
        unsqueeze_265: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_420, 0);  mul_420 = None
        unsqueeze_266: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
        unsqueeze_267: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
        mul_421: "f32[232]" = torch.ops.aten.mul.Tensor(sum_9, 3.985969387755102e-05)
        mul_422: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
        mul_423: "f32[232]" = torch.ops.aten.mul.Tensor(mul_421, mul_422);  mul_421 = mul_422 = None
        unsqueeze_268: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
        unsqueeze_269: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 2);  unsqueeze_268 = None
        unsqueeze_270: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 3);  unsqueeze_269 = None
        mul_424: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_318);  primals_318 = None
        unsqueeze_271: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_424, 0);  mul_424 = None
        unsqueeze_272: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
        unsqueeze_273: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
        mul_425: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_270);  sub_68 = unsqueeze_270 = None
        sub_70: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_425);  where_2 = mul_425 = None
        sub_71: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_267);  sub_70 = unsqueeze_267 = None
        mul_426: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_273);  sub_71 = unsqueeze_273 = None
        mul_427: "f32[232]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_157);  sum_9 = squeeze_157 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_426, getitem_131, primals_314, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_426 = getitem_131 = primals_314 = None
        getitem_149: "f32[512, 232, 7, 7]" = convolution_backward_3[0]
        getitem_150: "f32[232, 232, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_16: "f32[512, 464, 7, 7]" = torch.ops.aten.cat.default([slice_1, getitem_149], 1);  slice_1 = getitem_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_35: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.reshape.default(cat_16, [512, 232, 2, 7, 7]);  cat_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_22: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3, 4]);  view_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_17: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.clone.default(permute_22, memory_format = torch.contiguous_format);  permute_22 = None
        view_36: "f32[512, 464, 7, 7]" = torch.ops.aten.reshape.default(clone_17, [512, 464, 7, 7]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_3: "f32[512, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_36, 1, 0, 232)
        slice_4: "f32[512, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_36, 1, 232, 464);  view_36 = None
        sub_51: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_129)
        mul_357: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        unsqueeze_204: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1)
        unsqueeze_205: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_207: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_259: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        relu_33: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_259);  add_259 = None
        le_3: "b8[512, 232, 7, 7]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_3: "f32[512, 232, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, slice_4);  le_3 = slice_4 = None
        squeeze_153: "f32[232]" = torch.ops.aten.squeeze.dims(getitem_129, [0, 2, 3]);  getitem_129 = None
        unsqueeze_274: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_275: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
        unsqueeze_276: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
        sum_10: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_72: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_276);  convolution_51 = unsqueeze_276 = None
        mul_428: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_72)
        sum_11: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 2, 3]);  mul_428 = None
        mul_429: "f32[232]" = torch.ops.aten.mul.Tensor(sum_10, 3.985969387755102e-05)
        unsqueeze_277: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_429, 0);  mul_429 = None
        unsqueeze_278: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
        unsqueeze_279: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
        mul_430: "f32[232]" = torch.ops.aten.mul.Tensor(sum_11, 3.985969387755102e-05)
        squeeze_154: "f32[232]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_431: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_432: "f32[232]" = torch.ops.aten.mul.Tensor(mul_430, mul_431);  mul_430 = mul_431 = None
        unsqueeze_280: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
        unsqueeze_281: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 2);  unsqueeze_280 = None
        unsqueeze_282: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 3);  unsqueeze_281 = None
        mul_433: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_312);  primals_312 = None
        unsqueeze_283: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_433, 0);  mul_433 = None
        unsqueeze_284: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
        unsqueeze_285: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
        mul_434: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_282);  sub_72 = unsqueeze_282 = None
        sub_74: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_434);  where_3 = mul_434 = None
        sub_75: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_279);  sub_74 = unsqueeze_279 = None
        mul_435: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_285);  sub_75 = unsqueeze_285 = None
        mul_436: "f32[232]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_154);  sum_11 = squeeze_154 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_435, add_254, primals_308, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_435 = add_254 = primals_308 = None
        getitem_152: "f32[512, 232, 7, 7]" = convolution_backward_4[0]
        getitem_153: "f32[232, 232, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
        sum_12: "f32[232]" = torch.ops.aten.sum.dim_IntList(getitem_152, [0, 2, 3])
        sub_76: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_288);  convolution_50 = unsqueeze_288 = None
        mul_437: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_152, sub_76)
        sum_13: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3]);  mul_437 = None
        mul_438: "f32[232]" = torch.ops.aten.mul.Tensor(sum_12, 3.985969387755102e-05)
        unsqueeze_289: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_438, 0);  mul_438 = None
        unsqueeze_290: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
        unsqueeze_291: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
        mul_439: "f32[232]" = torch.ops.aten.mul.Tensor(sum_13, 3.985969387755102e-05)
        mul_440: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_441: "f32[232]" = torch.ops.aten.mul.Tensor(mul_439, mul_440);  mul_439 = mul_440 = None
        unsqueeze_292: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
        unsqueeze_293: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 2);  unsqueeze_292 = None
        unsqueeze_294: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 3);  unsqueeze_293 = None
        mul_442: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_306);  primals_306 = None
        unsqueeze_295: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
        unsqueeze_296: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
        unsqueeze_297: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
        mul_443: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_294);  sub_76 = unsqueeze_294 = None
        sub_78: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_152, mul_443);  getitem_152 = mul_443 = None
        sub_79: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_291);  sub_78 = unsqueeze_291 = None
        mul_444: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_297);  sub_79 = unsqueeze_297 = None
        mul_445: "f32[232]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_151);  sum_13 = squeeze_151 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_444, relu_32, primals_302, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  mul_444 = primals_302 = None
        getitem_155: "f32[512, 232, 7, 7]" = convolution_backward_5[0]
        getitem_156: "f32[232, 1, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
        le_4: "b8[512, 232, 7, 7]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_4: "f32[512, 232, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, getitem_155);  le_4 = getitem_155 = None
        sum_14: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_80: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_300);  convolution_49 = unsqueeze_300 = None
        mul_446: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_80)
        sum_15: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 2, 3]);  mul_446 = None
        mul_447: "f32[232]" = torch.ops.aten.mul.Tensor(sum_14, 3.985969387755102e-05)
        unsqueeze_301: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_447, 0);  mul_447 = None
        unsqueeze_302: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
        unsqueeze_303: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
        mul_448: "f32[232]" = torch.ops.aten.mul.Tensor(sum_15, 3.985969387755102e-05)
        mul_449: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_450: "f32[232]" = torch.ops.aten.mul.Tensor(mul_448, mul_449);  mul_448 = mul_449 = None
        unsqueeze_304: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
        unsqueeze_305: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 2);  unsqueeze_304 = None
        unsqueeze_306: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 3);  unsqueeze_305 = None
        mul_451: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_300);  primals_300 = None
        unsqueeze_307: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
        unsqueeze_308: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
        unsqueeze_309: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
        mul_452: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_306);  sub_80 = unsqueeze_306 = None
        sub_82: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_452);  where_4 = mul_452 = None
        sub_83: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_303);  sub_82 = unsqueeze_303 = None
        mul_453: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_309);  sub_83 = unsqueeze_309 = None
        mul_454: "f32[232]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_148);  sum_15 = squeeze_148 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_453, getitem_123, primals_296, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_453 = getitem_123 = primals_296 = None
        getitem_158: "f32[512, 232, 7, 7]" = convolution_backward_6[0]
        getitem_159: "f32[232, 232, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_17: "f32[512, 464, 7, 7]" = torch.ops.aten.cat.default([slice_3, getitem_158], 1);  slice_3 = getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_37: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.reshape.default(cat_17, [512, 232, 2, 7, 7]);  cat_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_23: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.permute.default(view_37, [0, 2, 1, 3, 4]);  view_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_18: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
        view_38: "f32[512, 464, 7, 7]" = torch.ops.aten.reshape.default(clone_18, [512, 464, 7, 7]);  clone_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_5: "f32[512, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_38, 1, 0, 232)
        slice_6: "f32[512, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_38, 1, 232, 464);  view_38 = None
        sub_48: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, getitem_121)
        mul_336: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = None
        unsqueeze_192: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_294, -1)
        unsqueeze_193: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        mul_342: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_336, unsqueeze_193);  mul_336 = unsqueeze_193 = None
        unsqueeze_194: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_295, -1);  primals_295 = None
        unsqueeze_195: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        add_244: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_342, unsqueeze_195);  mul_342 = unsqueeze_195 = None
        relu_31: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_244);  add_244 = None
        le_5: "b8[512, 232, 7, 7]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_5: "f32[512, 232, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, slice_6);  le_5 = slice_6 = None
        squeeze_144: "f32[232]" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        unsqueeze_310: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(squeeze_144, 0);  squeeze_144 = None
        unsqueeze_311: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
        unsqueeze_312: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
        sum_16: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_84: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_312);  convolution_48 = unsqueeze_312 = None
        mul_455: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_84)
        sum_17: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 2, 3]);  mul_455 = None
        mul_456: "f32[232]" = torch.ops.aten.mul.Tensor(sum_16, 3.985969387755102e-05)
        unsqueeze_313: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
        unsqueeze_314: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
        unsqueeze_315: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
        mul_457: "f32[232]" = torch.ops.aten.mul.Tensor(sum_17, 3.985969387755102e-05)
        squeeze_145: "f32[232]" = torch.ops.aten.squeeze.dims(rsqrt_48, [0, 2, 3]);  rsqrt_48 = None
        mul_458: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_459: "f32[232]" = torch.ops.aten.mul.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
        unsqueeze_316: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
        unsqueeze_317: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 2);  unsqueeze_316 = None
        unsqueeze_318: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 3);  unsqueeze_317 = None
        mul_460: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_294);  primals_294 = None
        unsqueeze_319: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_460, 0);  mul_460 = None
        unsqueeze_320: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
        unsqueeze_321: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
        mul_461: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_318);  sub_84 = unsqueeze_318 = None
        sub_86: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_461);  where_5 = mul_461 = None
        sub_87: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_315);  sub_86 = unsqueeze_315 = None
        mul_462: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_321);  sub_87 = unsqueeze_321 = None
        mul_463: "f32[232]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_145);  sum_17 = squeeze_145 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_462, add_239, primals_290, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_462 = add_239 = primals_290 = None
        getitem_161: "f32[512, 232, 7, 7]" = convolution_backward_7[0]
        getitem_162: "f32[232, 232, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
        sum_18: "f32[232]" = torch.ops.aten.sum.dim_IntList(getitem_161, [0, 2, 3])
        sub_88: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_324);  convolution_47 = unsqueeze_324 = None
        mul_464: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_161, sub_88)
        sum_19: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3]);  mul_464 = None
        mul_465: "f32[232]" = torch.ops.aten.mul.Tensor(sum_18, 3.985969387755102e-05)
        unsqueeze_325: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
        unsqueeze_326: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
        unsqueeze_327: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
        mul_466: "f32[232]" = torch.ops.aten.mul.Tensor(sum_19, 3.985969387755102e-05)
        mul_467: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_468: "f32[232]" = torch.ops.aten.mul.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        unsqueeze_328: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
        unsqueeze_329: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 2);  unsqueeze_328 = None
        unsqueeze_330: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 3);  unsqueeze_329 = None
        mul_469: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_288);  primals_288 = None
        unsqueeze_331: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
        unsqueeze_332: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
        unsqueeze_333: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
        mul_470: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_330);  sub_88 = unsqueeze_330 = None
        sub_90: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_161, mul_470);  getitem_161 = mul_470 = None
        sub_91: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_327);  sub_90 = unsqueeze_327 = None
        mul_471: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_333);  sub_91 = unsqueeze_333 = None
        mul_472: "f32[232]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_142);  sum_19 = squeeze_142 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_471, relu_30, primals_284, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  mul_471 = primals_284 = None
        getitem_164: "f32[512, 232, 7, 7]" = convolution_backward_8[0]
        getitem_165: "f32[232, 1, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
        le_6: "b8[512, 232, 7, 7]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_6: "f32[512, 232, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, getitem_164);  le_6 = getitem_164 = None
        sum_20: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_92: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_336);  convolution_46 = unsqueeze_336 = None
        mul_473: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_92)
        sum_21: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3]);  mul_473 = None
        mul_474: "f32[232]" = torch.ops.aten.mul.Tensor(sum_20, 3.985969387755102e-05)
        unsqueeze_337: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
        unsqueeze_338: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
        unsqueeze_339: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
        mul_475: "f32[232]" = torch.ops.aten.mul.Tensor(sum_21, 3.985969387755102e-05)
        mul_476: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_477: "f32[232]" = torch.ops.aten.mul.Tensor(mul_475, mul_476);  mul_475 = mul_476 = None
        unsqueeze_340: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
        unsqueeze_341: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 2);  unsqueeze_340 = None
        unsqueeze_342: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 3);  unsqueeze_341 = None
        mul_478: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_282);  primals_282 = None
        unsqueeze_343: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
        unsqueeze_344: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
        unsqueeze_345: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
        mul_479: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_342);  sub_92 = unsqueeze_342 = None
        sub_94: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_479);  where_6 = mul_479 = None
        sub_95: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_339);  sub_94 = unsqueeze_339 = None
        mul_480: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_345);  sub_95 = unsqueeze_345 = None
        mul_481: "f32[232]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_139);  sum_21 = squeeze_139 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_480, getitem_115, primals_278, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_480 = getitem_115 = primals_278 = None
        getitem_167: "f32[512, 232, 7, 7]" = convolution_backward_9[0]
        getitem_168: "f32[232, 232, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_18: "f32[512, 464, 7, 7]" = torch.ops.aten.cat.default([slice_5, getitem_167], 1);  slice_5 = getitem_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_39: "f32[512, 232, 2, 7, 7]" = torch.ops.aten.reshape.default(cat_18, [512, 232, 2, 7, 7]);  cat_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_24: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3, 4]);  view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_19: "f32[512, 2, 232, 7, 7]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_40: "f32[512, 464, 7, 7]" = torch.ops.aten.reshape.default(clone_19, [512, 464, 7, 7]);  clone_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        slice_7: "f32[512, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_40, 1, 0, 232)
        slice_8: "f32[512, 232, 7, 7]" = torch.ops.aten.slice.Tensor(view_40, 1, 232, 464);  view_40 = None
        sub_45: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, getitem_113)
        mul_315: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = None
        unsqueeze_180: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_276, -1)
        unsqueeze_181: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_321: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_315, unsqueeze_181);  mul_315 = unsqueeze_181 = None
        unsqueeze_182: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_277, -1);  primals_277 = None
        unsqueeze_183: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_229: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_321, unsqueeze_183);  mul_321 = unsqueeze_183 = None
        relu_29: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_229);  add_229 = None
        le_7: "b8[512, 232, 7, 7]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_7: "f32[512, 232, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, slice_8);  le_7 = slice_8 = None
        squeeze_135: "f32[232]" = torch.ops.aten.squeeze.dims(getitem_113, [0, 2, 3]);  getitem_113 = None
        unsqueeze_346: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(squeeze_135, 0);  squeeze_135 = None
        unsqueeze_347: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
        unsqueeze_348: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
        sum_22: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_96: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_348);  convolution_45 = unsqueeze_348 = None
        mul_482: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_96)
        sum_23: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3]);  mul_482 = None
        mul_483: "f32[232]" = torch.ops.aten.mul.Tensor(sum_22, 3.985969387755102e-05)
        unsqueeze_349: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
        unsqueeze_350: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
        unsqueeze_351: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
        mul_484: "f32[232]" = torch.ops.aten.mul.Tensor(sum_23, 3.985969387755102e-05)
        squeeze_136: "f32[232]" = torch.ops.aten.squeeze.dims(rsqrt_45, [0, 2, 3]);  rsqrt_45 = None
        mul_485: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_486: "f32[232]" = torch.ops.aten.mul.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
        unsqueeze_352: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
        unsqueeze_353: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 2);  unsqueeze_352 = None
        unsqueeze_354: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 3);  unsqueeze_353 = None
        mul_487: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_276);  primals_276 = None
        unsqueeze_355: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
        unsqueeze_356: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
        unsqueeze_357: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
        mul_488: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_354);  sub_96 = unsqueeze_354 = None
        sub_98: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_488);  where_7 = mul_488 = None
        sub_99: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_351);  sub_98 = unsqueeze_351 = None
        mul_489: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_357);  sub_99 = unsqueeze_357 = None
        mul_490: "f32[232]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_136);  sum_23 = squeeze_136 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_489, add_224, primals_272, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_489 = add_224 = primals_272 = None
        getitem_170: "f32[512, 232, 7, 7]" = convolution_backward_10[0]
        getitem_171: "f32[232, 232, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
        sum_24: "f32[232]" = torch.ops.aten.sum.dim_IntList(getitem_170, [0, 2, 3])
        sub_100: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_360);  convolution_44 = unsqueeze_360 = None
        mul_491: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_170, sub_100)
        sum_25: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
        mul_492: "f32[232]" = torch.ops.aten.mul.Tensor(sum_24, 3.985969387755102e-05)
        unsqueeze_361: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
        unsqueeze_362: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
        unsqueeze_363: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
        mul_493: "f32[232]" = torch.ops.aten.mul.Tensor(sum_25, 3.985969387755102e-05)
        mul_494: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_495: "f32[232]" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
        unsqueeze_364: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
        unsqueeze_365: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 2);  unsqueeze_364 = None
        unsqueeze_366: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 3);  unsqueeze_365 = None
        mul_496: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_270);  primals_270 = None
        unsqueeze_367: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
        unsqueeze_368: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
        unsqueeze_369: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
        mul_497: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_366);  sub_100 = unsqueeze_366 = None
        sub_102: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_170, mul_497);  getitem_170 = mul_497 = None
        sub_103: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_363);  sub_102 = unsqueeze_363 = None
        mul_498: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_369);  sub_103 = unsqueeze_369 = None
        mul_499: "f32[232]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_133);  sum_25 = squeeze_133 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_498, relu_28, primals_266, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  mul_498 = primals_266 = None
        getitem_173: "f32[512, 232, 14, 14]" = convolution_backward_11[0]
        getitem_174: "f32[232, 1, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
        le_8: "b8[512, 232, 14, 14]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_8: "f32[512, 232, 14, 14]" = torch.ops.aten.where.self(le_8, full_default, getitem_173);  le_8 = getitem_173 = None
        sum_26: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_104: "f32[512, 232, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_372);  convolution_43 = unsqueeze_372 = None
        mul_500: "f32[512, 232, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_104)
        sum_27: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
        mul_501: "f32[232]" = torch.ops.aten.mul.Tensor(sum_26, 9.964923469387754e-06)
        unsqueeze_373: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
        unsqueeze_374: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
        unsqueeze_375: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
        mul_502: "f32[232]" = torch.ops.aten.mul.Tensor(sum_27, 9.964923469387754e-06)
        mul_503: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_504: "f32[232]" = torch.ops.aten.mul.Tensor(mul_502, mul_503);  mul_502 = mul_503 = None
        unsqueeze_376: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
        unsqueeze_377: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 2);  unsqueeze_376 = None
        unsqueeze_378: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 3);  unsqueeze_377 = None
        mul_505: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_264);  primals_264 = None
        unsqueeze_379: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
        unsqueeze_380: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
        unsqueeze_381: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
        mul_506: "f32[512, 232, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_378);  sub_104 = unsqueeze_378 = None
        sub_106: "f32[512, 232, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_506);  where_8 = mul_506 = None
        sub_107: "f32[512, 232, 14, 14]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_375);  sub_106 = unsqueeze_375 = None
        mul_507: "f32[512, 232, 14, 14]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_381);  sub_107 = unsqueeze_381 = None
        mul_508: "f32[232]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_130);  sum_27 = squeeze_130 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_507, view_23, primals_260, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_507 = primals_260 = None
        getitem_176: "f32[512, 232, 14, 14]" = convolution_backward_12[0]
        getitem_177: "f32[232, 232, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
        sub_42: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, getitem_107)
        mul_294: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        unsqueeze_168: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_258, -1)
        unsqueeze_169: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        mul_300: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(mul_294, unsqueeze_169);  mul_294 = unsqueeze_169 = None
        unsqueeze_170: "f32[232, 1]" = torch.ops.aten.unsqueeze.default(primals_259, -1);  primals_259 = None
        unsqueeze_171: "f32[232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        add_214: "f32[512, 232, 7, 7]" = torch.ops.aten.add.Tensor(mul_300, unsqueeze_171);  mul_300 = unsqueeze_171 = None
        relu_27: "f32[512, 232, 7, 7]" = torch.ops.aten.relu.default(add_214);  add_214 = None
        le_9: "b8[512, 232, 7, 7]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_9: "f32[512, 232, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, slice_7);  le_9 = slice_7 = None
        squeeze_126: "f32[232]" = torch.ops.aten.squeeze.dims(getitem_107, [0, 2, 3]);  getitem_107 = None
        unsqueeze_382: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(squeeze_126, 0);  squeeze_126 = None
        unsqueeze_383: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
        unsqueeze_384: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
        sum_28: "f32[232]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_108: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_384);  convolution_42 = unsqueeze_384 = None
        mul_509: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_108)
        sum_29: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 2, 3]);  mul_509 = None
        mul_510: "f32[232]" = torch.ops.aten.mul.Tensor(sum_28, 3.985969387755102e-05)
        unsqueeze_385: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
        unsqueeze_386: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
        unsqueeze_387: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
        mul_511: "f32[232]" = torch.ops.aten.mul.Tensor(sum_29, 3.985969387755102e-05)
        squeeze_127: "f32[232]" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_512: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_513: "f32[232]" = torch.ops.aten.mul.Tensor(mul_511, mul_512);  mul_511 = mul_512 = None
        unsqueeze_388: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
        unsqueeze_389: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 2);  unsqueeze_388 = None
        unsqueeze_390: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 3);  unsqueeze_389 = None
        mul_514: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_258);  primals_258 = None
        unsqueeze_391: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
        unsqueeze_392: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
        unsqueeze_393: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
        mul_515: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_390);  sub_108 = unsqueeze_390 = None
        sub_110: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_515);  where_9 = mul_515 = None
        sub_111: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_387);  sub_110 = unsqueeze_387 = None
        mul_516: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_393);  sub_111 = unsqueeze_393 = None
        mul_517: "f32[232]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_127);  sum_29 = squeeze_127 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_516, add_209, primals_254, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_516 = add_209 = primals_254 = None
        getitem_179: "f32[512, 232, 7, 7]" = convolution_backward_13[0]
        getitem_180: "f32[232, 232, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
        sum_30: "f32[232]" = torch.ops.aten.sum.dim_IntList(getitem_179, [0, 2, 3])
        sub_112: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_396);  convolution_41 = unsqueeze_396 = None
        mul_518: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_179, sub_112)
        sum_31: "f32[232]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3]);  mul_518 = None
        mul_519: "f32[232]" = torch.ops.aten.mul.Tensor(sum_30, 3.985969387755102e-05)
        unsqueeze_397: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
        unsqueeze_398: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
        unsqueeze_399: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
        mul_520: "f32[232]" = torch.ops.aten.mul.Tensor(sum_31, 3.985969387755102e-05)
        mul_521: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_522: "f32[232]" = torch.ops.aten.mul.Tensor(mul_520, mul_521);  mul_520 = mul_521 = None
        unsqueeze_400: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
        unsqueeze_401: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 2);  unsqueeze_400 = None
        unsqueeze_402: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 3);  unsqueeze_401 = None
        mul_523: "f32[232]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_252);  primals_252 = None
        unsqueeze_403: "f32[1, 232]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
        unsqueeze_404: "f32[1, 232, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
        unsqueeze_405: "f32[1, 232, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
        mul_524: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_402);  sub_112 = unsqueeze_402 = None
        sub_114: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_179, mul_524);  getitem_179 = mul_524 = None
        sub_115: "f32[512, 232, 7, 7]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_399);  sub_114 = unsqueeze_399 = None
        mul_525: "f32[512, 232, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_405);  sub_115 = unsqueeze_405 = None
        mul_526: "f32[232]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_124);  sum_31 = squeeze_124 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_525, view_23, primals_248, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 232, [True, True, False]);  mul_525 = view_23 = primals_248 = None
        getitem_182: "f32[512, 232, 14, 14]" = convolution_backward_14[0]
        getitem_183: "f32[232, 1, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
        add_280: "f32[512, 232, 14, 14]" = torch.ops.aten.add.Tensor(getitem_176, getitem_182);  getitem_176 = getitem_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_41: "f32[512, 116, 2, 14, 14]" = torch.ops.aten.reshape.default(add_280, [512, 116, 2, 14, 14]);  add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_25: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_41, [0, 2, 1, 3, 4]);  view_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_20: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_25, memory_format = torch.contiguous_format);  permute_25 = None
        view_42: "f32[512, 232, 14, 14]" = torch.ops.aten.reshape.default(clone_20, [512, 232, 14, 14]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_9: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_42, 1, 0, 116)
        slice_10: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_42, 1, 116, 232);  view_42 = None
        sub_40: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, getitem_103)
        mul_280: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        unsqueeze_160: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_246, -1)
        unsqueeze_161: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        mul_286: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_161);  mul_280 = unsqueeze_161 = None
        unsqueeze_162: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_247, -1);  primals_247 = None
        unsqueeze_163: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        add_204: "f32[512, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_286, unsqueeze_163);  mul_286 = unsqueeze_163 = None
        relu_26: "f32[512, 116, 14, 14]" = torch.ops.aten.relu.default(add_204);  add_204 = None
        le_10: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_10: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, slice_10);  le_10 = slice_10 = None
        squeeze_120: "f32[116]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        unsqueeze_406: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(squeeze_120, 0);  squeeze_120 = None
        unsqueeze_407: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
        unsqueeze_408: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
        sum_32: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_116: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_408);  convolution_40 = unsqueeze_408 = None
        mul_527: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_116)
        sum_33: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3]);  mul_527 = None
        mul_528: "f32[116]" = torch.ops.aten.mul.Tensor(sum_32, 9.964923469387754e-06)
        unsqueeze_409: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
        unsqueeze_410: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
        unsqueeze_411: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
        mul_529: "f32[116]" = torch.ops.aten.mul.Tensor(sum_33, 9.964923469387754e-06)
        squeeze_121: "f32[116]" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_530: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_531: "f32[116]" = torch.ops.aten.mul.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
        unsqueeze_412: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
        unsqueeze_413: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 2);  unsqueeze_412 = None
        unsqueeze_414: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 3);  unsqueeze_413 = None
        mul_532: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_246);  primals_246 = None
        unsqueeze_415: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
        unsqueeze_416: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
        unsqueeze_417: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
        mul_533: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_414);  sub_116 = unsqueeze_414 = None
        sub_118: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_533);  where_10 = mul_533 = None
        sub_119: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_411);  sub_118 = unsqueeze_411 = None
        mul_534: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_417);  sub_119 = unsqueeze_417 = None
        mul_535: "f32[116]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_121);  sum_33 = squeeze_121 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_534, add_199, primals_242, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_534 = add_199 = primals_242 = None
        getitem_185: "f32[512, 116, 14, 14]" = convolution_backward_15[0]
        getitem_186: "f32[116, 116, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
        sum_34: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_185, [0, 2, 3])
        sub_120: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_420);  convolution_39 = unsqueeze_420 = None
        mul_536: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_185, sub_120)
        sum_35: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2, 3]);  mul_536 = None
        mul_537: "f32[116]" = torch.ops.aten.mul.Tensor(sum_34, 9.964923469387754e-06)
        unsqueeze_421: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
        unsqueeze_422: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
        unsqueeze_423: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
        mul_538: "f32[116]" = torch.ops.aten.mul.Tensor(sum_35, 9.964923469387754e-06)
        mul_539: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_540: "f32[116]" = torch.ops.aten.mul.Tensor(mul_538, mul_539);  mul_538 = mul_539 = None
        unsqueeze_424: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
        unsqueeze_425: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 2);  unsqueeze_424 = None
        unsqueeze_426: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 3);  unsqueeze_425 = None
        mul_541: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_240);  primals_240 = None
        unsqueeze_427: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
        unsqueeze_428: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
        unsqueeze_429: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
        mul_542: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_426);  sub_120 = unsqueeze_426 = None
        sub_122: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_185, mul_542);  getitem_185 = mul_542 = None
        sub_123: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_423);  sub_122 = unsqueeze_423 = None
        mul_543: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_429);  sub_123 = unsqueeze_429 = None
        mul_544: "f32[116]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_118);  sum_35 = squeeze_118 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_543, relu_25, primals_236, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_543 = primals_236 = None
        getitem_188: "f32[512, 116, 14, 14]" = convolution_backward_16[0]
        getitem_189: "f32[116, 1, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
        le_11: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_11: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, getitem_188);  le_11 = getitem_188 = None
        sum_36: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_124: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_432);  convolution_38 = unsqueeze_432 = None
        mul_545: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_124)
        sum_37: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 2, 3]);  mul_545 = None
        mul_546: "f32[116]" = torch.ops.aten.mul.Tensor(sum_36, 9.964923469387754e-06)
        unsqueeze_433: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
        unsqueeze_434: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
        unsqueeze_435: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
        mul_547: "f32[116]" = torch.ops.aten.mul.Tensor(sum_37, 9.964923469387754e-06)
        mul_548: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_549: "f32[116]" = torch.ops.aten.mul.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
        unsqueeze_436: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
        unsqueeze_437: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 2);  unsqueeze_436 = None
        unsqueeze_438: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 3);  unsqueeze_437 = None
        mul_550: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_234);  primals_234 = None
        unsqueeze_439: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
        unsqueeze_440: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
        unsqueeze_441: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
        mul_551: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_438);  sub_124 = unsqueeze_438 = None
        sub_126: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_551);  where_11 = mul_551 = None
        sub_127: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_435);  sub_126 = unsqueeze_435 = None
        mul_552: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_441);  sub_127 = unsqueeze_441 = None
        mul_553: "f32[116]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_115);  sum_37 = squeeze_115 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_552, getitem_97, primals_230, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_552 = getitem_97 = primals_230 = None
        getitem_191: "f32[512, 116, 14, 14]" = convolution_backward_17[0]
        getitem_192: "f32[116, 116, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_19: "f32[512, 232, 14, 14]" = torch.ops.aten.cat.default([slice_9, getitem_191], 1);  slice_9 = getitem_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_43: "f32[512, 116, 2, 14, 14]" = torch.ops.aten.reshape.default(cat_19, [512, 116, 2, 14, 14]);  cat_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_26: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3, 4]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_21: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_44: "f32[512, 232, 14, 14]" = torch.ops.aten.reshape.default(clone_21, [512, 232, 14, 14]);  clone_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_11: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_44, 1, 0, 116)
        slice_12: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_44, 1, 116, 232);  view_44 = None
        sub_37: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_95)
        mul_259: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        unsqueeze_148: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_149: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_151: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_189: "f32[512, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None
        relu_24: "f32[512, 116, 14, 14]" = torch.ops.aten.relu.default(add_189);  add_189 = None
        le_12: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_12: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, slice_12);  le_12 = slice_12 = None
        squeeze_111: "f32[116]" = torch.ops.aten.squeeze.dims(getitem_95, [0, 2, 3]);  getitem_95 = None
        unsqueeze_442: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_443: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
        unsqueeze_444: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
        sum_38: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_128: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_444);  convolution_37 = unsqueeze_444 = None
        mul_554: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_128)
        sum_39: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 2, 3]);  mul_554 = None
        mul_555: "f32[116]" = torch.ops.aten.mul.Tensor(sum_38, 9.964923469387754e-06)
        unsqueeze_445: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
        unsqueeze_446: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
        unsqueeze_447: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
        mul_556: "f32[116]" = torch.ops.aten.mul.Tensor(sum_39, 9.964923469387754e-06)
        squeeze_112: "f32[116]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_557: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_558: "f32[116]" = torch.ops.aten.mul.Tensor(mul_556, mul_557);  mul_556 = mul_557 = None
        unsqueeze_448: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
        unsqueeze_449: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 2);  unsqueeze_448 = None
        unsqueeze_450: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 3);  unsqueeze_449 = None
        mul_559: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_228);  primals_228 = None
        unsqueeze_451: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
        unsqueeze_452: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
        unsqueeze_453: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
        mul_560: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_450);  sub_128 = unsqueeze_450 = None
        sub_130: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_560);  where_12 = mul_560 = None
        sub_131: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_447);  sub_130 = unsqueeze_447 = None
        mul_561: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_453);  sub_131 = unsqueeze_453 = None
        mul_562: "f32[116]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_112);  sum_39 = squeeze_112 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_561, add_184, primals_224, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_561 = add_184 = primals_224 = None
        getitem_194: "f32[512, 116, 14, 14]" = convolution_backward_18[0]
        getitem_195: "f32[116, 116, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
        sum_40: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_194, [0, 2, 3])
        sub_132: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_456);  convolution_36 = unsqueeze_456 = None
        mul_563: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_194, sub_132)
        sum_41: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3]);  mul_563 = None
        mul_564: "f32[116]" = torch.ops.aten.mul.Tensor(sum_40, 9.964923469387754e-06)
        unsqueeze_457: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
        unsqueeze_458: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
        unsqueeze_459: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
        mul_565: "f32[116]" = torch.ops.aten.mul.Tensor(sum_41, 9.964923469387754e-06)
        mul_566: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_567: "f32[116]" = torch.ops.aten.mul.Tensor(mul_565, mul_566);  mul_565 = mul_566 = None
        unsqueeze_460: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
        unsqueeze_461: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 2);  unsqueeze_460 = None
        unsqueeze_462: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 3);  unsqueeze_461 = None
        mul_568: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_222);  primals_222 = None
        unsqueeze_463: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
        unsqueeze_464: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
        unsqueeze_465: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
        mul_569: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_462);  sub_132 = unsqueeze_462 = None
        sub_134: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_194, mul_569);  getitem_194 = mul_569 = None
        sub_135: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_459);  sub_134 = unsqueeze_459 = None
        mul_570: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_465);  sub_135 = unsqueeze_465 = None
        mul_571: "f32[116]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_109);  sum_41 = squeeze_109 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_570, relu_23, primals_218, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_570 = primals_218 = None
        getitem_197: "f32[512, 116, 14, 14]" = convolution_backward_19[0]
        getitem_198: "f32[116, 1, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
        le_13: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_13: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, getitem_197);  le_13 = getitem_197 = None
        sum_42: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_136: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_468);  convolution_35 = unsqueeze_468 = None
        mul_572: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_136)
        sum_43: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 2, 3]);  mul_572 = None
        mul_573: "f32[116]" = torch.ops.aten.mul.Tensor(sum_42, 9.964923469387754e-06)
        unsqueeze_469: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
        unsqueeze_470: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
        unsqueeze_471: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
        mul_574: "f32[116]" = torch.ops.aten.mul.Tensor(sum_43, 9.964923469387754e-06)
        mul_575: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_576: "f32[116]" = torch.ops.aten.mul.Tensor(mul_574, mul_575);  mul_574 = mul_575 = None
        unsqueeze_472: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
        unsqueeze_473: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 2);  unsqueeze_472 = None
        unsqueeze_474: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 3);  unsqueeze_473 = None
        mul_577: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_216);  primals_216 = None
        unsqueeze_475: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
        unsqueeze_476: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
        unsqueeze_477: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
        mul_578: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_474);  sub_136 = unsqueeze_474 = None
        sub_138: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_578);  where_13 = mul_578 = None
        sub_139: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_471);  sub_138 = unsqueeze_471 = None
        mul_579: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_477);  sub_139 = unsqueeze_477 = None
        mul_580: "f32[116]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_106);  sum_43 = squeeze_106 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_579, getitem_89, primals_212, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_579 = getitem_89 = primals_212 = None
        getitem_200: "f32[512, 116, 14, 14]" = convolution_backward_20[0]
        getitem_201: "f32[116, 116, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_20: "f32[512, 232, 14, 14]" = torch.ops.aten.cat.default([slice_11, getitem_200], 1);  slice_11 = getitem_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_45: "f32[512, 116, 2, 14, 14]" = torch.ops.aten.reshape.default(cat_20, [512, 116, 2, 14, 14]);  cat_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_27: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3, 4]);  view_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_22: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_46: "f32[512, 232, 14, 14]" = torch.ops.aten.reshape.default(clone_22, [512, 232, 14, 14]);  clone_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_13: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_46, 1, 0, 116)
        slice_14: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_46, 1, 116, 232);  view_46 = None
        sub_34: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, getitem_87)
        mul_238: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = None
        unsqueeze_136: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_210, -1)
        unsqueeze_137: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        mul_244: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_137);  mul_238 = unsqueeze_137 = None
        unsqueeze_138: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_211, -1);  primals_211 = None
        unsqueeze_139: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        add_174: "f32[512, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_244, unsqueeze_139);  mul_244 = unsqueeze_139 = None
        relu_22: "f32[512, 116, 14, 14]" = torch.ops.aten.relu.default(add_174);  add_174 = None
        le_14: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_14: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, slice_14);  le_14 = slice_14 = None
        squeeze_102: "f32[116]" = torch.ops.aten.squeeze.dims(getitem_87, [0, 2, 3]);  getitem_87 = None
        unsqueeze_478: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(squeeze_102, 0);  squeeze_102 = None
        unsqueeze_479: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
        unsqueeze_480: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
        sum_44: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_140: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_480);  convolution_34 = unsqueeze_480 = None
        mul_581: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_140)
        sum_45: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 2, 3]);  mul_581 = None
        mul_582: "f32[116]" = torch.ops.aten.mul.Tensor(sum_44, 9.964923469387754e-06)
        unsqueeze_481: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
        unsqueeze_482: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
        unsqueeze_483: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
        mul_583: "f32[116]" = torch.ops.aten.mul.Tensor(sum_45, 9.964923469387754e-06)
        squeeze_103: "f32[116]" = torch.ops.aten.squeeze.dims(rsqrt_34, [0, 2, 3]);  rsqrt_34 = None
        mul_584: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_585: "f32[116]" = torch.ops.aten.mul.Tensor(mul_583, mul_584);  mul_583 = mul_584 = None
        unsqueeze_484: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
        unsqueeze_485: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 2);  unsqueeze_484 = None
        unsqueeze_486: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 3);  unsqueeze_485 = None
        mul_586: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_210);  primals_210 = None
        unsqueeze_487: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
        unsqueeze_488: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
        unsqueeze_489: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
        mul_587: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_486);  sub_140 = unsqueeze_486 = None
        sub_142: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_587);  where_14 = mul_587 = None
        sub_143: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_483);  sub_142 = unsqueeze_483 = None
        mul_588: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_489);  sub_143 = unsqueeze_489 = None
        mul_589: "f32[116]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_103);  sum_45 = squeeze_103 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_588, add_169, primals_206, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_588 = add_169 = primals_206 = None
        getitem_203: "f32[512, 116, 14, 14]" = convolution_backward_21[0]
        getitem_204: "f32[116, 116, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
        sum_46: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_203, [0, 2, 3])
        sub_144: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_492);  convolution_33 = unsqueeze_492 = None
        mul_590: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_203, sub_144)
        sum_47: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 2, 3]);  mul_590 = None
        mul_591: "f32[116]" = torch.ops.aten.mul.Tensor(sum_46, 9.964923469387754e-06)
        unsqueeze_493: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
        unsqueeze_494: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
        unsqueeze_495: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
        mul_592: "f32[116]" = torch.ops.aten.mul.Tensor(sum_47, 9.964923469387754e-06)
        mul_593: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_594: "f32[116]" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
        unsqueeze_496: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
        unsqueeze_497: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 2);  unsqueeze_496 = None
        unsqueeze_498: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 3);  unsqueeze_497 = None
        mul_595: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_204);  primals_204 = None
        unsqueeze_499: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
        unsqueeze_500: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
        unsqueeze_501: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
        mul_596: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_498);  sub_144 = unsqueeze_498 = None
        sub_146: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_203, mul_596);  getitem_203 = mul_596 = None
        sub_147: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_495);  sub_146 = unsqueeze_495 = None
        mul_597: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_501);  sub_147 = unsqueeze_501 = None
        mul_598: "f32[116]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_100);  sum_47 = squeeze_100 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_597, relu_21, primals_200, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_597 = primals_200 = None
        getitem_206: "f32[512, 116, 14, 14]" = convolution_backward_22[0]
        getitem_207: "f32[116, 1, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
        le_15: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_15: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, getitem_206);  le_15 = getitem_206 = None
        sum_48: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_148: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_504);  convolution_32 = unsqueeze_504 = None
        mul_599: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_148)
        sum_49: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3]);  mul_599 = None
        mul_600: "f32[116]" = torch.ops.aten.mul.Tensor(sum_48, 9.964923469387754e-06)
        unsqueeze_505: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
        unsqueeze_506: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
        unsqueeze_507: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
        mul_601: "f32[116]" = torch.ops.aten.mul.Tensor(sum_49, 9.964923469387754e-06)
        mul_602: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_603: "f32[116]" = torch.ops.aten.mul.Tensor(mul_601, mul_602);  mul_601 = mul_602 = None
        unsqueeze_508: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
        unsqueeze_509: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 2);  unsqueeze_508 = None
        unsqueeze_510: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 3);  unsqueeze_509 = None
        mul_604: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_198);  primals_198 = None
        unsqueeze_511: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
        unsqueeze_512: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
        unsqueeze_513: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
        mul_605: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_510);  sub_148 = unsqueeze_510 = None
        sub_150: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_605);  where_15 = mul_605 = None
        sub_151: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_507);  sub_150 = unsqueeze_507 = None
        mul_606: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_513);  sub_151 = unsqueeze_513 = None
        mul_607: "f32[116]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_97);  sum_49 = squeeze_97 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_606, getitem_81, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_606 = getitem_81 = primals_194 = None
        getitem_209: "f32[512, 116, 14, 14]" = convolution_backward_23[0]
        getitem_210: "f32[116, 116, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_21: "f32[512, 232, 14, 14]" = torch.ops.aten.cat.default([slice_13, getitem_209], 1);  slice_13 = getitem_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_47: "f32[512, 116, 2, 14, 14]" = torch.ops.aten.reshape.default(cat_21, [512, 116, 2, 14, 14]);  cat_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_28: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3, 4]);  view_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_23: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_48: "f32[512, 232, 14, 14]" = torch.ops.aten.reshape.default(clone_23, [512, 232, 14, 14]);  clone_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_15: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_48, 1, 0, 116)
        slice_16: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_48, 1, 116, 232);  view_48 = None
        sub_31: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_79)
        mul_217: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        unsqueeze_124: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_125: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_193, -1);  primals_193 = None
        unsqueeze_127: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_159: "f32[512, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None
        relu_20: "f32[512, 116, 14, 14]" = torch.ops.aten.relu.default(add_159);  add_159 = None
        le_16: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_16: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, slice_16);  le_16 = slice_16 = None
        squeeze_93: "f32[116]" = torch.ops.aten.squeeze.dims(getitem_79, [0, 2, 3]);  getitem_79 = None
        unsqueeze_514: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_515: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
        unsqueeze_516: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
        sum_50: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_152: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_516);  convolution_31 = unsqueeze_516 = None
        mul_608: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_152)
        sum_51: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
        mul_609: "f32[116]" = torch.ops.aten.mul.Tensor(sum_50, 9.964923469387754e-06)
        unsqueeze_517: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
        unsqueeze_518: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
        unsqueeze_519: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
        mul_610: "f32[116]" = torch.ops.aten.mul.Tensor(sum_51, 9.964923469387754e-06)
        squeeze_94: "f32[116]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_611: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_612: "f32[116]" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
        unsqueeze_520: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
        unsqueeze_521: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 2);  unsqueeze_520 = None
        unsqueeze_522: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 3);  unsqueeze_521 = None
        mul_613: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_192);  primals_192 = None
        unsqueeze_523: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
        unsqueeze_524: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
        unsqueeze_525: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
        mul_614: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_522);  sub_152 = unsqueeze_522 = None
        sub_154: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_614);  where_16 = mul_614 = None
        sub_155: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_519);  sub_154 = unsqueeze_519 = None
        mul_615: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_525);  sub_155 = unsqueeze_525 = None
        mul_616: "f32[116]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_94);  sum_51 = squeeze_94 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_615, add_154, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_615 = add_154 = primals_188 = None
        getitem_212: "f32[512, 116, 14, 14]" = convolution_backward_24[0]
        getitem_213: "f32[116, 116, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
        sum_52: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_212, [0, 2, 3])
        sub_156: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_528);  convolution_30 = unsqueeze_528 = None
        mul_617: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_212, sub_156)
        sum_53: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3]);  mul_617 = None
        mul_618: "f32[116]" = torch.ops.aten.mul.Tensor(sum_52, 9.964923469387754e-06)
        unsqueeze_529: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
        unsqueeze_530: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
        unsqueeze_531: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
        mul_619: "f32[116]" = torch.ops.aten.mul.Tensor(sum_53, 9.964923469387754e-06)
        mul_620: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_621: "f32[116]" = torch.ops.aten.mul.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
        unsqueeze_532: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
        unsqueeze_533: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 2);  unsqueeze_532 = None
        unsqueeze_534: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 3);  unsqueeze_533 = None
        mul_622: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_186);  primals_186 = None
        unsqueeze_535: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
        unsqueeze_536: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
        unsqueeze_537: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
        mul_623: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_534);  sub_156 = unsqueeze_534 = None
        sub_158: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_212, mul_623);  getitem_212 = mul_623 = None
        sub_159: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_531);  sub_158 = unsqueeze_531 = None
        mul_624: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_537);  sub_159 = unsqueeze_537 = None
        mul_625: "f32[116]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_91);  sum_53 = squeeze_91 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_624, relu_19, primals_182, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_624 = primals_182 = None
        getitem_215: "f32[512, 116, 14, 14]" = convolution_backward_25[0]
        getitem_216: "f32[116, 1, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
        le_17: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_17: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, getitem_215);  le_17 = getitem_215 = None
        sum_54: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_160: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_540);  convolution_29 = unsqueeze_540 = None
        mul_626: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_160)
        sum_55: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 2, 3]);  mul_626 = None
        mul_627: "f32[116]" = torch.ops.aten.mul.Tensor(sum_54, 9.964923469387754e-06)
        unsqueeze_541: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
        unsqueeze_542: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
        unsqueeze_543: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
        mul_628: "f32[116]" = torch.ops.aten.mul.Tensor(sum_55, 9.964923469387754e-06)
        mul_629: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_630: "f32[116]" = torch.ops.aten.mul.Tensor(mul_628, mul_629);  mul_628 = mul_629 = None
        unsqueeze_544: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
        unsqueeze_545: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 2);  unsqueeze_544 = None
        unsqueeze_546: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 3);  unsqueeze_545 = None
        mul_631: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_180);  primals_180 = None
        unsqueeze_547: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
        unsqueeze_548: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
        unsqueeze_549: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
        mul_632: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_546);  sub_160 = unsqueeze_546 = None
        sub_162: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_632);  where_17 = mul_632 = None
        sub_163: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_543);  sub_162 = unsqueeze_543 = None
        mul_633: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_549);  sub_163 = unsqueeze_549 = None
        mul_634: "f32[116]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_88);  sum_55 = squeeze_88 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_633, getitem_73, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_633 = getitem_73 = primals_176 = None
        getitem_218: "f32[512, 116, 14, 14]" = convolution_backward_26[0]
        getitem_219: "f32[116, 116, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_22: "f32[512, 232, 14, 14]" = torch.ops.aten.cat.default([slice_15, getitem_218], 1);  slice_15 = getitem_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_49: "f32[512, 116, 2, 14, 14]" = torch.ops.aten.reshape.default(cat_22, [512, 116, 2, 14, 14]);  cat_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_29: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3, 4]);  view_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_24: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_50: "f32[512, 232, 14, 14]" = torch.ops.aten.reshape.default(clone_24, [512, 232, 14, 14]);  clone_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_17: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_50, 1, 0, 116)
        slice_18: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_50, 1, 116, 232);  view_50 = None
        sub_28: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, getitem_71)
        mul_196: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = None
        unsqueeze_112: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_174, -1)
        unsqueeze_113: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_202: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_113);  mul_196 = unsqueeze_113 = None
        unsqueeze_114: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_175, -1);  primals_175 = None
        unsqueeze_115: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_144: "f32[512, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_115);  mul_202 = unsqueeze_115 = None
        relu_18: "f32[512, 116, 14, 14]" = torch.ops.aten.relu.default(add_144);  add_144 = None
        le_18: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_18: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, slice_18);  le_18 = slice_18 = None
        squeeze_84: "f32[116]" = torch.ops.aten.squeeze.dims(getitem_71, [0, 2, 3]);  getitem_71 = None
        unsqueeze_550: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_551: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
        unsqueeze_552: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
        sum_56: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        sub_164: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_552);  convolution_28 = unsqueeze_552 = None
        mul_635: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_164)
        sum_57: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_635, [0, 2, 3]);  mul_635 = None
        mul_636: "f32[116]" = torch.ops.aten.mul.Tensor(sum_56, 9.964923469387754e-06)
        unsqueeze_553: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
        unsqueeze_554: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
        unsqueeze_555: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
        mul_637: "f32[116]" = torch.ops.aten.mul.Tensor(sum_57, 9.964923469387754e-06)
        squeeze_85: "f32[116]" = torch.ops.aten.squeeze.dims(rsqrt_28, [0, 2, 3]);  rsqrt_28 = None
        mul_638: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_639: "f32[116]" = torch.ops.aten.mul.Tensor(mul_637, mul_638);  mul_637 = mul_638 = None
        unsqueeze_556: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
        unsqueeze_557: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 2);  unsqueeze_556 = None
        unsqueeze_558: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 3);  unsqueeze_557 = None
        mul_640: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_174);  primals_174 = None
        unsqueeze_559: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
        unsqueeze_560: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
        unsqueeze_561: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
        mul_641: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_558);  sub_164 = unsqueeze_558 = None
        sub_166: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_641);  where_18 = mul_641 = None
        sub_167: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_555);  sub_166 = unsqueeze_555 = None
        mul_642: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_561);  sub_167 = unsqueeze_561 = None
        mul_643: "f32[116]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_85);  sum_57 = squeeze_85 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_642, add_139, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_642 = add_139 = primals_170 = None
        getitem_221: "f32[512, 116, 14, 14]" = convolution_backward_27[0]
        getitem_222: "f32[116, 116, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
        sum_58: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_221, [0, 2, 3])
        sub_168: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_564);  convolution_27 = unsqueeze_564 = None
        mul_644: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_221, sub_168)
        sum_59: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 2, 3]);  mul_644 = None
        mul_645: "f32[116]" = torch.ops.aten.mul.Tensor(sum_58, 9.964923469387754e-06)
        unsqueeze_565: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
        unsqueeze_566: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
        unsqueeze_567: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
        mul_646: "f32[116]" = torch.ops.aten.mul.Tensor(sum_59, 9.964923469387754e-06)
        mul_647: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_648: "f32[116]" = torch.ops.aten.mul.Tensor(mul_646, mul_647);  mul_646 = mul_647 = None
        unsqueeze_568: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
        unsqueeze_569: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 2);  unsqueeze_568 = None
        unsqueeze_570: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 3);  unsqueeze_569 = None
        mul_649: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  primals_168 = None
        unsqueeze_571: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
        unsqueeze_572: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
        unsqueeze_573: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
        mul_650: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_570);  sub_168 = unsqueeze_570 = None
        sub_170: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_221, mul_650);  getitem_221 = mul_650 = None
        sub_171: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_567);  sub_170 = unsqueeze_567 = None
        mul_651: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_573);  sub_171 = unsqueeze_573 = None
        mul_652: "f32[116]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_82);  sum_59 = squeeze_82 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_651, relu_17, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_651 = primals_164 = None
        getitem_224: "f32[512, 116, 14, 14]" = convolution_backward_28[0]
        getitem_225: "f32[116, 1, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
        le_19: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_19: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_224);  le_19 = getitem_224 = None
        sum_60: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_172: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_576);  convolution_26 = unsqueeze_576 = None
        mul_653: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_172)
        sum_61: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_653, [0, 2, 3]);  mul_653 = None
        mul_654: "f32[116]" = torch.ops.aten.mul.Tensor(sum_60, 9.964923469387754e-06)
        unsqueeze_577: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
        unsqueeze_578: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
        unsqueeze_579: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
        mul_655: "f32[116]" = torch.ops.aten.mul.Tensor(sum_61, 9.964923469387754e-06)
        mul_656: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_657: "f32[116]" = torch.ops.aten.mul.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
        unsqueeze_580: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
        unsqueeze_581: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 2);  unsqueeze_580 = None
        unsqueeze_582: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 3);  unsqueeze_581 = None
        mul_658: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_162);  primals_162 = None
        unsqueeze_583: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
        unsqueeze_584: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
        unsqueeze_585: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
        mul_659: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_582);  sub_172 = unsqueeze_582 = None
        sub_174: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_659);  where_19 = mul_659 = None
        sub_175: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_579);  sub_174 = unsqueeze_579 = None
        mul_660: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_585);  sub_175 = unsqueeze_585 = None
        mul_661: "f32[116]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_79);  sum_61 = squeeze_79 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_660, getitem_65, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_660 = getitem_65 = primals_158 = None
        getitem_227: "f32[512, 116, 14, 14]" = convolution_backward_29[0]
        getitem_228: "f32[116, 116, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_23: "f32[512, 232, 14, 14]" = torch.ops.aten.cat.default([slice_17, getitem_227], 1);  slice_17 = getitem_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_51: "f32[512, 116, 2, 14, 14]" = torch.ops.aten.reshape.default(cat_23, [512, 116, 2, 14, 14]);  cat_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_30: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3, 4]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_25: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_52: "f32[512, 232, 14, 14]" = torch.ops.aten.reshape.default(clone_25, [512, 232, 14, 14]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_19: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_52, 1, 0, 116)
        slice_20: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_52, 1, 116, 232);  view_52 = None
        sub_25: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_63)
        mul_175: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_100: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_103: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_129: "f32[512, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None
        relu_16: "f32[512, 116, 14, 14]" = torch.ops.aten.relu.default(add_129);  add_129 = None
        le_20: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_20: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, slice_20);  le_20 = slice_20 = None
        squeeze_75: "f32[116]" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        unsqueeze_586: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_587: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
        unsqueeze_588: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
        sum_62: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        sub_176: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_588);  convolution_25 = unsqueeze_588 = None
        mul_662: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_176)
        sum_63: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3]);  mul_662 = None
        mul_663: "f32[116]" = torch.ops.aten.mul.Tensor(sum_62, 9.964923469387754e-06)
        unsqueeze_589: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
        unsqueeze_590: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
        unsqueeze_591: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
        mul_664: "f32[116]" = torch.ops.aten.mul.Tensor(sum_63, 9.964923469387754e-06)
        squeeze_76: "f32[116]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_665: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_666: "f32[116]" = torch.ops.aten.mul.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
        unsqueeze_592: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
        unsqueeze_593: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 2);  unsqueeze_592 = None
        unsqueeze_594: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 3);  unsqueeze_593 = None
        mul_667: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_156);  primals_156 = None
        unsqueeze_595: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
        unsqueeze_596: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
        unsqueeze_597: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
        mul_668: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_594);  sub_176 = unsqueeze_594 = None
        sub_178: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_668);  where_20 = mul_668 = None
        sub_179: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_591);  sub_178 = unsqueeze_591 = None
        mul_669: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_597);  sub_179 = unsqueeze_597 = None
        mul_670: "f32[116]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_76);  sum_63 = squeeze_76 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_669, add_124, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_669 = add_124 = primals_152 = None
        getitem_230: "f32[512, 116, 14, 14]" = convolution_backward_30[0]
        getitem_231: "f32[116, 116, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
        sum_64: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_230, [0, 2, 3])
        sub_180: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_600);  convolution_24 = unsqueeze_600 = None
        mul_671: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_230, sub_180)
        sum_65: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 2, 3]);  mul_671 = None
        mul_672: "f32[116]" = torch.ops.aten.mul.Tensor(sum_64, 9.964923469387754e-06)
        unsqueeze_601: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
        unsqueeze_602: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
        unsqueeze_603: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
        mul_673: "f32[116]" = torch.ops.aten.mul.Tensor(sum_65, 9.964923469387754e-06)
        mul_674: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_675: "f32[116]" = torch.ops.aten.mul.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
        unsqueeze_604: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
        unsqueeze_605: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 2);  unsqueeze_604 = None
        unsqueeze_606: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 3);  unsqueeze_605 = None
        mul_676: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_150);  primals_150 = None
        unsqueeze_607: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
        unsqueeze_608: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
        unsqueeze_609: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
        mul_677: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_606);  sub_180 = unsqueeze_606 = None
        sub_182: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_230, mul_677);  getitem_230 = mul_677 = None
        sub_183: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_603);  sub_182 = unsqueeze_603 = None
        mul_678: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_609);  sub_183 = unsqueeze_609 = None
        mul_679: "f32[116]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_73);  sum_65 = squeeze_73 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_678, relu_15, primals_146, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_678 = primals_146 = None
        getitem_233: "f32[512, 116, 14, 14]" = convolution_backward_31[0]
        getitem_234: "f32[116, 1, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
        le_21: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_21: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, getitem_233);  le_21 = getitem_233 = None
        sum_66: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        sub_184: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_612);  convolution_23 = unsqueeze_612 = None
        mul_680: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_184)
        sum_67: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3]);  mul_680 = None
        mul_681: "f32[116]" = torch.ops.aten.mul.Tensor(sum_66, 9.964923469387754e-06)
        unsqueeze_613: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
        unsqueeze_614: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
        unsqueeze_615: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
        mul_682: "f32[116]" = torch.ops.aten.mul.Tensor(sum_67, 9.964923469387754e-06)
        mul_683: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_684: "f32[116]" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
        unsqueeze_616: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
        unsqueeze_617: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 2);  unsqueeze_616 = None
        unsqueeze_618: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 3);  unsqueeze_617 = None
        mul_685: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_144);  primals_144 = None
        unsqueeze_619: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
        unsqueeze_620: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
        unsqueeze_621: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
        mul_686: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_618);  sub_184 = unsqueeze_618 = None
        sub_186: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_686);  where_21 = mul_686 = None
        sub_187: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_615);  sub_186 = unsqueeze_615 = None
        mul_687: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_621);  sub_187 = unsqueeze_621 = None
        mul_688: "f32[116]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_70);  sum_67 = squeeze_70 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_687, getitem_57, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_687 = getitem_57 = primals_140 = None
        getitem_236: "f32[512, 116, 14, 14]" = convolution_backward_32[0]
        getitem_237: "f32[116, 116, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_24: "f32[512, 232, 14, 14]" = torch.ops.aten.cat.default([slice_19, getitem_236], 1);  slice_19 = getitem_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_53: "f32[512, 116, 2, 14, 14]" = torch.ops.aten.reshape.default(cat_24, [512, 116, 2, 14, 14]);  cat_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_31: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3, 4]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_26: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
        view_54: "f32[512, 232, 14, 14]" = torch.ops.aten.reshape.default(clone_26, [512, 232, 14, 14]);  clone_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_21: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_54, 1, 0, 116)
        slice_22: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_54, 1, 116, 232);  view_54 = None
        sub_22: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, getitem_55)
        mul_154: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        unsqueeze_88: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_138, -1)
        unsqueeze_89: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_160: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_89);  mul_154 = unsqueeze_89 = None
        unsqueeze_90: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_139, -1);  primals_139 = None
        unsqueeze_91: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_114: "f32[512, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_160, unsqueeze_91);  mul_160 = unsqueeze_91 = None
        relu_14: "f32[512, 116, 14, 14]" = torch.ops.aten.relu.default(add_114);  add_114 = None
        le_22: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_22: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, slice_22);  le_22 = slice_22 = None
        squeeze_66: "f32[116]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        unsqueeze_622: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_623: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
        unsqueeze_624: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
        sum_68: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        sub_188: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_624);  convolution_22 = unsqueeze_624 = None
        mul_689: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_188)
        sum_69: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None
        mul_690: "f32[116]" = torch.ops.aten.mul.Tensor(sum_68, 9.964923469387754e-06)
        unsqueeze_625: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
        unsqueeze_626: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
        unsqueeze_627: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
        mul_691: "f32[116]" = torch.ops.aten.mul.Tensor(sum_69, 9.964923469387754e-06)
        squeeze_67: "f32[116]" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_692: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_693: "f32[116]" = torch.ops.aten.mul.Tensor(mul_691, mul_692);  mul_691 = mul_692 = None
        unsqueeze_628: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
        unsqueeze_629: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 2);  unsqueeze_628 = None
        unsqueeze_630: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 3);  unsqueeze_629 = None
        mul_694: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_138);  primals_138 = None
        unsqueeze_631: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
        unsqueeze_632: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
        unsqueeze_633: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
        mul_695: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_630);  sub_188 = unsqueeze_630 = None
        sub_190: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_695);  where_22 = mul_695 = None
        sub_191: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_627);  sub_190 = unsqueeze_627 = None
        mul_696: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_633);  sub_191 = unsqueeze_633 = None
        mul_697: "f32[116]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_67);  sum_69 = squeeze_67 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_696, add_109, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_696 = add_109 = primals_134 = None
        getitem_239: "f32[512, 116, 14, 14]" = convolution_backward_33[0]
        getitem_240: "f32[116, 116, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
        sum_70: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_239, [0, 2, 3])
        sub_192: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_636);  convolution_21 = unsqueeze_636 = None
        mul_698: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_239, sub_192)
        sum_71: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 2, 3]);  mul_698 = None
        mul_699: "f32[116]" = torch.ops.aten.mul.Tensor(sum_70, 9.964923469387754e-06)
        unsqueeze_637: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
        unsqueeze_638: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
        unsqueeze_639: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
        mul_700: "f32[116]" = torch.ops.aten.mul.Tensor(sum_71, 9.964923469387754e-06)
        mul_701: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_702: "f32[116]" = torch.ops.aten.mul.Tensor(mul_700, mul_701);  mul_700 = mul_701 = None
        unsqueeze_640: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
        unsqueeze_641: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 2);  unsqueeze_640 = None
        unsqueeze_642: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 3);  unsqueeze_641 = None
        mul_703: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_132);  primals_132 = None
        unsqueeze_643: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
        unsqueeze_644: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
        unsqueeze_645: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
        mul_704: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_642);  sub_192 = unsqueeze_642 = None
        sub_194: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_239, mul_704);  getitem_239 = mul_704 = None
        sub_195: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_639);  sub_194 = unsqueeze_639 = None
        mul_705: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_645);  sub_195 = unsqueeze_645 = None
        mul_706: "f32[116]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_64);  sum_71 = squeeze_64 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_705, relu_13, primals_128, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_705 = primals_128 = None
        getitem_242: "f32[512, 116, 14, 14]" = convolution_backward_34[0]
        getitem_243: "f32[116, 1, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
        le_23: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_23: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, getitem_242);  le_23 = getitem_242 = None
        sum_72: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        sub_196: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_648);  convolution_20 = unsqueeze_648 = None
        mul_707: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_196)
        sum_73: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3]);  mul_707 = None
        mul_708: "f32[116]" = torch.ops.aten.mul.Tensor(sum_72, 9.964923469387754e-06)
        unsqueeze_649: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
        unsqueeze_650: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
        unsqueeze_651: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
        mul_709: "f32[116]" = torch.ops.aten.mul.Tensor(sum_73, 9.964923469387754e-06)
        mul_710: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_711: "f32[116]" = torch.ops.aten.mul.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
        unsqueeze_652: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_653: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 2);  unsqueeze_652 = None
        unsqueeze_654: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 3);  unsqueeze_653 = None
        mul_712: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  primals_126 = None
        unsqueeze_655: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
        unsqueeze_656: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
        unsqueeze_657: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
        mul_713: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_654);  sub_196 = unsqueeze_654 = None
        sub_198: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_713);  where_23 = mul_713 = None
        sub_199: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_651);  sub_198 = unsqueeze_651 = None
        mul_714: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_657);  sub_199 = unsqueeze_657 = None
        mul_715: "f32[116]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_61);  sum_73 = squeeze_61 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_714, getitem_49, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_714 = getitem_49 = primals_122 = None
        getitem_245: "f32[512, 116, 14, 14]" = convolution_backward_35[0]
        getitem_246: "f32[116, 116, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_25: "f32[512, 232, 14, 14]" = torch.ops.aten.cat.default([slice_21, getitem_245], 1);  slice_21 = getitem_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_55: "f32[512, 116, 2, 14, 14]" = torch.ops.aten.reshape.default(cat_25, [512, 116, 2, 14, 14]);  cat_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_32: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3, 4]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_27: "f32[512, 2, 116, 14, 14]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_56: "f32[512, 232, 14, 14]" = torch.ops.aten.reshape.default(clone_27, [512, 232, 14, 14]);  clone_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        slice_23: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_56, 1, 0, 116)
        slice_24: "f32[512, 116, 14, 14]" = torch.ops.aten.slice.Tensor(view_56, 1, 116, 232);  view_56 = None
        sub_19: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_47)
        mul_133: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        unsqueeze_76: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_79: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_99: "f32[512, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None
        relu_12: "f32[512, 116, 14, 14]" = torch.ops.aten.relu.default(add_99);  add_99 = None
        le_24: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_24: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, slice_24);  le_24 = slice_24 = None
        squeeze_57: "f32[116]" = torch.ops.aten.squeeze.dims(getitem_47, [0, 2, 3]);  getitem_47 = None
        unsqueeze_658: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_659: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
        unsqueeze_660: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
        sum_74: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_200: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_660);  convolution_19 = unsqueeze_660 = None
        mul_716: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_200)
        sum_75: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 2, 3]);  mul_716 = None
        mul_717: "f32[116]" = torch.ops.aten.mul.Tensor(sum_74, 9.964923469387754e-06)
        unsqueeze_661: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
        unsqueeze_662: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
        unsqueeze_663: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
        mul_718: "f32[116]" = torch.ops.aten.mul.Tensor(sum_75, 9.964923469387754e-06)
        squeeze_58: "f32[116]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_719: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_720: "f32[116]" = torch.ops.aten.mul.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
        unsqueeze_664: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
        unsqueeze_665: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 2);  unsqueeze_664 = None
        unsqueeze_666: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 3);  unsqueeze_665 = None
        mul_721: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_667: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
        unsqueeze_668: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
        unsqueeze_669: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
        mul_722: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_666);  sub_200 = unsqueeze_666 = None
        sub_202: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_722);  where_24 = mul_722 = None
        sub_203: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_663);  sub_202 = unsqueeze_663 = None
        mul_723: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_669);  sub_203 = unsqueeze_669 = None
        mul_724: "f32[116]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_58);  sum_75 = squeeze_58 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_723, add_94, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_723 = add_94 = primals_116 = None
        getitem_248: "f32[512, 116, 14, 14]" = convolution_backward_36[0]
        getitem_249: "f32[116, 116, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
        sum_76: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_248, [0, 2, 3])
        sub_204: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_672);  convolution_18 = unsqueeze_672 = None
        mul_725: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_248, sub_204)
        sum_77: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3]);  mul_725 = None
        mul_726: "f32[116]" = torch.ops.aten.mul.Tensor(sum_76, 9.964923469387754e-06)
        unsqueeze_673: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
        unsqueeze_674: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
        unsqueeze_675: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
        mul_727: "f32[116]" = torch.ops.aten.mul.Tensor(sum_77, 9.964923469387754e-06)
        mul_728: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_729: "f32[116]" = torch.ops.aten.mul.Tensor(mul_727, mul_728);  mul_727 = mul_728 = None
        unsqueeze_676: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
        unsqueeze_677: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 2);  unsqueeze_676 = None
        unsqueeze_678: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 3);  unsqueeze_677 = None
        mul_730: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_679: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
        unsqueeze_680: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
        unsqueeze_681: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
        mul_731: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_678);  sub_204 = unsqueeze_678 = None
        sub_206: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_248, mul_731);  getitem_248 = mul_731 = None
        sub_207: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_675);  sub_206 = unsqueeze_675 = None
        mul_732: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_681);  sub_207 = unsqueeze_681 = None
        mul_733: "f32[116]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_55);  sum_77 = squeeze_55 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_732, relu_11, primals_110, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_732 = primals_110 = None
        getitem_251: "f32[512, 116, 28, 28]" = convolution_backward_37[0]
        getitem_252: "f32[116, 1, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
        le_25: "b8[512, 116, 28, 28]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_25: "f32[512, 116, 28, 28]" = torch.ops.aten.where.self(le_25, full_default, getitem_251);  le_25 = getitem_251 = None
        sum_78: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        sub_208: "f32[512, 116, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_684);  convolution_17 = unsqueeze_684 = None
        mul_734: "f32[512, 116, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_208)
        sum_79: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_734, [0, 2, 3]);  mul_734 = None
        mul_735: "f32[116]" = torch.ops.aten.mul.Tensor(sum_78, 2.4912308673469386e-06)
        unsqueeze_685: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
        unsqueeze_686: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
        unsqueeze_687: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
        mul_736: "f32[116]" = torch.ops.aten.mul.Tensor(sum_79, 2.4912308673469386e-06)
        mul_737: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_738: "f32[116]" = torch.ops.aten.mul.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
        unsqueeze_688: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
        unsqueeze_689: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 2);  unsqueeze_688 = None
        unsqueeze_690: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 3);  unsqueeze_689 = None
        mul_739: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_691: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
        unsqueeze_692: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
        unsqueeze_693: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
        mul_740: "f32[512, 116, 28, 28]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_690);  sub_208 = unsqueeze_690 = None
        sub_210: "f32[512, 116, 28, 28]" = torch.ops.aten.sub.Tensor(where_25, mul_740);  where_25 = mul_740 = None
        sub_211: "f32[512, 116, 28, 28]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_687);  sub_210 = unsqueeze_687 = None
        mul_741: "f32[512, 116, 28, 28]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_693);  sub_211 = unsqueeze_693 = None
        mul_742: "f32[116]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_52);  sum_79 = squeeze_52 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_741, view_7, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_741 = primals_104 = None
        getitem_254: "f32[512, 116, 28, 28]" = convolution_backward_38[0]
        getitem_255: "f32[116, 116, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
        sub_16: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, getitem_41)
        mul_112: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        unsqueeze_64: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[116, 1]" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_84: "f32[512, 116, 14, 14]" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None
        relu_10: "f32[512, 116, 14, 14]" = torch.ops.aten.relu.default(add_84);  add_84 = None
        le_26: "b8[512, 116, 14, 14]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_26: "f32[512, 116, 14, 14]" = torch.ops.aten.where.self(le_26, full_default, slice_23);  le_26 = slice_23 = None
        squeeze_48: "f32[116]" = torch.ops.aten.squeeze.dims(getitem_41, [0, 2, 3]);  getitem_41 = None
        unsqueeze_694: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_695: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
        unsqueeze_696: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
        sum_80: "f32[116]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
        sub_212: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_696);  convolution_16 = unsqueeze_696 = None
        mul_743: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(where_26, sub_212)
        sum_81: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_743, [0, 2, 3]);  mul_743 = None
        mul_744: "f32[116]" = torch.ops.aten.mul.Tensor(sum_80, 9.964923469387754e-06)
        unsqueeze_697: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
        unsqueeze_698: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
        unsqueeze_699: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
        mul_745: "f32[116]" = torch.ops.aten.mul.Tensor(sum_81, 9.964923469387754e-06)
        squeeze_49: "f32[116]" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_746: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_747: "f32[116]" = torch.ops.aten.mul.Tensor(mul_745, mul_746);  mul_745 = mul_746 = None
        unsqueeze_700: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
        unsqueeze_701: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 2);  unsqueeze_700 = None
        unsqueeze_702: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 3);  unsqueeze_701 = None
        mul_748: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_703: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
        unsqueeze_704: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
        unsqueeze_705: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
        mul_749: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_702);  sub_212 = unsqueeze_702 = None
        sub_214: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(where_26, mul_749);  where_26 = mul_749 = None
        sub_215: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_699);  sub_214 = unsqueeze_699 = None
        mul_750: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_705);  sub_215 = unsqueeze_705 = None
        mul_751: "f32[116]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_49);  sum_81 = squeeze_49 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_750, add_79, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_750 = add_79 = primals_98 = None
        getitem_257: "f32[512, 116, 14, 14]" = convolution_backward_39[0]
        getitem_258: "f32[116, 116, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
        sum_82: "f32[116]" = torch.ops.aten.sum.dim_IntList(getitem_257, [0, 2, 3])
        sub_216: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_708);  convolution_15 = unsqueeze_708 = None
        mul_752: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_257, sub_216)
        sum_83: "f32[116]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 2, 3]);  mul_752 = None
        mul_753: "f32[116]" = torch.ops.aten.mul.Tensor(sum_82, 9.964923469387754e-06)
        unsqueeze_709: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
        unsqueeze_710: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
        unsqueeze_711: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
        mul_754: "f32[116]" = torch.ops.aten.mul.Tensor(sum_83, 9.964923469387754e-06)
        mul_755: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_756: "f32[116]" = torch.ops.aten.mul.Tensor(mul_754, mul_755);  mul_754 = mul_755 = None
        unsqueeze_712: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
        unsqueeze_713: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 2);  unsqueeze_712 = None
        unsqueeze_714: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 3);  unsqueeze_713 = None
        mul_757: "f32[116]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_715: "f32[1, 116]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
        unsqueeze_716: "f32[1, 116, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
        unsqueeze_717: "f32[1, 116, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
        mul_758: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_714);  sub_216 = unsqueeze_714 = None
        sub_218: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_257, mul_758);  getitem_257 = mul_758 = None
        sub_219: "f32[512, 116, 14, 14]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_711);  sub_218 = unsqueeze_711 = None
        mul_759: "f32[512, 116, 14, 14]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_717);  sub_219 = unsqueeze_717 = None
        mul_760: "f32[116]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_46);  sum_83 = squeeze_46 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_759, view_7, primals_92, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 116, [True, True, False]);  mul_759 = view_7 = primals_92 = None
        getitem_260: "f32[512, 116, 28, 28]" = convolution_backward_40[0]
        getitem_261: "f32[116, 1, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
        add_281: "f32[512, 116, 28, 28]" = torch.ops.aten.add.Tensor(getitem_254, getitem_260);  getitem_254 = getitem_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_57: "f32[512, 58, 2, 28, 28]" = torch.ops.aten.reshape.default(add_281, [512, 58, 2, 28, 28]);  add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_33: "f32[512, 2, 58, 28, 28]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3, 4]);  view_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_28: "f32[512, 2, 58, 28, 28]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_58: "f32[512, 116, 28, 28]" = torch.ops.aten.reshape.default(clone_28, [512, 116, 28, 28]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_25: "f32[512, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_58, 1, 0, 58)
        slice_26: "f32[512, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_58, 1, 58, 116);  view_58 = None
        sub_14: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_37)
        mul_98: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        unsqueeze_56: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_74: "f32[512, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None
        relu_9: "f32[512, 58, 28, 28]" = torch.ops.aten.relu.default(add_74);  add_74 = None
        le_27: "b8[512, 58, 28, 28]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_27: "f32[512, 58, 28, 28]" = torch.ops.aten.where.self(le_27, full_default, slice_26);  le_27 = slice_26 = None
        squeeze_42: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_37, [0, 2, 3]);  getitem_37 = None
        unsqueeze_718: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_719: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
        unsqueeze_720: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
        sum_84: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
        sub_220: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_720);  convolution_14 = unsqueeze_720 = None
        mul_761: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_220)
        sum_85: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 2, 3]);  mul_761 = None
        mul_762: "f32[58]" = torch.ops.aten.mul.Tensor(sum_84, 2.4912308673469386e-06)
        unsqueeze_721: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
        unsqueeze_722: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
        unsqueeze_723: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
        mul_763: "f32[58]" = torch.ops.aten.mul.Tensor(sum_85, 2.4912308673469386e-06)
        squeeze_43: "f32[58]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_764: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_765: "f32[58]" = torch.ops.aten.mul.Tensor(mul_763, mul_764);  mul_763 = mul_764 = None
        unsqueeze_724: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
        unsqueeze_725: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 2);  unsqueeze_724 = None
        unsqueeze_726: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 3);  unsqueeze_725 = None
        mul_766: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_727: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
        unsqueeze_728: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
        unsqueeze_729: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
        mul_767: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_726);  sub_220 = unsqueeze_726 = None
        sub_222: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(where_27, mul_767);  where_27 = mul_767 = None
        sub_223: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_723);  sub_222 = unsqueeze_723 = None
        mul_768: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_729);  sub_223 = unsqueeze_729 = None
        mul_769: "f32[58]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_43);  sum_85 = squeeze_43 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_768, add_69, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_768 = add_69 = primals_86 = None
        getitem_263: "f32[512, 58, 28, 28]" = convolution_backward_41[0]
        getitem_264: "f32[58, 58, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
        sum_86: "f32[58]" = torch.ops.aten.sum.dim_IntList(getitem_263, [0, 2, 3])
        sub_224: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_732);  convolution_13 = unsqueeze_732 = None
        mul_770: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_263, sub_224)
        sum_87: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3]);  mul_770 = None
        mul_771: "f32[58]" = torch.ops.aten.mul.Tensor(sum_86, 2.4912308673469386e-06)
        unsqueeze_733: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
        unsqueeze_734: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
        unsqueeze_735: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
        mul_772: "f32[58]" = torch.ops.aten.mul.Tensor(sum_87, 2.4912308673469386e-06)
        mul_773: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_774: "f32[58]" = torch.ops.aten.mul.Tensor(mul_772, mul_773);  mul_772 = mul_773 = None
        unsqueeze_736: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
        unsqueeze_737: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 2);  unsqueeze_736 = None
        unsqueeze_738: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 3);  unsqueeze_737 = None
        mul_775: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_739: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
        unsqueeze_740: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
        unsqueeze_741: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
        mul_776: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_738);  sub_224 = unsqueeze_738 = None
        sub_226: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_263, mul_776);  getitem_263 = mul_776 = None
        sub_227: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_735);  sub_226 = unsqueeze_735 = None
        mul_777: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_741);  sub_227 = unsqueeze_741 = None
        mul_778: "f32[58]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_40);  sum_87 = squeeze_40 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_777, relu_8, primals_80, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  mul_777 = primals_80 = None
        getitem_266: "f32[512, 58, 28, 28]" = convolution_backward_42[0]
        getitem_267: "f32[58, 1, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
        le_28: "b8[512, 58, 28, 28]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_28: "f32[512, 58, 28, 28]" = torch.ops.aten.where.self(le_28, full_default, getitem_266);  le_28 = getitem_266 = None
        sum_88: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
        sub_228: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_744);  convolution_12 = unsqueeze_744 = None
        mul_779: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, sub_228)
        sum_89: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_779, [0, 2, 3]);  mul_779 = None
        mul_780: "f32[58]" = torch.ops.aten.mul.Tensor(sum_88, 2.4912308673469386e-06)
        unsqueeze_745: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
        unsqueeze_746: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
        unsqueeze_747: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
        mul_781: "f32[58]" = torch.ops.aten.mul.Tensor(sum_89, 2.4912308673469386e-06)
        mul_782: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_783: "f32[58]" = torch.ops.aten.mul.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
        unsqueeze_748: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
        unsqueeze_749: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 2);  unsqueeze_748 = None
        unsqueeze_750: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 3);  unsqueeze_749 = None
        mul_784: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_751: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
        unsqueeze_752: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
        unsqueeze_753: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
        mul_785: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_750);  sub_228 = unsqueeze_750 = None
        sub_230: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(where_28, mul_785);  where_28 = mul_785 = None
        sub_231: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_747);  sub_230 = unsqueeze_747 = None
        mul_786: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_753);  sub_231 = unsqueeze_753 = None
        mul_787: "f32[58]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_37);  sum_89 = squeeze_37 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_786, getitem_31, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_786 = getitem_31 = primals_74 = None
        getitem_269: "f32[512, 58, 28, 28]" = convolution_backward_43[0]
        getitem_270: "f32[58, 58, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_26: "f32[512, 116, 28, 28]" = torch.ops.aten.cat.default([slice_25, getitem_269], 1);  slice_25 = getitem_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_59: "f32[512, 58, 2, 28, 28]" = torch.ops.aten.reshape.default(cat_26, [512, 58, 2, 28, 28]);  cat_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_34: "f32[512, 2, 58, 28, 28]" = torch.ops.aten.permute.default(view_59, [0, 2, 1, 3, 4]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_29: "f32[512, 2, 58, 28, 28]" = torch.ops.aten.clone.default(permute_34, memory_format = torch.contiguous_format);  permute_34 = None
        view_60: "f32[512, 116, 28, 28]" = torch.ops.aten.reshape.default(clone_29, [512, 116, 28, 28]);  clone_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_27: "f32[512, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_60, 1, 0, 58)
        slice_28: "f32[512, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_60, 1, 58, 116);  view_60 = None
        sub_11: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, getitem_29)
        mul_77: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        unsqueeze_44: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_59: "f32[512, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None
        relu_7: "f32[512, 58, 28, 28]" = torch.ops.aten.relu.default(add_59);  add_59 = None
        le_29: "b8[512, 58, 28, 28]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_29: "f32[512, 58, 28, 28]" = torch.ops.aten.where.self(le_29, full_default, slice_28);  le_29 = slice_28 = None
        squeeze_33: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        unsqueeze_754: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_755: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
        unsqueeze_756: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
        sum_90: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        sub_232: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_756);  convolution_11 = unsqueeze_756 = None
        mul_788: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, sub_232)
        sum_91: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_788, [0, 2, 3]);  mul_788 = None
        mul_789: "f32[58]" = torch.ops.aten.mul.Tensor(sum_90, 2.4912308673469386e-06)
        unsqueeze_757: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
        unsqueeze_758: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
        unsqueeze_759: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
        mul_790: "f32[58]" = torch.ops.aten.mul.Tensor(sum_91, 2.4912308673469386e-06)
        squeeze_34: "f32[58]" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_791: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_792: "f32[58]" = torch.ops.aten.mul.Tensor(mul_790, mul_791);  mul_790 = mul_791 = None
        unsqueeze_760: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
        unsqueeze_761: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 2);  unsqueeze_760 = None
        unsqueeze_762: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 3);  unsqueeze_761 = None
        mul_793: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_763: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
        unsqueeze_764: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
        unsqueeze_765: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
        mul_794: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_762);  sub_232 = unsqueeze_762 = None
        sub_234: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(where_29, mul_794);  where_29 = mul_794 = None
        sub_235: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_759);  sub_234 = unsqueeze_759 = None
        mul_795: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_765);  sub_235 = unsqueeze_765 = None
        mul_796: "f32[58]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_34);  sum_91 = squeeze_34 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_795, add_54, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_795 = add_54 = primals_68 = None
        getitem_272: "f32[512, 58, 28, 28]" = convolution_backward_44[0]
        getitem_273: "f32[58, 58, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
        sum_92: "f32[58]" = torch.ops.aten.sum.dim_IntList(getitem_272, [0, 2, 3])
        sub_236: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_768);  convolution_10 = unsqueeze_768 = None
        mul_797: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_272, sub_236)
        sum_93: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 2, 3]);  mul_797 = None
        mul_798: "f32[58]" = torch.ops.aten.mul.Tensor(sum_92, 2.4912308673469386e-06)
        unsqueeze_769: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
        unsqueeze_770: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
        unsqueeze_771: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
        mul_799: "f32[58]" = torch.ops.aten.mul.Tensor(sum_93, 2.4912308673469386e-06)
        mul_800: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_801: "f32[58]" = torch.ops.aten.mul.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
        unsqueeze_772: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
        unsqueeze_773: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 2);  unsqueeze_772 = None
        unsqueeze_774: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 3);  unsqueeze_773 = None
        mul_802: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_775: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
        unsqueeze_776: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
        unsqueeze_777: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
        mul_803: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_774);  sub_236 = unsqueeze_774 = None
        sub_238: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_272, mul_803);  getitem_272 = mul_803 = None
        sub_239: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_771);  sub_238 = unsqueeze_771 = None
        mul_804: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_777);  sub_239 = unsqueeze_777 = None
        mul_805: "f32[58]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_31);  sum_93 = squeeze_31 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_804, relu_6, primals_62, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  mul_804 = primals_62 = None
        getitem_275: "f32[512, 58, 28, 28]" = convolution_backward_45[0]
        getitem_276: "f32[58, 1, 3, 3]" = convolution_backward_45[1];  convolution_backward_45 = None
        le_30: "b8[512, 58, 28, 28]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_30: "f32[512, 58, 28, 28]" = torch.ops.aten.where.self(le_30, full_default, getitem_275);  le_30 = getitem_275 = None
        sum_94: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
        sub_240: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_780);  convolution_9 = unsqueeze_780 = None
        mul_806: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_30, sub_240)
        sum_95: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_806, [0, 2, 3]);  mul_806 = None
        mul_807: "f32[58]" = torch.ops.aten.mul.Tensor(sum_94, 2.4912308673469386e-06)
        unsqueeze_781: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_782: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
        unsqueeze_783: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
        mul_808: "f32[58]" = torch.ops.aten.mul.Tensor(sum_95, 2.4912308673469386e-06)
        mul_809: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_810: "f32[58]" = torch.ops.aten.mul.Tensor(mul_808, mul_809);  mul_808 = mul_809 = None
        unsqueeze_784: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
        unsqueeze_785: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 2);  unsqueeze_784 = None
        unsqueeze_786: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 3);  unsqueeze_785 = None
        mul_811: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_787: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
        unsqueeze_788: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
        unsqueeze_789: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
        mul_812: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_786);  sub_240 = unsqueeze_786 = None
        sub_242: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(where_30, mul_812);  where_30 = mul_812 = None
        sub_243: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_783);  sub_242 = unsqueeze_783 = None
        mul_813: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_789);  sub_243 = unsqueeze_789 = None
        mul_814: "f32[58]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_28);  sum_95 = squeeze_28 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_813, getitem_23, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_813 = getitem_23 = primals_56 = None
        getitem_278: "f32[512, 58, 28, 28]" = convolution_backward_46[0]
        getitem_279: "f32[58, 58, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_27: "f32[512, 116, 28, 28]" = torch.ops.aten.cat.default([slice_27, getitem_278], 1);  slice_27 = getitem_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_61: "f32[512, 58, 2, 28, 28]" = torch.ops.aten.reshape.default(cat_27, [512, 58, 2, 28, 28]);  cat_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_35: "f32[512, 2, 58, 28, 28]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3, 4]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_30: "f32[512, 2, 58, 28, 28]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_62: "f32[512, 116, 28, 28]" = torch.ops.aten.reshape.default(clone_30, [512, 116, 28, 28]);  clone_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:95 in forward, code: out = torch.cat((x1, self.branch2(x2)), dim=1)
        slice_29: "f32[512, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_62, 1, 0, 58)
        slice_30: "f32[512, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_62, 1, 58, 116);  view_62 = None
        sub_8: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_21)
        mul_56: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        unsqueeze_32: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_44: "f32[512, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None
        relu_5: "f32[512, 58, 28, 28]" = torch.ops.aten.relu.default(add_44);  add_44 = None
        le_31: "b8[512, 58, 28, 28]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_31: "f32[512, 58, 28, 28]" = torch.ops.aten.where.self(le_31, full_default, slice_30);  le_31 = slice_30 = None
        squeeze_24: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_790: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_791: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
        unsqueeze_792: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
        sum_96: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
        sub_244: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_792);  convolution_8 = unsqueeze_792 = None
        mul_815: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_31, sub_244)
        sum_97: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 2, 3]);  mul_815 = None
        mul_816: "f32[58]" = torch.ops.aten.mul.Tensor(sum_96, 2.4912308673469386e-06)
        unsqueeze_793: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
        unsqueeze_794: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
        unsqueeze_795: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
        mul_817: "f32[58]" = torch.ops.aten.mul.Tensor(sum_97, 2.4912308673469386e-06)
        squeeze_25: "f32[58]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_818: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_819: "f32[58]" = torch.ops.aten.mul.Tensor(mul_817, mul_818);  mul_817 = mul_818 = None
        unsqueeze_796: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
        unsqueeze_797: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 2);  unsqueeze_796 = None
        unsqueeze_798: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 3);  unsqueeze_797 = None
        mul_820: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_799: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
        unsqueeze_800: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
        unsqueeze_801: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
        mul_821: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_798);  sub_244 = unsqueeze_798 = None
        sub_246: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(where_31, mul_821);  where_31 = mul_821 = None
        sub_247: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_795);  sub_246 = unsqueeze_795 = None
        mul_822: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_801);  sub_247 = unsqueeze_801 = None
        mul_823: "f32[58]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_25);  sum_97 = squeeze_25 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_822, add_39, primals_50, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_822 = add_39 = primals_50 = None
        getitem_281: "f32[512, 58, 28, 28]" = convolution_backward_47[0]
        getitem_282: "f32[58, 58, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
        sum_98: "f32[58]" = torch.ops.aten.sum.dim_IntList(getitem_281, [0, 2, 3])
        sub_248: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_804);  convolution_7 = unsqueeze_804 = None
        mul_824: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_281, sub_248)
        sum_99: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_824, [0, 2, 3]);  mul_824 = None
        mul_825: "f32[58]" = torch.ops.aten.mul.Tensor(sum_98, 2.4912308673469386e-06)
        unsqueeze_805: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
        unsqueeze_806: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
        unsqueeze_807: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
        mul_826: "f32[58]" = torch.ops.aten.mul.Tensor(sum_99, 2.4912308673469386e-06)
        mul_827: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_828: "f32[58]" = torch.ops.aten.mul.Tensor(mul_826, mul_827);  mul_826 = mul_827 = None
        unsqueeze_808: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
        unsqueeze_809: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 2);  unsqueeze_808 = None
        unsqueeze_810: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 3);  unsqueeze_809 = None
        mul_829: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_811: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
        unsqueeze_812: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
        unsqueeze_813: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
        mul_830: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_810);  sub_248 = unsqueeze_810 = None
        sub_250: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_281, mul_830);  getitem_281 = mul_830 = None
        sub_251: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_807);  sub_250 = unsqueeze_807 = None
        mul_831: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_813);  sub_251 = unsqueeze_813 = None
        mul_832: "f32[58]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_22);  sum_99 = squeeze_22 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_831, relu_4, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  mul_831 = primals_44 = None
        getitem_284: "f32[512, 58, 28, 28]" = convolution_backward_48[0]
        getitem_285: "f32[58, 1, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
        le_32: "b8[512, 58, 28, 28]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_32: "f32[512, 58, 28, 28]" = torch.ops.aten.where.self(le_32, full_default, getitem_284);  le_32 = getitem_284 = None
        sum_100: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
        sub_252: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_816);  convolution_6 = unsqueeze_816 = None
        mul_833: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, sub_252)
        sum_101: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_833, [0, 2, 3]);  mul_833 = None
        mul_834: "f32[58]" = torch.ops.aten.mul.Tensor(sum_100, 2.4912308673469386e-06)
        unsqueeze_817: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
        unsqueeze_818: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
        unsqueeze_819: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
        mul_835: "f32[58]" = torch.ops.aten.mul.Tensor(sum_101, 2.4912308673469386e-06)
        mul_836: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_837: "f32[58]" = torch.ops.aten.mul.Tensor(mul_835, mul_836);  mul_835 = mul_836 = None
        unsqueeze_820: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
        unsqueeze_821: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 2);  unsqueeze_820 = None
        unsqueeze_822: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 3);  unsqueeze_821 = None
        mul_838: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_823: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
        unsqueeze_824: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
        unsqueeze_825: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
        mul_839: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_822);  sub_252 = unsqueeze_822 = None
        sub_254: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(where_32, mul_839);  where_32 = mul_839 = None
        sub_255: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_819);  sub_254 = unsqueeze_819 = None
        mul_840: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_825);  sub_255 = unsqueeze_825 = None
        mul_841: "f32[58]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_19);  sum_101 = squeeze_19 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_840, getitem_15, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_840 = getitem_15 = primals_38 = None
        getitem_287: "f32[512, 58, 28, 28]" = convolution_backward_49[0]
        getitem_288: "f32[58, 58, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:94 in forward, code: x1, x2 = x.chunk(2, dim=1)
        cat_28: "f32[512, 116, 28, 28]" = torch.ops.aten.cat.default([slice_29, getitem_287], 1);  slice_29 = getitem_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:38 in channel_shuffle, code: x = x.view(batchsize, num_channels, height, width)
        view_63: "f32[512, 58, 2, 28, 28]" = torch.ops.aten.reshape.default(cat_28, [512, 58, 2, 28, 28]);  cat_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:35 in channel_shuffle, code: x = torch.transpose(x, 1, 2).contiguous()
        permute_36: "f32[512, 2, 58, 28, 28]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3, 4]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:33 in channel_shuffle, code: x = x.view(batchsize, groups, channels_per_group, height, width)
        clone_31: "f32[512, 2, 58, 28, 28]" = torch.ops.aten.clone.default(permute_36, memory_format = torch.contiguous_format);  permute_36 = None
        view_64: "f32[512, 116, 28, 28]" = torch.ops.aten.reshape.default(clone_31, [512, 116, 28, 28]);  clone_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:97 in forward, code: out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        slice_31: "f32[512, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_64, 1, 0, 58)
        slice_32: "f32[512, 58, 28, 28]" = torch.ops.aten.slice.Tensor(view_64, 1, 58, 116);  view_64 = None
        sub_5: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_5, getitem_13)
        mul_35: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        unsqueeze_20: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_29: "f32[512, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None
        relu_3: "f32[512, 58, 28, 28]" = torch.ops.aten.relu.default(add_29);  add_29 = None
        le_33: "b8[512, 58, 28, 28]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_33: "f32[512, 58, 28, 28]" = torch.ops.aten.where.self(le_33, full_default, slice_32);  le_33 = slice_32 = None
        squeeze_15: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_13, [0, 2, 3]);  getitem_13 = None
        unsqueeze_826: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_827: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
        unsqueeze_828: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
        sum_102: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
        sub_256: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_828);  convolution_5 = unsqueeze_828 = None
        mul_842: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, sub_256)
        sum_103: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_842, [0, 2, 3]);  mul_842 = None
        mul_843: "f32[58]" = torch.ops.aten.mul.Tensor(sum_102, 2.4912308673469386e-06)
        unsqueeze_829: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
        unsqueeze_830: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
        unsqueeze_831: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
        mul_844: "f32[58]" = torch.ops.aten.mul.Tensor(sum_103, 2.4912308673469386e-06)
        squeeze_16: "f32[58]" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_845: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_846: "f32[58]" = torch.ops.aten.mul.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
        unsqueeze_832: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
        unsqueeze_833: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 2);  unsqueeze_832 = None
        unsqueeze_834: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 3);  unsqueeze_833 = None
        mul_847: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_835: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
        unsqueeze_836: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
        unsqueeze_837: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
        mul_848: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_834);  sub_256 = unsqueeze_834 = None
        sub_258: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(where_33, mul_848);  where_33 = mul_848 = None
        sub_259: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_831);  sub_258 = unsqueeze_831 = None
        mul_849: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_837);  sub_259 = unsqueeze_837 = None
        mul_850: "f32[58]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_16);  sum_103 = squeeze_16 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_849, add_24, primals_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_849 = add_24 = primals_32 = None
        getitem_290: "f32[512, 58, 28, 28]" = convolution_backward_50[0]
        getitem_291: "f32[58, 58, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
        sum_104: "f32[58]" = torch.ops.aten.sum.dim_IntList(getitem_290, [0, 2, 3])
        sub_260: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_840);  convolution_4 = unsqueeze_840 = None
        mul_851: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_290, sub_260)
        sum_105: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_851, [0, 2, 3]);  mul_851 = None
        mul_852: "f32[58]" = torch.ops.aten.mul.Tensor(sum_104, 2.4912308673469386e-06)
        unsqueeze_841: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
        unsqueeze_842: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
        unsqueeze_843: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
        mul_853: "f32[58]" = torch.ops.aten.mul.Tensor(sum_105, 2.4912308673469386e-06)
        mul_854: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_855: "f32[58]" = torch.ops.aten.mul.Tensor(mul_853, mul_854);  mul_853 = mul_854 = None
        unsqueeze_844: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
        unsqueeze_845: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, 2);  unsqueeze_844 = None
        unsqueeze_846: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 3);  unsqueeze_845 = None
        mul_856: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_847: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
        unsqueeze_848: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
        unsqueeze_849: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
        mul_857: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_846);  sub_260 = unsqueeze_846 = None
        sub_262: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_290, mul_857);  getitem_290 = mul_857 = None
        sub_263: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_843);  sub_262 = unsqueeze_843 = None
        mul_858: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_849);  sub_263 = unsqueeze_849 = None
        mul_859: "f32[58]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_13);  sum_105 = squeeze_13 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_858, relu_2, primals_26, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 58, [True, True, False]);  mul_858 = primals_26 = None
        getitem_293: "f32[512, 58, 56, 56]" = convolution_backward_51[0]
        getitem_294: "f32[58, 1, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
        le_34: "b8[512, 58, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_34: "f32[512, 58, 56, 56]" = torch.ops.aten.where.self(le_34, full_default, getitem_293);  le_34 = getitem_293 = None
        sum_106: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
        sub_264: "f32[512, 58, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_852);  convolution_3 = unsqueeze_852 = None
        mul_860: "f32[512, 58, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, sub_264)
        sum_107: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_860, [0, 2, 3]);  mul_860 = None
        mul_861: "f32[58]" = torch.ops.aten.mul.Tensor(sum_106, 6.228077168367346e-07)
        unsqueeze_853: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
        unsqueeze_854: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
        unsqueeze_855: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
        mul_862: "f32[58]" = torch.ops.aten.mul.Tensor(sum_107, 6.228077168367346e-07)
        mul_863: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_864: "f32[58]" = torch.ops.aten.mul.Tensor(mul_862, mul_863);  mul_862 = mul_863 = None
        unsqueeze_856: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
        unsqueeze_857: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 2);  unsqueeze_856 = None
        unsqueeze_858: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 3);  unsqueeze_857 = None
        mul_865: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_859: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
        unsqueeze_860: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
        unsqueeze_861: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
        mul_866: "f32[512, 58, 56, 56]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_858);  sub_264 = unsqueeze_858 = None
        sub_266: "f32[512, 58, 56, 56]" = torch.ops.aten.sub.Tensor(where_34, mul_866);  where_34 = mul_866 = None
        sub_267: "f32[512, 58, 56, 56]" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_855);  sub_266 = unsqueeze_855 = None
        mul_867: "f32[512, 58, 56, 56]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_861);  sub_267 = unsqueeze_861 = None
        mul_868: "f32[58]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_10);  sum_107 = squeeze_10 = None
        convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_867, getitem_2, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_867 = primals_20 = None
        getitem_296: "f32[512, 24, 56, 56]" = convolution_backward_52[0]
        getitem_297: "f32[58, 24, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
        sub_2: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, getitem_7)
        mul_14: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        unsqueeze_8: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[58, 1]" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_14: "f32[512, 58, 28, 28]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None
        relu_1: "f32[512, 58, 28, 28]" = torch.ops.aten.relu.default(add_14);  add_14 = None
        le_35: "b8[512, 58, 28, 28]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_35: "f32[512, 58, 28, 28]" = torch.ops.aten.where.self(le_35, full_default, slice_31);  le_35 = slice_31 = None
        squeeze_6: "f32[58]" = torch.ops.aten.squeeze.dims(getitem_7, [0, 2, 3]);  getitem_7 = None
        unsqueeze_862: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_863: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
        unsqueeze_864: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
        sum_108: "f32[58]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
        sub_268: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_864);  convolution_2 = unsqueeze_864 = None
        mul_869: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(where_35, sub_268)
        sum_109: "f32[58]" = torch.ops.aten.sum.dim_IntList(mul_869, [0, 2, 3]);  mul_869 = None
        mul_870: "f32[58]" = torch.ops.aten.mul.Tensor(sum_108, 2.4912308673469386e-06)
        unsqueeze_865: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
        unsqueeze_866: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
        unsqueeze_867: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
        mul_871: "f32[58]" = torch.ops.aten.mul.Tensor(sum_109, 2.4912308673469386e-06)
        squeeze_7: "f32[58]" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_872: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_873: "f32[58]" = torch.ops.aten.mul.Tensor(mul_871, mul_872);  mul_871 = mul_872 = None
        unsqueeze_868: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
        unsqueeze_869: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 2);  unsqueeze_868 = None
        unsqueeze_870: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 3);  unsqueeze_869 = None
        mul_874: "f32[58]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_871: "f32[1, 58]" = torch.ops.aten.unsqueeze.default(mul_874, 0);  mul_874 = None
        unsqueeze_872: "f32[1, 58, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
        unsqueeze_873: "f32[1, 58, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
        mul_875: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_870);  sub_268 = unsqueeze_870 = None
        sub_270: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(where_35, mul_875);  where_35 = mul_875 = None
        sub_271: "f32[512, 58, 28, 28]" = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_867);  sub_270 = unsqueeze_867 = None
        mul_876: "f32[512, 58, 28, 28]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_873);  sub_271 = unsqueeze_873 = None
        mul_877: "f32[58]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_7);  sum_109 = squeeze_7 = None
        convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_876, add_9, primals_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_876 = add_9 = primals_14 = None
        getitem_299: "f32[512, 24, 28, 28]" = convolution_backward_53[0]
        getitem_300: "f32[58, 24, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
        sum_110: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_299, [0, 2, 3])
        sub_272: "f32[512, 24, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_876);  convolution_1 = unsqueeze_876 = None
        mul_878: "f32[512, 24, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_299, sub_272)
        sum_111: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 2, 3]);  mul_878 = None
        mul_879: "f32[24]" = torch.ops.aten.mul.Tensor(sum_110, 2.4912308673469386e-06)
        unsqueeze_877: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
        unsqueeze_878: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
        unsqueeze_879: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
        mul_880: "f32[24]" = torch.ops.aten.mul.Tensor(sum_111, 2.4912308673469386e-06)
        mul_881: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_882: "f32[24]" = torch.ops.aten.mul.Tensor(mul_880, mul_881);  mul_880 = mul_881 = None
        unsqueeze_880: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
        unsqueeze_881: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, 2);  unsqueeze_880 = None
        unsqueeze_882: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 3);  unsqueeze_881 = None
        mul_883: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_883: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
        unsqueeze_884: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
        unsqueeze_885: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
        mul_884: "f32[512, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_882);  sub_272 = unsqueeze_882 = None
        sub_274: "f32[512, 24, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_299, mul_884);  getitem_299 = mul_884 = None
        sub_275: "f32[512, 24, 28, 28]" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_879);  sub_274 = unsqueeze_879 = None
        mul_885: "f32[512, 24, 28, 28]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_885);  sub_275 = unsqueeze_885 = None
        mul_886: "f32[24]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_4);  sum_111 = squeeze_4 = None
        convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_885, getitem_2, primals_8, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False]);  mul_885 = getitem_2 = primals_8 = None
        getitem_302: "f32[512, 24, 56, 56]" = convolution_backward_54[0]
        getitem_303: "f32[24, 1, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
        add_282: "f32[512, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_296, getitem_302);  getitem_296 = getitem_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:156 in _forward_impl, code: x = self.maxpool(x)
        full_default_36: "f32[12288, 12544]" = torch.ops.aten.full.default([12288, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_65: "f32[12288, 3136]" = torch.ops.aten.reshape.default(add_282, [12288, 3136]);  add_282 = None
        _low_memory_max_pool_offsets_to_indices: "i64[512, 24, 56, 56]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_3 = None
        view_66: "i64[12288, 3136]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [12288, 3136]);  _low_memory_max_pool_offsets_to_indices = None
        scatter_add: "f32[12288, 12544]" = torch.ops.aten.scatter_add.default(full_default_36, 1, view_66, view_65);  full_default_36 = view_66 = view_65 = None
        view_67: "f32[512, 24, 112, 112]" = torch.ops.aten.reshape.default(scatter_add, [512, 24, 112, 112]);  scatter_add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/shufflenetv2.py:155 in _forward_impl, code: x = self.conv1(x)
        sub: "f32[512, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[512, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[512, 24, 112, 112]" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[24, 1]" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[512, 24, 112, 112]" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        relu: "f32[512, 24, 112, 112]" = torch.ops.aten.relu.default(add_4);  add_4 = None
        le_36: "b8[512, 24, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_36: "f32[512, 24, 112, 112]" = torch.ops.aten.where.self(le_36, full_default, view_67);  le_36 = full_default = view_67 = None
        squeeze: "f32[24]" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_886: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_887: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
        unsqueeze_888: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
        sum_112: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
        sub_276: "f32[512, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_888);  convolution = unsqueeze_888 = None
        mul_887: "f32[512, 24, 112, 112]" = torch.ops.aten.mul.Tensor(where_36, sub_276)
        sum_113: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 2, 3]);  mul_887 = None
        mul_888: "f32[24]" = torch.ops.aten.mul.Tensor(sum_112, 1.5570192920918366e-07)
        unsqueeze_889: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
        unsqueeze_890: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
        unsqueeze_891: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
        mul_889: "f32[24]" = torch.ops.aten.mul.Tensor(sum_113, 1.5570192920918366e-07)
        squeeze_1: "f32[24]" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_890: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_891: "f32[24]" = torch.ops.aten.mul.Tensor(mul_889, mul_890);  mul_889 = mul_890 = None
        unsqueeze_892: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
        unsqueeze_893: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 2);  unsqueeze_892 = None
        unsqueeze_894: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 3);  unsqueeze_893 = None
        mul_892: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_895: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
        unsqueeze_896: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
        unsqueeze_897: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
        mul_893: "f32[512, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_894);  sub_276 = unsqueeze_894 = None
        sub_278: "f32[512, 24, 112, 112]" = torch.ops.aten.sub.Tensor(where_36, mul_893);  where_36 = mul_893 = None
        sub_279: "f32[512, 24, 112, 112]" = torch.ops.aten.sub.Tensor(sub_278, unsqueeze_891);  sub_278 = unsqueeze_891 = None
        mul_894: "f32[512, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_897);  sub_279 = unsqueeze_897 = None
        mul_895: "f32[24]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_1);  sum_113 = squeeze_1 = None
        convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_894, primals_2, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_894 = primals_2 = primals_1 = None
        getitem_306: "f32[24, 3, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
        return (getitem_306, None, None, None, None, mul_895, sum_112, getitem_303, None, None, None, mul_886, sum_110, getitem_300, None, None, None, mul_877, sum_108, getitem_297, None, None, None, mul_868, sum_106, getitem_294, None, None, None, mul_859, sum_104, getitem_291, None, None, None, mul_850, sum_102, getitem_288, None, None, None, mul_841, sum_100, getitem_285, None, None, None, mul_832, sum_98, getitem_282, None, None, None, mul_823, sum_96, getitem_279, None, None, None, mul_814, sum_94, getitem_276, None, None, None, mul_805, sum_92, getitem_273, None, None, None, mul_796, sum_90, getitem_270, None, None, None, mul_787, sum_88, getitem_267, None, None, None, mul_778, sum_86, getitem_264, None, None, None, mul_769, sum_84, getitem_261, None, None, None, mul_760, sum_82, getitem_258, None, None, None, mul_751, sum_80, getitem_255, None, None, None, mul_742, sum_78, getitem_252, None, None, None, mul_733, sum_76, getitem_249, None, None, None, mul_724, sum_74, getitem_246, None, None, None, mul_715, sum_72, getitem_243, None, None, None, mul_706, sum_70, getitem_240, None, None, None, mul_697, sum_68, getitem_237, None, None, None, mul_688, sum_66, getitem_234, None, None, None, mul_679, sum_64, getitem_231, None, None, None, mul_670, sum_62, getitem_228, None, None, None, mul_661, sum_60, getitem_225, None, None, None, mul_652, sum_58, getitem_222, None, None, None, mul_643, sum_56, getitem_219, None, None, None, mul_634, sum_54, getitem_216, None, None, None, mul_625, sum_52, getitem_213, None, None, None, mul_616, sum_50, getitem_210, None, None, None, mul_607, sum_48, getitem_207, None, None, None, mul_598, sum_46, getitem_204, None, None, None, mul_589, sum_44, getitem_201, None, None, None, mul_580, sum_42, getitem_198, None, None, None, mul_571, sum_40, getitem_195, None, None, None, mul_562, sum_38, getitem_192, None, None, None, mul_553, sum_36, getitem_189, None, None, None, mul_544, sum_34, getitem_186, None, None, None, mul_535, sum_32, getitem_183, None, None, None, mul_526, sum_30, getitem_180, None, None, None, mul_517, sum_28, getitem_177, None, None, None, mul_508, sum_26, getitem_174, None, None, None, mul_499, sum_24, getitem_171, None, None, None, mul_490, sum_22, getitem_168, None, None, None, mul_481, sum_20, getitem_165, None, None, None, mul_472, sum_18, getitem_162, None, None, None, mul_463, sum_16, getitem_159, None, None, None, mul_454, sum_14, getitem_156, None, None, None, mul_445, sum_12, getitem_153, None, None, None, mul_436, sum_10, getitem_150, None, None, None, mul_427, sum_8, getitem_147, None, None, None, mul_418, sum_6, getitem_144, None, None, None, mul_409, sum_4, getitem_141, None, None, None, mul_400, sum_2, mm_1, view_32)
