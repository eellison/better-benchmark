import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[256, 3, 224, 224]", primals_6: "f32[32]", primals_8: "f32[32, 1, 3, 3]", primals_12: "f32[32]", primals_14: "f32[16, 32, 1, 1]", primals_18: "f32[16]", primals_20: "f32[48, 16, 1, 1]", primals_24: "f32[48]", primals_26: "f32[48, 1, 3, 3]", primals_30: "f32[48]", primals_32: "f32[24, 48, 1, 1]", primals_36: "f32[24]", primals_38: "f32[72, 24, 1, 1]", primals_42: "f32[72]", primals_44: "f32[72, 1, 3, 3]", primals_48: "f32[72]", primals_50: "f32[24, 72, 1, 1]", primals_54: "f32[24]", primals_56: "f32[72, 24, 1, 1]", primals_60: "f32[72]", primals_62: "f32[72, 1, 3, 3]", primals_66: "f32[72]", primals_68: "f32[24, 72, 1, 1]", primals_72: "f32[24]", primals_74: "f32[72, 24, 1, 1]", primals_78: "f32[72]", primals_80: "f32[72, 1, 5, 5]", primals_84: "f32[72]", primals_86: "f32[40, 72, 1, 1]", primals_90: "f32[40]", primals_92: "f32[120, 40, 1, 1]", primals_96: "f32[120]", primals_98: "f32[120, 1, 5, 5]", primals_102: "f32[120]", primals_104: "f32[40, 120, 1, 1]", primals_108: "f32[40]", primals_110: "f32[120, 40, 1, 1]", primals_114: "f32[120]", primals_116: "f32[120, 1, 5, 5]", primals_120: "f32[120]", primals_122: "f32[40, 120, 1, 1]", primals_126: "f32[40]", primals_128: "f32[240, 40, 1, 1]", primals_132: "f32[240]", primals_134: "f32[240, 1, 5, 5]", primals_138: "f32[240]", primals_140: "f32[80, 240, 1, 1]", primals_144: "f32[80]", primals_146: "f32[480, 80, 1, 1]", primals_150: "f32[480]", primals_152: "f32[480, 1, 5, 5]", primals_156: "f32[480]", primals_158: "f32[80, 480, 1, 1]", primals_162: "f32[80]", primals_164: "f32[480, 80, 1, 1]", primals_168: "f32[480]", primals_170: "f32[480, 1, 5, 5]", primals_174: "f32[480]", primals_176: "f32[80, 480, 1, 1]", primals_180: "f32[80]", primals_182: "f32[480, 80, 1, 1]", primals_186: "f32[480]", primals_188: "f32[480, 1, 3, 3]", primals_192: "f32[480]", primals_194: "f32[96, 480, 1, 1]", primals_198: "f32[96]", primals_200: "f32[576, 96, 1, 1]", primals_204: "f32[576]", primals_206: "f32[576, 1, 3, 3]", primals_210: "f32[576]", primals_212: "f32[96, 576, 1, 1]", primals_216: "f32[96]", primals_218: "f32[576, 96, 1, 1]", primals_222: "f32[576]", primals_224: "f32[576, 1, 5, 5]", primals_228: "f32[576]", primals_230: "f32[192, 576, 1, 1]", primals_234: "f32[192]", primals_236: "f32[1152, 192, 1, 1]", primals_240: "f32[1152]", primals_242: "f32[1152, 1, 5, 5]", primals_246: "f32[1152]", primals_248: "f32[192, 1152, 1, 1]", primals_252: "f32[192]", primals_254: "f32[1152, 192, 1, 1]", primals_258: "f32[1152]", primals_260: "f32[1152, 1, 5, 5]", primals_264: "f32[1152]", primals_266: "f32[192, 1152, 1, 1]", primals_270: "f32[192]", primals_272: "f32[1152, 192, 1, 1]", primals_276: "f32[1152]", primals_278: "f32[1152, 1, 5, 5]", primals_282: "f32[1152]", primals_284: "f32[192, 1152, 1, 1]", primals_288: "f32[192]", primals_290: "f32[1152, 192, 1, 1]", primals_294: "f32[1152]", primals_296: "f32[1152, 1, 3, 3]", primals_300: "f32[1152]", primals_302: "f32[320, 1152, 1, 1]", primals_306: "f32[320]", primals_308: "f32[1280, 320, 1, 1]", primals_312: "f32[1280]", primals_313: "f32[1280]", primals_314: "f32[1000, 1280]", convolution: "f32[256, 32, 112, 112]", squeeze_1: "f32[32]", relu: "f32[256, 32, 112, 112]", convolution_1: "f32[256, 32, 112, 112]", squeeze_4: "f32[32]", relu_1: "f32[256, 32, 112, 112]", convolution_2: "f32[256, 16, 112, 112]", squeeze_7: "f32[16]", add_14: "f32[256, 16, 112, 112]", convolution_3: "f32[256, 48, 112, 112]", squeeze_10: "f32[48]", relu_2: "f32[256, 48, 112, 112]", convolution_4: "f32[256, 48, 56, 56]", squeeze_13: "f32[48]", relu_3: "f32[256, 48, 56, 56]", convolution_5: "f32[256, 24, 56, 56]", squeeze_16: "f32[24]", add_29: "f32[256, 24, 56, 56]", convolution_6: "f32[256, 72, 56, 56]", squeeze_19: "f32[72]", relu_4: "f32[256, 72, 56, 56]", convolution_7: "f32[256, 72, 56, 56]", squeeze_22: "f32[72]", relu_5: "f32[256, 72, 56, 56]", convolution_8: "f32[256, 24, 56, 56]", squeeze_25: "f32[24]", add_45: "f32[256, 24, 56, 56]", convolution_9: "f32[256, 72, 56, 56]", squeeze_28: "f32[72]", relu_6: "f32[256, 72, 56, 56]", convolution_10: "f32[256, 72, 56, 56]", squeeze_31: "f32[72]", relu_7: "f32[256, 72, 56, 56]", convolution_11: "f32[256, 24, 56, 56]", squeeze_34: "f32[24]", add_61: "f32[256, 24, 56, 56]", convolution_12: "f32[256, 72, 56, 56]", squeeze_37: "f32[72]", relu_8: "f32[256, 72, 56, 56]", convolution_13: "f32[256, 72, 28, 28]", squeeze_40: "f32[72]", relu_9: "f32[256, 72, 28, 28]", convolution_14: "f32[256, 40, 28, 28]", squeeze_43: "f32[40]", add_76: "f32[256, 40, 28, 28]", convolution_15: "f32[256, 120, 28, 28]", squeeze_46: "f32[120]", relu_10: "f32[256, 120, 28, 28]", convolution_16: "f32[256, 120, 28, 28]", squeeze_49: "f32[120]", relu_11: "f32[256, 120, 28, 28]", convolution_17: "f32[256, 40, 28, 28]", squeeze_52: "f32[40]", add_92: "f32[256, 40, 28, 28]", convolution_18: "f32[256, 120, 28, 28]", squeeze_55: "f32[120]", relu_12: "f32[256, 120, 28, 28]", convolution_19: "f32[256, 120, 28, 28]", squeeze_58: "f32[120]", relu_13: "f32[256, 120, 28, 28]", convolution_20: "f32[256, 40, 28, 28]", squeeze_61: "f32[40]", add_108: "f32[256, 40, 28, 28]", convolution_21: "f32[256, 240, 28, 28]", squeeze_64: "f32[240]", relu_14: "f32[256, 240, 28, 28]", convolution_22: "f32[256, 240, 14, 14]", squeeze_67: "f32[240]", relu_15: "f32[256, 240, 14, 14]", convolution_23: "f32[256, 80, 14, 14]", squeeze_70: "f32[80]", add_123: "f32[256, 80, 14, 14]", convolution_24: "f32[256, 480, 14, 14]", squeeze_73: "f32[480]", relu_16: "f32[256, 480, 14, 14]", convolution_25: "f32[256, 480, 14, 14]", squeeze_76: "f32[480]", relu_17: "f32[256, 480, 14, 14]", convolution_26: "f32[256, 80, 14, 14]", squeeze_79: "f32[80]", add_139: "f32[256, 80, 14, 14]", convolution_27: "f32[256, 480, 14, 14]", squeeze_82: "f32[480]", relu_18: "f32[256, 480, 14, 14]", convolution_28: "f32[256, 480, 14, 14]", squeeze_85: "f32[480]", relu_19: "f32[256, 480, 14, 14]", convolution_29: "f32[256, 80, 14, 14]", squeeze_88: "f32[80]", add_155: "f32[256, 80, 14, 14]", convolution_30: "f32[256, 480, 14, 14]", squeeze_91: "f32[480]", relu_20: "f32[256, 480, 14, 14]", convolution_31: "f32[256, 480, 14, 14]", squeeze_94: "f32[480]", relu_21: "f32[256, 480, 14, 14]", convolution_32: "f32[256, 96, 14, 14]", squeeze_97: "f32[96]", add_170: "f32[256, 96, 14, 14]", convolution_33: "f32[256, 576, 14, 14]", squeeze_100: "f32[576]", relu_22: "f32[256, 576, 14, 14]", convolution_34: "f32[256, 576, 14, 14]", squeeze_103: "f32[576]", relu_23: "f32[256, 576, 14, 14]", convolution_35: "f32[256, 96, 14, 14]", squeeze_106: "f32[96]", add_186: "f32[256, 96, 14, 14]", convolution_36: "f32[256, 576, 14, 14]", squeeze_109: "f32[576]", relu_24: "f32[256, 576, 14, 14]", convolution_37: "f32[256, 576, 7, 7]", squeeze_112: "f32[576]", relu_25: "f32[256, 576, 7, 7]", convolution_38: "f32[256, 192, 7, 7]", squeeze_115: "f32[192]", add_201: "f32[256, 192, 7, 7]", convolution_39: "f32[256, 1152, 7, 7]", squeeze_118: "f32[1152]", relu_26: "f32[256, 1152, 7, 7]", convolution_40: "f32[256, 1152, 7, 7]", squeeze_121: "f32[1152]", relu_27: "f32[256, 1152, 7, 7]", convolution_41: "f32[256, 192, 7, 7]", squeeze_124: "f32[192]", add_217: "f32[256, 192, 7, 7]", convolution_42: "f32[256, 1152, 7, 7]", squeeze_127: "f32[1152]", relu_28: "f32[256, 1152, 7, 7]", convolution_43: "f32[256, 1152, 7, 7]", squeeze_130: "f32[1152]", relu_29: "f32[256, 1152, 7, 7]", convolution_44: "f32[256, 192, 7, 7]", squeeze_133: "f32[192]", add_233: "f32[256, 192, 7, 7]", convolution_45: "f32[256, 1152, 7, 7]", squeeze_136: "f32[1152]", relu_30: "f32[256, 1152, 7, 7]", convolution_46: "f32[256, 1152, 7, 7]", squeeze_139: "f32[1152]", relu_31: "f32[256, 1152, 7, 7]", convolution_47: "f32[256, 192, 7, 7]", squeeze_142: "f32[192]", add_249: "f32[256, 192, 7, 7]", convolution_48: "f32[256, 1152, 7, 7]", squeeze_145: "f32[1152]", relu_32: "f32[256, 1152, 7, 7]", convolution_49: "f32[256, 1152, 7, 7]", squeeze_148: "f32[1152]", relu_33: "f32[256, 1152, 7, 7]", convolution_50: "f32[256, 320, 7, 7]", squeeze_151: "f32[320]", add_264: "f32[256, 320, 7, 7]", convolution_51: "f32[256, 1280, 7, 7]", getitem_103: "f32[1, 1280, 1, 1]", rsqrt_51: "f32[1, 1280, 1, 1]", lt: "b8[256, 1280]", mul_364: "f32[256, 1280]", unsqueeze_224: "f32[1, 320, 1, 1]", unsqueeze_236: "f32[1, 1152, 1, 1]", unsqueeze_248: "f32[1, 1152, 1, 1]", unsqueeze_260: "f32[1, 192, 1, 1]", unsqueeze_272: "f32[1, 1152, 1, 1]", unsqueeze_284: "f32[1, 1152, 1, 1]", unsqueeze_296: "f32[1, 192, 1, 1]", unsqueeze_308: "f32[1, 1152, 1, 1]", unsqueeze_320: "f32[1, 1152, 1, 1]", unsqueeze_332: "f32[1, 192, 1, 1]", unsqueeze_344: "f32[1, 1152, 1, 1]", unsqueeze_356: "f32[1, 1152, 1, 1]", unsqueeze_368: "f32[1, 192, 1, 1]", unsqueeze_380: "f32[1, 576, 1, 1]", unsqueeze_392: "f32[1, 576, 1, 1]", unsqueeze_404: "f32[1, 96, 1, 1]", unsqueeze_416: "f32[1, 576, 1, 1]", unsqueeze_428: "f32[1, 576, 1, 1]", unsqueeze_440: "f32[1, 96, 1, 1]", unsqueeze_452: "f32[1, 480, 1, 1]", unsqueeze_464: "f32[1, 480, 1, 1]", unsqueeze_476: "f32[1, 80, 1, 1]", unsqueeze_488: "f32[1, 480, 1, 1]", unsqueeze_500: "f32[1, 480, 1, 1]", unsqueeze_512: "f32[1, 80, 1, 1]", unsqueeze_524: "f32[1, 480, 1, 1]", unsqueeze_536: "f32[1, 480, 1, 1]", unsqueeze_548: "f32[1, 80, 1, 1]", unsqueeze_560: "f32[1, 240, 1, 1]", unsqueeze_572: "f32[1, 240, 1, 1]", unsqueeze_584: "f32[1, 40, 1, 1]", unsqueeze_596: "f32[1, 120, 1, 1]", unsqueeze_608: "f32[1, 120, 1, 1]", unsqueeze_620: "f32[1, 40, 1, 1]", unsqueeze_632: "f32[1, 120, 1, 1]", unsqueeze_644: "f32[1, 120, 1, 1]", unsqueeze_656: "f32[1, 40, 1, 1]", unsqueeze_668: "f32[1, 72, 1, 1]", unsqueeze_680: "f32[1, 72, 1, 1]", unsqueeze_692: "f32[1, 24, 1, 1]", unsqueeze_704: "f32[1, 72, 1, 1]", unsqueeze_716: "f32[1, 72, 1, 1]", unsqueeze_728: "f32[1, 24, 1, 1]", unsqueeze_740: "f32[1, 72, 1, 1]", unsqueeze_752: "f32[1, 72, 1, 1]", unsqueeze_764: "f32[1, 24, 1, 1]", unsqueeze_776: "f32[1, 48, 1, 1]", unsqueeze_788: "f32[1, 48, 1, 1]", unsqueeze_800: "f32[1, 16, 1, 1]", unsqueeze_812: "f32[1, 32, 1, 1]", unsqueeze_824: "f32[1, 32, 1, 1]", tangents_1: "f32[256, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:162 in forward, code: return self.classifier(x)
        permute: "f32[1280, 1000]" = torch.ops.aten.permute.default(primals_314, [1, 0]);  primals_314 = None
        permute_1: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[256, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "f32[1000, 256]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, mul_364);  permute_2 = mul_364 = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type: "f32[256, 1280]" = torch.ops.prims.convert_element_type.default(lt, torch.float32);  lt = None
        div: "f32[256, 1280]" = torch.ops.aten.div.Scalar(convert_element_type, 0.8);  convert_element_type = None
        mul_365: "f32[256, 1280]" = torch.ops.aten.mul.Tensor(mm, div);  mm = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:161 in forward, code: x = x.mean([2, 3])
        unsqueeze_208: "f32[256, 1280, 1]" = torch.ops.aten.unsqueeze.default(mul_365, 2);  mul_365 = None
        unsqueeze_209: "f32[256, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 3);  unsqueeze_208 = None
        expand: "f32[256, 1280, 7, 7]" = torch.ops.aten.expand.default(unsqueeze_209, [256, 1280, 7, 7]);  unsqueeze_209 = None
        div_1: "f32[256, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:159 in forward, code: x = self.layers(x)
        sub_51: "f32[256, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, getitem_103)
        mul_357: "f32[256, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        unsqueeze_204: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_312, -1)
        unsqueeze_205: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_363: "f32[256, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(mul_357, unsqueeze_205);  mul_357 = unsqueeze_205 = None
        unsqueeze_206: "f32[1280, 1]" = torch.ops.aten.unsqueeze.default(primals_313, -1);  primals_313 = None
        unsqueeze_207: "f32[1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_269: "f32[256, 1280, 7, 7]" = torch.ops.aten.add.Tensor(mul_363, unsqueeze_207);  mul_363 = unsqueeze_207 = None
        relu_34: "f32[256, 1280, 7, 7]" = torch.ops.aten.relu.default(add_269);  add_269 = None
        le: "b8[256, 1280, 7, 7]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[256, 1280, 7, 7]" = torch.ops.aten.where.self(le, full_default, div_1);  le = div_1 = None
        squeeze_153: "f32[1280]" = torch.ops.aten.squeeze.dims(getitem_103, [0, 2, 3]);  getitem_103 = None
        unsqueeze_210: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(squeeze_153, 0);  squeeze_153 = None
        unsqueeze_211: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 2);  unsqueeze_210 = None
        unsqueeze_212: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 3);  unsqueeze_211 = None
        sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_52: "f32[256, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_212);  convolution_51 = unsqueeze_212 = None
        mul_366: "f32[256, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_52)
        sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 2, 3]);  mul_366 = None
        mul_367: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, 7.971938775510203e-05)
        unsqueeze_213: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
        unsqueeze_214: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
        unsqueeze_215: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
        mul_368: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, 7.971938775510203e-05)
        squeeze_154: "f32[1280]" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_369: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
        mul_370: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_368, mul_369);  mul_368 = mul_369 = None
        unsqueeze_216: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
        unsqueeze_217: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 2);  unsqueeze_216 = None
        unsqueeze_218: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 3);  unsqueeze_217 = None
        mul_371: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_312);  primals_312 = None
        unsqueeze_219: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
        unsqueeze_220: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
        unsqueeze_221: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 3);  unsqueeze_220 = None
        mul_372: "f32[256, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_218);  sub_52 = unsqueeze_218 = None
        sub_54: "f32[256, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_372);  where = mul_372 = None
        sub_55: "f32[256, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_54, unsqueeze_215);  sub_54 = unsqueeze_215 = None
        mul_373: "f32[256, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_221);  sub_55 = unsqueeze_221 = None
        mul_374: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_154);  sum_3 = squeeze_154 = None
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_373, add_264, primals_308, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_373 = add_264 = primals_308 = None
        getitem_104: "f32[256, 320, 7, 7]" = convolution_backward[0]
        getitem_105: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:63 in forward, code: return self.layers(input)
        sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem_104, [0, 2, 3])
        sub_56: "f32[256, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_224);  convolution_50 = unsqueeze_224 = None
        mul_375: "f32[256, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_104, sub_56)
        sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 2, 3]);  mul_375 = None
        mul_376: "f32[320]" = torch.ops.aten.mul.Tensor(sum_4, 7.971938775510203e-05)
        unsqueeze_225: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
        unsqueeze_226: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
        unsqueeze_227: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
        mul_377: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, 7.971938775510203e-05)
        mul_378: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
        mul_379: "f32[320]" = torch.ops.aten.mul.Tensor(mul_377, mul_378);  mul_377 = mul_378 = None
        unsqueeze_228: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
        unsqueeze_229: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 2);  unsqueeze_228 = None
        unsqueeze_230: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 3);  unsqueeze_229 = None
        mul_380: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_306);  primals_306 = None
        unsqueeze_231: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
        unsqueeze_232: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
        unsqueeze_233: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
        mul_381: "f32[256, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_230);  sub_56 = unsqueeze_230 = None
        sub_58: "f32[256, 320, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_104, mul_381);  getitem_104 = mul_381 = None
        sub_59: "f32[256, 320, 7, 7]" = torch.ops.aten.sub.Tensor(sub_58, unsqueeze_227);  sub_58 = unsqueeze_227 = None
        mul_382: "f32[256, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_233);  sub_59 = unsqueeze_233 = None
        mul_383: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_151);  sum_5 = squeeze_151 = None
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_382, relu_33, primals_302, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_382 = primals_302 = None
        getitem_107: "f32[256, 1152, 7, 7]" = convolution_backward_1[0]
        getitem_108: "f32[320, 1152, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
        le_1: "b8[256, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_1: "f32[256, 1152, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, getitem_107);  le_1 = getitem_107 = None
        sum_6: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_60: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_236);  convolution_49 = unsqueeze_236 = None
        mul_384: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_60)
        sum_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 2, 3]);  mul_384 = None
        mul_385: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_6, 7.971938775510203e-05)
        unsqueeze_237: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
        unsqueeze_238: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
        unsqueeze_239: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
        mul_386: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_7, 7.971938775510203e-05)
        mul_387: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
        mul_388: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
        unsqueeze_240: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
        unsqueeze_241: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
        unsqueeze_242: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
        mul_389: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_300);  primals_300 = None
        unsqueeze_243: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_389, 0);  mul_389 = None
        unsqueeze_244: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
        unsqueeze_245: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
        mul_390: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_242);  sub_60 = unsqueeze_242 = None
        sub_62: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_390);  where_1 = mul_390 = None
        sub_63: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_239);  sub_62 = unsqueeze_239 = None
        mul_391: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_245);  sub_63 = unsqueeze_245 = None
        mul_392: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_148);  sum_7 = squeeze_148 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_391, relu_32, primals_296, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_391 = primals_296 = None
        getitem_110: "f32[256, 1152, 7, 7]" = convolution_backward_2[0]
        getitem_111: "f32[1152, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
        le_2: "b8[256, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_2: "f32[256, 1152, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_110);  le_2 = getitem_110 = None
        sum_8: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_64: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_248);  convolution_48 = unsqueeze_248 = None
        mul_393: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_64)
        sum_9: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 2, 3]);  mul_393 = None
        mul_394: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_8, 7.971938775510203e-05)
        unsqueeze_249: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
        unsqueeze_250: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
        unsqueeze_251: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
        mul_395: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, 7.971938775510203e-05)
        mul_396: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
        mul_397: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_395, mul_396);  mul_395 = mul_396 = None
        unsqueeze_252: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
        unsqueeze_253: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 2);  unsqueeze_252 = None
        unsqueeze_254: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 3);  unsqueeze_253 = None
        mul_398: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_294);  primals_294 = None
        unsqueeze_255: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_398, 0);  mul_398 = None
        unsqueeze_256: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
        unsqueeze_257: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
        mul_399: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_254);  sub_64 = unsqueeze_254 = None
        sub_66: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_399);  where_2 = mul_399 = None
        sub_67: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_251);  sub_66 = unsqueeze_251 = None
        mul_400: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_257);  sub_67 = unsqueeze_257 = None
        mul_401: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_145);  sum_9 = squeeze_145 = None
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_400, add_249, primals_290, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_400 = add_249 = primals_290 = None
        getitem_113: "f32[256, 192, 7, 7]" = convolution_backward_3[0]
        getitem_114: "f32[1152, 192, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:61 in forward, code: return self.layers(input) + input
        sum_10: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3])
        sub_68: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_260);  convolution_47 = unsqueeze_260 = None
        mul_402: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, sub_68)
        sum_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_402, [0, 2, 3]);  mul_402 = None
        mul_403: "f32[192]" = torch.ops.aten.mul.Tensor(sum_10, 7.971938775510203e-05)
        unsqueeze_261: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_403, 0);  mul_403 = None
        unsqueeze_262: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
        unsqueeze_263: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
        mul_404: "f32[192]" = torch.ops.aten.mul.Tensor(sum_11, 7.971938775510203e-05)
        mul_405: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
        mul_406: "f32[192]" = torch.ops.aten.mul.Tensor(mul_404, mul_405);  mul_404 = mul_405 = None
        unsqueeze_264: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_406, 0);  mul_406 = None
        unsqueeze_265: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 2);  unsqueeze_264 = None
        unsqueeze_266: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 3);  unsqueeze_265 = None
        mul_407: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_288);  primals_288 = None
        unsqueeze_267: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_407, 0);  mul_407 = None
        unsqueeze_268: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
        unsqueeze_269: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
        mul_408: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_266);  sub_68 = unsqueeze_266 = None
        sub_70: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_113, mul_408);  mul_408 = None
        sub_71: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_263);  sub_70 = unsqueeze_263 = None
        mul_409: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_269);  sub_71 = unsqueeze_269 = None
        mul_410: "f32[192]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_142);  sum_11 = squeeze_142 = None
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_409, relu_31, primals_284, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_409 = primals_284 = None
        getitem_116: "f32[256, 1152, 7, 7]" = convolution_backward_4[0]
        getitem_117: "f32[192, 1152, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
        le_3: "b8[256, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_3: "f32[256, 1152, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, getitem_116);  le_3 = getitem_116 = None
        sum_12: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_72: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_272);  convolution_46 = unsqueeze_272 = None
        mul_411: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_72)
        sum_13: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_411, [0, 2, 3]);  mul_411 = None
        mul_412: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_12, 7.971938775510203e-05)
        unsqueeze_273: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
        unsqueeze_274: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
        unsqueeze_275: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
        mul_413: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_13, 7.971938775510203e-05)
        mul_414: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
        mul_415: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_413, mul_414);  mul_413 = mul_414 = None
        unsqueeze_276: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_415, 0);  mul_415 = None
        unsqueeze_277: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 2);  unsqueeze_276 = None
        unsqueeze_278: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 3);  unsqueeze_277 = None
        mul_416: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_282);  primals_282 = None
        unsqueeze_279: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_416, 0);  mul_416 = None
        unsqueeze_280: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
        unsqueeze_281: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
        mul_417: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_278);  sub_72 = unsqueeze_278 = None
        sub_74: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_417);  where_3 = mul_417 = None
        sub_75: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_275);  sub_74 = unsqueeze_275 = None
        mul_418: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_281);  sub_75 = unsqueeze_281 = None
        mul_419: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_139);  sum_13 = squeeze_139 = None
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_418, relu_30, primals_278, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_418 = primals_278 = None
        getitem_119: "f32[256, 1152, 7, 7]" = convolution_backward_5[0]
        getitem_120: "f32[1152, 1, 5, 5]" = convolution_backward_5[1];  convolution_backward_5 = None
        le_4: "b8[256, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_4: "f32[256, 1152, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, getitem_119);  le_4 = getitem_119 = None
        sum_14: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_76: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_284);  convolution_45 = unsqueeze_284 = None
        mul_420: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_76)
        sum_15: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_420, [0, 2, 3]);  mul_420 = None
        mul_421: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_14, 7.971938775510203e-05)
        unsqueeze_285: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
        unsqueeze_286: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
        unsqueeze_287: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
        mul_422: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, 7.971938775510203e-05)
        mul_423: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
        mul_424: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_422, mul_423);  mul_422 = mul_423 = None
        unsqueeze_288: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_424, 0);  mul_424 = None
        unsqueeze_289: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 2);  unsqueeze_288 = None
        unsqueeze_290: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 3);  unsqueeze_289 = None
        mul_425: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_276);  primals_276 = None
        unsqueeze_291: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_425, 0);  mul_425 = None
        unsqueeze_292: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
        unsqueeze_293: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
        mul_426: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_290);  sub_76 = unsqueeze_290 = None
        sub_78: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_426);  where_4 = mul_426 = None
        sub_79: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_287);  sub_78 = unsqueeze_287 = None
        mul_427: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_293);  sub_79 = unsqueeze_293 = None
        mul_428: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_136);  sum_15 = squeeze_136 = None
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_427, add_233, primals_272, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_427 = add_233 = primals_272 = None
        getitem_122: "f32[256, 192, 7, 7]" = convolution_backward_6[0]
        getitem_123: "f32[1152, 192, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
        add_270: "f32[256, 192, 7, 7]" = torch.ops.aten.add.Tensor(getitem_113, getitem_122);  getitem_113 = getitem_122 = None
        sum_16: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_270, [0, 2, 3])
        sub_80: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_296);  convolution_44 = unsqueeze_296 = None
        mul_429: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_270, sub_80)
        sum_17: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 2, 3]);  mul_429 = None
        mul_430: "f32[192]" = torch.ops.aten.mul.Tensor(sum_16, 7.971938775510203e-05)
        unsqueeze_297: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_430, 0);  mul_430 = None
        unsqueeze_298: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
        unsqueeze_299: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
        mul_431: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, 7.971938775510203e-05)
        mul_432: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
        mul_433: "f32[192]" = torch.ops.aten.mul.Tensor(mul_431, mul_432);  mul_431 = mul_432 = None
        unsqueeze_300: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_433, 0);  mul_433 = None
        unsqueeze_301: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
        unsqueeze_302: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
        mul_434: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_270);  primals_270 = None
        unsqueeze_303: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_434, 0);  mul_434 = None
        unsqueeze_304: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
        unsqueeze_305: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
        mul_435: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_302);  sub_80 = unsqueeze_302 = None
        sub_82: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_270, mul_435);  mul_435 = None
        sub_83: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_299);  sub_82 = unsqueeze_299 = None
        mul_436: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_305);  sub_83 = unsqueeze_305 = None
        mul_437: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_133);  sum_17 = squeeze_133 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_436, relu_29, primals_266, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_436 = primals_266 = None
        getitem_125: "f32[256, 1152, 7, 7]" = convolution_backward_7[0]
        getitem_126: "f32[192, 1152, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
        le_5: "b8[256, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_5: "f32[256, 1152, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, getitem_125);  le_5 = getitem_125 = None
        sum_18: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_84: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_308);  convolution_43 = unsqueeze_308 = None
        mul_438: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_84)
        sum_19: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 2, 3]);  mul_438 = None
        mul_439: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_18, 7.971938775510203e-05)
        unsqueeze_309: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
        unsqueeze_310: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
        unsqueeze_311: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
        mul_440: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, 7.971938775510203e-05)
        mul_441: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
        mul_442: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_440, mul_441);  mul_440 = mul_441 = None
        unsqueeze_312: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
        unsqueeze_313: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
        unsqueeze_314: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
        mul_443: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_264);  primals_264 = None
        unsqueeze_315: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_443, 0);  mul_443 = None
        unsqueeze_316: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
        unsqueeze_317: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 3);  unsqueeze_316 = None
        mul_444: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_314);  sub_84 = unsqueeze_314 = None
        sub_86: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_444);  where_5 = mul_444 = None
        sub_87: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_311);  sub_86 = unsqueeze_311 = None
        mul_445: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_317);  sub_87 = unsqueeze_317 = None
        mul_446: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_130);  sum_19 = squeeze_130 = None
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_445, relu_28, primals_260, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_445 = primals_260 = None
        getitem_128: "f32[256, 1152, 7, 7]" = convolution_backward_8[0]
        getitem_129: "f32[1152, 1, 5, 5]" = convolution_backward_8[1];  convolution_backward_8 = None
        le_6: "b8[256, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_6: "f32[256, 1152, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, getitem_128);  le_6 = getitem_128 = None
        sum_20: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_88: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_320);  convolution_42 = unsqueeze_320 = None
        mul_447: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_88)
        sum_21: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_447, [0, 2, 3]);  mul_447 = None
        mul_448: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_20, 7.971938775510203e-05)
        unsqueeze_321: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
        unsqueeze_322: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
        unsqueeze_323: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
        mul_449: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, 7.971938775510203e-05)
        mul_450: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
        mul_451: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
        unsqueeze_324: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_451, 0);  mul_451 = None
        unsqueeze_325: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 2);  unsqueeze_324 = None
        unsqueeze_326: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 3);  unsqueeze_325 = None
        mul_452: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_258);  primals_258 = None
        unsqueeze_327: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
        unsqueeze_328: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
        unsqueeze_329: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 3);  unsqueeze_328 = None
        mul_453: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_326);  sub_88 = unsqueeze_326 = None
        sub_90: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_453);  where_6 = mul_453 = None
        sub_91: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_323);  sub_90 = unsqueeze_323 = None
        mul_454: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_329);  sub_91 = unsqueeze_329 = None
        mul_455: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_127);  sum_21 = squeeze_127 = None
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_454, add_217, primals_254, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_454 = add_217 = primals_254 = None
        getitem_131: "f32[256, 192, 7, 7]" = convolution_backward_9[0]
        getitem_132: "f32[1152, 192, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
        add_271: "f32[256, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_270, getitem_131);  add_270 = getitem_131 = None
        sum_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_271, [0, 2, 3])
        sub_92: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_332);  convolution_41 = unsqueeze_332 = None
        mul_456: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_271, sub_92)
        sum_23: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_456, [0, 2, 3]);  mul_456 = None
        mul_457: "f32[192]" = torch.ops.aten.mul.Tensor(sum_22, 7.971938775510203e-05)
        unsqueeze_333: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_457, 0);  mul_457 = None
        unsqueeze_334: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
        unsqueeze_335: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
        mul_458: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, 7.971938775510203e-05)
        mul_459: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
        mul_460: "f32[192]" = torch.ops.aten.mul.Tensor(mul_458, mul_459);  mul_458 = mul_459 = None
        unsqueeze_336: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_460, 0);  mul_460 = None
        unsqueeze_337: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 2);  unsqueeze_336 = None
        unsqueeze_338: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 3);  unsqueeze_337 = None
        mul_461: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_252);  primals_252 = None
        unsqueeze_339: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_461, 0);  mul_461 = None
        unsqueeze_340: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
        unsqueeze_341: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 3);  unsqueeze_340 = None
        mul_462: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_338);  sub_92 = unsqueeze_338 = None
        sub_94: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_271, mul_462);  mul_462 = None
        sub_95: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_335);  sub_94 = unsqueeze_335 = None
        mul_463: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_341);  sub_95 = unsqueeze_341 = None
        mul_464: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_124);  sum_23 = squeeze_124 = None
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_463, relu_27, primals_248, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_463 = primals_248 = None
        getitem_134: "f32[256, 1152, 7, 7]" = convolution_backward_10[0]
        getitem_135: "f32[192, 1152, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
        le_7: "b8[256, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_7: "f32[256, 1152, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, getitem_134);  le_7 = getitem_134 = None
        sum_24: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_96: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_344);  convolution_40 = unsqueeze_344 = None
        mul_465: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_96)
        sum_25: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_465, [0, 2, 3]);  mul_465 = None
        mul_466: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_24, 7.971938775510203e-05)
        unsqueeze_345: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_466, 0);  mul_466 = None
        unsqueeze_346: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
        unsqueeze_347: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
        mul_467: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_25, 7.971938775510203e-05)
        mul_468: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
        mul_469: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_467, mul_468);  mul_467 = mul_468 = None
        unsqueeze_348: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
        unsqueeze_349: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 2);  unsqueeze_348 = None
        unsqueeze_350: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 3);  unsqueeze_349 = None
        mul_470: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_246);  primals_246 = None
        unsqueeze_351: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_470, 0);  mul_470 = None
        unsqueeze_352: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
        unsqueeze_353: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
        mul_471: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_350);  sub_96 = unsqueeze_350 = None
        sub_98: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_471);  where_7 = mul_471 = None
        sub_99: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_347);  sub_98 = unsqueeze_347 = None
        mul_472: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_353);  sub_99 = unsqueeze_353 = None
        mul_473: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_121);  sum_25 = squeeze_121 = None
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_472, relu_26, primals_242, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_472 = primals_242 = None
        getitem_137: "f32[256, 1152, 7, 7]" = convolution_backward_11[0]
        getitem_138: "f32[1152, 1, 5, 5]" = convolution_backward_11[1];  convolution_backward_11 = None
        le_8: "b8[256, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_8: "f32[256, 1152, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, getitem_137);  le_8 = getitem_137 = None
        sum_26: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_100: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_356);  convolution_39 = unsqueeze_356 = None
        mul_474: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_100)
        sum_27: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_474, [0, 2, 3]);  mul_474 = None
        mul_475: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_26, 7.971938775510203e-05)
        unsqueeze_357: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
        unsqueeze_358: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
        unsqueeze_359: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
        mul_476: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, 7.971938775510203e-05)
        mul_477: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
        mul_478: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
        unsqueeze_360: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
        unsqueeze_361: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 2);  unsqueeze_360 = None
        unsqueeze_362: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 3);  unsqueeze_361 = None
        mul_479: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_240);  primals_240 = None
        unsqueeze_363: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_479, 0);  mul_479 = None
        unsqueeze_364: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
        unsqueeze_365: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
        mul_480: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_362);  sub_100 = unsqueeze_362 = None
        sub_102: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_480);  where_8 = mul_480 = None
        sub_103: "f32[256, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_359);  sub_102 = unsqueeze_359 = None
        mul_481: "f32[256, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_365);  sub_103 = unsqueeze_365 = None
        mul_482: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_118);  sum_27 = squeeze_118 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_481, add_201, primals_236, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_481 = add_201 = primals_236 = None
        getitem_140: "f32[256, 192, 7, 7]" = convolution_backward_12[0]
        getitem_141: "f32[1152, 192, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
        add_272: "f32[256, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_271, getitem_140);  add_271 = getitem_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:63 in forward, code: return self.layers(input)
        sum_28: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 2, 3])
        sub_104: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_368);  convolution_38 = unsqueeze_368 = None
        mul_483: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_272, sub_104)
        sum_29: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 2, 3]);  mul_483 = None
        mul_484: "f32[192]" = torch.ops.aten.mul.Tensor(sum_28, 7.971938775510203e-05)
        unsqueeze_369: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
        unsqueeze_370: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
        unsqueeze_371: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
        mul_485: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, 7.971938775510203e-05)
        mul_486: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_487: "f32[192]" = torch.ops.aten.mul.Tensor(mul_485, mul_486);  mul_485 = mul_486 = None
        unsqueeze_372: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
        unsqueeze_373: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 2);  unsqueeze_372 = None
        unsqueeze_374: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 3);  unsqueeze_373 = None
        mul_488: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_234);  primals_234 = None
        unsqueeze_375: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
        unsqueeze_376: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
        unsqueeze_377: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
        mul_489: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_374);  sub_104 = unsqueeze_374 = None
        sub_106: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_272, mul_489);  add_272 = mul_489 = None
        sub_107: "f32[256, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_371);  sub_106 = unsqueeze_371 = None
        mul_490: "f32[256, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_377);  sub_107 = unsqueeze_377 = None
        mul_491: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_115);  sum_29 = squeeze_115 = None
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_490, relu_25, primals_230, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_490 = primals_230 = None
        getitem_143: "f32[256, 576, 7, 7]" = convolution_backward_13[0]
        getitem_144: "f32[192, 576, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
        le_9: "b8[256, 576, 7, 7]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_9: "f32[256, 576, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, getitem_143);  le_9 = getitem_143 = None
        sum_30: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_108: "f32[256, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_380);  convolution_37 = unsqueeze_380 = None
        mul_492: "f32[256, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_108)
        sum_31: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_492, [0, 2, 3]);  mul_492 = None
        mul_493: "f32[576]" = torch.ops.aten.mul.Tensor(sum_30, 7.971938775510203e-05)
        unsqueeze_381: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
        unsqueeze_382: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
        unsqueeze_383: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
        mul_494: "f32[576]" = torch.ops.aten.mul.Tensor(sum_31, 7.971938775510203e-05)
        mul_495: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_496: "f32[576]" = torch.ops.aten.mul.Tensor(mul_494, mul_495);  mul_494 = mul_495 = None
        unsqueeze_384: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
        unsqueeze_385: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
        unsqueeze_386: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
        mul_497: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_228);  primals_228 = None
        unsqueeze_387: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_497, 0);  mul_497 = None
        unsqueeze_388: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
        unsqueeze_389: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
        mul_498: "f32[256, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_386);  sub_108 = unsqueeze_386 = None
        sub_110: "f32[256, 576, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_498);  where_9 = mul_498 = None
        sub_111: "f32[256, 576, 7, 7]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_383);  sub_110 = unsqueeze_383 = None
        mul_499: "f32[256, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_389);  sub_111 = unsqueeze_389 = None
        mul_500: "f32[576]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_112);  sum_31 = squeeze_112 = None
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_499, relu_24, primals_224, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_499 = primals_224 = None
        getitem_146: "f32[256, 576, 14, 14]" = convolution_backward_14[0]
        getitem_147: "f32[576, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
        le_10: "b8[256, 576, 14, 14]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_10: "f32[256, 576, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, getitem_146);  le_10 = getitem_146 = None
        sum_32: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_112: "f32[256, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_392);  convolution_36 = unsqueeze_392 = None
        mul_501: "f32[256, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_112)
        sum_33: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_501, [0, 2, 3]);  mul_501 = None
        mul_502: "f32[576]" = torch.ops.aten.mul.Tensor(sum_32, 1.992984693877551e-05)
        unsqueeze_393: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
        unsqueeze_394: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
        unsqueeze_395: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
        mul_503: "f32[576]" = torch.ops.aten.mul.Tensor(sum_33, 1.992984693877551e-05)
        mul_504: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_505: "f32[576]" = torch.ops.aten.mul.Tensor(mul_503, mul_504);  mul_503 = mul_504 = None
        unsqueeze_396: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
        unsqueeze_397: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 2);  unsqueeze_396 = None
        unsqueeze_398: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 3);  unsqueeze_397 = None
        mul_506: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_222);  primals_222 = None
        unsqueeze_399: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_506, 0);  mul_506 = None
        unsqueeze_400: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
        unsqueeze_401: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
        mul_507: "f32[256, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_398);  sub_112 = unsqueeze_398 = None
        sub_114: "f32[256, 576, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_507);  where_10 = mul_507 = None
        sub_115: "f32[256, 576, 14, 14]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_395);  sub_114 = unsqueeze_395 = None
        mul_508: "f32[256, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_401);  sub_115 = unsqueeze_401 = None
        mul_509: "f32[576]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_109);  sum_33 = squeeze_109 = None
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_508, add_186, primals_218, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_508 = add_186 = primals_218 = None
        getitem_149: "f32[256, 96, 14, 14]" = convolution_backward_15[0]
        getitem_150: "f32[576, 96, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:61 in forward, code: return self.layers(input) + input
        sum_34: "f32[96]" = torch.ops.aten.sum.dim_IntList(getitem_149, [0, 2, 3])
        sub_116: "f32[256, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_404);  convolution_35 = unsqueeze_404 = None
        mul_510: "f32[256, 96, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_149, sub_116)
        sum_35: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 2, 3]);  mul_510 = None
        mul_511: "f32[96]" = torch.ops.aten.mul.Tensor(sum_34, 1.992984693877551e-05)
        unsqueeze_405: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_406: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
        unsqueeze_407: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
        mul_512: "f32[96]" = torch.ops.aten.mul.Tensor(sum_35, 1.992984693877551e-05)
        mul_513: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_514: "f32[96]" = torch.ops.aten.mul.Tensor(mul_512, mul_513);  mul_512 = mul_513 = None
        unsqueeze_408: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
        unsqueeze_409: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 2);  unsqueeze_408 = None
        unsqueeze_410: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 3);  unsqueeze_409 = None
        mul_515: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_216);  primals_216 = None
        unsqueeze_411: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_515, 0);  mul_515 = None
        unsqueeze_412: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
        unsqueeze_413: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
        mul_516: "f32[256, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_410);  sub_116 = unsqueeze_410 = None
        sub_118: "f32[256, 96, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_149, mul_516);  mul_516 = None
        sub_119: "f32[256, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_407);  sub_118 = unsqueeze_407 = None
        mul_517: "f32[256, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_413);  sub_119 = unsqueeze_413 = None
        mul_518: "f32[96]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_106);  sum_35 = squeeze_106 = None
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_517, relu_23, primals_212, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_517 = primals_212 = None
        getitem_152: "f32[256, 576, 14, 14]" = convolution_backward_16[0]
        getitem_153: "f32[96, 576, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
        le_11: "b8[256, 576, 14, 14]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_11: "f32[256, 576, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, getitem_152);  le_11 = getitem_152 = None
        sum_36: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_120: "f32[256, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_416);  convolution_34 = unsqueeze_416 = None
        mul_519: "f32[256, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_120)
        sum_37: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_519, [0, 2, 3]);  mul_519 = None
        mul_520: "f32[576]" = torch.ops.aten.mul.Tensor(sum_36, 1.992984693877551e-05)
        unsqueeze_417: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_418: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
        unsqueeze_419: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
        mul_521: "f32[576]" = torch.ops.aten.mul.Tensor(sum_37, 1.992984693877551e-05)
        mul_522: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_523: "f32[576]" = torch.ops.aten.mul.Tensor(mul_521, mul_522);  mul_521 = mul_522 = None
        unsqueeze_420: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
        unsqueeze_421: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 2);  unsqueeze_420 = None
        unsqueeze_422: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 3);  unsqueeze_421 = None
        mul_524: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_210);  primals_210 = None
        unsqueeze_423: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
        unsqueeze_424: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
        unsqueeze_425: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
        mul_525: "f32[256, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_422);  sub_120 = unsqueeze_422 = None
        sub_122: "f32[256, 576, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_525);  where_11 = mul_525 = None
        sub_123: "f32[256, 576, 14, 14]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_419);  sub_122 = unsqueeze_419 = None
        mul_526: "f32[256, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_425);  sub_123 = unsqueeze_425 = None
        mul_527: "f32[576]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_103);  sum_37 = squeeze_103 = None
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_526, relu_22, primals_206, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_526 = primals_206 = None
        getitem_155: "f32[256, 576, 14, 14]" = convolution_backward_17[0]
        getitem_156: "f32[576, 1, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
        le_12: "b8[256, 576, 14, 14]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_12: "f32[256, 576, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, getitem_155);  le_12 = getitem_155 = None
        sum_38: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_124: "f32[256, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_428);  convolution_33 = unsqueeze_428 = None
        mul_528: "f32[256, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_124)
        sum_39: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 2, 3]);  mul_528 = None
        mul_529: "f32[576]" = torch.ops.aten.mul.Tensor(sum_38, 1.992984693877551e-05)
        unsqueeze_429: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
        unsqueeze_430: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
        unsqueeze_431: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
        mul_530: "f32[576]" = torch.ops.aten.mul.Tensor(sum_39, 1.992984693877551e-05)
        mul_531: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_532: "f32[576]" = torch.ops.aten.mul.Tensor(mul_530, mul_531);  mul_530 = mul_531 = None
        unsqueeze_432: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
        unsqueeze_433: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 2);  unsqueeze_432 = None
        unsqueeze_434: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 3);  unsqueeze_433 = None
        mul_533: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_204);  primals_204 = None
        unsqueeze_435: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
        unsqueeze_436: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
        unsqueeze_437: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
        mul_534: "f32[256, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_434);  sub_124 = unsqueeze_434 = None
        sub_126: "f32[256, 576, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_534);  where_12 = mul_534 = None
        sub_127: "f32[256, 576, 14, 14]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_431);  sub_126 = unsqueeze_431 = None
        mul_535: "f32[256, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_437);  sub_127 = unsqueeze_437 = None
        mul_536: "f32[576]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_100);  sum_39 = squeeze_100 = None
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_535, add_170, primals_200, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_535 = add_170 = primals_200 = None
        getitem_158: "f32[256, 96, 14, 14]" = convolution_backward_18[0]
        getitem_159: "f32[576, 96, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
        add_273: "f32[256, 96, 14, 14]" = torch.ops.aten.add.Tensor(getitem_149, getitem_158);  getitem_149 = getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:63 in forward, code: return self.layers(input)
        sum_40: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_273, [0, 2, 3])
        sub_128: "f32[256, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_440);  convolution_32 = unsqueeze_440 = None
        mul_537: "f32[256, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_273, sub_128)
        sum_41: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_537, [0, 2, 3]);  mul_537 = None
        mul_538: "f32[96]" = torch.ops.aten.mul.Tensor(sum_40, 1.992984693877551e-05)
        unsqueeze_441: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
        unsqueeze_442: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
        unsqueeze_443: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
        mul_539: "f32[96]" = torch.ops.aten.mul.Tensor(sum_41, 1.992984693877551e-05)
        mul_540: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_541: "f32[96]" = torch.ops.aten.mul.Tensor(mul_539, mul_540);  mul_539 = mul_540 = None
        unsqueeze_444: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
        unsqueeze_445: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
        unsqueeze_446: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
        mul_542: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_198);  primals_198 = None
        unsqueeze_447: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
        unsqueeze_448: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
        unsqueeze_449: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
        mul_543: "f32[256, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_446);  sub_128 = unsqueeze_446 = None
        sub_130: "f32[256, 96, 14, 14]" = torch.ops.aten.sub.Tensor(add_273, mul_543);  add_273 = mul_543 = None
        sub_131: "f32[256, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_443);  sub_130 = unsqueeze_443 = None
        mul_544: "f32[256, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_449);  sub_131 = unsqueeze_449 = None
        mul_545: "f32[96]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_97);  sum_41 = squeeze_97 = None
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_544, relu_21, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_544 = primals_194 = None
        getitem_161: "f32[256, 480, 14, 14]" = convolution_backward_19[0]
        getitem_162: "f32[96, 480, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
        le_13: "b8[256, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_13: "f32[256, 480, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, getitem_161);  le_13 = getitem_161 = None
        sum_42: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_132: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_452);  convolution_31 = unsqueeze_452 = None
        mul_546: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_132)
        sum_43: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_546, [0, 2, 3]);  mul_546 = None
        mul_547: "f32[480]" = torch.ops.aten.mul.Tensor(sum_42, 1.992984693877551e-05)
        unsqueeze_453: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
        unsqueeze_454: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
        unsqueeze_455: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
        mul_548: "f32[480]" = torch.ops.aten.mul.Tensor(sum_43, 1.992984693877551e-05)
        mul_549: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_550: "f32[480]" = torch.ops.aten.mul.Tensor(mul_548, mul_549);  mul_548 = mul_549 = None
        unsqueeze_456: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
        unsqueeze_457: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
        unsqueeze_458: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
        mul_551: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_192);  primals_192 = None
        unsqueeze_459: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
        unsqueeze_460: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
        unsqueeze_461: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
        mul_552: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_458);  sub_132 = unsqueeze_458 = None
        sub_134: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_552);  where_13 = mul_552 = None
        sub_135: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_455);  sub_134 = unsqueeze_455 = None
        mul_553: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_461);  sub_135 = unsqueeze_461 = None
        mul_554: "f32[480]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_94);  sum_43 = squeeze_94 = None
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_553, relu_20, primals_188, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_553 = primals_188 = None
        getitem_164: "f32[256, 480, 14, 14]" = convolution_backward_20[0]
        getitem_165: "f32[480, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
        le_14: "b8[256, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_14: "f32[256, 480, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, getitem_164);  le_14 = getitem_164 = None
        sum_44: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_136: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_464);  convolution_30 = unsqueeze_464 = None
        mul_555: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_136)
        sum_45: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_555, [0, 2, 3]);  mul_555 = None
        mul_556: "f32[480]" = torch.ops.aten.mul.Tensor(sum_44, 1.992984693877551e-05)
        unsqueeze_465: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
        unsqueeze_466: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
        unsqueeze_467: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
        mul_557: "f32[480]" = torch.ops.aten.mul.Tensor(sum_45, 1.992984693877551e-05)
        mul_558: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_559: "f32[480]" = torch.ops.aten.mul.Tensor(mul_557, mul_558);  mul_557 = mul_558 = None
        unsqueeze_468: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
        unsqueeze_469: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 2);  unsqueeze_468 = None
        unsqueeze_470: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 3);  unsqueeze_469 = None
        mul_560: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_186);  primals_186 = None
        unsqueeze_471: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
        unsqueeze_472: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
        unsqueeze_473: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
        mul_561: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_470);  sub_136 = unsqueeze_470 = None
        sub_138: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_561);  where_14 = mul_561 = None
        sub_139: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_467);  sub_138 = unsqueeze_467 = None
        mul_562: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_473);  sub_139 = unsqueeze_473 = None
        mul_563: "f32[480]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_91);  sum_45 = squeeze_91 = None
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_562, add_155, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_562 = add_155 = primals_182 = None
        getitem_167: "f32[256, 80, 14, 14]" = convolution_backward_21[0]
        getitem_168: "f32[480, 80, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:61 in forward, code: return self.layers(input) + input
        sum_46: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_167, [0, 2, 3])
        sub_140: "f32[256, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_476);  convolution_29 = unsqueeze_476 = None
        mul_564: "f32[256, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_167, sub_140)
        sum_47: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_564, [0, 2, 3]);  mul_564 = None
        mul_565: "f32[80]" = torch.ops.aten.mul.Tensor(sum_46, 1.992984693877551e-05)
        unsqueeze_477: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
        unsqueeze_478: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
        unsqueeze_479: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
        mul_566: "f32[80]" = torch.ops.aten.mul.Tensor(sum_47, 1.992984693877551e-05)
        mul_567: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_568: "f32[80]" = torch.ops.aten.mul.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
        unsqueeze_480: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
        unsqueeze_481: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 2);  unsqueeze_480 = None
        unsqueeze_482: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 3);  unsqueeze_481 = None
        mul_569: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_180);  primals_180 = None
        unsqueeze_483: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
        unsqueeze_484: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
        unsqueeze_485: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
        mul_570: "f32[256, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_482);  sub_140 = unsqueeze_482 = None
        sub_142: "f32[256, 80, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_167, mul_570);  mul_570 = None
        sub_143: "f32[256, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_479);  sub_142 = unsqueeze_479 = None
        mul_571: "f32[256, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_485);  sub_143 = unsqueeze_485 = None
        mul_572: "f32[80]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_88);  sum_47 = squeeze_88 = None
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_571, relu_19, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_571 = primals_176 = None
        getitem_170: "f32[256, 480, 14, 14]" = convolution_backward_22[0]
        getitem_171: "f32[80, 480, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
        le_15: "b8[256, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_15: "f32[256, 480, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, getitem_170);  le_15 = getitem_170 = None
        sum_48: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_144: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_488);  convolution_28 = unsqueeze_488 = None
        mul_573: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_144)
        sum_49: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_573, [0, 2, 3]);  mul_573 = None
        mul_574: "f32[480]" = torch.ops.aten.mul.Tensor(sum_48, 1.992984693877551e-05)
        unsqueeze_489: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
        unsqueeze_490: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
        unsqueeze_491: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
        mul_575: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, 1.992984693877551e-05)
        mul_576: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_577: "f32[480]" = torch.ops.aten.mul.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
        unsqueeze_492: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
        unsqueeze_493: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 2);  unsqueeze_492 = None
        unsqueeze_494: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 3);  unsqueeze_493 = None
        mul_578: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_174);  primals_174 = None
        unsqueeze_495: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_578, 0);  mul_578 = None
        unsqueeze_496: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
        unsqueeze_497: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
        mul_579: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_494);  sub_144 = unsqueeze_494 = None
        sub_146: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_579);  where_15 = mul_579 = None
        sub_147: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_491);  sub_146 = unsqueeze_491 = None
        mul_580: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_497);  sub_147 = unsqueeze_497 = None
        mul_581: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_85);  sum_49 = squeeze_85 = None
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_580, relu_18, primals_170, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_580 = primals_170 = None
        getitem_173: "f32[256, 480, 14, 14]" = convolution_backward_23[0]
        getitem_174: "f32[480, 1, 5, 5]" = convolution_backward_23[1];  convolution_backward_23 = None
        le_16: "b8[256, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_16: "f32[256, 480, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, getitem_173);  le_16 = getitem_173 = None
        sum_50: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_148: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_500);  convolution_27 = unsqueeze_500 = None
        mul_582: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_148)
        sum_51: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_582, [0, 2, 3]);  mul_582 = None
        mul_583: "f32[480]" = torch.ops.aten.mul.Tensor(sum_50, 1.992984693877551e-05)
        unsqueeze_501: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
        unsqueeze_502: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
        unsqueeze_503: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
        mul_584: "f32[480]" = torch.ops.aten.mul.Tensor(sum_51, 1.992984693877551e-05)
        mul_585: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_586: "f32[480]" = torch.ops.aten.mul.Tensor(mul_584, mul_585);  mul_584 = mul_585 = None
        unsqueeze_504: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
        unsqueeze_505: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 2);  unsqueeze_504 = None
        unsqueeze_506: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 3);  unsqueeze_505 = None
        mul_587: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  primals_168 = None
        unsqueeze_507: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_587, 0);  mul_587 = None
        unsqueeze_508: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
        unsqueeze_509: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
        mul_588: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_506);  sub_148 = unsqueeze_506 = None
        sub_150: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_588);  where_16 = mul_588 = None
        sub_151: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_503);  sub_150 = unsqueeze_503 = None
        mul_589: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_509);  sub_151 = unsqueeze_509 = None
        mul_590: "f32[480]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_82);  sum_51 = squeeze_82 = None
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_589, add_139, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_589 = add_139 = primals_164 = None
        getitem_176: "f32[256, 80, 14, 14]" = convolution_backward_24[0]
        getitem_177: "f32[480, 80, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
        add_274: "f32[256, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_167, getitem_176);  getitem_167 = getitem_176 = None
        sum_52: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 2, 3])
        sub_152: "f32[256, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_512);  convolution_26 = unsqueeze_512 = None
        mul_591: "f32[256, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_274, sub_152)
        sum_53: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 2, 3]);  mul_591 = None
        mul_592: "f32[80]" = torch.ops.aten.mul.Tensor(sum_52, 1.992984693877551e-05)
        unsqueeze_513: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
        unsqueeze_514: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
        unsqueeze_515: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
        mul_593: "f32[80]" = torch.ops.aten.mul.Tensor(sum_53, 1.992984693877551e-05)
        mul_594: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_595: "f32[80]" = torch.ops.aten.mul.Tensor(mul_593, mul_594);  mul_593 = mul_594 = None
        unsqueeze_516: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
        unsqueeze_517: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
        unsqueeze_518: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
        mul_596: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_162);  primals_162 = None
        unsqueeze_519: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
        unsqueeze_520: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
        unsqueeze_521: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
        mul_597: "f32[256, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_518);  sub_152 = unsqueeze_518 = None
        sub_154: "f32[256, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_274, mul_597);  mul_597 = None
        sub_155: "f32[256, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_515);  sub_154 = unsqueeze_515 = None
        mul_598: "f32[256, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_521);  sub_155 = unsqueeze_521 = None
        mul_599: "f32[80]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_79);  sum_53 = squeeze_79 = None
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_598, relu_17, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_598 = primals_158 = None
        getitem_179: "f32[256, 480, 14, 14]" = convolution_backward_25[0]
        getitem_180: "f32[80, 480, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
        le_17: "b8[256, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_17: "f32[256, 480, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, getitem_179);  le_17 = getitem_179 = None
        sum_54: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_156: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_524);  convolution_25 = unsqueeze_524 = None
        mul_600: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_156)
        sum_55: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 2, 3]);  mul_600 = None
        mul_601: "f32[480]" = torch.ops.aten.mul.Tensor(sum_54, 1.992984693877551e-05)
        unsqueeze_525: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
        unsqueeze_526: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
        unsqueeze_527: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
        mul_602: "f32[480]" = torch.ops.aten.mul.Tensor(sum_55, 1.992984693877551e-05)
        mul_603: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_604: "f32[480]" = torch.ops.aten.mul.Tensor(mul_602, mul_603);  mul_602 = mul_603 = None
        unsqueeze_528: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
        unsqueeze_529: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 2);  unsqueeze_528 = None
        unsqueeze_530: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 3);  unsqueeze_529 = None
        mul_605: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_156);  primals_156 = None
        unsqueeze_531: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
        unsqueeze_532: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
        unsqueeze_533: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
        mul_606: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_530);  sub_156 = unsqueeze_530 = None
        sub_158: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_606);  where_17 = mul_606 = None
        sub_159: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_527);  sub_158 = unsqueeze_527 = None
        mul_607: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_533);  sub_159 = unsqueeze_533 = None
        mul_608: "f32[480]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_76);  sum_55 = squeeze_76 = None
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_607, relu_16, primals_152, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_607 = primals_152 = None
        getitem_182: "f32[256, 480, 14, 14]" = convolution_backward_26[0]
        getitem_183: "f32[480, 1, 5, 5]" = convolution_backward_26[1];  convolution_backward_26 = None
        le_18: "b8[256, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_18: "f32[256, 480, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, getitem_182);  le_18 = getitem_182 = None
        sum_56: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        sub_160: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_536);  convolution_24 = unsqueeze_536 = None
        mul_609: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_160)
        sum_57: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_609, [0, 2, 3]);  mul_609 = None
        mul_610: "f32[480]" = torch.ops.aten.mul.Tensor(sum_56, 1.992984693877551e-05)
        unsqueeze_537: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
        unsqueeze_538: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
        unsqueeze_539: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
        mul_611: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, 1.992984693877551e-05)
        mul_612: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_613: "f32[480]" = torch.ops.aten.mul.Tensor(mul_611, mul_612);  mul_611 = mul_612 = None
        unsqueeze_540: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
        unsqueeze_541: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 2);  unsqueeze_540 = None
        unsqueeze_542: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 3);  unsqueeze_541 = None
        mul_614: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_150);  primals_150 = None
        unsqueeze_543: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_614, 0);  mul_614 = None
        unsqueeze_544: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 2);  unsqueeze_543 = None
        unsqueeze_545: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 3);  unsqueeze_544 = None
        mul_615: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_542);  sub_160 = unsqueeze_542 = None
        sub_162: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_615);  where_18 = mul_615 = None
        sub_163: "f32[256, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_539);  sub_162 = unsqueeze_539 = None
        mul_616: "f32[256, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_545);  sub_163 = unsqueeze_545 = None
        mul_617: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_73);  sum_57 = squeeze_73 = None
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_616, add_123, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_616 = add_123 = primals_146 = None
        getitem_185: "f32[256, 80, 14, 14]" = convolution_backward_27[0]
        getitem_186: "f32[480, 80, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
        add_275: "f32[256, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_274, getitem_185);  add_274 = getitem_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:63 in forward, code: return self.layers(input)
        sum_58: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 2, 3])
        sub_164: "f32[256, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_548);  convolution_23 = unsqueeze_548 = None
        mul_618: "f32[256, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_275, sub_164)
        sum_59: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_618, [0, 2, 3]);  mul_618 = None
        mul_619: "f32[80]" = torch.ops.aten.mul.Tensor(sum_58, 1.992984693877551e-05)
        unsqueeze_549: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
        unsqueeze_550: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
        unsqueeze_551: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
        mul_620: "f32[80]" = torch.ops.aten.mul.Tensor(sum_59, 1.992984693877551e-05)
        mul_621: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_622: "f32[80]" = torch.ops.aten.mul.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
        unsqueeze_552: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
        unsqueeze_553: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 2);  unsqueeze_552 = None
        unsqueeze_554: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 3);  unsqueeze_553 = None
        mul_623: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_144);  primals_144 = None
        unsqueeze_555: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
        unsqueeze_556: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
        unsqueeze_557: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
        mul_624: "f32[256, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_554);  sub_164 = unsqueeze_554 = None
        sub_166: "f32[256, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_275, mul_624);  add_275 = mul_624 = None
        sub_167: "f32[256, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_551);  sub_166 = unsqueeze_551 = None
        mul_625: "f32[256, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_557);  sub_167 = unsqueeze_557 = None
        mul_626: "f32[80]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_70);  sum_59 = squeeze_70 = None
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_625, relu_15, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_625 = primals_140 = None
        getitem_188: "f32[256, 240, 14, 14]" = convolution_backward_28[0]
        getitem_189: "f32[80, 240, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
        le_19: "b8[256, 240, 14, 14]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_19: "f32[256, 240, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_188);  le_19 = getitem_188 = None
        sum_60: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_168: "f32[256, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_560);  convolution_22 = unsqueeze_560 = None
        mul_627: "f32[256, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_168)
        sum_61: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_627, [0, 2, 3]);  mul_627 = None
        mul_628: "f32[240]" = torch.ops.aten.mul.Tensor(sum_60, 1.992984693877551e-05)
        unsqueeze_561: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
        unsqueeze_562: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
        unsqueeze_563: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
        mul_629: "f32[240]" = torch.ops.aten.mul.Tensor(sum_61, 1.992984693877551e-05)
        mul_630: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_631: "f32[240]" = torch.ops.aten.mul.Tensor(mul_629, mul_630);  mul_629 = mul_630 = None
        unsqueeze_564: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
        unsqueeze_565: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 2);  unsqueeze_564 = None
        unsqueeze_566: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 3);  unsqueeze_565 = None
        mul_632: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_138);  primals_138 = None
        unsqueeze_567: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
        unsqueeze_568: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
        unsqueeze_569: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
        mul_633: "f32[256, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_566);  sub_168 = unsqueeze_566 = None
        sub_170: "f32[256, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_633);  where_19 = mul_633 = None
        sub_171: "f32[256, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_563);  sub_170 = unsqueeze_563 = None
        mul_634: "f32[256, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_569);  sub_171 = unsqueeze_569 = None
        mul_635: "f32[240]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_67);  sum_61 = squeeze_67 = None
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_634, relu_14, primals_134, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_634 = primals_134 = None
        getitem_191: "f32[256, 240, 28, 28]" = convolution_backward_29[0]
        getitem_192: "f32[240, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
        le_20: "b8[256, 240, 28, 28]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_20: "f32[256, 240, 28, 28]" = torch.ops.aten.where.self(le_20, full_default, getitem_191);  le_20 = getitem_191 = None
        sum_62: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        sub_172: "f32[256, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_572);  convolution_21 = unsqueeze_572 = None
        mul_636: "f32[256, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_20, sub_172)
        sum_63: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_636, [0, 2, 3]);  mul_636 = None
        mul_637: "f32[240]" = torch.ops.aten.mul.Tensor(sum_62, 4.982461734693877e-06)
        unsqueeze_573: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_637, 0);  mul_637 = None
        unsqueeze_574: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
        unsqueeze_575: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
        mul_638: "f32[240]" = torch.ops.aten.mul.Tensor(sum_63, 4.982461734693877e-06)
        mul_639: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_640: "f32[240]" = torch.ops.aten.mul.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
        unsqueeze_576: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
        unsqueeze_577: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 2);  unsqueeze_576 = None
        unsqueeze_578: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 3);  unsqueeze_577 = None
        mul_641: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_132);  primals_132 = None
        unsqueeze_579: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_641, 0);  mul_641 = None
        unsqueeze_580: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
        unsqueeze_581: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
        mul_642: "f32[256, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_578);  sub_172 = unsqueeze_578 = None
        sub_174: "f32[256, 240, 28, 28]" = torch.ops.aten.sub.Tensor(where_20, mul_642);  where_20 = mul_642 = None
        sub_175: "f32[256, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_575);  sub_174 = unsqueeze_575 = None
        mul_643: "f32[256, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_581);  sub_175 = unsqueeze_581 = None
        mul_644: "f32[240]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_64);  sum_63 = squeeze_64 = None
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_643, add_108, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_643 = add_108 = primals_128 = None
        getitem_194: "f32[256, 40, 28, 28]" = convolution_backward_30[0]
        getitem_195: "f32[240, 40, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:61 in forward, code: return self.layers(input) + input
        sum_64: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_194, [0, 2, 3])
        sub_176: "f32[256, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_584);  convolution_20 = unsqueeze_584 = None
        mul_645: "f32[256, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_194, sub_176)
        sum_65: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_645, [0, 2, 3]);  mul_645 = None
        mul_646: "f32[40]" = torch.ops.aten.mul.Tensor(sum_64, 4.982461734693877e-06)
        unsqueeze_585: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
        unsqueeze_586: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
        unsqueeze_587: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
        mul_647: "f32[40]" = torch.ops.aten.mul.Tensor(sum_65, 4.982461734693877e-06)
        mul_648: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_649: "f32[40]" = torch.ops.aten.mul.Tensor(mul_647, mul_648);  mul_647 = mul_648 = None
        unsqueeze_588: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
        unsqueeze_589: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 2);  unsqueeze_588 = None
        unsqueeze_590: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 3);  unsqueeze_589 = None
        mul_650: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  primals_126 = None
        unsqueeze_591: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_650, 0);  mul_650 = None
        unsqueeze_592: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
        unsqueeze_593: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
        mul_651: "f32[256, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_590);  sub_176 = unsqueeze_590 = None
        sub_178: "f32[256, 40, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_194, mul_651);  mul_651 = None
        sub_179: "f32[256, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_587);  sub_178 = unsqueeze_587 = None
        mul_652: "f32[256, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_593);  sub_179 = unsqueeze_593 = None
        mul_653: "f32[40]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_61);  sum_65 = squeeze_61 = None
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_652, relu_13, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_652 = primals_122 = None
        getitem_197: "f32[256, 120, 28, 28]" = convolution_backward_31[0]
        getitem_198: "f32[40, 120, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
        le_21: "b8[256, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_21: "f32[256, 120, 28, 28]" = torch.ops.aten.where.self(le_21, full_default, getitem_197);  le_21 = getitem_197 = None
        sum_66: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        sub_180: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_596);  convolution_19 = unsqueeze_596 = None
        mul_654: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_21, sub_180)
        sum_67: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_654, [0, 2, 3]);  mul_654 = None
        mul_655: "f32[120]" = torch.ops.aten.mul.Tensor(sum_66, 4.982461734693877e-06)
        unsqueeze_597: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
        unsqueeze_598: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
        unsqueeze_599: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
        mul_656: "f32[120]" = torch.ops.aten.mul.Tensor(sum_67, 4.982461734693877e-06)
        mul_657: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_658: "f32[120]" = torch.ops.aten.mul.Tensor(mul_656, mul_657);  mul_656 = mul_657 = None
        unsqueeze_600: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
        unsqueeze_601: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 2);  unsqueeze_600 = None
        unsqueeze_602: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 3);  unsqueeze_601 = None
        mul_659: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_603: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
        unsqueeze_604: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
        unsqueeze_605: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
        mul_660: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_602);  sub_180 = unsqueeze_602 = None
        sub_182: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_21, mul_660);  where_21 = mul_660 = None
        sub_183: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_599);  sub_182 = unsqueeze_599 = None
        mul_661: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_605);  sub_183 = unsqueeze_605 = None
        mul_662: "f32[120]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_58);  sum_67 = squeeze_58 = None
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_661, relu_12, primals_116, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_661 = primals_116 = None
        getitem_200: "f32[256, 120, 28, 28]" = convolution_backward_32[0]
        getitem_201: "f32[120, 1, 5, 5]" = convolution_backward_32[1];  convolution_backward_32 = None
        le_22: "b8[256, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_22: "f32[256, 120, 28, 28]" = torch.ops.aten.where.self(le_22, full_default, getitem_200);  le_22 = getitem_200 = None
        sum_68: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        sub_184: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_608);  convolution_18 = unsqueeze_608 = None
        mul_663: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, sub_184)
        sum_69: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_663, [0, 2, 3]);  mul_663 = None
        mul_664: "f32[120]" = torch.ops.aten.mul.Tensor(sum_68, 4.982461734693877e-06)
        unsqueeze_609: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
        unsqueeze_610: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
        unsqueeze_611: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
        mul_665: "f32[120]" = torch.ops.aten.mul.Tensor(sum_69, 4.982461734693877e-06)
        mul_666: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_667: "f32[120]" = torch.ops.aten.mul.Tensor(mul_665, mul_666);  mul_665 = mul_666 = None
        unsqueeze_612: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
        unsqueeze_613: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 2);  unsqueeze_612 = None
        unsqueeze_614: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 3);  unsqueeze_613 = None
        mul_668: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_615: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
        unsqueeze_616: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
        unsqueeze_617: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
        mul_669: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_614);  sub_184 = unsqueeze_614 = None
        sub_186: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_22, mul_669);  where_22 = mul_669 = None
        sub_187: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_611);  sub_186 = unsqueeze_611 = None
        mul_670: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_617);  sub_187 = unsqueeze_617 = None
        mul_671: "f32[120]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_55);  sum_69 = squeeze_55 = None
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_670, add_92, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_670 = add_92 = primals_110 = None
        getitem_203: "f32[256, 40, 28, 28]" = convolution_backward_33[0]
        getitem_204: "f32[120, 40, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
        add_276: "f32[256, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_194, getitem_203);  getitem_194 = getitem_203 = None
        sum_70: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_276, [0, 2, 3])
        sub_188: "f32[256, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_620);  convolution_17 = unsqueeze_620 = None
        mul_672: "f32[256, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_276, sub_188)
        sum_71: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_672, [0, 2, 3]);  mul_672 = None
        mul_673: "f32[40]" = torch.ops.aten.mul.Tensor(sum_70, 4.982461734693877e-06)
        unsqueeze_621: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
        unsqueeze_622: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
        unsqueeze_623: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
        mul_674: "f32[40]" = torch.ops.aten.mul.Tensor(sum_71, 4.982461734693877e-06)
        mul_675: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_676: "f32[40]" = torch.ops.aten.mul.Tensor(mul_674, mul_675);  mul_674 = mul_675 = None
        unsqueeze_624: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
        unsqueeze_625: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 2);  unsqueeze_624 = None
        unsqueeze_626: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 3);  unsqueeze_625 = None
        mul_677: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_627: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
        unsqueeze_628: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
        unsqueeze_629: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
        mul_678: "f32[256, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_626);  sub_188 = unsqueeze_626 = None
        sub_190: "f32[256, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_276, mul_678);  mul_678 = None
        sub_191: "f32[256, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_623);  sub_190 = unsqueeze_623 = None
        mul_679: "f32[256, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_629);  sub_191 = unsqueeze_629 = None
        mul_680: "f32[40]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_52);  sum_71 = squeeze_52 = None
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_679, relu_11, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_679 = primals_104 = None
        getitem_206: "f32[256, 120, 28, 28]" = convolution_backward_34[0]
        getitem_207: "f32[40, 120, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
        le_23: "b8[256, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_23: "f32[256, 120, 28, 28]" = torch.ops.aten.where.self(le_23, full_default, getitem_206);  le_23 = getitem_206 = None
        sum_72: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        sub_192: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_632);  convolution_16 = unsqueeze_632 = None
        mul_681: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_23, sub_192)
        sum_73: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_681, [0, 2, 3]);  mul_681 = None
        mul_682: "f32[120]" = torch.ops.aten.mul.Tensor(sum_72, 4.982461734693877e-06)
        unsqueeze_633: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
        unsqueeze_634: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
        unsqueeze_635: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
        mul_683: "f32[120]" = torch.ops.aten.mul.Tensor(sum_73, 4.982461734693877e-06)
        mul_684: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_685: "f32[120]" = torch.ops.aten.mul.Tensor(mul_683, mul_684);  mul_683 = mul_684 = None
        unsqueeze_636: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
        unsqueeze_637: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 2);  unsqueeze_636 = None
        unsqueeze_638: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 3);  unsqueeze_637 = None
        mul_686: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_639: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
        unsqueeze_640: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
        unsqueeze_641: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
        mul_687: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_638);  sub_192 = unsqueeze_638 = None
        sub_194: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_23, mul_687);  where_23 = mul_687 = None
        sub_195: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_635);  sub_194 = unsqueeze_635 = None
        mul_688: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_641);  sub_195 = unsqueeze_641 = None
        mul_689: "f32[120]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_49);  sum_73 = squeeze_49 = None
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_688, relu_10, primals_98, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_688 = primals_98 = None
        getitem_209: "f32[256, 120, 28, 28]" = convolution_backward_35[0]
        getitem_210: "f32[120, 1, 5, 5]" = convolution_backward_35[1];  convolution_backward_35 = None
        le_24: "b8[256, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_24: "f32[256, 120, 28, 28]" = torch.ops.aten.where.self(le_24, full_default, getitem_209);  le_24 = getitem_209 = None
        sum_74: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_196: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_644);  convolution_15 = unsqueeze_644 = None
        mul_690: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, sub_196)
        sum_75: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_690, [0, 2, 3]);  mul_690 = None
        mul_691: "f32[120]" = torch.ops.aten.mul.Tensor(sum_74, 4.982461734693877e-06)
        unsqueeze_645: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
        unsqueeze_646: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
        unsqueeze_647: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
        mul_692: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, 4.982461734693877e-06)
        mul_693: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_694: "f32[120]" = torch.ops.aten.mul.Tensor(mul_692, mul_693);  mul_692 = mul_693 = None
        unsqueeze_648: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
        unsqueeze_649: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 2);  unsqueeze_648 = None
        unsqueeze_650: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 3);  unsqueeze_649 = None
        mul_695: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_651: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
        unsqueeze_652: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
        unsqueeze_653: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
        mul_696: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_650);  sub_196 = unsqueeze_650 = None
        sub_198: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_24, mul_696);  where_24 = mul_696 = None
        sub_199: "f32[256, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_647);  sub_198 = unsqueeze_647 = None
        mul_697: "f32[256, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_653);  sub_199 = unsqueeze_653 = None
        mul_698: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_46);  sum_75 = squeeze_46 = None
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_697, add_76, primals_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_697 = add_76 = primals_92 = None
        getitem_212: "f32[256, 40, 28, 28]" = convolution_backward_36[0]
        getitem_213: "f32[120, 40, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
        add_277: "f32[256, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_276, getitem_212);  add_276 = getitem_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:63 in forward, code: return self.layers(input)
        sum_76: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_277, [0, 2, 3])
        sub_200: "f32[256, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_656);  convolution_14 = unsqueeze_656 = None
        mul_699: "f32[256, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_277, sub_200)
        sum_77: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_699, [0, 2, 3]);  mul_699 = None
        mul_700: "f32[40]" = torch.ops.aten.mul.Tensor(sum_76, 4.982461734693877e-06)
        unsqueeze_657: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_700, 0);  mul_700 = None
        unsqueeze_658: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
        unsqueeze_659: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
        mul_701: "f32[40]" = torch.ops.aten.mul.Tensor(sum_77, 4.982461734693877e-06)
        mul_702: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_703: "f32[40]" = torch.ops.aten.mul.Tensor(mul_701, mul_702);  mul_701 = mul_702 = None
        unsqueeze_660: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
        unsqueeze_661: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 2);  unsqueeze_660 = None
        unsqueeze_662: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 3);  unsqueeze_661 = None
        mul_704: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_663: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
        unsqueeze_664: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
        unsqueeze_665: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
        mul_705: "f32[256, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_662);  sub_200 = unsqueeze_662 = None
        sub_202: "f32[256, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_277, mul_705);  add_277 = mul_705 = None
        sub_203: "f32[256, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_659);  sub_202 = unsqueeze_659 = None
        mul_706: "f32[256, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_665);  sub_203 = unsqueeze_665 = None
        mul_707: "f32[40]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_43);  sum_77 = squeeze_43 = None
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_706, relu_9, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_706 = primals_86 = None
        getitem_215: "f32[256, 72, 28, 28]" = convolution_backward_37[0]
        getitem_216: "f32[40, 72, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
        le_25: "b8[256, 72, 28, 28]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_25: "f32[256, 72, 28, 28]" = torch.ops.aten.where.self(le_25, full_default, getitem_215);  le_25 = getitem_215 = None
        sum_78: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        sub_204: "f32[256, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_668);  convolution_13 = unsqueeze_668 = None
        mul_708: "f32[256, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_204)
        sum_79: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_708, [0, 2, 3]);  mul_708 = None
        mul_709: "f32[72]" = torch.ops.aten.mul.Tensor(sum_78, 4.982461734693877e-06)
        unsqueeze_669: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
        unsqueeze_670: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
        unsqueeze_671: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
        mul_710: "f32[72]" = torch.ops.aten.mul.Tensor(sum_79, 4.982461734693877e-06)
        mul_711: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_712: "f32[72]" = torch.ops.aten.mul.Tensor(mul_710, mul_711);  mul_710 = mul_711 = None
        unsqueeze_672: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
        unsqueeze_673: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 2);  unsqueeze_672 = None
        unsqueeze_674: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 3);  unsqueeze_673 = None
        mul_713: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_675: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_713, 0);  mul_713 = None
        unsqueeze_676: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
        unsqueeze_677: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
        mul_714: "f32[256, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_674);  sub_204 = unsqueeze_674 = None
        sub_206: "f32[256, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_25, mul_714);  where_25 = mul_714 = None
        sub_207: "f32[256, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_671);  sub_206 = unsqueeze_671 = None
        mul_715: "f32[256, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_677);  sub_207 = unsqueeze_677 = None
        mul_716: "f32[72]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_40);  sum_79 = squeeze_40 = None
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_715, relu_8, primals_80, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_715 = primals_80 = None
        getitem_218: "f32[256, 72, 56, 56]" = convolution_backward_38[0]
        getitem_219: "f32[72, 1, 5, 5]" = convolution_backward_38[1];  convolution_backward_38 = None
        le_26: "b8[256, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_26: "f32[256, 72, 56, 56]" = torch.ops.aten.where.self(le_26, full_default, getitem_218);  le_26 = getitem_218 = None
        sum_80: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
        sub_208: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_680);  convolution_12 = unsqueeze_680 = None
        mul_717: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_26, sub_208)
        sum_81: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_717, [0, 2, 3]);  mul_717 = None
        mul_718: "f32[72]" = torch.ops.aten.mul.Tensor(sum_80, 1.2456154336734693e-06)
        unsqueeze_681: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
        unsqueeze_682: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
        unsqueeze_683: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
        mul_719: "f32[72]" = torch.ops.aten.mul.Tensor(sum_81, 1.2456154336734693e-06)
        mul_720: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_721: "f32[72]" = torch.ops.aten.mul.Tensor(mul_719, mul_720);  mul_719 = mul_720 = None
        unsqueeze_684: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
        unsqueeze_685: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 2);  unsqueeze_684 = None
        unsqueeze_686: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 3);  unsqueeze_685 = None
        mul_722: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_687: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
        unsqueeze_688: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
        unsqueeze_689: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
        mul_723: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_686);  sub_208 = unsqueeze_686 = None
        sub_210: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_26, mul_723);  where_26 = mul_723 = None
        sub_211: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_683);  sub_210 = unsqueeze_683 = None
        mul_724: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_689);  sub_211 = unsqueeze_689 = None
        mul_725: "f32[72]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_37);  sum_81 = squeeze_37 = None
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_724, add_61, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_724 = add_61 = primals_74 = None
        getitem_221: "f32[256, 24, 56, 56]" = convolution_backward_39[0]
        getitem_222: "f32[72, 24, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:61 in forward, code: return self.layers(input) + input
        sum_82: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_221, [0, 2, 3])
        sub_212: "f32[256, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_692);  convolution_11 = unsqueeze_692 = None
        mul_726: "f32[256, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_221, sub_212)
        sum_83: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_726, [0, 2, 3]);  mul_726 = None
        mul_727: "f32[24]" = torch.ops.aten.mul.Tensor(sum_82, 1.2456154336734693e-06)
        unsqueeze_693: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
        unsqueeze_694: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
        unsqueeze_695: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
        mul_728: "f32[24]" = torch.ops.aten.mul.Tensor(sum_83, 1.2456154336734693e-06)
        mul_729: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_730: "f32[24]" = torch.ops.aten.mul.Tensor(mul_728, mul_729);  mul_728 = mul_729 = None
        unsqueeze_696: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
        unsqueeze_697: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 2);  unsqueeze_696 = None
        unsqueeze_698: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 3);  unsqueeze_697 = None
        mul_731: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_699: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
        unsqueeze_700: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
        unsqueeze_701: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
        mul_732: "f32[256, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_698);  sub_212 = unsqueeze_698 = None
        sub_214: "f32[256, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_221, mul_732);  mul_732 = None
        sub_215: "f32[256, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_695);  sub_214 = unsqueeze_695 = None
        mul_733: "f32[256, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_701);  sub_215 = unsqueeze_701 = None
        mul_734: "f32[24]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_34);  sum_83 = squeeze_34 = None
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_733, relu_7, primals_68, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_733 = primals_68 = None
        getitem_224: "f32[256, 72, 56, 56]" = convolution_backward_40[0]
        getitem_225: "f32[24, 72, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
        le_27: "b8[256, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_27: "f32[256, 72, 56, 56]" = torch.ops.aten.where.self(le_27, full_default, getitem_224);  le_27 = getitem_224 = None
        sum_84: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
        sub_216: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_704);  convolution_10 = unsqueeze_704 = None
        mul_735: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_27, sub_216)
        sum_85: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_735, [0, 2, 3]);  mul_735 = None
        mul_736: "f32[72]" = torch.ops.aten.mul.Tensor(sum_84, 1.2456154336734693e-06)
        unsqueeze_705: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
        unsqueeze_706: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
        unsqueeze_707: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
        mul_737: "f32[72]" = torch.ops.aten.mul.Tensor(sum_85, 1.2456154336734693e-06)
        mul_738: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_739: "f32[72]" = torch.ops.aten.mul.Tensor(mul_737, mul_738);  mul_737 = mul_738 = None
        unsqueeze_708: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
        unsqueeze_709: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 2);  unsqueeze_708 = None
        unsqueeze_710: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 3);  unsqueeze_709 = None
        mul_740: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_711: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
        unsqueeze_712: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
        unsqueeze_713: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
        mul_741: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_710);  sub_216 = unsqueeze_710 = None
        sub_218: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_27, mul_741);  where_27 = mul_741 = None
        sub_219: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_707);  sub_218 = unsqueeze_707 = None
        mul_742: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_713);  sub_219 = unsqueeze_713 = None
        mul_743: "f32[72]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_31);  sum_85 = squeeze_31 = None
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_742, relu_6, primals_62, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_742 = primals_62 = None
        getitem_227: "f32[256, 72, 56, 56]" = convolution_backward_41[0]
        getitem_228: "f32[72, 1, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
        le_28: "b8[256, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_28: "f32[256, 72, 56, 56]" = torch.ops.aten.where.self(le_28, full_default, getitem_227);  le_28 = getitem_227 = None
        sum_86: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
        sub_220: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_716);  convolution_9 = unsqueeze_716 = None
        mul_744: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_28, sub_220)
        sum_87: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 2, 3]);  mul_744 = None
        mul_745: "f32[72]" = torch.ops.aten.mul.Tensor(sum_86, 1.2456154336734693e-06)
        unsqueeze_717: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
        unsqueeze_718: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
        unsqueeze_719: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
        mul_746: "f32[72]" = torch.ops.aten.mul.Tensor(sum_87, 1.2456154336734693e-06)
        mul_747: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_748: "f32[72]" = torch.ops.aten.mul.Tensor(mul_746, mul_747);  mul_746 = mul_747 = None
        unsqueeze_720: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
        unsqueeze_721: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 2);  unsqueeze_720 = None
        unsqueeze_722: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 3);  unsqueeze_721 = None
        mul_749: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_723: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
        unsqueeze_724: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
        unsqueeze_725: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
        mul_750: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_722);  sub_220 = unsqueeze_722 = None
        sub_222: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_28, mul_750);  where_28 = mul_750 = None
        sub_223: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_719);  sub_222 = unsqueeze_719 = None
        mul_751: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_725);  sub_223 = unsqueeze_725 = None
        mul_752: "f32[72]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_28);  sum_87 = squeeze_28 = None
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_751, add_45, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_751 = add_45 = primals_56 = None
        getitem_230: "f32[256, 24, 56, 56]" = convolution_backward_42[0]
        getitem_231: "f32[72, 24, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
        add_278: "f32[256, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_221, getitem_230);  getitem_221 = getitem_230 = None
        sum_88: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 2, 3])
        sub_224: "f32[256, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_728);  convolution_8 = unsqueeze_728 = None
        mul_753: "f32[256, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_278, sub_224)
        sum_89: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_753, [0, 2, 3]);  mul_753 = None
        mul_754: "f32[24]" = torch.ops.aten.mul.Tensor(sum_88, 1.2456154336734693e-06)
        unsqueeze_729: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
        unsqueeze_730: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
        unsqueeze_731: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
        mul_755: "f32[24]" = torch.ops.aten.mul.Tensor(sum_89, 1.2456154336734693e-06)
        mul_756: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_757: "f32[24]" = torch.ops.aten.mul.Tensor(mul_755, mul_756);  mul_755 = mul_756 = None
        unsqueeze_732: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
        unsqueeze_733: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 2);  unsqueeze_732 = None
        unsqueeze_734: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 3);  unsqueeze_733 = None
        mul_758: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_735: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
        unsqueeze_736: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
        unsqueeze_737: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
        mul_759: "f32[256, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_734);  sub_224 = unsqueeze_734 = None
        sub_226: "f32[256, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_278, mul_759);  mul_759 = None
        sub_227: "f32[256, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_731);  sub_226 = unsqueeze_731 = None
        mul_760: "f32[256, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_737);  sub_227 = unsqueeze_737 = None
        mul_761: "f32[24]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_25);  sum_89 = squeeze_25 = None
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_760, relu_5, primals_50, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_760 = primals_50 = None
        getitem_233: "f32[256, 72, 56, 56]" = convolution_backward_43[0]
        getitem_234: "f32[24, 72, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
        le_29: "b8[256, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_29: "f32[256, 72, 56, 56]" = torch.ops.aten.where.self(le_29, full_default, getitem_233);  le_29 = getitem_233 = None
        sum_90: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        sub_228: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_740);  convolution_7 = unsqueeze_740 = None
        mul_762: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_29, sub_228)
        sum_91: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_762, [0, 2, 3]);  mul_762 = None
        mul_763: "f32[72]" = torch.ops.aten.mul.Tensor(sum_90, 1.2456154336734693e-06)
        unsqueeze_741: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
        unsqueeze_742: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
        unsqueeze_743: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
        mul_764: "f32[72]" = torch.ops.aten.mul.Tensor(sum_91, 1.2456154336734693e-06)
        mul_765: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_766: "f32[72]" = torch.ops.aten.mul.Tensor(mul_764, mul_765);  mul_764 = mul_765 = None
        unsqueeze_744: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
        unsqueeze_745: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 2);  unsqueeze_744 = None
        unsqueeze_746: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 3);  unsqueeze_745 = None
        mul_767: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_747: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
        unsqueeze_748: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
        unsqueeze_749: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
        mul_768: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_746);  sub_228 = unsqueeze_746 = None
        sub_230: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_29, mul_768);  where_29 = mul_768 = None
        sub_231: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_743);  sub_230 = unsqueeze_743 = None
        mul_769: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_749);  sub_231 = unsqueeze_749 = None
        mul_770: "f32[72]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_22);  sum_91 = squeeze_22 = None
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_769, relu_4, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_769 = primals_44 = None
        getitem_236: "f32[256, 72, 56, 56]" = convolution_backward_44[0]
        getitem_237: "f32[72, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
        le_30: "b8[256, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_30: "f32[256, 72, 56, 56]" = torch.ops.aten.where.self(le_30, full_default, getitem_236);  le_30 = getitem_236 = None
        sum_92: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
        sub_232: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_752);  convolution_6 = unsqueeze_752 = None
        mul_771: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, sub_232)
        sum_93: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_771, [0, 2, 3]);  mul_771 = None
        mul_772: "f32[72]" = torch.ops.aten.mul.Tensor(sum_92, 1.2456154336734693e-06)
        unsqueeze_753: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
        unsqueeze_754: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
        unsqueeze_755: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
        mul_773: "f32[72]" = torch.ops.aten.mul.Tensor(sum_93, 1.2456154336734693e-06)
        mul_774: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_775: "f32[72]" = torch.ops.aten.mul.Tensor(mul_773, mul_774);  mul_773 = mul_774 = None
        unsqueeze_756: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
        unsqueeze_757: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 2);  unsqueeze_756 = None
        unsqueeze_758: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 3);  unsqueeze_757 = None
        mul_776: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_759: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
        unsqueeze_760: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
        unsqueeze_761: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
        mul_777: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_758);  sub_232 = unsqueeze_758 = None
        sub_234: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_30, mul_777);  where_30 = mul_777 = None
        sub_235: "f32[256, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_755);  sub_234 = unsqueeze_755 = None
        mul_778: "f32[256, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_761);  sub_235 = unsqueeze_761 = None
        mul_779: "f32[72]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_19);  sum_93 = squeeze_19 = None
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_778, add_29, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_778 = add_29 = primals_38 = None
        getitem_239: "f32[256, 24, 56, 56]" = convolution_backward_45[0]
        getitem_240: "f32[72, 24, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
        add_279: "f32[256, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_278, getitem_239);  add_278 = getitem_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:63 in forward, code: return self.layers(input)
        sum_94: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_279, [0, 2, 3])
        sub_236: "f32[256, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_764);  convolution_5 = unsqueeze_764 = None
        mul_780: "f32[256, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_279, sub_236)
        sum_95: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 2, 3]);  mul_780 = None
        mul_781: "f32[24]" = torch.ops.aten.mul.Tensor(sum_94, 1.2456154336734693e-06)
        unsqueeze_765: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
        unsqueeze_766: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
        unsqueeze_767: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
        mul_782: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, 1.2456154336734693e-06)
        mul_783: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_784: "f32[24]" = torch.ops.aten.mul.Tensor(mul_782, mul_783);  mul_782 = mul_783 = None
        unsqueeze_768: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
        unsqueeze_769: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 2);  unsqueeze_768 = None
        unsqueeze_770: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 3);  unsqueeze_769 = None
        mul_785: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_771: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
        unsqueeze_772: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 2);  unsqueeze_771 = None
        unsqueeze_773: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, 3);  unsqueeze_772 = None
        mul_786: "f32[256, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_770);  sub_236 = unsqueeze_770 = None
        sub_238: "f32[256, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_279, mul_786);  add_279 = mul_786 = None
        sub_239: "f32[256, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_767);  sub_238 = unsqueeze_767 = None
        mul_787: "f32[256, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_773);  sub_239 = unsqueeze_773 = None
        mul_788: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_16);  sum_95 = squeeze_16 = None
        convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_787, relu_3, primals_32, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_787 = primals_32 = None
        getitem_242: "f32[256, 48, 56, 56]" = convolution_backward_46[0]
        getitem_243: "f32[24, 48, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
        le_31: "b8[256, 48, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_31: "f32[256, 48, 56, 56]" = torch.ops.aten.where.self(le_31, full_default, getitem_242);  le_31 = getitem_242 = None
        sum_96: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
        sub_240: "f32[256, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_776);  convolution_4 = unsqueeze_776 = None
        mul_789: "f32[256, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, sub_240)
        sum_97: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_789, [0, 2, 3]);  mul_789 = None
        mul_790: "f32[48]" = torch.ops.aten.mul.Tensor(sum_96, 1.2456154336734693e-06)
        unsqueeze_777: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
        unsqueeze_778: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
        unsqueeze_779: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
        mul_791: "f32[48]" = torch.ops.aten.mul.Tensor(sum_97, 1.2456154336734693e-06)
        mul_792: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_793: "f32[48]" = torch.ops.aten.mul.Tensor(mul_791, mul_792);  mul_791 = mul_792 = None
        unsqueeze_780: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
        unsqueeze_781: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 2);  unsqueeze_780 = None
        unsqueeze_782: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 3);  unsqueeze_781 = None
        mul_794: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_783: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
        unsqueeze_784: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
        unsqueeze_785: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
        mul_795: "f32[256, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_782);  sub_240 = unsqueeze_782 = None
        sub_242: "f32[256, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_31, mul_795);  where_31 = mul_795 = None
        sub_243: "f32[256, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_779);  sub_242 = unsqueeze_779 = None
        mul_796: "f32[256, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_785);  sub_243 = unsqueeze_785 = None
        mul_797: "f32[48]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_13);  sum_97 = squeeze_13 = None
        convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_796, relu_2, primals_26, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_796 = primals_26 = None
        getitem_245: "f32[256, 48, 112, 112]" = convolution_backward_47[0]
        getitem_246: "f32[48, 1, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
        le_32: "b8[256, 48, 112, 112]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_32: "f32[256, 48, 112, 112]" = torch.ops.aten.where.self(le_32, full_default, getitem_245);  le_32 = getitem_245 = None
        sum_98: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
        sub_244: "f32[256, 48, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_788);  convolution_3 = unsqueeze_788 = None
        mul_798: "f32[256, 48, 112, 112]" = torch.ops.aten.mul.Tensor(where_32, sub_244)
        sum_99: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_798, [0, 2, 3]);  mul_798 = None
        mul_799: "f32[48]" = torch.ops.aten.mul.Tensor(sum_98, 3.114038584183673e-07)
        unsqueeze_789: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
        unsqueeze_790: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
        unsqueeze_791: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
        mul_800: "f32[48]" = torch.ops.aten.mul.Tensor(sum_99, 3.114038584183673e-07)
        mul_801: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_802: "f32[48]" = torch.ops.aten.mul.Tensor(mul_800, mul_801);  mul_800 = mul_801 = None
        unsqueeze_792: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
        unsqueeze_793: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 2);  unsqueeze_792 = None
        unsqueeze_794: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 3);  unsqueeze_793 = None
        mul_803: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_795: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
        unsqueeze_796: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
        unsqueeze_797: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
        mul_804: "f32[256, 48, 112, 112]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_794);  sub_244 = unsqueeze_794 = None
        sub_246: "f32[256, 48, 112, 112]" = torch.ops.aten.sub.Tensor(where_32, mul_804);  where_32 = mul_804 = None
        sub_247: "f32[256, 48, 112, 112]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_791);  sub_246 = unsqueeze_791 = None
        mul_805: "f32[256, 48, 112, 112]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_797);  sub_247 = unsqueeze_797 = None
        mul_806: "f32[48]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_10);  sum_99 = squeeze_10 = None
        convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_805, add_14, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_805 = add_14 = primals_20 = None
        getitem_248: "f32[256, 16, 112, 112]" = convolution_backward_48[0]
        getitem_249: "f32[48, 16, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/mnasnet.py:159 in forward, code: x = self.layers(x)
        sum_100: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_248, [0, 2, 3])
        sub_248: "f32[256, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_800);  convolution_2 = unsqueeze_800 = None
        mul_807: "f32[256, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_248, sub_248)
        sum_101: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 2, 3]);  mul_807 = None
        mul_808: "f32[16]" = torch.ops.aten.mul.Tensor(sum_100, 3.114038584183673e-07)
        unsqueeze_801: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
        unsqueeze_802: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
        unsqueeze_803: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
        mul_809: "f32[16]" = torch.ops.aten.mul.Tensor(sum_101, 3.114038584183673e-07)
        mul_810: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_811: "f32[16]" = torch.ops.aten.mul.Tensor(mul_809, mul_810);  mul_809 = mul_810 = None
        unsqueeze_804: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
        unsqueeze_805: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 2);  unsqueeze_804 = None
        unsqueeze_806: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 3);  unsqueeze_805 = None
        mul_812: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_807: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
        unsqueeze_808: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
        unsqueeze_809: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
        mul_813: "f32[256, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_806);  sub_248 = unsqueeze_806 = None
        sub_250: "f32[256, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_248, mul_813);  getitem_248 = mul_813 = None
        sub_251: "f32[256, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_803);  sub_250 = unsqueeze_803 = None
        mul_814: "f32[256, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_809);  sub_251 = unsqueeze_809 = None
        mul_815: "f32[16]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_7);  sum_101 = squeeze_7 = None
        convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_814, relu_1, primals_14, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_814 = primals_14 = None
        getitem_251: "f32[256, 32, 112, 112]" = convolution_backward_49[0]
        getitem_252: "f32[16, 32, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
        le_33: "b8[256, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_33: "f32[256, 32, 112, 112]" = torch.ops.aten.where.self(le_33, full_default, getitem_251);  le_33 = getitem_251 = None
        sum_102: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
        sub_252: "f32[256, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_812);  convolution_1 = unsqueeze_812 = None
        mul_816: "f32[256, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_33, sub_252)
        sum_103: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_816, [0, 2, 3]);  mul_816 = None
        mul_817: "f32[32]" = torch.ops.aten.mul.Tensor(sum_102, 3.114038584183673e-07)
        unsqueeze_813: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
        unsqueeze_814: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
        unsqueeze_815: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
        mul_818: "f32[32]" = torch.ops.aten.mul.Tensor(sum_103, 3.114038584183673e-07)
        mul_819: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_820: "f32[32]" = torch.ops.aten.mul.Tensor(mul_818, mul_819);  mul_818 = mul_819 = None
        unsqueeze_816: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
        unsqueeze_817: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 2);  unsqueeze_816 = None
        unsqueeze_818: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 3);  unsqueeze_817 = None
        mul_821: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_819: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
        unsqueeze_820: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
        unsqueeze_821: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
        mul_822: "f32[256, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_818);  sub_252 = unsqueeze_818 = None
        sub_254: "f32[256, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_33, mul_822);  where_33 = mul_822 = None
        sub_255: "f32[256, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_815);  sub_254 = unsqueeze_815 = None
        mul_823: "f32[256, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_821);  sub_255 = unsqueeze_821 = None
        mul_824: "f32[32]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_4);  sum_103 = squeeze_4 = None
        convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_823, relu, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_823 = primals_8 = None
        getitem_254: "f32[256, 32, 112, 112]" = convolution_backward_50[0]
        getitem_255: "f32[32, 1, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
        le_34: "b8[256, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_34: "f32[256, 32, 112, 112]" = torch.ops.aten.where.self(le_34, full_default, getitem_254);  le_34 = full_default = getitem_254 = None
        sum_104: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
        sub_256: "f32[256, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_824);  convolution = unsqueeze_824 = None
        mul_825: "f32[256, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_34, sub_256)
        sum_105: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_825, [0, 2, 3]);  mul_825 = None
        mul_826: "f32[32]" = torch.ops.aten.mul.Tensor(sum_104, 3.114038584183673e-07)
        unsqueeze_825: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
        unsqueeze_826: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
        unsqueeze_827: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
        mul_827: "f32[32]" = torch.ops.aten.mul.Tensor(sum_105, 3.114038584183673e-07)
        mul_828: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_829: "f32[32]" = torch.ops.aten.mul.Tensor(mul_827, mul_828);  mul_827 = mul_828 = None
        unsqueeze_828: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
        unsqueeze_829: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 2);  unsqueeze_828 = None
        unsqueeze_830: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 3);  unsqueeze_829 = None
        mul_830: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_831: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
        unsqueeze_832: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
        unsqueeze_833: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
        mul_831: "f32[256, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_830);  sub_256 = unsqueeze_830 = None
        sub_258: "f32[256, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_34, mul_831);  where_34 = mul_831 = None
        sub_259: "f32[256, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_827);  sub_258 = unsqueeze_827 = None
        mul_832: "f32[256, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_833);  sub_259 = unsqueeze_833 = None
        mul_833: "f32[32]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_1);  sum_105 = squeeze_1 = None
        convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_832, primals_2, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_832 = primals_2 = primals_1 = None
        getitem_258: "f32[32, 3, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
        return (getitem_258, None, None, None, None, mul_833, sum_104, getitem_255, None, None, None, mul_824, sum_102, getitem_252, None, None, None, mul_815, sum_100, getitem_249, None, None, None, mul_806, sum_98, getitem_246, None, None, None, mul_797, sum_96, getitem_243, None, None, None, mul_788, sum_94, getitem_240, None, None, None, mul_779, sum_92, getitem_237, None, None, None, mul_770, sum_90, getitem_234, None, None, None, mul_761, sum_88, getitem_231, None, None, None, mul_752, sum_86, getitem_228, None, None, None, mul_743, sum_84, getitem_225, None, None, None, mul_734, sum_82, getitem_222, None, None, None, mul_725, sum_80, getitem_219, None, None, None, mul_716, sum_78, getitem_216, None, None, None, mul_707, sum_76, getitem_213, None, None, None, mul_698, sum_74, getitem_210, None, None, None, mul_689, sum_72, getitem_207, None, None, None, mul_680, sum_70, getitem_204, None, None, None, mul_671, sum_68, getitem_201, None, None, None, mul_662, sum_66, getitem_198, None, None, None, mul_653, sum_64, getitem_195, None, None, None, mul_644, sum_62, getitem_192, None, None, None, mul_635, sum_60, getitem_189, None, None, None, mul_626, sum_58, getitem_186, None, None, None, mul_617, sum_56, getitem_183, None, None, None, mul_608, sum_54, getitem_180, None, None, None, mul_599, sum_52, getitem_177, None, None, None, mul_590, sum_50, getitem_174, None, None, None, mul_581, sum_48, getitem_171, None, None, None, mul_572, sum_46, getitem_168, None, None, None, mul_563, sum_44, getitem_165, None, None, None, mul_554, sum_42, getitem_162, None, None, None, mul_545, sum_40, getitem_159, None, None, None, mul_536, sum_38, getitem_156, None, None, None, mul_527, sum_36, getitem_153, None, None, None, mul_518, sum_34, getitem_150, None, None, None, mul_509, sum_32, getitem_147, None, None, None, mul_500, sum_30, getitem_144, None, None, None, mul_491, sum_28, getitem_141, None, None, None, mul_482, sum_26, getitem_138, None, None, None, mul_473, sum_24, getitem_135, None, None, None, mul_464, sum_22, getitem_132, None, None, None, mul_455, sum_20, getitem_129, None, None, None, mul_446, sum_18, getitem_126, None, None, None, mul_437, sum_16, getitem_123, None, None, None, mul_428, sum_14, getitem_120, None, None, None, mul_419, sum_12, getitem_117, None, None, None, mul_410, sum_10, getitem_114, None, None, None, mul_401, sum_8, getitem_111, None, None, None, mul_392, sum_6, getitem_108, None, None, None, mul_383, sum_4, getitem_105, None, None, None, mul_374, sum_2, mm_1, view)
