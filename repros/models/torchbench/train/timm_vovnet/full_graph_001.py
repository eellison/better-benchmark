import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[64, 3, 3, 3]", primals_2: "f32[32, 3, 224, 224]", primals_6: "f32[64]", primals_8: "f32[64, 64, 3, 3]", primals_12: "f32[64]", primals_14: "f32[128, 64, 3, 3]", primals_18: "f32[128]", primals_20: "f32[128, 128, 3, 3]", primals_24: "f32[128]", primals_26: "f32[128, 128, 3, 3]", primals_30: "f32[128]", primals_32: "f32[128, 128, 3, 3]", primals_36: "f32[128]", primals_38: "f32[128, 128, 3, 3]", primals_42: "f32[128]", primals_44: "f32[128, 128, 3, 3]", primals_48: "f32[128]", primals_49: "f32[128]", primals_50: "f32[256, 768, 1, 1]", primals_54: "f32[256]", primals_55: "f32[256]", primals_56: "f32[160, 256, 3, 3]", primals_60: "f32[160]", primals_62: "f32[160, 160, 3, 3]", primals_66: "f32[160]", primals_68: "f32[160, 160, 3, 3]", primals_72: "f32[160]", primals_74: "f32[160, 160, 3, 3]", primals_78: "f32[160]", primals_80: "f32[160, 160, 3, 3]", primals_84: "f32[160]", primals_85: "f32[160]", primals_86: "f32[512, 1056, 1, 1]", primals_90: "f32[512]", primals_91: "f32[512]", primals_92: "f32[192, 512, 3, 3]", primals_96: "f32[192]", primals_98: "f32[192, 192, 3, 3]", primals_102: "f32[192]", primals_104: "f32[192, 192, 3, 3]", primals_108: "f32[192]", primals_110: "f32[192, 192, 3, 3]", primals_114: "f32[192]", primals_116: "f32[192, 192, 3, 3]", primals_120: "f32[192]", primals_121: "f32[192]", primals_122: "f32[768, 1472, 1, 1]", primals_126: "f32[768]", primals_128: "f32[192, 768, 3, 3]", primals_132: "f32[192]", primals_134: "f32[192, 192, 3, 3]", primals_138: "f32[192]", primals_140: "f32[192, 192, 3, 3]", primals_144: "f32[192]", primals_146: "f32[192, 192, 3, 3]", primals_150: "f32[192]", primals_152: "f32[192, 192, 3, 3]", primals_156: "f32[192]", primals_157: "f32[192]", primals_158: "f32[768, 1728, 1, 1]", primals_162: "f32[768]", primals_163: "f32[768]", primals_164: "f32[224, 768, 3, 3]", primals_168: "f32[224]", primals_170: "f32[224, 224, 3, 3]", primals_174: "f32[224]", primals_176: "f32[224, 224, 3, 3]", primals_180: "f32[224]", primals_182: "f32[224, 224, 3, 3]", primals_186: "f32[224]", primals_188: "f32[224, 224, 3, 3]", primals_192: "f32[224]", primals_193: "f32[224]", primals_194: "f32[1024, 1888, 1, 1]", primals_198: "f32[1024]", primals_200: "f32[224, 1024, 3, 3]", primals_204: "f32[224]", primals_206: "f32[224, 224, 3, 3]", primals_210: "f32[224]", primals_212: "f32[224, 224, 3, 3]", primals_216: "f32[224]", primals_218: "f32[224, 224, 3, 3]", primals_222: "f32[224]", primals_224: "f32[224, 224, 3, 3]", primals_228: "f32[224]", primals_229: "f32[224]", primals_230: "f32[1024, 2144, 1, 1]", primals_234: "f32[1024]", primals_235: "f32[1024]", primals_236: "f32[1000, 1024]", convolution: "f32[32, 64, 112, 112]", squeeze_1: "f32[64]", relu: "f32[32, 64, 112, 112]", convolution_1: "f32[32, 64, 112, 112]", squeeze_4: "f32[64]", relu_1: "f32[32, 64, 112, 112]", convolution_2: "f32[32, 128, 56, 56]", squeeze_7: "f32[128]", relu_2: "f32[32, 128, 56, 56]", convolution_3: "f32[32, 128, 56, 56]", squeeze_10: "f32[128]", relu_3: "f32[32, 128, 56, 56]", convolution_4: "f32[32, 128, 56, 56]", squeeze_13: "f32[128]", relu_4: "f32[32, 128, 56, 56]", convolution_5: "f32[32, 128, 56, 56]", squeeze_16: "f32[128]", relu_5: "f32[32, 128, 56, 56]", convolution_6: "f32[32, 128, 56, 56]", squeeze_19: "f32[128]", relu_6: "f32[32, 128, 56, 56]", convolution_7: "f32[32, 128, 56, 56]", getitem_15: "f32[1, 128, 1, 1]", rsqrt_7: "f32[1, 128, 1, 1]", cat: "f32[32, 768, 56, 56]", convolution_8: "f32[32, 256, 56, 56]", getitem_17: "f32[1, 256, 1, 1]", rsqrt_8: "f32[1, 256, 1, 1]", getitem_18: "f32[32, 256, 28, 28]", getitem_19: "i8[32, 256, 28, 28]", convolution_9: "f32[32, 160, 28, 28]", squeeze_28: "f32[160]", relu_9: "f32[32, 160, 28, 28]", convolution_10: "f32[32, 160, 28, 28]", squeeze_31: "f32[160]", relu_10: "f32[32, 160, 28, 28]", convolution_11: "f32[32, 160, 28, 28]", squeeze_34: "f32[160]", relu_11: "f32[32, 160, 28, 28]", convolution_12: "f32[32, 160, 28, 28]", squeeze_37: "f32[160]", relu_12: "f32[32, 160, 28, 28]", convolution_13: "f32[32, 160, 28, 28]", getitem_29: "f32[1, 160, 1, 1]", rsqrt_13: "f32[1, 160, 1, 1]", cat_1: "f32[32, 1056, 28, 28]", convolution_14: "f32[32, 512, 28, 28]", getitem_31: "f32[1, 512, 1, 1]", rsqrt_14: "f32[1, 512, 1, 1]", getitem_32: "f32[32, 512, 14, 14]", getitem_33: "i8[32, 512, 14, 14]", convolution_15: "f32[32, 192, 14, 14]", squeeze_46: "f32[192]", relu_15: "f32[32, 192, 14, 14]", convolution_16: "f32[32, 192, 14, 14]", squeeze_49: "f32[192]", relu_16: "f32[32, 192, 14, 14]", convolution_17: "f32[32, 192, 14, 14]", squeeze_52: "f32[192]", relu_17: "f32[32, 192, 14, 14]", convolution_18: "f32[32, 192, 14, 14]", squeeze_55: "f32[192]", relu_18: "f32[32, 192, 14, 14]", convolution_19: "f32[32, 192, 14, 14]", getitem_43: "f32[1, 192, 1, 1]", rsqrt_19: "f32[1, 192, 1, 1]", cat_2: "f32[32, 1472, 14, 14]", convolution_20: "f32[32, 768, 14, 14]", squeeze_61: "f32[768]", relu_20: "f32[32, 768, 14, 14]", convolution_21: "f32[32, 192, 14, 14]", squeeze_64: "f32[192]", relu_21: "f32[32, 192, 14, 14]", convolution_22: "f32[32, 192, 14, 14]", squeeze_67: "f32[192]", relu_22: "f32[32, 192, 14, 14]", convolution_23: "f32[32, 192, 14, 14]", squeeze_70: "f32[192]", relu_23: "f32[32, 192, 14, 14]", convolution_24: "f32[32, 192, 14, 14]", squeeze_73: "f32[192]", relu_24: "f32[32, 192, 14, 14]", convolution_25: "f32[32, 192, 14, 14]", getitem_55: "f32[1, 192, 1, 1]", rsqrt_25: "f32[1, 192, 1, 1]", cat_3: "f32[32, 1728, 14, 14]", convolution_26: "f32[32, 768, 14, 14]", getitem_57: "f32[1, 768, 1, 1]", rsqrt_26: "f32[1, 768, 1, 1]", getitem_58: "f32[32, 768, 7, 7]", getitem_59: "i8[32, 768, 7, 7]", convolution_27: "f32[32, 224, 7, 7]", squeeze_82: "f32[224]", relu_27: "f32[32, 224, 7, 7]", convolution_28: "f32[32, 224, 7, 7]", squeeze_85: "f32[224]", relu_28: "f32[32, 224, 7, 7]", convolution_29: "f32[32, 224, 7, 7]", squeeze_88: "f32[224]", relu_29: "f32[32, 224, 7, 7]", convolution_30: "f32[32, 224, 7, 7]", squeeze_91: "f32[224]", relu_30: "f32[32, 224, 7, 7]", convolution_31: "f32[32, 224, 7, 7]", getitem_69: "f32[1, 224, 1, 1]", rsqrt_31: "f32[1, 224, 1, 1]", cat_4: "f32[32, 1888, 7, 7]", convolution_32: "f32[32, 1024, 7, 7]", squeeze_97: "f32[1024]", relu_32: "f32[32, 1024, 7, 7]", convolution_33: "f32[32, 224, 7, 7]", squeeze_100: "f32[224]", relu_33: "f32[32, 224, 7, 7]", convolution_34: "f32[32, 224, 7, 7]", squeeze_103: "f32[224]", relu_34: "f32[32, 224, 7, 7]", convolution_35: "f32[32, 224, 7, 7]", squeeze_106: "f32[224]", relu_35: "f32[32, 224, 7, 7]", convolution_36: "f32[32, 224, 7, 7]", squeeze_109: "f32[224]", relu_36: "f32[32, 224, 7, 7]", convolution_37: "f32[32, 224, 7, 7]", getitem_81: "f32[1, 224, 1, 1]", rsqrt_37: "f32[1, 224, 1, 1]", cat_5: "f32[32, 2144, 7, 7]", convolution_38: "f32[32, 1024, 7, 7]", getitem_83: "f32[1, 1024, 1, 1]", rsqrt_38: "f32[1, 1024, 1, 1]", view: "f32[32, 1024]", unsqueeze_182: "f32[1, 224, 1, 1]", unsqueeze_194: "f32[1, 224, 1, 1]", unsqueeze_206: "f32[1, 224, 1, 1]", unsqueeze_218: "f32[1, 224, 1, 1]", unsqueeze_230: "f32[1, 1024, 1, 1]", unsqueeze_254: "f32[1, 224, 1, 1]", unsqueeze_266: "f32[1, 224, 1, 1]", unsqueeze_278: "f32[1, 224, 1, 1]", unsqueeze_290: "f32[1, 224, 1, 1]", unsqueeze_326: "f32[1, 192, 1, 1]", unsqueeze_338: "f32[1, 192, 1, 1]", unsqueeze_350: "f32[1, 192, 1, 1]", unsqueeze_362: "f32[1, 192, 1, 1]", unsqueeze_374: "f32[1, 768, 1, 1]", unsqueeze_398: "f32[1, 192, 1, 1]", unsqueeze_410: "f32[1, 192, 1, 1]", unsqueeze_422: "f32[1, 192, 1, 1]", unsqueeze_434: "f32[1, 192, 1, 1]", unsqueeze_470: "f32[1, 160, 1, 1]", unsqueeze_482: "f32[1, 160, 1, 1]", unsqueeze_494: "f32[1, 160, 1, 1]", unsqueeze_506: "f32[1, 160, 1, 1]", unsqueeze_542: "f32[1, 128, 1, 1]", unsqueeze_554: "f32[1, 128, 1, 1]", unsqueeze_566: "f32[1, 128, 1, 1]", unsqueeze_578: "f32[1, 128, 1, 1]", unsqueeze_590: "f32[1, 128, 1, 1]", unsqueeze_602: "f32[1, 64, 1, 1]", unsqueeze_614: "f32[1, 64, 1, 1]", tangents_1: "f32[32, 1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_236, [1, 0]);  primals_236 = None
        permute_1: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm: "f32[32, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "f32[1000, 32]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_2: "f32[32, 1024, 1, 1]" = torch.ops.aten.reshape.default(mm, [32, 1024, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        expand: "f32[32, 1024, 7, 7]" = torch.ops.aten.expand.default(view_2, [32, 1024, 7, 7]);  view_2 = None
        div: "f32[32, 1024, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_38: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, getitem_83)
        mul_266: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        unsqueeze_152: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_153: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        mul_272: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(mul_266, unsqueeze_153);  mul_266 = unsqueeze_153 = None
        unsqueeze_154: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
        unsqueeze_155: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        add_194: "f32[32, 1024, 7, 7]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_155);  mul_272 = unsqueeze_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_38: "f32[32, 1024, 7, 7]" = torch.ops.aten.relu.default(add_194);  add_194 = None
        le: "b8[32, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1024, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_114: "f32[1024]" = torch.ops.aten.squeeze.dims(getitem_83, [0, 2, 3]);  getitem_83 = None
        unsqueeze_156: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(squeeze_114, 0);  squeeze_114 = None
        unsqueeze_157: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 2);  unsqueeze_156 = None
        unsqueeze_158: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 3);  unsqueeze_157 = None
        sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
        sub_39: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_158);  convolution_38 = unsqueeze_158 = None
        mul_273: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_39)
        sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_273, [0, 2, 3]);  mul_273 = None
        mul_274: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_2, 0.0006377551020408163)
        unsqueeze_159: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
        unsqueeze_160: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 2);  unsqueeze_159 = None
        unsqueeze_161: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 3);  unsqueeze_160 = None
        mul_275: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, 0.0006377551020408163)
        squeeze_115: "f32[1024]" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_276: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
        mul_277: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
        unsqueeze_162: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_277, 0);  mul_277 = None
        unsqueeze_163: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 2);  unsqueeze_162 = None
        unsqueeze_164: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 3);  unsqueeze_163 = None
        mul_278: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_234);  primals_234 = None
        unsqueeze_165: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_278, 0);  mul_278 = None
        unsqueeze_166: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 2);  unsqueeze_165 = None
        unsqueeze_167: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 3);  unsqueeze_166 = None
        mul_279: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_164);  sub_39 = unsqueeze_164 = None
        sub_41: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_279);  where = mul_279 = None
        sub_42: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_41, unsqueeze_161);  sub_41 = unsqueeze_161 = None
        mul_280: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_167);  sub_42 = unsqueeze_167 = None
        mul_281: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_115);  sum_3 = squeeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_280, cat_5, primals_230, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_280 = cat_5 = primals_230 = None
        getitem_84: "f32[32, 2144, 7, 7]" = convolution_backward[0]
        getitem_85: "f32[1024, 2144, 1, 1]" = convolution_backward[1];  convolution_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        slice_1: "f32[32, 1024, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_84, 1, 0, 1024)
        slice_2: "f32[32, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_84, 1, 1024, 1248)
        slice_3: "f32[32, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_84, 1, 1248, 1472)
        slice_4: "f32[32, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_84, 1, 1472, 1696)
        slice_5: "f32[32, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_84, 1, 1696, 1920)
        slice_6: "f32[32, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_84, 1, 1920, 2144);  getitem_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_37: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, getitem_81)
        mul_259: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        unsqueeze_148: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_149: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_265: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_149);  mul_259 = unsqueeze_149 = None
        unsqueeze_150: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_151: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_189: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_151);  mul_265 = unsqueeze_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_37: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_189);  add_189 = None
        le_1: "b8[32, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        where_1: "f32[32, 224, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, slice_6);  le_1 = slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_111: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_81, [0, 2, 3]);  getitem_81 = None
        unsqueeze_168: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_111, 0);  squeeze_111 = None
        unsqueeze_169: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 2);  unsqueeze_168 = None
        unsqueeze_170: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 3);  unsqueeze_169 = None
        sum_4: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
        sub_43: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_170);  convolution_37 = unsqueeze_170 = None
        mul_282: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_43)
        sum_5: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3]);  mul_282 = None
        mul_283: "f32[224]" = torch.ops.aten.mul.Tensor(sum_4, 0.0006377551020408163)
        unsqueeze_171: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_283, 0);  mul_283 = None
        unsqueeze_172: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 2);  unsqueeze_171 = None
        unsqueeze_173: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, 3);  unsqueeze_172 = None
        mul_284: "f32[224]" = torch.ops.aten.mul.Tensor(sum_5, 0.0006377551020408163)
        squeeze_112: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_285: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
        mul_286: "f32[224]" = torch.ops.aten.mul.Tensor(mul_284, mul_285);  mul_284 = mul_285 = None
        unsqueeze_174: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_286, 0);  mul_286 = None
        unsqueeze_175: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 2);  unsqueeze_174 = None
        unsqueeze_176: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_175, 3);  unsqueeze_175 = None
        mul_287: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_228);  primals_228 = None
        unsqueeze_177: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
        unsqueeze_178: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 2);  unsqueeze_177 = None
        unsqueeze_179: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 3);  unsqueeze_178 = None
        mul_288: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_176);  sub_43 = unsqueeze_176 = None
        sub_45: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_288);  where_1 = mul_288 = None
        sub_46: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_173);  sub_45 = unsqueeze_173 = None
        mul_289: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_179);  sub_46 = unsqueeze_179 = None
        mul_290: "f32[224]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_112);  sum_5 = squeeze_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_289, relu_36, primals_224, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_289 = primals_224 = None
        getitem_87: "f32[32, 224, 7, 7]" = convolution_backward_1[0]
        getitem_88: "f32[224, 224, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
        add_195: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_5, getitem_87);  slice_5 = getitem_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_2: "b8[32, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        where_2: "f32[32, 224, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, add_195);  le_2 = add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_6: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
        sub_47: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_182);  convolution_36 = unsqueeze_182 = None
        mul_291: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_47)
        sum_7: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 2, 3]);  mul_291 = None
        mul_292: "f32[224]" = torch.ops.aten.mul.Tensor(sum_6, 0.0006377551020408163)
        unsqueeze_183: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
        unsqueeze_184: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 2);  unsqueeze_183 = None
        unsqueeze_185: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 3);  unsqueeze_184 = None
        mul_293: "f32[224]" = torch.ops.aten.mul.Tensor(sum_7, 0.0006377551020408163)
        mul_294: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
        mul_295: "f32[224]" = torch.ops.aten.mul.Tensor(mul_293, mul_294);  mul_293 = mul_294 = None
        unsqueeze_186: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_295, 0);  mul_295 = None
        unsqueeze_187: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 2);  unsqueeze_186 = None
        unsqueeze_188: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 3);  unsqueeze_187 = None
        mul_296: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_222);  primals_222 = None
        unsqueeze_189: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_296, 0);  mul_296 = None
        unsqueeze_190: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 2);  unsqueeze_189 = None
        unsqueeze_191: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 3);  unsqueeze_190 = None
        mul_297: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_188);  sub_47 = unsqueeze_188 = None
        sub_49: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_297);  where_2 = mul_297 = None
        sub_50: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_49, unsqueeze_185);  sub_49 = unsqueeze_185 = None
        mul_298: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_191);  sub_50 = unsqueeze_191 = None
        mul_299: "f32[224]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_109);  sum_7 = squeeze_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_298, relu_35, primals_218, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_298 = primals_218 = None
        getitem_90: "f32[32, 224, 7, 7]" = convolution_backward_2[0]
        getitem_91: "f32[224, 224, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
        add_196: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_4, getitem_90);  slice_4 = getitem_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_3: "b8[32, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        where_3: "f32[32, 224, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, add_196);  le_3 = add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_8: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
        sub_51: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_194);  convolution_35 = unsqueeze_194 = None
        mul_300: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_51)
        sum_9: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 2, 3]);  mul_300 = None
        mul_301: "f32[224]" = torch.ops.aten.mul.Tensor(sum_8, 0.0006377551020408163)
        unsqueeze_195: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
        unsqueeze_196: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 2);  unsqueeze_195 = None
        unsqueeze_197: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, 3);  unsqueeze_196 = None
        mul_302: "f32[224]" = torch.ops.aten.mul.Tensor(sum_9, 0.0006377551020408163)
        mul_303: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
        mul_304: "f32[224]" = torch.ops.aten.mul.Tensor(mul_302, mul_303);  mul_302 = mul_303 = None
        unsqueeze_198: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_304, 0);  mul_304 = None
        unsqueeze_199: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 2);  unsqueeze_198 = None
        unsqueeze_200: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 3);  unsqueeze_199 = None
        mul_305: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_216);  primals_216 = None
        unsqueeze_201: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_305, 0);  mul_305 = None
        unsqueeze_202: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 2);  unsqueeze_201 = None
        unsqueeze_203: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 3);  unsqueeze_202 = None
        mul_306: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_200);  sub_51 = unsqueeze_200 = None
        sub_53: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_306);  where_3 = mul_306 = None
        sub_54: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_197);  sub_53 = unsqueeze_197 = None
        mul_307: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_203);  sub_54 = unsqueeze_203 = None
        mul_308: "f32[224]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_106);  sum_9 = squeeze_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_307, relu_34, primals_212, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_307 = primals_212 = None
        getitem_93: "f32[32, 224, 7, 7]" = convolution_backward_3[0]
        getitem_94: "f32[224, 224, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
        add_197: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_3, getitem_93);  slice_3 = getitem_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_4: "b8[32, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        where_4: "f32[32, 224, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, add_197);  le_4 = add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_10: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
        sub_55: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_206);  convolution_34 = unsqueeze_206 = None
        mul_309: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_55)
        sum_11: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 2, 3]);  mul_309 = None
        mul_310: "f32[224]" = torch.ops.aten.mul.Tensor(sum_10, 0.0006377551020408163)
        unsqueeze_207: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_310, 0);  mul_310 = None
        unsqueeze_208: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 2);  unsqueeze_207 = None
        unsqueeze_209: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, 3);  unsqueeze_208 = None
        mul_311: "f32[224]" = torch.ops.aten.mul.Tensor(sum_11, 0.0006377551020408163)
        mul_312: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
        mul_313: "f32[224]" = torch.ops.aten.mul.Tensor(mul_311, mul_312);  mul_311 = mul_312 = None
        unsqueeze_210: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_313, 0);  mul_313 = None
        unsqueeze_211: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 2);  unsqueeze_210 = None
        unsqueeze_212: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 3);  unsqueeze_211 = None
        mul_314: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_210);  primals_210 = None
        unsqueeze_213: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
        unsqueeze_214: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 2);  unsqueeze_213 = None
        unsqueeze_215: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 3);  unsqueeze_214 = None
        mul_315: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_212);  sub_55 = unsqueeze_212 = None
        sub_57: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_315);  where_4 = mul_315 = None
        sub_58: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_209);  sub_57 = unsqueeze_209 = None
        mul_316: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_215);  sub_58 = unsqueeze_215 = None
        mul_317: "f32[224]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_103);  sum_11 = squeeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_316, relu_33, primals_206, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_316 = primals_206 = None
        getitem_96: "f32[32, 224, 7, 7]" = convolution_backward_4[0]
        getitem_97: "f32[224, 224, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
        add_198: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_2, getitem_96);  slice_2 = getitem_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_5: "b8[32, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        where_5: "f32[32, 224, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, add_198);  le_5 = add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_12: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
        sub_59: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_218);  convolution_33 = unsqueeze_218 = None
        mul_318: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_59)
        sum_13: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_318, [0, 2, 3]);  mul_318 = None
        mul_319: "f32[224]" = torch.ops.aten.mul.Tensor(sum_12, 0.0006377551020408163)
        unsqueeze_219: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
        unsqueeze_220: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 2);  unsqueeze_219 = None
        unsqueeze_221: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, 3);  unsqueeze_220 = None
        mul_320: "f32[224]" = torch.ops.aten.mul.Tensor(sum_13, 0.0006377551020408163)
        mul_321: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
        mul_322: "f32[224]" = torch.ops.aten.mul.Tensor(mul_320, mul_321);  mul_320 = mul_321 = None
        unsqueeze_222: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
        unsqueeze_223: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 2);  unsqueeze_222 = None
        unsqueeze_224: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 3);  unsqueeze_223 = None
        mul_323: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_204);  primals_204 = None
        unsqueeze_225: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_323, 0);  mul_323 = None
        unsqueeze_226: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 2);  unsqueeze_225 = None
        unsqueeze_227: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 3);  unsqueeze_226 = None
        mul_324: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_224);  sub_59 = unsqueeze_224 = None
        sub_61: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_324);  where_5 = mul_324 = None
        sub_62: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_221);  sub_61 = unsqueeze_221 = None
        mul_325: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_227);  sub_62 = unsqueeze_227 = None
        mul_326: "f32[224]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_100);  sum_13 = squeeze_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_325, relu_32, primals_200, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_325 = primals_200 = None
        getitem_99: "f32[32, 1024, 7, 7]" = convolution_backward_5[0]
        getitem_100: "f32[224, 1024, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
        add_199: "f32[32, 1024, 7, 7]" = torch.ops.aten.add.Tensor(slice_1, getitem_99);  slice_1 = getitem_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_6: "b8[32, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        where_6: "f32[32, 1024, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, add_199);  le_6 = add_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_14: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
        sub_63: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_230);  convolution_32 = unsqueeze_230 = None
        mul_327: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_63)
        sum_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 2, 3]);  mul_327 = None
        mul_328: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_14, 0.0006377551020408163)
        unsqueeze_231: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_328, 0);  mul_328 = None
        unsqueeze_232: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
        unsqueeze_233: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
        mul_329: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_15, 0.0006377551020408163)
        mul_330: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
        mul_331: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_329, mul_330);  mul_329 = mul_330 = None
        unsqueeze_234: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_331, 0);  mul_331 = None
        unsqueeze_235: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
        unsqueeze_236: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
        mul_332: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_198);  primals_198 = None
        unsqueeze_237: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_332, 0);  mul_332 = None
        unsqueeze_238: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
        unsqueeze_239: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
        mul_333: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_236);  sub_63 = unsqueeze_236 = None
        sub_65: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_333);  where_6 = mul_333 = None
        sub_66: "f32[32, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_65, unsqueeze_233);  sub_65 = unsqueeze_233 = None
        mul_334: "f32[32, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_239);  sub_66 = unsqueeze_239 = None
        mul_335: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_97);  sum_15 = squeeze_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_334, cat_4, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_334 = cat_4 = primals_194 = None
        getitem_102: "f32[32, 1888, 7, 7]" = convolution_backward_6[0]
        getitem_103: "f32[1024, 1888, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        slice_7: "f32[32, 768, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_102, 1, 0, 768)
        slice_8: "f32[32, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_102, 1, 768, 992)
        slice_9: "f32[32, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_102, 1, 992, 1216)
        slice_10: "f32[32, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_102, 1, 1216, 1440)
        slice_11: "f32[32, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_102, 1, 1440, 1664)
        slice_12: "f32[32, 224, 7, 7]" = torch.ops.aten.slice.Tensor(getitem_102, 1, 1664, 1888);  getitem_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_31: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_31, getitem_69)
        mul_217: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = None
        unsqueeze_124: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_192, -1)
        unsqueeze_125: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_223: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_125);  mul_217 = unsqueeze_125 = None
        unsqueeze_126: "f32[224, 1]" = torch.ops.aten.unsqueeze.default(primals_193, -1);  primals_193 = None
        unsqueeze_127: "f32[224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_159: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(mul_223, unsqueeze_127);  mul_223 = unsqueeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_31: "f32[32, 224, 7, 7]" = torch.ops.aten.relu.default(add_159);  add_159 = None
        le_7: "b8[32, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        where_7: "f32[32, 224, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, slice_12);  le_7 = slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_93: "f32[224]" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        unsqueeze_240: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_241: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 2);  unsqueeze_240 = None
        unsqueeze_242: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 3);  unsqueeze_241 = None
        sum_16: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
        sub_67: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_242);  convolution_31 = unsqueeze_242 = None
        mul_336: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_67)
        sum_17: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2, 3]);  mul_336 = None
        mul_337: "f32[224]" = torch.ops.aten.mul.Tensor(sum_16, 0.0006377551020408163)
        unsqueeze_243: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_337, 0);  mul_337 = None
        unsqueeze_244: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
        unsqueeze_245: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
        mul_338: "f32[224]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
        squeeze_94: "f32[224]" = torch.ops.aten.squeeze.dims(rsqrt_31, [0, 2, 3]);  rsqrt_31 = None
        mul_339: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_340: "f32[224]" = torch.ops.aten.mul.Tensor(mul_338, mul_339);  mul_338 = mul_339 = None
        unsqueeze_246: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_340, 0);  mul_340 = None
        unsqueeze_247: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
        unsqueeze_248: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
        mul_341: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_192);  primals_192 = None
        unsqueeze_249: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
        unsqueeze_250: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
        unsqueeze_251: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
        mul_342: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_248);  sub_67 = unsqueeze_248 = None
        sub_69: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_342);  where_7 = mul_342 = None
        sub_70: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_69, unsqueeze_245);  sub_69 = unsqueeze_245 = None
        mul_343: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_251);  sub_70 = unsqueeze_251 = None
        mul_344: "f32[224]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_94);  sum_17 = squeeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_343, relu_30, primals_188, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_343 = primals_188 = None
        getitem_105: "f32[32, 224, 7, 7]" = convolution_backward_7[0]
        getitem_106: "f32[224, 224, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
        add_200: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_11, getitem_105);  slice_11 = getitem_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_8: "b8[32, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        where_8: "f32[32, 224, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, add_200);  le_8 = add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_18: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
        sub_71: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_254);  convolution_30 = unsqueeze_254 = None
        mul_345: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_71)
        sum_19: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 2, 3]);  mul_345 = None
        mul_346: "f32[224]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
        unsqueeze_255: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
        unsqueeze_256: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
        unsqueeze_257: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
        mul_347: "f32[224]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
        mul_348: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_349: "f32[224]" = torch.ops.aten.mul.Tensor(mul_347, mul_348);  mul_347 = mul_348 = None
        unsqueeze_258: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_349, 0);  mul_349 = None
        unsqueeze_259: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
        unsqueeze_260: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
        mul_350: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_186);  primals_186 = None
        unsqueeze_261: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_350, 0);  mul_350 = None
        unsqueeze_262: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
        unsqueeze_263: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
        mul_351: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_260);  sub_71 = unsqueeze_260 = None
        sub_73: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_351);  where_8 = mul_351 = None
        sub_74: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_73, unsqueeze_257);  sub_73 = unsqueeze_257 = None
        mul_352: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_263);  sub_74 = unsqueeze_263 = None
        mul_353: "f32[224]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_91);  sum_19 = squeeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_352, relu_29, primals_182, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_352 = primals_182 = None
        getitem_108: "f32[32, 224, 7, 7]" = convolution_backward_8[0]
        getitem_109: "f32[224, 224, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
        add_201: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_10, getitem_108);  slice_10 = getitem_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_9: "b8[32, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        where_9: "f32[32, 224, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, add_201);  le_9 = add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_20: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
        sub_75: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_266);  convolution_29 = unsqueeze_266 = None
        mul_354: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_75)
        sum_21: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 2, 3]);  mul_354 = None
        mul_355: "f32[224]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
        unsqueeze_267: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_355, 0);  mul_355 = None
        unsqueeze_268: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
        unsqueeze_269: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
        mul_356: "f32[224]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
        mul_357: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_358: "f32[224]" = torch.ops.aten.mul.Tensor(mul_356, mul_357);  mul_356 = mul_357 = None
        unsqueeze_270: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_358, 0);  mul_358 = None
        unsqueeze_271: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
        unsqueeze_272: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
        mul_359: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_180);  primals_180 = None
        unsqueeze_273: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
        unsqueeze_274: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
        unsqueeze_275: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
        mul_360: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_272);  sub_75 = unsqueeze_272 = None
        sub_77: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_360);  where_9 = mul_360 = None
        sub_78: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_269);  sub_77 = unsqueeze_269 = None
        mul_361: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_275);  sub_78 = unsqueeze_275 = None
        mul_362: "f32[224]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_88);  sum_21 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_361, relu_28, primals_176, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_361 = primals_176 = None
        getitem_111: "f32[32, 224, 7, 7]" = convolution_backward_9[0]
        getitem_112: "f32[224, 224, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
        add_202: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_9, getitem_111);  slice_9 = getitem_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_10: "b8[32, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        where_10: "f32[32, 224, 7, 7]" = torch.ops.aten.where.self(le_10, full_default, add_202);  le_10 = add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_22: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
        sub_79: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_278);  convolution_28 = unsqueeze_278 = None
        mul_363: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_79)
        sum_23: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 2, 3]);  mul_363 = None
        mul_364: "f32[224]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
        unsqueeze_279: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_364, 0);  mul_364 = None
        unsqueeze_280: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
        unsqueeze_281: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
        mul_365: "f32[224]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
        mul_366: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_367: "f32[224]" = torch.ops.aten.mul.Tensor(mul_365, mul_366);  mul_365 = mul_366 = None
        unsqueeze_282: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_367, 0);  mul_367 = None
        unsqueeze_283: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
        unsqueeze_284: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
        mul_368: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_174);  primals_174 = None
        unsqueeze_285: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
        unsqueeze_286: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
        unsqueeze_287: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
        mul_369: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_284);  sub_79 = unsqueeze_284 = None
        sub_81: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_369);  where_10 = mul_369 = None
        sub_82: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_281);  sub_81 = unsqueeze_281 = None
        mul_370: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_287);  sub_82 = unsqueeze_287 = None
        mul_371: "f32[224]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_85);  sum_23 = squeeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_370, relu_27, primals_170, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_370 = primals_170 = None
        getitem_114: "f32[32, 224, 7, 7]" = convolution_backward_10[0]
        getitem_115: "f32[224, 224, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
        add_203: "f32[32, 224, 7, 7]" = torch.ops.aten.add.Tensor(slice_8, getitem_114);  slice_8 = getitem_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_11: "b8[32, 224, 7, 7]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        where_11: "f32[32, 224, 7, 7]" = torch.ops.aten.where.self(le_11, full_default, add_203);  le_11 = add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_24: "f32[224]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
        sub_83: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_290);  convolution_27 = unsqueeze_290 = None
        mul_372: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, sub_83)
        sum_25: "f32[224]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 2, 3]);  mul_372 = None
        mul_373: "f32[224]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
        unsqueeze_291: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
        unsqueeze_292: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
        unsqueeze_293: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
        mul_374: "f32[224]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
        mul_375: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_376: "f32[224]" = torch.ops.aten.mul.Tensor(mul_374, mul_375);  mul_374 = mul_375 = None
        unsqueeze_294: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_376, 0);  mul_376 = None
        unsqueeze_295: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
        unsqueeze_296: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
        mul_377: "f32[224]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_168);  primals_168 = None
        unsqueeze_297: "f32[1, 224]" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
        unsqueeze_298: "f32[1, 224, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
        unsqueeze_299: "f32[1, 224, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
        mul_378: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_296);  sub_83 = unsqueeze_296 = None
        sub_85: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(where_11, mul_378);  where_11 = mul_378 = None
        sub_86: "f32[32, 224, 7, 7]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_293);  sub_85 = unsqueeze_293 = None
        mul_379: "f32[32, 224, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_299);  sub_86 = unsqueeze_299 = None
        mul_380: "f32[224]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_82);  sum_25 = squeeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_379, getitem_58, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_379 = getitem_58 = primals_164 = None
        getitem_117: "f32[32, 768, 7, 7]" = convolution_backward_11[0]
        getitem_118: "f32[224, 768, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
        add_204: "f32[32, 768, 7, 7]" = torch.ops.aten.add.Tensor(slice_7, getitem_117);  slice_7 = getitem_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:161 in forward, code: x = self.pool(x)
        full_default_12: "f32[24576, 196]" = torch.ops.aten.full.default([24576, 196], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_3: "f32[24576, 49]" = torch.ops.aten.reshape.default(add_204, [24576, 49]);  add_204 = None
        _low_memory_max_pool_offsets_to_indices_2: "i64[32, 768, 7, 7]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_59, [3, 3], [14, 14], [2, 2], [0, 0], [1, 1]);  getitem_59 = None
        view_4: "i64[24576, 49]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_2, [24576, 49]);  _low_memory_max_pool_offsets_to_indices_2 = None
        scatter_add: "f32[24576, 196]" = torch.ops.aten.scatter_add.default(full_default_12, 1, view_4, view_3);  full_default_12 = view_4 = view_3 = None
        view_5: "f32[32, 768, 14, 14]" = torch.ops.aten.reshape.default(scatter_add, [32, 768, 14, 14]);  scatter_add = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_26: "f32[32, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, getitem_57)
        mul_182: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
        unsqueeze_104: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_162, -1)
        unsqueeze_105: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_188: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(mul_182, unsqueeze_105);  mul_182 = unsqueeze_105 = None
        unsqueeze_106: "f32[768, 1]" = torch.ops.aten.unsqueeze.default(primals_163, -1);  primals_163 = None
        unsqueeze_107: "f32[768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_134: "f32[32, 768, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_107);  mul_188 = unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_26: "f32[32, 768, 14, 14]" = torch.ops.aten.relu.default(add_134);  add_134 = None
        le_12: "b8[32, 768, 14, 14]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        where_12: "f32[32, 768, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, view_5);  le_12 = view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_78: "f32[768]" = torch.ops.aten.squeeze.dims(getitem_57, [0, 2, 3]);  getitem_57 = None
        unsqueeze_300: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_301: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 2);  unsqueeze_300 = None
        unsqueeze_302: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 3);  unsqueeze_301 = None
        sum_26: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
        sub_87: "f32[32, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_302);  convolution_26 = unsqueeze_302 = None
        mul_381: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_87)
        sum_27: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 2, 3]);  mul_381 = None
        mul_382: "f32[768]" = torch.ops.aten.mul.Tensor(sum_26, 0.00015943877551020407)
        unsqueeze_303: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
        unsqueeze_304: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
        unsqueeze_305: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
        mul_383: "f32[768]" = torch.ops.aten.mul.Tensor(sum_27, 0.00015943877551020407)
        squeeze_79: "f32[768]" = torch.ops.aten.squeeze.dims(rsqrt_26, [0, 2, 3]);  rsqrt_26 = None
        mul_384: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_385: "f32[768]" = torch.ops.aten.mul.Tensor(mul_383, mul_384);  mul_383 = mul_384 = None
        unsqueeze_306: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_385, 0);  mul_385 = None
        unsqueeze_307: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 2);  unsqueeze_306 = None
        unsqueeze_308: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 3);  unsqueeze_307 = None
        mul_386: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_162);  primals_162 = None
        unsqueeze_309: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
        unsqueeze_310: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
        unsqueeze_311: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
        mul_387: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_308);  sub_87 = unsqueeze_308 = None
        sub_89: "f32[32, 768, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_387);  where_12 = mul_387 = None
        sub_90: "f32[32, 768, 14, 14]" = torch.ops.aten.sub.Tensor(sub_89, unsqueeze_305);  sub_89 = unsqueeze_305 = None
        mul_388: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_311);  sub_90 = unsqueeze_311 = None
        mul_389: "f32[768]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_79);  sum_27 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_388, cat_3, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_388 = cat_3 = primals_158 = None
        getitem_120: "f32[32, 1728, 14, 14]" = convolution_backward_12[0]
        getitem_121: "f32[768, 1728, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        slice_13: "f32[32, 768, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_120, 1, 0, 768)
        slice_14: "f32[32, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_120, 1, 768, 960)
        slice_15: "f32[32, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_120, 1, 960, 1152)
        slice_16: "f32[32, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_120, 1, 1152, 1344)
        slice_17: "f32[32, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_120, 1, 1344, 1536)
        slice_18: "f32[32, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_120, 1, 1536, 1728);  getitem_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_25: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, getitem_55)
        mul_175: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_100: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_156, -1)
        unsqueeze_101: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_181: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_101);  mul_175 = unsqueeze_101 = None
        unsqueeze_102: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_157, -1);  primals_157 = None
        unsqueeze_103: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_129: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_181, unsqueeze_103);  mul_181 = unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_25: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_129);  add_129 = None
        le_13: "b8[32, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        where_13: "f32[32, 192, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, slice_18);  le_13 = slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_75: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_55, [0, 2, 3]);  getitem_55 = None
        unsqueeze_312: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_313: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 2);  unsqueeze_312 = None
        unsqueeze_314: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 3);  unsqueeze_313 = None
        sum_28: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
        sub_91: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_314);  convolution_25 = unsqueeze_314 = None
        mul_390: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_91)
        sum_29: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 2, 3]);  mul_390 = None
        mul_391: "f32[192]" = torch.ops.aten.mul.Tensor(sum_28, 0.00015943877551020407)
        unsqueeze_315: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
        unsqueeze_316: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
        unsqueeze_317: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 3);  unsqueeze_316 = None
        mul_392: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, 0.00015943877551020407)
        squeeze_76: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_393: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_394: "f32[192]" = torch.ops.aten.mul.Tensor(mul_392, mul_393);  mul_392 = mul_393 = None
        unsqueeze_318: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_394, 0);  mul_394 = None
        unsqueeze_319: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 2);  unsqueeze_318 = None
        unsqueeze_320: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 3);  unsqueeze_319 = None
        mul_395: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_156);  primals_156 = None
        unsqueeze_321: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
        unsqueeze_322: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
        unsqueeze_323: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
        mul_396: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_320);  sub_91 = unsqueeze_320 = None
        sub_93: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_396);  where_13 = mul_396 = None
        sub_94: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_93, unsqueeze_317);  sub_93 = unsqueeze_317 = None
        mul_397: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_323);  sub_94 = unsqueeze_323 = None
        mul_398: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_76);  sum_29 = squeeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_397, relu_24, primals_152, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_397 = primals_152 = None
        getitem_123: "f32[32, 192, 14, 14]" = convolution_backward_13[0]
        getitem_124: "f32[192, 192, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
        add_205: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_17, getitem_123);  slice_17 = getitem_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_14: "b8[32, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        where_14: "f32[32, 192, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, add_205);  le_14 = add_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_30: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
        sub_95: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_326);  convolution_24 = unsqueeze_326 = None
        mul_399: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_95)
        sum_31: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3]);  mul_399 = None
        mul_400: "f32[192]" = torch.ops.aten.mul.Tensor(sum_30, 0.00015943877551020407)
        unsqueeze_327: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
        unsqueeze_328: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
        unsqueeze_329: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 3);  unsqueeze_328 = None
        mul_401: "f32[192]" = torch.ops.aten.mul.Tensor(sum_31, 0.00015943877551020407)
        mul_402: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_403: "f32[192]" = torch.ops.aten.mul.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
        unsqueeze_330: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_403, 0);  mul_403 = None
        unsqueeze_331: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 2);  unsqueeze_330 = None
        unsqueeze_332: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 3);  unsqueeze_331 = None
        mul_404: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_150);  primals_150 = None
        unsqueeze_333: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
        unsqueeze_334: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
        unsqueeze_335: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
        mul_405: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_332);  sub_95 = unsqueeze_332 = None
        sub_97: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_405);  where_14 = mul_405 = None
        sub_98: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_97, unsqueeze_329);  sub_97 = unsqueeze_329 = None
        mul_406: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_335);  sub_98 = unsqueeze_335 = None
        mul_407: "f32[192]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_73);  sum_31 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_406, relu_23, primals_146, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = primals_146 = None
        getitem_126: "f32[32, 192, 14, 14]" = convolution_backward_14[0]
        getitem_127: "f32[192, 192, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
        add_206: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_16, getitem_126);  slice_16 = getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_15: "b8[32, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        where_15: "f32[32, 192, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, add_206);  le_15 = add_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_32: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
        sub_99: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_338);  convolution_23 = unsqueeze_338 = None
        mul_408: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_99)
        sum_33: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 2, 3]);  mul_408 = None
        mul_409: "f32[192]" = torch.ops.aten.mul.Tensor(sum_32, 0.00015943877551020407)
        unsqueeze_339: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
        unsqueeze_340: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
        unsqueeze_341: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 3);  unsqueeze_340 = None
        mul_410: "f32[192]" = torch.ops.aten.mul.Tensor(sum_33, 0.00015943877551020407)
        mul_411: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_412: "f32[192]" = torch.ops.aten.mul.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
        unsqueeze_342: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
        unsqueeze_343: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 2);  unsqueeze_342 = None
        unsqueeze_344: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 3);  unsqueeze_343 = None
        mul_413: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_144);  primals_144 = None
        unsqueeze_345: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
        unsqueeze_346: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
        unsqueeze_347: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
        mul_414: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_344);  sub_99 = unsqueeze_344 = None
        sub_101: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_414);  where_15 = mul_414 = None
        sub_102: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_101, unsqueeze_341);  sub_101 = unsqueeze_341 = None
        mul_415: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_347);  sub_102 = unsqueeze_347 = None
        mul_416: "f32[192]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_70);  sum_33 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_415, relu_22, primals_140, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_415 = primals_140 = None
        getitem_129: "f32[32, 192, 14, 14]" = convolution_backward_15[0]
        getitem_130: "f32[192, 192, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
        add_207: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_15, getitem_129);  slice_15 = getitem_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_16: "b8[32, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        where_16: "f32[32, 192, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, add_207);  le_16 = add_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_34: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
        sub_103: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_350);  convolution_22 = unsqueeze_350 = None
        mul_417: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_103)
        sum_35: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3]);  mul_417 = None
        mul_418: "f32[192]" = torch.ops.aten.mul.Tensor(sum_34, 0.00015943877551020407)
        unsqueeze_351: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
        unsqueeze_352: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
        unsqueeze_353: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
        mul_419: "f32[192]" = torch.ops.aten.mul.Tensor(sum_35, 0.00015943877551020407)
        mul_420: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_421: "f32[192]" = torch.ops.aten.mul.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
        unsqueeze_354: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
        unsqueeze_355: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 2);  unsqueeze_354 = None
        unsqueeze_356: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 3);  unsqueeze_355 = None
        mul_422: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_138);  primals_138 = None
        unsqueeze_357: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
        unsqueeze_358: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
        unsqueeze_359: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
        mul_423: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_356);  sub_103 = unsqueeze_356 = None
        sub_105: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_423);  where_16 = mul_423 = None
        sub_106: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_105, unsqueeze_353);  sub_105 = unsqueeze_353 = None
        mul_424: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_359);  sub_106 = unsqueeze_359 = None
        mul_425: "f32[192]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_67);  sum_35 = squeeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_424, relu_21, primals_134, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_424 = primals_134 = None
        getitem_132: "f32[32, 192, 14, 14]" = convolution_backward_16[0]
        getitem_133: "f32[192, 192, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
        add_208: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_14, getitem_132);  slice_14 = getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_17: "b8[32, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        where_17: "f32[32, 192, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, add_208);  le_17 = add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_36: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
        sub_107: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_362);  convolution_21 = unsqueeze_362 = None
        mul_426: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_107)
        sum_37: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 2, 3]);  mul_426 = None
        mul_427: "f32[192]" = torch.ops.aten.mul.Tensor(sum_36, 0.00015943877551020407)
        unsqueeze_363: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_427, 0);  mul_427 = None
        unsqueeze_364: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
        unsqueeze_365: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
        mul_428: "f32[192]" = torch.ops.aten.mul.Tensor(sum_37, 0.00015943877551020407)
        mul_429: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_430: "f32[192]" = torch.ops.aten.mul.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
        unsqueeze_366: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_430, 0);  mul_430 = None
        unsqueeze_367: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 2);  unsqueeze_366 = None
        unsqueeze_368: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 3);  unsqueeze_367 = None
        mul_431: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_132);  primals_132 = None
        unsqueeze_369: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
        unsqueeze_370: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
        unsqueeze_371: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
        mul_432: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_368);  sub_107 = unsqueeze_368 = None
        sub_109: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_432);  where_17 = mul_432 = None
        sub_110: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_365);  sub_109 = unsqueeze_365 = None
        mul_433: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_371);  sub_110 = unsqueeze_371 = None
        mul_434: "f32[192]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_64);  sum_37 = squeeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_433, relu_20, primals_128, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_433 = primals_128 = None
        getitem_135: "f32[32, 768, 14, 14]" = convolution_backward_17[0]
        getitem_136: "f32[192, 768, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
        add_209: "f32[32, 768, 14, 14]" = torch.ops.aten.add.Tensor(slice_13, getitem_135);  slice_13 = getitem_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_18: "b8[32, 768, 14, 14]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        where_18: "f32[32, 768, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, add_209);  le_18 = add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_38: "f32[768]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
        sub_111: "f32[32, 768, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_374);  convolution_20 = unsqueeze_374 = None
        mul_435: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_111)
        sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 2, 3]);  mul_435 = None
        mul_436: "f32[768]" = torch.ops.aten.mul.Tensor(sum_38, 0.00015943877551020407)
        unsqueeze_375: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
        unsqueeze_376: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
        unsqueeze_377: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
        mul_437: "f32[768]" = torch.ops.aten.mul.Tensor(sum_39, 0.00015943877551020407)
        mul_438: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_439: "f32[768]" = torch.ops.aten.mul.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
        unsqueeze_378: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
        unsqueeze_379: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 2);  unsqueeze_378 = None
        unsqueeze_380: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 3);  unsqueeze_379 = None
        mul_440: "f32[768]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_126);  primals_126 = None
        unsqueeze_381: "f32[1, 768]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
        unsqueeze_382: "f32[1, 768, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
        unsqueeze_383: "f32[1, 768, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
        mul_441: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_380);  sub_111 = unsqueeze_380 = None
        sub_113: "f32[32, 768, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_441);  where_18 = mul_441 = None
        sub_114: "f32[32, 768, 14, 14]" = torch.ops.aten.sub.Tensor(sub_113, unsqueeze_377);  sub_113 = unsqueeze_377 = None
        mul_442: "f32[32, 768, 14, 14]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_383);  sub_114 = unsqueeze_383 = None
        mul_443: "f32[768]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_61);  sum_39 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_442, cat_2, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_442 = cat_2 = primals_122 = None
        getitem_138: "f32[32, 1472, 14, 14]" = convolution_backward_18[0]
        getitem_139: "f32[768, 1472, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        slice_19: "f32[32, 512, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_138, 1, 0, 512)
        slice_20: "f32[32, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_138, 1, 512, 704)
        slice_21: "f32[32, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_138, 1, 704, 896)
        slice_22: "f32[32, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_138, 1, 896, 1088)
        slice_23: "f32[32, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_138, 1, 1088, 1280)
        slice_24: "f32[32, 192, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_138, 1, 1280, 1472);  getitem_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_19: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, getitem_43)
        mul_133: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
        unsqueeze_76: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_120, -1)
        unsqueeze_77: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_139: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_77);  mul_133 = unsqueeze_77 = None
        unsqueeze_78: "f32[192, 1]" = torch.ops.aten.unsqueeze.default(primals_121, -1);  primals_121 = None
        unsqueeze_79: "f32[192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_99: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(mul_139, unsqueeze_79);  mul_139 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_19: "f32[32, 192, 14, 14]" = torch.ops.aten.relu.default(add_99);  add_99 = None
        le_19: "b8[32, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        where_19: "f32[32, 192, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, slice_24);  le_19 = slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_57: "f32[192]" = torch.ops.aten.squeeze.dims(getitem_43, [0, 2, 3]);  getitem_43 = None
        unsqueeze_384: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_385: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 2);  unsqueeze_384 = None
        unsqueeze_386: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 3);  unsqueeze_385 = None
        sum_40: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
        sub_115: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_386);  convolution_19 = unsqueeze_386 = None
        mul_444: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_115)
        sum_41: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 2, 3]);  mul_444 = None
        mul_445: "f32[192]" = torch.ops.aten.mul.Tensor(sum_40, 0.00015943877551020407)
        unsqueeze_387: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_445, 0);  mul_445 = None
        unsqueeze_388: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
        unsqueeze_389: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
        mul_446: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, 0.00015943877551020407)
        squeeze_58: "f32[192]" = torch.ops.aten.squeeze.dims(rsqrt_19, [0, 2, 3]);  rsqrt_19 = None
        mul_447: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_448: "f32[192]" = torch.ops.aten.mul.Tensor(mul_446, mul_447);  mul_446 = mul_447 = None
        unsqueeze_390: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
        unsqueeze_391: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
        unsqueeze_392: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
        mul_449: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_393: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
        unsqueeze_394: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
        unsqueeze_395: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
        mul_450: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_392);  sub_115 = unsqueeze_392 = None
        sub_117: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_450);  where_19 = mul_450 = None
        sub_118: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_117, unsqueeze_389);  sub_117 = unsqueeze_389 = None
        mul_451: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_395);  sub_118 = unsqueeze_395 = None
        mul_452: "f32[192]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_58);  sum_41 = squeeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_451, relu_18, primals_116, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_451 = primals_116 = None
        getitem_141: "f32[32, 192, 14, 14]" = convolution_backward_19[0]
        getitem_142: "f32[192, 192, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
        add_210: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_23, getitem_141);  slice_23 = getitem_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_20: "b8[32, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        where_20: "f32[32, 192, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, add_210);  le_20 = add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_42: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
        sub_119: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_398);  convolution_18 = unsqueeze_398 = None
        mul_453: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_119)
        sum_43: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_453, [0, 2, 3]);  mul_453 = None
        mul_454: "f32[192]" = torch.ops.aten.mul.Tensor(sum_42, 0.00015943877551020407)
        unsqueeze_399: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
        unsqueeze_400: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
        unsqueeze_401: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
        mul_455: "f32[192]" = torch.ops.aten.mul.Tensor(sum_43, 0.00015943877551020407)
        mul_456: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_457: "f32[192]" = torch.ops.aten.mul.Tensor(mul_455, mul_456);  mul_455 = mul_456 = None
        unsqueeze_402: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_457, 0);  mul_457 = None
        unsqueeze_403: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
        unsqueeze_404: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
        mul_458: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_405: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
        unsqueeze_406: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
        unsqueeze_407: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
        mul_459: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_404);  sub_119 = unsqueeze_404 = None
        sub_121: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_459);  where_20 = mul_459 = None
        sub_122: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_121, unsqueeze_401);  sub_121 = unsqueeze_401 = None
        mul_460: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_407);  sub_122 = unsqueeze_407 = None
        mul_461: "f32[192]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_55);  sum_43 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_460, relu_17, primals_110, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_460 = primals_110 = None
        getitem_144: "f32[32, 192, 14, 14]" = convolution_backward_20[0]
        getitem_145: "f32[192, 192, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
        add_211: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_22, getitem_144);  slice_22 = getitem_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_21: "b8[32, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        where_21: "f32[32, 192, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, add_211);  le_21 = add_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_44: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
        sub_123: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_410);  convolution_17 = unsqueeze_410 = None
        mul_462: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_123)
        sum_45: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_462, [0, 2, 3]);  mul_462 = None
        mul_463: "f32[192]" = torch.ops.aten.mul.Tensor(sum_44, 0.00015943877551020407)
        unsqueeze_411: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
        unsqueeze_412: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
        unsqueeze_413: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
        mul_464: "f32[192]" = torch.ops.aten.mul.Tensor(sum_45, 0.00015943877551020407)
        mul_465: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_466: "f32[192]" = torch.ops.aten.mul.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None
        unsqueeze_414: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_466, 0);  mul_466 = None
        unsqueeze_415: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
        unsqueeze_416: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
        mul_467: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_417: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
        unsqueeze_418: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
        unsqueeze_419: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
        mul_468: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_416);  sub_123 = unsqueeze_416 = None
        sub_125: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_468);  where_21 = mul_468 = None
        sub_126: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_125, unsqueeze_413);  sub_125 = unsqueeze_413 = None
        mul_469: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_419);  sub_126 = unsqueeze_419 = None
        mul_470: "f32[192]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_52);  sum_45 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_469, relu_16, primals_104, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_469 = primals_104 = None
        getitem_147: "f32[32, 192, 14, 14]" = convolution_backward_21[0]
        getitem_148: "f32[192, 192, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
        add_212: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_21, getitem_147);  slice_21 = getitem_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_22: "b8[32, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        where_22: "f32[32, 192, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, add_212);  le_22 = add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_46: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
        sub_127: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_422);  convolution_16 = unsqueeze_422 = None
        mul_471: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_127)
        sum_47: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_471, [0, 2, 3]);  mul_471 = None
        mul_472: "f32[192]" = torch.ops.aten.mul.Tensor(sum_46, 0.00015943877551020407)
        unsqueeze_423: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
        unsqueeze_424: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
        unsqueeze_425: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
        mul_473: "f32[192]" = torch.ops.aten.mul.Tensor(sum_47, 0.00015943877551020407)
        mul_474: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_475: "f32[192]" = torch.ops.aten.mul.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
        unsqueeze_426: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
        unsqueeze_427: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
        unsqueeze_428: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
        mul_476: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_429: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
        unsqueeze_430: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
        unsqueeze_431: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
        mul_477: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_428);  sub_127 = unsqueeze_428 = None
        sub_129: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_477);  where_22 = mul_477 = None
        sub_130: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_129, unsqueeze_425);  sub_129 = unsqueeze_425 = None
        mul_478: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_431);  sub_130 = unsqueeze_431 = None
        mul_479: "f32[192]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_49);  sum_47 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_478, relu_15, primals_98, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_478 = primals_98 = None
        getitem_150: "f32[32, 192, 14, 14]" = convolution_backward_22[0]
        getitem_151: "f32[192, 192, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
        add_213: "f32[32, 192, 14, 14]" = torch.ops.aten.add.Tensor(slice_20, getitem_150);  slice_20 = getitem_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_23: "b8[32, 192, 14, 14]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_23: "f32[32, 192, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, add_213);  le_23 = add_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_48: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
        sub_131: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_434);  convolution_15 = unsqueeze_434 = None
        mul_480: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_131)
        sum_49: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 2, 3]);  mul_480 = None
        mul_481: "f32[192]" = torch.ops.aten.mul.Tensor(sum_48, 0.00015943877551020407)
        unsqueeze_435: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
        unsqueeze_436: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
        unsqueeze_437: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
        mul_482: "f32[192]" = torch.ops.aten.mul.Tensor(sum_49, 0.00015943877551020407)
        mul_483: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_484: "f32[192]" = torch.ops.aten.mul.Tensor(mul_482, mul_483);  mul_482 = mul_483 = None
        unsqueeze_438: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
        unsqueeze_439: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
        unsqueeze_440: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
        mul_485: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_441: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
        unsqueeze_442: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
        unsqueeze_443: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
        mul_486: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_440);  sub_131 = unsqueeze_440 = None
        sub_133: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_486);  where_23 = mul_486 = None
        sub_134: "f32[32, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_437);  sub_133 = unsqueeze_437 = None
        mul_487: "f32[32, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_443);  sub_134 = unsqueeze_443 = None
        mul_488: "f32[192]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_46);  sum_49 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_487, getitem_32, primals_92, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_487 = getitem_32 = primals_92 = None
        getitem_153: "f32[32, 512, 14, 14]" = convolution_backward_23[0]
        getitem_154: "f32[192, 512, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
        add_214: "f32[32, 512, 14, 14]" = torch.ops.aten.add.Tensor(slice_19, getitem_153);  slice_19 = getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:161 in forward, code: x = self.pool(x)
        full_default_25: "f32[16384, 784]" = torch.ops.aten.full.default([16384, 784], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_6: "f32[16384, 196]" = torch.ops.aten.reshape.default(add_214, [16384, 196]);  add_214 = None
        _low_memory_max_pool_offsets_to_indices_1: "i64[32, 512, 14, 14]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_33, [3, 3], [28, 28], [2, 2], [0, 0], [1, 1]);  getitem_33 = None
        view_7: "i64[16384, 196]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices_1, [16384, 196]);  _low_memory_max_pool_offsets_to_indices_1 = None
        scatter_add_1: "f32[16384, 784]" = torch.ops.aten.scatter_add.default(full_default_25, 1, view_7, view_6);  full_default_25 = view_7 = view_6 = None
        view_8: "f32[32, 512, 28, 28]" = torch.ops.aten.reshape.default(scatter_add_1, [32, 512, 28, 28]);  scatter_add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_14: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, getitem_31)
        mul_98: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        unsqueeze_56: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_74: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_14: "f32[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_74);  add_74 = None
        le_24: "b8[32, 512, 28, 28]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_24: "f32[32, 512, 28, 28]" = torch.ops.aten.where.self(le_24, full_default, view_8);  le_24 = view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_42: "f32[512]" = torch.ops.aten.squeeze.dims(getitem_31, [0, 2, 3]);  getitem_31 = None
        unsqueeze_444: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_445: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 2);  unsqueeze_444 = None
        unsqueeze_446: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 3);  unsqueeze_445 = None
        sum_50: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
        sub_135: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_446);  convolution_14 = unsqueeze_446 = None
        mul_489: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, sub_135)
        sum_51: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_489, [0, 2, 3]);  mul_489 = None
        mul_490: "f32[512]" = torch.ops.aten.mul.Tensor(sum_50, 3.985969387755102e-05)
        unsqueeze_447: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
        unsqueeze_448: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
        unsqueeze_449: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
        mul_491: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, 3.985969387755102e-05)
        squeeze_43: "f32[512]" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_492: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_493: "f32[512]" = torch.ops.aten.mul.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
        unsqueeze_450: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
        unsqueeze_451: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
        unsqueeze_452: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
        mul_494: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_453: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
        unsqueeze_454: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
        unsqueeze_455: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
        mul_495: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_452);  sub_135 = unsqueeze_452 = None
        sub_137: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(where_24, mul_495);  where_24 = mul_495 = None
        sub_138: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_449);  sub_137 = unsqueeze_449 = None
        mul_496: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_455);  sub_138 = unsqueeze_455 = None
        mul_497: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_43);  sum_51 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_496, cat_1, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_496 = cat_1 = primals_86 = None
        getitem_156: "f32[32, 1056, 28, 28]" = convolution_backward_24[0]
        getitem_157: "f32[512, 1056, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        slice_25: "f32[32, 256, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_156, 1, 0, 256)
        slice_26: "f32[32, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_156, 1, 256, 416)
        slice_27: "f32[32, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_156, 1, 416, 576)
        slice_28: "f32[32, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_156, 1, 576, 736)
        slice_29: "f32[32, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_156, 1, 736, 896)
        slice_30: "f32[32, 160, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_156, 1, 896, 1056);  getitem_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_13: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, getitem_29)
        mul_91: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        unsqueeze_52: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[160, 1]" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_69: "f32[32, 160, 28, 28]" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_13: "f32[32, 160, 28, 28]" = torch.ops.aten.relu.default(add_69);  add_69 = None
        le_25: "b8[32, 160, 28, 28]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_25: "f32[32, 160, 28, 28]" = torch.ops.aten.where.self(le_25, full_default, slice_30);  le_25 = slice_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_39: "f32[160]" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        unsqueeze_456: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_457: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 2);  unsqueeze_456 = None
        unsqueeze_458: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 3);  unsqueeze_457 = None
        sum_52: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
        sub_139: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_458);  convolution_13 = unsqueeze_458 = None
        mul_498: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_139)
        sum_53: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_498, [0, 2, 3]);  mul_498 = None
        mul_499: "f32[160]" = torch.ops.aten.mul.Tensor(sum_52, 3.985969387755102e-05)
        unsqueeze_459: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
        unsqueeze_460: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
        unsqueeze_461: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
        mul_500: "f32[160]" = torch.ops.aten.mul.Tensor(sum_53, 3.985969387755102e-05)
        squeeze_40: "f32[160]" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_501: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_502: "f32[160]" = torch.ops.aten.mul.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
        unsqueeze_462: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
        unsqueeze_463: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
        unsqueeze_464: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
        mul_503: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_465: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
        unsqueeze_466: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
        unsqueeze_467: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
        mul_504: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_464);  sub_139 = unsqueeze_464 = None
        sub_141: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_25, mul_504);  where_25 = mul_504 = None
        sub_142: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_141, unsqueeze_461);  sub_141 = unsqueeze_461 = None
        mul_505: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_467);  sub_142 = unsqueeze_467 = None
        mul_506: "f32[160]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_40);  sum_53 = squeeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_505, relu_12, primals_80, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_505 = primals_80 = None
        getitem_159: "f32[32, 160, 28, 28]" = convolution_backward_25[0]
        getitem_160: "f32[160, 160, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
        add_215: "f32[32, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_29, getitem_159);  slice_29 = getitem_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_26: "b8[32, 160, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_26: "f32[32, 160, 28, 28]" = torch.ops.aten.where.self(le_26, full_default, add_215);  le_26 = add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_54: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
        sub_143: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_470);  convolution_12 = unsqueeze_470 = None
        mul_507: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, sub_143)
        sum_55: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
        mul_508: "f32[160]" = torch.ops.aten.mul.Tensor(sum_54, 3.985969387755102e-05)
        unsqueeze_471: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
        unsqueeze_472: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
        unsqueeze_473: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
        mul_509: "f32[160]" = torch.ops.aten.mul.Tensor(sum_55, 3.985969387755102e-05)
        mul_510: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_511: "f32[160]" = torch.ops.aten.mul.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
        unsqueeze_474: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
        unsqueeze_475: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
        unsqueeze_476: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
        mul_512: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_477: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
        unsqueeze_478: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
        unsqueeze_479: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
        mul_513: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_476);  sub_143 = unsqueeze_476 = None
        sub_145: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_26, mul_513);  where_26 = mul_513 = None
        sub_146: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_145, unsqueeze_473);  sub_145 = unsqueeze_473 = None
        mul_514: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_479);  sub_146 = unsqueeze_479 = None
        mul_515: "f32[160]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_37);  sum_55 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_514, relu_11, primals_74, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_514 = primals_74 = None
        getitem_162: "f32[32, 160, 28, 28]" = convolution_backward_26[0]
        getitem_163: "f32[160, 160, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
        add_216: "f32[32, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_28, getitem_162);  slice_28 = getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_27: "b8[32, 160, 28, 28]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_27: "f32[32, 160, 28, 28]" = torch.ops.aten.where.self(le_27, full_default, add_216);  le_27 = add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_56: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
        sub_147: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_482);  convolution_11 = unsqueeze_482 = None
        mul_516: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_147)
        sum_57: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2, 3]);  mul_516 = None
        mul_517: "f32[160]" = torch.ops.aten.mul.Tensor(sum_56, 3.985969387755102e-05)
        unsqueeze_483: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
        unsqueeze_484: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
        unsqueeze_485: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
        mul_518: "f32[160]" = torch.ops.aten.mul.Tensor(sum_57, 3.985969387755102e-05)
        mul_519: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_520: "f32[160]" = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
        unsqueeze_486: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
        unsqueeze_487: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
        unsqueeze_488: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
        mul_521: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_489: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
        unsqueeze_490: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
        unsqueeze_491: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
        mul_522: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_488);  sub_147 = unsqueeze_488 = None
        sub_149: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_27, mul_522);  where_27 = mul_522 = None
        sub_150: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_149, unsqueeze_485);  sub_149 = unsqueeze_485 = None
        mul_523: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_491);  sub_150 = unsqueeze_491 = None
        mul_524: "f32[160]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_34);  sum_57 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_523, relu_10, primals_68, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_523 = primals_68 = None
        getitem_165: "f32[32, 160, 28, 28]" = convolution_backward_27[0]
        getitem_166: "f32[160, 160, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
        add_217: "f32[32, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_27, getitem_165);  slice_27 = getitem_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_28: "b8[32, 160, 28, 28]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_28: "f32[32, 160, 28, 28]" = torch.ops.aten.where.self(le_28, full_default, add_217);  le_28 = add_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_58: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
        sub_151: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_494);  convolution_10 = unsqueeze_494 = None
        mul_525: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, sub_151)
        sum_59: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_525, [0, 2, 3]);  mul_525 = None
        mul_526: "f32[160]" = torch.ops.aten.mul.Tensor(sum_58, 3.985969387755102e-05)
        unsqueeze_495: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
        unsqueeze_496: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
        unsqueeze_497: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
        mul_527: "f32[160]" = torch.ops.aten.mul.Tensor(sum_59, 3.985969387755102e-05)
        mul_528: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_529: "f32[160]" = torch.ops.aten.mul.Tensor(mul_527, mul_528);  mul_527 = mul_528 = None
        unsqueeze_498: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
        unsqueeze_499: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
        unsqueeze_500: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
        mul_530: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_501: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
        unsqueeze_502: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
        unsqueeze_503: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
        mul_531: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_500);  sub_151 = unsqueeze_500 = None
        sub_153: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_28, mul_531);  where_28 = mul_531 = None
        sub_154: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_153, unsqueeze_497);  sub_153 = unsqueeze_497 = None
        mul_532: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_503);  sub_154 = unsqueeze_503 = None
        mul_533: "f32[160]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_31);  sum_59 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_532, relu_9, primals_62, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_532 = primals_62 = None
        getitem_168: "f32[32, 160, 28, 28]" = convolution_backward_28[0]
        getitem_169: "f32[160, 160, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
        add_218: "f32[32, 160, 28, 28]" = torch.ops.aten.add.Tensor(slice_26, getitem_168);  slice_26 = getitem_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_29: "b8[32, 160, 28, 28]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_29: "f32[32, 160, 28, 28]" = torch.ops.aten.where.self(le_29, full_default, add_218);  le_29 = add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_60: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
        sub_155: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_506);  convolution_9 = unsqueeze_506 = None
        mul_534: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, sub_155)
        sum_61: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_534, [0, 2, 3]);  mul_534 = None
        mul_535: "f32[160]" = torch.ops.aten.mul.Tensor(sum_60, 3.985969387755102e-05)
        unsqueeze_507: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
        unsqueeze_508: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
        unsqueeze_509: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
        mul_536: "f32[160]" = torch.ops.aten.mul.Tensor(sum_61, 3.985969387755102e-05)
        mul_537: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_538: "f32[160]" = torch.ops.aten.mul.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
        unsqueeze_510: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
        unsqueeze_511: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
        unsqueeze_512: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
        mul_539: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_513: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
        unsqueeze_514: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
        unsqueeze_515: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
        mul_540: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_512);  sub_155 = unsqueeze_512 = None
        sub_157: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(where_29, mul_540);  where_29 = mul_540 = None
        sub_158: "f32[32, 160, 28, 28]" = torch.ops.aten.sub.Tensor(sub_157, unsqueeze_509);  sub_157 = unsqueeze_509 = None
        mul_541: "f32[32, 160, 28, 28]" = torch.ops.aten.mul.Tensor(sub_158, unsqueeze_515);  sub_158 = unsqueeze_515 = None
        mul_542: "f32[160]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_28);  sum_61 = squeeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_541, getitem_18, primals_56, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_541 = getitem_18 = primals_56 = None
        getitem_171: "f32[32, 256, 28, 28]" = convolution_backward_29[0]
        getitem_172: "f32[160, 256, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
        add_219: "f32[32, 256, 28, 28]" = torch.ops.aten.add.Tensor(slice_25, getitem_171);  slice_25 = getitem_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:161 in forward, code: x = self.pool(x)
        full_default_32: "f32[8192, 3136]" = torch.ops.aten.full.default([8192, 3136], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_9: "f32[8192, 784]" = torch.ops.aten.reshape.default(add_219, [8192, 784]);  add_219 = None
        _low_memory_max_pool_offsets_to_indices: "i64[32, 256, 28, 28]" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_19, [3, 3], [56, 56], [2, 2], [0, 0], [1, 1]);  getitem_19 = None
        view_10: "i64[8192, 784]" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [8192, 784]);  _low_memory_max_pool_offsets_to_indices = None
        scatter_add_2: "f32[8192, 3136]" = torch.ops.aten.scatter_add.default(full_default_32, 1, view_10, view_9);  full_default_32 = view_10 = view_9 = None
        view_11: "f32[32, 256, 56, 56]" = torch.ops.aten.reshape.default(scatter_add_2, [32, 256, 56, 56]);  scatter_add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_8: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_56: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_44: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_8: "f32[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_44);  add_44 = None
        le_30: "b8[32, 256, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_30: "f32[32, 256, 56, 56]" = torch.ops.aten.where.self(le_30, full_default, view_11);  le_30 = view_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_24: "f32[256]" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        unsqueeze_516: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_517: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 2);  unsqueeze_516 = None
        unsqueeze_518: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 3);  unsqueeze_517 = None
        sum_62: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
        sub_159: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_518);  convolution_8 = unsqueeze_518 = None
        mul_543: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, sub_159)
        sum_63: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 2, 3]);  mul_543 = None
        mul_544: "f32[256]" = torch.ops.aten.mul.Tensor(sum_62, 9.964923469387754e-06)
        unsqueeze_519: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
        unsqueeze_520: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
        unsqueeze_521: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
        mul_545: "f32[256]" = torch.ops.aten.mul.Tensor(sum_63, 9.964923469387754e-06)
        squeeze_25: "f32[256]" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_546: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_547: "f32[256]" = torch.ops.aten.mul.Tensor(mul_545, mul_546);  mul_545 = mul_546 = None
        unsqueeze_522: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
        unsqueeze_523: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
        unsqueeze_524: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
        mul_548: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_525: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
        unsqueeze_526: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
        unsqueeze_527: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
        mul_549: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_524);  sub_159 = unsqueeze_524 = None
        sub_161: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(where_30, mul_549);  where_30 = mul_549 = None
        sub_162: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(sub_161, unsqueeze_521);  sub_161 = unsqueeze_521 = None
        mul_550: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_527);  sub_162 = unsqueeze_527 = None
        mul_551: "f32[256]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_25);  sum_63 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_550, cat, primals_50, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_550 = cat = primals_50 = None
        getitem_174: "f32[32, 768, 56, 56]" = convolution_backward_30[0]
        getitem_175: "f32[256, 768, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vovnet.py:40 in forward, code: x = torch.cat(concat_list, dim=1)
        slice_31: "f32[32, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_174, 1, 0, 128)
        slice_32: "f32[32, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_174, 1, 128, 256)
        slice_33: "f32[32, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_174, 1, 256, 384)
        slice_34: "f32[32, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_174, 1, 384, 512)
        slice_35: "f32[32, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_174, 1, 512, 640)
        slice_36: "f32[32, 128, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_174, 1, 640, 768);  getitem_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_7: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        unsqueeze_28: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_39: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        relu_7: "f32[32, 128, 56, 56]" = torch.ops.aten.relu.default(add_39);  add_39 = None
        le_31: "b8[32, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_31: "f32[32, 128, 56, 56]" = torch.ops.aten.where.self(le_31, full_default, slice_36);  le_31 = slice_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_21: "f32[128]" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        unsqueeze_528: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_529: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 2);  unsqueeze_528 = None
        unsqueeze_530: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 3);  unsqueeze_529 = None
        sum_64: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
        sub_163: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_530);  convolution_7 = unsqueeze_530 = None
        mul_552: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, sub_163)
        sum_65: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_552, [0, 2, 3]);  mul_552 = None
        mul_553: "f32[128]" = torch.ops.aten.mul.Tensor(sum_64, 9.964923469387754e-06)
        unsqueeze_531: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
        unsqueeze_532: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
        unsqueeze_533: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
        mul_554: "f32[128]" = torch.ops.aten.mul.Tensor(sum_65, 9.964923469387754e-06)
        squeeze_22: "f32[128]" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_555: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_556: "f32[128]" = torch.ops.aten.mul.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
        unsqueeze_534: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
        unsqueeze_535: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
        unsqueeze_536: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
        mul_557: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_537: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
        unsqueeze_538: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
        unsqueeze_539: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
        mul_558: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_536);  sub_163 = unsqueeze_536 = None
        sub_165: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_31, mul_558);  where_31 = mul_558 = None
        sub_166: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_165, unsqueeze_533);  sub_165 = unsqueeze_533 = None
        mul_559: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_166, unsqueeze_539);  sub_166 = unsqueeze_539 = None
        mul_560: "f32[128]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_22);  sum_65 = squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_559, relu_6, primals_44, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_559 = primals_44 = None
        getitem_177: "f32[32, 128, 56, 56]" = convolution_backward_31[0]
        getitem_178: "f32[128, 128, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
        add_220: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_35, getitem_177);  slice_35 = getitem_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_32: "b8[32, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_32: "f32[32, 128, 56, 56]" = torch.ops.aten.where.self(le_32, full_default, add_220);  le_32 = add_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_66: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
        sub_167: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_542);  convolution_6 = unsqueeze_542 = None
        mul_561: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_32, sub_167)
        sum_67: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_561, [0, 2, 3]);  mul_561 = None
        mul_562: "f32[128]" = torch.ops.aten.mul.Tensor(sum_66, 9.964923469387754e-06)
        unsqueeze_543: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
        unsqueeze_544: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 2);  unsqueeze_543 = None
        unsqueeze_545: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, 3);  unsqueeze_544 = None
        mul_563: "f32[128]" = torch.ops.aten.mul.Tensor(sum_67, 9.964923469387754e-06)
        mul_564: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_565: "f32[128]" = torch.ops.aten.mul.Tensor(mul_563, mul_564);  mul_563 = mul_564 = None
        unsqueeze_546: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
        unsqueeze_547: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
        unsqueeze_548: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
        mul_566: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_549: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
        unsqueeze_550: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
        unsqueeze_551: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
        mul_567: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_548);  sub_167 = unsqueeze_548 = None
        sub_169: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_32, mul_567);  where_32 = mul_567 = None
        sub_170: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_545);  sub_169 = unsqueeze_545 = None
        mul_568: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_551);  sub_170 = unsqueeze_551 = None
        mul_569: "f32[128]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_19);  sum_67 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_568, relu_5, primals_38, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_568 = primals_38 = None
        getitem_180: "f32[32, 128, 56, 56]" = convolution_backward_32[0]
        getitem_181: "f32[128, 128, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
        add_221: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_34, getitem_180);  slice_34 = getitem_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_33: "b8[32, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_33: "f32[32, 128, 56, 56]" = torch.ops.aten.where.self(le_33, full_default, add_221);  le_33 = add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_68: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
        sub_171: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_554);  convolution_5 = unsqueeze_554 = None
        mul_570: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_33, sub_171)
        sum_69: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 2, 3]);  mul_570 = None
        mul_571: "f32[128]" = torch.ops.aten.mul.Tensor(sum_68, 9.964923469387754e-06)
        unsqueeze_555: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
        unsqueeze_556: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
        unsqueeze_557: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
        mul_572: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, 9.964923469387754e-06)
        mul_573: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_574: "f32[128]" = torch.ops.aten.mul.Tensor(mul_572, mul_573);  mul_572 = mul_573 = None
        unsqueeze_558: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
        unsqueeze_559: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
        unsqueeze_560: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
        mul_575: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_561: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
        unsqueeze_562: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
        unsqueeze_563: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
        mul_576: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_560);  sub_171 = unsqueeze_560 = None
        sub_173: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_33, mul_576);  where_33 = mul_576 = None
        sub_174: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_173, unsqueeze_557);  sub_173 = unsqueeze_557 = None
        mul_577: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_563);  sub_174 = unsqueeze_563 = None
        mul_578: "f32[128]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_16);  sum_69 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_577, relu_4, primals_32, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_577 = primals_32 = None
        getitem_183: "f32[32, 128, 56, 56]" = convolution_backward_33[0]
        getitem_184: "f32[128, 128, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
        add_222: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_33, getitem_183);  slice_33 = getitem_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_34: "b8[32, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_34: "f32[32, 128, 56, 56]" = torch.ops.aten.where.self(le_34, full_default, add_222);  le_34 = add_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_70: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
        sub_175: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_566);  convolution_4 = unsqueeze_566 = None
        mul_579: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, sub_175)
        sum_71: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 2, 3]);  mul_579 = None
        mul_580: "f32[128]" = torch.ops.aten.mul.Tensor(sum_70, 9.964923469387754e-06)
        unsqueeze_567: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
        unsqueeze_568: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
        unsqueeze_569: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
        mul_581: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, 9.964923469387754e-06)
        mul_582: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_583: "f32[128]" = torch.ops.aten.mul.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
        unsqueeze_570: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
        unsqueeze_571: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
        unsqueeze_572: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
        mul_584: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_573: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
        unsqueeze_574: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
        unsqueeze_575: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
        mul_585: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_572);  sub_175 = unsqueeze_572 = None
        sub_177: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_34, mul_585);  where_34 = mul_585 = None
        sub_178: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_177, unsqueeze_569);  sub_177 = unsqueeze_569 = None
        mul_586: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_575);  sub_178 = unsqueeze_575 = None
        mul_587: "f32[128]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_13);  sum_71 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_586, relu_3, primals_26, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_586 = primals_26 = None
        getitem_186: "f32[32, 128, 56, 56]" = convolution_backward_34[0]
        getitem_187: "f32[128, 128, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
        add_223: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_32, getitem_186);  slice_32 = getitem_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_35: "b8[32, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_35: "f32[32, 128, 56, 56]" = torch.ops.aten.where.self(le_35, full_default, add_223);  le_35 = add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_72: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
        sub_179: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_578);  convolution_3 = unsqueeze_578 = None
        mul_588: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_35, sub_179)
        sum_73: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_588, [0, 2, 3]);  mul_588 = None
        mul_589: "f32[128]" = torch.ops.aten.mul.Tensor(sum_72, 9.964923469387754e-06)
        unsqueeze_579: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
        unsqueeze_580: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
        unsqueeze_581: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
        mul_590: "f32[128]" = torch.ops.aten.mul.Tensor(sum_73, 9.964923469387754e-06)
        mul_591: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_592: "f32[128]" = torch.ops.aten.mul.Tensor(mul_590, mul_591);  mul_590 = mul_591 = None
        unsqueeze_582: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
        unsqueeze_583: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
        unsqueeze_584: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
        mul_593: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_585: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
        unsqueeze_586: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
        unsqueeze_587: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
        mul_594: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_584);  sub_179 = unsqueeze_584 = None
        sub_181: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_35, mul_594);  where_35 = mul_594 = None
        sub_182: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_181, unsqueeze_581);  sub_181 = unsqueeze_581 = None
        mul_595: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_587);  sub_182 = unsqueeze_587 = None
        mul_596: "f32[128]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_10);  sum_73 = squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_595, relu_2, primals_20, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_595 = primals_20 = None
        getitem_189: "f32[32, 128, 56, 56]" = convolution_backward_35[0]
        getitem_190: "f32[128, 128, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
        add_224: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(slice_31, getitem_189);  slice_31 = getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_36: "b8[32, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_36: "f32[32, 128, 56, 56]" = torch.ops.aten.where.self(le_36, full_default, add_224);  le_36 = add_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_74: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
        sub_183: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_590);  convolution_2 = unsqueeze_590 = None
        mul_597: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_36, sub_183)
        sum_75: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 2, 3]);  mul_597 = None
        mul_598: "f32[128]" = torch.ops.aten.mul.Tensor(sum_74, 9.964923469387754e-06)
        unsqueeze_591: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
        unsqueeze_592: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
        unsqueeze_593: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
        mul_599: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, 9.964923469387754e-06)
        mul_600: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_601: "f32[128]" = torch.ops.aten.mul.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
        unsqueeze_594: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
        unsqueeze_595: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
        unsqueeze_596: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
        mul_602: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_597: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
        unsqueeze_598: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
        unsqueeze_599: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
        mul_603: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_596);  sub_183 = unsqueeze_596 = None
        sub_185: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_36, mul_603);  where_36 = mul_603 = None
        sub_186: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_185, unsqueeze_593);  sub_185 = unsqueeze_593 = None
        mul_604: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_186, unsqueeze_599);  sub_186 = unsqueeze_599 = None
        mul_605: "f32[128]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_7);  sum_75 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_604, relu_1, primals_14, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_604 = primals_14 = None
        getitem_192: "f32[32, 64, 112, 112]" = convolution_backward_36[0]
        getitem_193: "f32[128, 64, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_37: "b8[32, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_37: "f32[32, 64, 112, 112]" = torch.ops.aten.where.self(le_37, full_default, getitem_192);  le_37 = getitem_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_76: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
        sub_187: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_602);  convolution_1 = unsqueeze_602 = None
        mul_606: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_37, sub_187)
        sum_77: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 2, 3]);  mul_606 = None
        mul_607: "f32[64]" = torch.ops.aten.mul.Tensor(sum_76, 2.4912308673469386e-06)
        unsqueeze_603: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
        unsqueeze_604: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
        unsqueeze_605: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
        mul_608: "f32[64]" = torch.ops.aten.mul.Tensor(sum_77, 2.4912308673469386e-06)
        mul_609: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_610: "f32[64]" = torch.ops.aten.mul.Tensor(mul_608, mul_609);  mul_608 = mul_609 = None
        unsqueeze_606: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
        unsqueeze_607: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
        unsqueeze_608: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
        mul_611: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_609: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
        unsqueeze_610: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
        unsqueeze_611: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
        mul_612: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_608);  sub_187 = unsqueeze_608 = None
        sub_189: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_37, mul_612);  where_37 = mul_612 = None
        sub_190: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_605);  sub_189 = unsqueeze_605 = None
        mul_613: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_611);  sub_190 = unsqueeze_611 = None
        mul_614: "f32[64]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_4);  sum_77 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_613, relu, primals_8, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_613 = primals_8 = None
        getitem_195: "f32[32, 64, 112, 112]" = convolution_backward_37[0]
        getitem_196: "f32[64, 64, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        le_38: "b8[32, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_38: "f32[32, 64, 112, 112]" = torch.ops.aten.where.self(le_38, full_default, getitem_195);  le_38 = full_default = getitem_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_78: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
        sub_191: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_614);  convolution = unsqueeze_614 = None
        mul_615: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_38, sub_191)
        sum_79: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_615, [0, 2, 3]);  mul_615 = None
        mul_616: "f32[64]" = torch.ops.aten.mul.Tensor(sum_78, 2.4912308673469386e-06)
        unsqueeze_615: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
        unsqueeze_616: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
        unsqueeze_617: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
        mul_617: "f32[64]" = torch.ops.aten.mul.Tensor(sum_79, 2.4912308673469386e-06)
        mul_618: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_619: "f32[64]" = torch.ops.aten.mul.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
        unsqueeze_618: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
        unsqueeze_619: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
        unsqueeze_620: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
        mul_620: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_621: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
        unsqueeze_622: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
        unsqueeze_623: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
        mul_621: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_620);  sub_191 = unsqueeze_620 = None
        sub_193: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_38, mul_621);  where_38 = mul_621 = None
        sub_194: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_193, unsqueeze_617);  sub_193 = unsqueeze_617 = None
        mul_622: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_623);  sub_194 = unsqueeze_623 = None
        mul_623: "f32[64]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_1);  sum_79 = squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_622, primals_2, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_622 = primals_2 = primals_1 = None
        getitem_199: "f32[64, 3, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
        return (getitem_199, None, None, None, None, mul_623, sum_78, getitem_196, None, None, None, mul_614, sum_76, getitem_193, None, None, None, mul_605, sum_74, getitem_190, None, None, None, mul_596, sum_72, getitem_187, None, None, None, mul_587, sum_70, getitem_184, None, None, None, mul_578, sum_68, getitem_181, None, None, None, mul_569, sum_66, getitem_178, None, None, None, mul_560, sum_64, getitem_175, None, None, None, mul_551, sum_62, getitem_172, None, None, None, mul_542, sum_60, getitem_169, None, None, None, mul_533, sum_58, getitem_166, None, None, None, mul_524, sum_56, getitem_163, None, None, None, mul_515, sum_54, getitem_160, None, None, None, mul_506, sum_52, getitem_157, None, None, None, mul_497, sum_50, getitem_154, None, None, None, mul_488, sum_48, getitem_151, None, None, None, mul_479, sum_46, getitem_148, None, None, None, mul_470, sum_44, getitem_145, None, None, None, mul_461, sum_42, getitem_142, None, None, None, mul_452, sum_40, getitem_139, None, None, None, mul_443, sum_38, getitem_136, None, None, None, mul_434, sum_36, getitem_133, None, None, None, mul_425, sum_34, getitem_130, None, None, None, mul_416, sum_32, getitem_127, None, None, None, mul_407, sum_30, getitem_124, None, None, None, mul_398, sum_28, getitem_121, None, None, None, mul_389, sum_26, getitem_118, None, None, None, mul_380, sum_24, getitem_115, None, None, None, mul_371, sum_22, getitem_112, None, None, None, mul_362, sum_20, getitem_109, None, None, None, mul_353, sum_18, getitem_106, None, None, None, mul_344, sum_16, getitem_103, None, None, None, mul_335, sum_14, getitem_100, None, None, None, mul_326, sum_12, getitem_97, None, None, None, mul_317, sum_10, getitem_94, None, None, None, mul_308, sum_8, getitem_91, None, None, None, mul_299, sum_6, getitem_88, None, None, None, mul_290, sum_4, getitem_85, None, None, None, mul_281, sum_2, mm_1, view_1)
