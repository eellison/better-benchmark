class GraphModule(torch.nn.Module):
    def forward(self, primals_6: "f32[64][1]cuda:0", primals_7: "f32[64][1]cuda:0", primals_12: "f32[64][1]cuda:0", primals_18: "f32[64][1]cuda:0", primals_24: "f32[64][1]cuda:0", primals_30: "f32[64][1]cuda:0", primals_36: "f32[128][1]cuda:0", primals_42: "f32[128][1]cuda:0", primals_48: "f32[128][1]cuda:0", primals_54: "f32[128][1]cuda:0", primals_60: "f32[128][1]cuda:0", primals_66: "f32[256][1]cuda:0", primals_72: "f32[256][1]cuda:0", primals_78: "f32[256][1]cuda:0", primals_84: "f32[256][1]cuda:0", primals_90: "f32[256][1]cuda:0", primals_96: "f32[512][1]cuda:0", primals_102: "f32[512][1]cuda:0", primals_108: "f32[512][1]cuda:0", primals_114: "f32[512][1]cuda:0", primals_120: "f32[512][1]cuda:0", convert_element_type: "bf16[64, 3, 7, 7][147, 1, 21, 3]cuda:0", convert_element_type_1: "bf16[16, 3, 224, 224][150528, 1, 672, 3]cuda:0", convolution: "bf16[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0", getitem_1: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", rsqrt: "f32[1, 64, 1, 1][64, 1, 64, 64]cuda:0", getitem_2: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0", getitem_3: "i8[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_4: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_1: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_4: "f32[64][1]cuda:0", relu_1: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_7: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_2: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_7: "f32[64][1]cuda:0", relu_2: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_10: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_3: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_10: "f32[64][1]cuda:0", relu_3: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_13: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_4: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0", squeeze_13: "f32[64][1]cuda:0", relu_4: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0", convert_element_type_16: "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0", convolution_5: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_16: "f32[128][1]cuda:0", relu_5: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_19: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_6: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_19: "f32[128][1]cuda:0", convert_element_type_22: "bf16[128, 64, 1, 1][64, 1, 64, 64]cuda:0", convolution_7: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_22: "f32[128][1]cuda:0", relu_6: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_25: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_8: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_25: "f32[128][1]cuda:0", relu_7: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_28: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_9: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0", squeeze_28: "f32[128][1]cuda:0", relu_8: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0", convert_element_type_31: "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0", convolution_10: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_31: "f32[256][1]cuda:0", relu_9: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_34: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_11: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_34: "f32[256][1]cuda:0", convert_element_type_37: "bf16[256, 128, 1, 1][128, 1, 128, 128]cuda:0", convolution_12: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_37: "f32[256][1]cuda:0", relu_10: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_40: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_13: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_40: "f32[256][1]cuda:0", relu_11: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_43: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_14: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0", squeeze_43: "f32[256][1]cuda:0", relu_12: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0", convert_element_type_46: "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0", convolution_15: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0", squeeze_46: "f32[512][1]cuda:0", relu_13: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0", convert_element_type_49: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", convolution_16: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0", squeeze_49: "f32[512][1]cuda:0", convert_element_type_52: "bf16[512, 256, 1, 1][256, 1, 256, 256]cuda:0", convolution_17: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0", squeeze_52: "f32[512][1]cuda:0", relu_14: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0", convert_element_type_55: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", convolution_18: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0", squeeze_55: "f32[512][1]cuda:0", relu_15: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0", convert_element_type_58: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0", convolution_19: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0", squeeze_58: "f32[512][1]cuda:0", view: "bf16[16, 512][512, 1]cuda:0", permute_1: "bf16[1000, 512][512, 1]cuda:0", le: "b8[16, 512, 7, 7][25088, 49, 7, 1]cuda:0", unsqueeze_82: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_94: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_106: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_118: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_130: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", unsqueeze_142: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_154: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_166: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_178: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_190: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", unsqueeze_202: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_214: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_226: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_238: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_250: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", unsqueeze_262: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_274: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_286: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_298: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", tangents_1: "bf16[16, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        mm: "bf16[16, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
        permute_2: "bf16[1000, 16][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_1: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
        convert_element_type_70: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1, torch.bfloat16);  view_1 = None
        convert_element_type_71: "f32[1000, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_72: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_70, torch.float32);  convert_element_type_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        view_2: "bf16[16, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [16, 512, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        expand: "bf16[16, 512, 7, 7][512, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(view_2, [16, 512, 7, 7]);  view_2 = None
        div: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand, 49);  expand = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        full_default: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.where.self(le, full_default, div);  le = div = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_73: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(where, torch.float32)
        sum_2: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_73, [0, 2, 3])
        convert_element_type_59: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        sub_20: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_59, unsqueeze_82);  convert_element_type_59 = unsqueeze_82 = None
        mul_140: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_73, sub_20)
        sum_3: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_140, [0, 2, 3]);  mul_140 = None
        mul_141: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.0012755102040816326)
        unsqueeze_83: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_141, 0);  mul_141 = None
        unsqueeze_84: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_83, 2);  unsqueeze_83 = None
        unsqueeze_85: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 3);  unsqueeze_84 = None
        mul_142: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.0012755102040816326)
        mul_143: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_144: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
        unsqueeze_86: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_144, 0);  mul_144 = None
        unsqueeze_87: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_86, 2);  unsqueeze_86 = None
        unsqueeze_88: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_87, 3);  unsqueeze_87 = None
        mul_145: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_120);  primals_120 = None
        unsqueeze_89: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_145, 0);  mul_145 = None
        unsqueeze_90: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_89, 2);  unsqueeze_89 = None
        unsqueeze_91: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, 3);  unsqueeze_90 = None
        mul_146: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_88);  sub_20 = unsqueeze_88 = None
        sub_22: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_73, mul_146);  convert_element_type_73 = mul_146 = None
        sub_23: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_22, unsqueeze_85);  sub_22 = unsqueeze_85 = None
        mul_147: "f32[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_91);  sub_23 = unsqueeze_91 = None
        mul_148: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_58);  sum_3 = squeeze_58 = None
        convert_element_type_75: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_147, torch.bfloat16);  mul_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward = torch.ops.aten.convolution_backward.default(convert_element_type_75, relu_15, convert_element_type_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_75 = convert_element_type_58 = None
        getitem_42: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = convolution_backward[0]
        getitem_43: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_76: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_1: "b8[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        where_1: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.where.self(le_1, full_default, getitem_42);  le_1 = getitem_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_77: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_1, torch.float32);  where_1 = None
        sum_4: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_77, [0, 2, 3])
        convert_element_type_56: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_18, torch.float32);  convolution_18 = None
        sub_24: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_56, unsqueeze_94);  convert_element_type_56 = unsqueeze_94 = None
        mul_149: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_77, sub_24)
        sum_5: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_149, [0, 2, 3]);  mul_149 = None
        mul_150: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.0012755102040816326)
        unsqueeze_95: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_150, 0);  mul_150 = None
        unsqueeze_96: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 2);  unsqueeze_95 = None
        unsqueeze_97: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, 3);  unsqueeze_96 = None
        mul_151: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.0012755102040816326)
        mul_152: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_153: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, mul_152);  mul_151 = mul_152 = None
        unsqueeze_98: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_153, 0);  mul_153 = None
        unsqueeze_99: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, 2);  unsqueeze_98 = None
        unsqueeze_100: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 3);  unsqueeze_99 = None
        mul_154: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_114);  primals_114 = None
        unsqueeze_101: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_154, 0);  mul_154 = None
        unsqueeze_102: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 2);  unsqueeze_101 = None
        unsqueeze_103: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, 3);  unsqueeze_102 = None
        mul_155: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_100);  sub_24 = unsqueeze_100 = None
        sub_26: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_77, mul_155);  convert_element_type_77 = mul_155 = None
        sub_27: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_26, unsqueeze_97);  sub_26 = unsqueeze_97 = None
        mul_156: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_103);  sub_27 = unsqueeze_103 = None
        mul_157: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_55);  sum_5 = squeeze_55 = None
        convert_element_type_79: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16);  mul_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_79, relu_14, convert_element_type_55, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_79 = convert_element_type_55 = None
        getitem_45: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = convolution_backward_1[0]
        getitem_46: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        add_108: "bf16[16, 512, 7, 7][25088, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(where, getitem_45);  where = getitem_45 = None
        convert_element_type_80: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_46, torch.float32);  getitem_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_2: "b8[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        where_2: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.where.self(le_2, full_default, add_108);  le_2 = add_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convert_element_type_81: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_2, torch.float32);  where_2 = None
        sum_6: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_81, [0, 2, 3])
        convert_element_type_53: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_17, torch.float32);  convolution_17 = None
        sub_28: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_53, unsqueeze_106);  convert_element_type_53 = unsqueeze_106 = None
        mul_158: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, sub_28)
        sum_7: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 2, 3]);  mul_158 = None
        mul_159: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.0012755102040816326)
        unsqueeze_107: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_159, 0);  mul_159 = None
        unsqueeze_108: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_107, 2);  unsqueeze_107 = None
        unsqueeze_109: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 3);  unsqueeze_108 = None
        mul_160: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 0.0012755102040816326)
        mul_161: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_162: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, mul_161);  mul_160 = mul_161 = None
        unsqueeze_110: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_162, 0);  mul_162 = None
        unsqueeze_111: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_110, 2);  unsqueeze_110 = None
        unsqueeze_112: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_111, 3);  unsqueeze_111 = None
        mul_163: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_108);  primals_108 = None
        unsqueeze_113: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_163, 0);  mul_163 = None
        unsqueeze_114: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 2);  unsqueeze_113 = None
        unsqueeze_115: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 3);  unsqueeze_114 = None
        mul_164: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_112);  sub_28 = unsqueeze_112 = None
        sub_30: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_81, mul_164);  mul_164 = None
        sub_31: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_30, unsqueeze_109);  sub_30 = unsqueeze_109 = None
        mul_165: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_115);  sub_31 = unsqueeze_115 = None
        mul_166: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_52);  sum_7 = squeeze_52 = None
        convert_element_type_83: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16);  mul_165 = None
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(convert_element_type_83, relu_12, convert_element_type_52, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_83 = convert_element_type_52 = None
        getitem_48: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_2[0]
        getitem_49: "bf16[512, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        convert_element_type_84: "f32[512, 256, 1, 1][256, 1, 256, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_49, torch.float32);  getitem_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        sum_8: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_81, [0, 2, 3])
        convert_element_type_50: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        sub_32: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_50, unsqueeze_118);  convert_element_type_50 = unsqueeze_118 = None
        mul_167: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_81, sub_32)
        sum_9: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_167, [0, 2, 3]);  mul_167 = None
        mul_168: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_8, 0.0012755102040816326)
        unsqueeze_119: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_168, 0);  mul_168 = None
        unsqueeze_120: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 2);  unsqueeze_119 = None
        unsqueeze_121: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, 3);  unsqueeze_120 = None
        mul_169: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, 0.0012755102040816326)
        mul_170: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_171: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, mul_170);  mul_169 = mul_170 = None
        unsqueeze_122: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_171, 0);  mul_171 = None
        unsqueeze_123: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, 2);  unsqueeze_122 = None
        unsqueeze_124: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_123, 3);  unsqueeze_123 = None
        mul_172: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_125: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_172, 0);  mul_172 = None
        unsqueeze_126: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 2);  unsqueeze_125 = None
        unsqueeze_127: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 3);  unsqueeze_126 = None
        mul_173: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_124);  sub_32 = unsqueeze_124 = None
        sub_34: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_81, mul_173);  convert_element_type_81 = mul_173 = None
        sub_35: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_34, unsqueeze_121);  sub_34 = unsqueeze_121 = None
        mul_174: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_127);  sub_35 = unsqueeze_127 = None
        mul_175: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_9, squeeze_49);  sum_9 = squeeze_49 = None
        convert_element_type_87: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_174, torch.bfloat16);  mul_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(convert_element_type_87, relu_13, convert_element_type_49, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_87 = convert_element_type_49 = None
        getitem_51: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = convolution_backward_3[0]
        getitem_52: "bf16[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_88: "f32[512, 512, 3, 3][4608, 1, 1536, 512]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_52, torch.float32);  getitem_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_3: "b8[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        where_3: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.where.self(le_3, full_default, getitem_51);  le_3 = getitem_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_89: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(where_3, torch.float32);  where_3 = None
        sum_10: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_89, [0, 2, 3])
        convert_element_type_47: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_15, torch.float32);  convolution_15 = None
        sub_36: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_47, unsqueeze_130);  convert_element_type_47 = unsqueeze_130 = None
        mul_176: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_89, sub_36)
        sum_11: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 2, 3]);  mul_176 = None
        mul_177: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_10, 0.0012755102040816326)
        unsqueeze_131: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_177, 0);  mul_177 = None
        unsqueeze_132: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
        unsqueeze_133: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
        mul_178: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, 0.0012755102040816326)
        mul_179: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_180: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
        unsqueeze_134: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_180, 0);  mul_180 = None
        unsqueeze_135: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
        unsqueeze_136: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
        mul_181: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_137: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_181, 0);  mul_181 = None
        unsqueeze_138: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
        unsqueeze_139: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
        mul_182: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_136);  sub_36 = unsqueeze_136 = None
        sub_38: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_89, mul_182);  convert_element_type_89 = mul_182 = None
        sub_39: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, unsqueeze_133);  sub_38 = unsqueeze_133 = None
        mul_183: "f32[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_139);  sub_39 = unsqueeze_139 = None
        mul_184: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_11, squeeze_46);  sum_11 = squeeze_46 = None
        convert_element_type_91: "bf16[16, 512, 7, 7][25088, 1, 3584, 512]cuda:0" = torch.ops.prims.convert_element_type.default(mul_183, torch.bfloat16);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_91, relu_12, convert_element_type_46, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_91 = convert_element_type_46 = None
        getitem_54: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_4[0]
        getitem_55: "bf16[512, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        add_109: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, getitem_54);  getitem_48 = getitem_54 = None
        convert_element_type_92: "f32[512, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_55, torch.float32);  getitem_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_4: "b8[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        where_4: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_4, full_default, add_109);  le_4 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_93: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_4, torch.float32)
        sum_12: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_93, [0, 2, 3])
        convert_element_type_44: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_14, torch.float32);  convolution_14 = None
        sub_40: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_44, unsqueeze_142);  convert_element_type_44 = unsqueeze_142 = None
        mul_185: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_93, sub_40)
        sum_13: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_185, [0, 2, 3]);  mul_185 = None
        mul_186: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_12, 0.00031887755102040814)
        unsqueeze_143: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_186, 0);  mul_186 = None
        unsqueeze_144: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
        unsqueeze_145: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
        mul_187: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, 0.00031887755102040814)
        mul_188: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_189: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, mul_188);  mul_187 = mul_188 = None
        unsqueeze_146: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_189, 0);  mul_189 = None
        unsqueeze_147: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
        unsqueeze_148: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 3);  unsqueeze_147 = None
        mul_190: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_149: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_190, 0);  mul_190 = None
        unsqueeze_150: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
        unsqueeze_151: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 3);  unsqueeze_150 = None
        mul_191: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_148);  sub_40 = unsqueeze_148 = None
        sub_42: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_93, mul_191);  convert_element_type_93 = mul_191 = None
        sub_43: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_42, unsqueeze_145);  sub_42 = unsqueeze_145 = None
        mul_192: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_151);  sub_43 = unsqueeze_151 = None
        mul_193: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_13, squeeze_43);  sum_13 = squeeze_43 = None
        convert_element_type_95: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_192, torch.bfloat16);  mul_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(convert_element_type_95, relu_11, convert_element_type_43, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_95 = convert_element_type_43 = None
        getitem_57: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_5[0]
        getitem_58: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        convert_element_type_96: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_58, torch.float32);  getitem_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_5: "b8[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        where_5: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_5, full_default, getitem_57);  le_5 = getitem_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_97: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_5, torch.float32);  where_5 = None
        sum_14: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_97, [0, 2, 3])
        convert_element_type_41: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_13, torch.float32);  convolution_13 = None
        sub_44: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_41, unsqueeze_154);  convert_element_type_41 = unsqueeze_154 = None
        mul_194: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_97, sub_44)
        sum_15: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_194, [0, 2, 3]);  mul_194 = None
        mul_195: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_14, 0.00031887755102040814)
        unsqueeze_155: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_195, 0);  mul_195 = None
        unsqueeze_156: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
        unsqueeze_157: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
        mul_196: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, 0.00031887755102040814)
        mul_197: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_198: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        unsqueeze_158: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_198, 0);  mul_198 = None
        unsqueeze_159: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
        unsqueeze_160: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
        mul_199: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_161: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_199, 0);  mul_199 = None
        unsqueeze_162: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
        unsqueeze_163: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
        mul_200: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_160);  sub_44 = unsqueeze_160 = None
        sub_46: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_97, mul_200);  convert_element_type_97 = mul_200 = None
        sub_47: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_46, unsqueeze_157);  sub_46 = unsqueeze_157 = None
        mul_201: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_163);  sub_47 = unsqueeze_163 = None
        mul_202: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_15, squeeze_40);  sum_15 = squeeze_40 = None
        convert_element_type_99: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(convert_element_type_99, relu_10, convert_element_type_40, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_99 = convert_element_type_40 = None
        getitem_60: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_6[0]
        getitem_61: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        add_110: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.add.Tensor(where_4, getitem_60);  where_4 = getitem_60 = None
        convert_element_type_100: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_6: "b8[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        where_6: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_6, full_default, add_110);  le_6 = add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convert_element_type_101: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_6, torch.float32);  where_6 = None
        sum_16: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_101, [0, 2, 3])
        convert_element_type_38: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        sub_48: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_38, unsqueeze_166);  convert_element_type_38 = unsqueeze_166 = None
        mul_203: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_101, sub_48)
        sum_17: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_203, [0, 2, 3]);  mul_203 = None
        mul_204: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_16, 0.00031887755102040814)
        unsqueeze_167: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_204, 0);  mul_204 = None
        unsqueeze_168: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
        unsqueeze_169: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
        mul_205: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, 0.00031887755102040814)
        mul_206: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_207: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, mul_206);  mul_205 = mul_206 = None
        unsqueeze_170: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_207, 0);  mul_207 = None
        unsqueeze_171: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
        unsqueeze_172: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
        mul_208: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_173: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_208, 0);  mul_208 = None
        unsqueeze_174: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
        unsqueeze_175: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
        mul_209: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_172);  sub_48 = unsqueeze_172 = None
        sub_50: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, mul_209);  mul_209 = None
        sub_51: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_50, unsqueeze_169);  sub_50 = unsqueeze_169 = None
        mul_210: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_175);  sub_51 = unsqueeze_175 = None
        mul_211: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_17, squeeze_37);  sum_17 = squeeze_37 = None
        convert_element_type_103: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_210, torch.bfloat16);  mul_210 = None
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(convert_element_type_103, relu_8, convert_element_type_37, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_103 = convert_element_type_37 = None
        getitem_63: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_7[0]
        getitem_64: "bf16[256, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_104: "f32[256, 128, 1, 1][128, 1, 128, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_64, torch.float32);  getitem_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        sum_18: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_101, [0, 2, 3])
        convert_element_type_35: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_11, torch.float32);  convolution_11 = None
        sub_52: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_35, unsqueeze_178);  convert_element_type_35 = unsqueeze_178 = None
        mul_212: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_101, sub_52)
        sum_19: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_212, [0, 2, 3]);  mul_212 = None
        mul_213: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_18, 0.00031887755102040814)
        unsqueeze_179: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_213, 0);  mul_213 = None
        unsqueeze_180: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
        unsqueeze_181: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
        mul_214: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, 0.00031887755102040814)
        mul_215: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_216: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
        unsqueeze_182: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_216, 0);  mul_216 = None
        unsqueeze_183: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
        unsqueeze_184: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
        mul_217: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_185: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_217, 0);  mul_217 = None
        unsqueeze_186: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
        unsqueeze_187: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
        mul_218: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_184);  sub_52 = unsqueeze_184 = None
        sub_54: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_101, mul_218);  convert_element_type_101 = mul_218 = None
        sub_55: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_54, unsqueeze_181);  sub_54 = unsqueeze_181 = None
        mul_219: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_187);  sub_55 = unsqueeze_187 = None
        mul_220: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_19, squeeze_34);  sum_19 = squeeze_34 = None
        convert_element_type_107: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_219, torch.bfloat16);  mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_107, relu_9, convert_element_type_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_107 = convert_element_type_34 = None
        getitem_66: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = convolution_backward_8[0]
        getitem_67: "bf16[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_108: "f32[256, 256, 3, 3][2304, 1, 768, 256]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_7: "b8[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        where_7: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.where.self(le_7, full_default, getitem_66);  le_7 = getitem_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_109: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(where_7, torch.float32);  where_7 = None
        sum_20: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_109, [0, 2, 3])
        convert_element_type_32: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_10, torch.float32);  convolution_10 = None
        sub_56: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_32, unsqueeze_190);  convert_element_type_32 = unsqueeze_190 = None
        mul_221: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_109, sub_56)
        sum_21: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_221, [0, 2, 3]);  mul_221 = None
        mul_222: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_20, 0.00031887755102040814)
        unsqueeze_191: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_222, 0);  mul_222 = None
        unsqueeze_192: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
        unsqueeze_193: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
        mul_223: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, 0.00031887755102040814)
        mul_224: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_225: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
        unsqueeze_194: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_225, 0);  mul_225 = None
        unsqueeze_195: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
        unsqueeze_196: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
        mul_226: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_197: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_226, 0);  mul_226 = None
        unsqueeze_198: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
        unsqueeze_199: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
        mul_227: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_196);  sub_56 = unsqueeze_196 = None
        sub_58: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_109, mul_227);  convert_element_type_109 = mul_227 = None
        sub_59: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_58, unsqueeze_193);  sub_58 = unsqueeze_193 = None
        mul_228: "f32[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_199);  sub_59 = unsqueeze_199 = None
        mul_229: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_21, squeeze_31);  sum_21 = squeeze_31 = None
        convert_element_type_111: "bf16[16, 256, 14, 14][50176, 1, 3584, 256]cuda:0" = torch.ops.prims.convert_element_type.default(mul_228, torch.bfloat16);  mul_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(convert_element_type_111, relu_8, convert_element_type_31, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_111 = convert_element_type_31 = None
        getitem_69: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_9[0]
        getitem_70: "bf16[256, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        add_111: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(getitem_63, getitem_69);  getitem_63 = getitem_69 = None
        convert_element_type_112: "f32[256, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_70, torch.float32);  getitem_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_8: "b8[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        where_8: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_8, full_default, add_111);  le_8 = add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_113: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_8, torch.float32)
        sum_22: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_113, [0, 2, 3])
        convert_element_type_29: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        sub_60: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_29, unsqueeze_202);  convert_element_type_29 = unsqueeze_202 = None
        mul_230: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_113, sub_60)
        sum_23: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 2, 3]);  mul_230 = None
        mul_231: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_22, 7.971938775510203e-05)
        unsqueeze_203: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_231, 0);  mul_231 = None
        unsqueeze_204: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
        unsqueeze_205: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
        mul_232: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, 7.971938775510203e-05)
        mul_233: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_234: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
        unsqueeze_206: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_234, 0);  mul_234 = None
        unsqueeze_207: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
        unsqueeze_208: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
        mul_235: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_209: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_235, 0);  mul_235 = None
        unsqueeze_210: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
        unsqueeze_211: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
        mul_236: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_208);  sub_60 = unsqueeze_208 = None
        sub_62: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_113, mul_236);  convert_element_type_113 = mul_236 = None
        sub_63: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_205);  sub_62 = unsqueeze_205 = None
        mul_237: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_211);  sub_63 = unsqueeze_211 = None
        mul_238: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_23, squeeze_28);  sum_23 = squeeze_28 = None
        convert_element_type_115: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_237, torch.bfloat16);  mul_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(convert_element_type_115, relu_7, convert_element_type_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_115 = convert_element_type_28 = None
        getitem_72: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_10[0]
        getitem_73: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_116: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_9: "b8[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        where_9: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_9, full_default, getitem_72);  le_9 = getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_117: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_9, torch.float32);  where_9 = None
        sum_24: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_117, [0, 2, 3])
        convert_element_type_26: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_8, torch.float32);  convolution_8 = None
        sub_64: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_26, unsqueeze_214);  convert_element_type_26 = unsqueeze_214 = None
        mul_239: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_117, sub_64)
        sum_25: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 2, 3]);  mul_239 = None
        mul_240: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_24, 7.971938775510203e-05)
        unsqueeze_215: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_240, 0);  mul_240 = None
        unsqueeze_216: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
        unsqueeze_217: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
        mul_241: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, 7.971938775510203e-05)
        mul_242: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_243: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
        unsqueeze_218: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_243, 0);  mul_243 = None
        unsqueeze_219: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
        unsqueeze_220: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
        mul_244: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_221: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_244, 0);  mul_244 = None
        unsqueeze_222: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
        unsqueeze_223: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
        mul_245: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_220);  sub_64 = unsqueeze_220 = None
        sub_66: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_117, mul_245);  convert_element_type_117 = mul_245 = None
        sub_67: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_217);  sub_66 = unsqueeze_217 = None
        mul_246: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_223);  sub_67 = unsqueeze_223 = None
        mul_247: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_25, squeeze_25);  sum_25 = squeeze_25 = None
        convert_element_type_119: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_246, torch.bfloat16);  mul_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_119, relu_6, convert_element_type_25, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_119 = convert_element_type_25 = None
        getitem_75: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_11[0]
        getitem_76: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        add_112: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.add.Tensor(where_8, getitem_75);  where_8 = getitem_75 = None
        convert_element_type_120: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_76, torch.float32);  getitem_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_10: "b8[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        where_10: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_10, full_default, add_112);  le_10 = add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:100 in forward, code: identity = self.downsample(x)
        convert_element_type_121: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_10, torch.float32);  where_10 = None
        sum_26: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_121, [0, 2, 3])
        convert_element_type_23: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_7, torch.float32);  convolution_7 = None
        sub_68: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_23, unsqueeze_226);  convert_element_type_23 = unsqueeze_226 = None
        mul_248: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_121, sub_68)
        sum_27: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 2, 3]);  mul_248 = None
        mul_249: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_26, 7.971938775510203e-05)
        unsqueeze_227: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_249, 0);  mul_249 = None
        unsqueeze_228: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
        unsqueeze_229: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
        mul_250: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, 7.971938775510203e-05)
        mul_251: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_252: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
        unsqueeze_230: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_252, 0);  mul_252 = None
        unsqueeze_231: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
        unsqueeze_232: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
        mul_253: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_233: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_253, 0);  mul_253 = None
        unsqueeze_234: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
        unsqueeze_235: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
        mul_254: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_232);  sub_68 = unsqueeze_232 = None
        sub_70: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_121, mul_254);  mul_254 = None
        sub_71: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_229);  sub_70 = unsqueeze_229 = None
        mul_255: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_235);  sub_71 = unsqueeze_235 = None
        mul_256: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_27, squeeze_22);  sum_27 = squeeze_22 = None
        convert_element_type_123: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_255, torch.bfloat16);  mul_255 = None
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(convert_element_type_123, relu_4, convert_element_type_22, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_123 = convert_element_type_22 = None
        getitem_78: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_12[0]
        getitem_79: "bf16[128, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        convert_element_type_124: "f32[128, 64, 1, 1][64, 1, 64, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_79, torch.float32);  getitem_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        sum_28: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_121, [0, 2, 3])
        convert_element_type_20: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_6, torch.float32);  convolution_6 = None
        sub_72: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, unsqueeze_238);  convert_element_type_20 = unsqueeze_238 = None
        mul_257: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_121, sub_72)
        sum_29: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 2, 3]);  mul_257 = None
        mul_258: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_28, 7.971938775510203e-05)
        unsqueeze_239: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_258, 0);  mul_258 = None
        unsqueeze_240: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
        unsqueeze_241: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
        mul_259: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, 7.971938775510203e-05)
        mul_260: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_261: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, mul_260);  mul_259 = mul_260 = None
        unsqueeze_242: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_261, 0);  mul_261 = None
        unsqueeze_243: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
        unsqueeze_244: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
        mul_262: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_245: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_262, 0);  mul_262 = None
        unsqueeze_246: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
        unsqueeze_247: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
        mul_263: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_244);  sub_72 = unsqueeze_244 = None
        sub_74: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_121, mul_263);  convert_element_type_121 = mul_263 = None
        sub_75: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_241);  sub_74 = unsqueeze_241 = None
        mul_264: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_247);  sub_75 = unsqueeze_247 = None
        mul_265: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_29, squeeze_19);  sum_29 = squeeze_19 = None
        convert_element_type_127: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_264, torch.bfloat16);  mul_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(convert_element_type_127, relu_5, convert_element_type_19, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_127 = convert_element_type_19 = None
        getitem_81: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = convolution_backward_13[0]
        getitem_82: "bf16[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_128: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_82, torch.float32);  getitem_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_11: "b8[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        where_11: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.where.self(le_11, full_default, getitem_81);  le_11 = getitem_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_129: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(where_11, torch.float32);  where_11 = None
        sum_30: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_129, [0, 2, 3])
        convert_element_type_17: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        sub_76: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_17, unsqueeze_250);  convert_element_type_17 = unsqueeze_250 = None
        mul_266: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_129, sub_76)
        sum_31: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 2, 3]);  mul_266 = None
        mul_267: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_30, 7.971938775510203e-05)
        unsqueeze_251: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_267, 0);  mul_267 = None
        unsqueeze_252: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
        unsqueeze_253: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
        mul_268: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, 7.971938775510203e-05)
        mul_269: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_270: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
        unsqueeze_254: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_270, 0);  mul_270 = None
        unsqueeze_255: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
        unsqueeze_256: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
        mul_271: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_257: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
        unsqueeze_258: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
        unsqueeze_259: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
        mul_272: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_256);  sub_76 = unsqueeze_256 = None
        sub_78: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_129, mul_272);  convert_element_type_129 = mul_272 = None
        sub_79: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_253);  sub_78 = unsqueeze_253 = None
        mul_273: "f32[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_259);  sub_79 = unsqueeze_259 = None
        mul_274: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_31, squeeze_16);  sum_31 = squeeze_16 = None
        convert_element_type_131: "bf16[16, 128, 28, 28][100352, 1, 3584, 128]cuda:0" = torch.ops.prims.convert_element_type.default(mul_273, torch.bfloat16);  mul_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_131, relu_4, convert_element_type_16, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_131 = convert_element_type_16 = None
        getitem_84: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_14[0]
        getitem_85: "bf16[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        add_113: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, getitem_84);  getitem_78 = getitem_84 = None
        convert_element_type_132: "f32[128, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_85, torch.float32);  getitem_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_12: "b8[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        where_12: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_12, full_default, add_113);  le_12 = add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_133: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_12, torch.float32)
        sum_32: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_133, [0, 2, 3])
        convert_element_type_14: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_4, torch.float32);  convolution_4 = None
        sub_80: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_14, unsqueeze_262);  convert_element_type_14 = unsqueeze_262 = None
        mul_275: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_133, sub_80)
        sum_33: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 2, 3]);  mul_275 = None
        mul_276: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_32, 1.992984693877551e-05)
        unsqueeze_263: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_276, 0);  mul_276 = None
        unsqueeze_264: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        unsqueeze_265: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
        mul_277: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, 1.992984693877551e-05)
        mul_278: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_279: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
        unsqueeze_266: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_279, 0);  mul_279 = None
        unsqueeze_267: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
        unsqueeze_268: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
        mul_280: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_269: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_280, 0);  mul_280 = None
        unsqueeze_270: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
        unsqueeze_271: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
        mul_281: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_268);  sub_80 = unsqueeze_268 = None
        sub_82: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_133, mul_281);  convert_element_type_133 = mul_281 = None
        sub_83: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_265);  sub_82 = unsqueeze_265 = None
        mul_282: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_271);  sub_83 = unsqueeze_271 = None
        mul_283: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_33, squeeze_13);  sum_33 = squeeze_13 = None
        convert_element_type_135: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16);  mul_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(convert_element_type_135, relu_3, convert_element_type_13, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_135 = convert_element_type_13 = None
        getitem_87: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_15[0]
        getitem_88: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        convert_element_type_136: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_88, torch.float32);  getitem_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_13: "b8[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        where_13: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_13, full_default, getitem_87);  le_13 = getitem_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_137: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_13, torch.float32);  where_13 = None
        sum_34: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_137, [0, 2, 3])
        convert_element_type_11: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_3, torch.float32);  convolution_3 = None
        sub_84: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_11, unsqueeze_274);  convert_element_type_11 = unsqueeze_274 = None
        mul_284: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_137, sub_84)
        sum_35: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_284, [0, 2, 3]);  mul_284 = None
        mul_285: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_34, 1.992984693877551e-05)
        unsqueeze_275: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_285, 0);  mul_285 = None
        unsqueeze_276: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        unsqueeze_277: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
        mul_286: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, 1.992984693877551e-05)
        mul_287: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_288: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_286, mul_287);  mul_286 = mul_287 = None
        unsqueeze_278: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_288, 0);  mul_288 = None
        unsqueeze_279: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
        unsqueeze_280: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
        mul_289: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_281: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_289, 0);  mul_289 = None
        unsqueeze_282: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
        unsqueeze_283: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
        mul_290: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_280);  sub_84 = unsqueeze_280 = None
        sub_86: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_137, mul_290);  convert_element_type_137 = mul_290 = None
        sub_87: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_277);  sub_86 = unsqueeze_277 = None
        mul_291: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_283);  sub_87 = unsqueeze_283 = None
        mul_292: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_35, squeeze_10);  sum_35 = squeeze_10 = None
        convert_element_type_139: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_291, torch.bfloat16);  mul_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(convert_element_type_139, relu_2, convert_element_type_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_139 = convert_element_type_10 = None
        getitem_90: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_16[0]
        getitem_91: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        add_114: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(where_12, getitem_90);  where_12 = getitem_90 = None
        convert_element_type_140: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:103 in forward, code: out = self.relu(out)
        le_14: "b8[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        where_14: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_14, full_default, add_114);  le_14 = add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:97 in forward, code: out = self.bn2(out)
        convert_element_type_141: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_14, torch.float32)
        sum_36: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_141, [0, 2, 3])
        convert_element_type_8: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        sub_88: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_8, unsqueeze_286);  convert_element_type_8 = unsqueeze_286 = None
        mul_293: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_141, sub_88)
        sum_37: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3]);  mul_293 = None
        mul_294: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_36, 1.992984693877551e-05)
        unsqueeze_287: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
        unsqueeze_288: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        unsqueeze_289: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
        mul_295: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, 1.992984693877551e-05)
        mul_296: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_297: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, mul_296);  mul_295 = mul_296 = None
        unsqueeze_290: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_297, 0);  mul_297 = None
        unsqueeze_291: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
        unsqueeze_292: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
        mul_298: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_293: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
        unsqueeze_294: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
        unsqueeze_295: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
        mul_299: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_292);  sub_88 = unsqueeze_292 = None
        sub_90: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_141, mul_299);  convert_element_type_141 = mul_299 = None
        sub_91: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_289);  sub_90 = unsqueeze_289 = None
        mul_300: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_295);  sub_91 = unsqueeze_295 = None
        mul_301: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_37, squeeze_7);  sum_37 = squeeze_7 = None
        convert_element_type_143: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_300, torch.bfloat16);  mul_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:96 in forward, code: out = self.conv2(out)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_143, relu_1, convert_element_type_7, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_143 = convert_element_type_7 = None
        getitem_93: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_17[0]
        getitem_94: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_144: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_94, torch.float32);  getitem_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:94 in forward, code: out = self.relu(out)
        le_15: "b8[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        where_15: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.where.self(le_15, full_default, getitem_93);  le_15 = getitem_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:93 in forward, code: out = self.bn1(out)
        convert_element_type_145: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_15, torch.float32);  where_15 = None
        sum_38: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_145, [0, 2, 3])
        convert_element_type_5: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_1, torch.float32);  convolution_1 = None
        sub_92: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_5, unsqueeze_298);  convert_element_type_5 = unsqueeze_298 = None
        mul_302: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_145, sub_92)
        sum_39: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 2, 3]);  mul_302 = None
        mul_303: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_38, 1.992984693877551e-05)
        unsqueeze_299: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_303, 0);  mul_303 = None
        unsqueeze_300: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
        unsqueeze_301: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
        mul_304: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, 1.992984693877551e-05)
        mul_305: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_306: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
        unsqueeze_302: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_306, 0);  mul_306 = None
        unsqueeze_303: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
        unsqueeze_304: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
        mul_307: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_305: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_307, 0);  mul_307 = None
        unsqueeze_306: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
        unsqueeze_307: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        mul_308: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_304);  sub_92 = unsqueeze_304 = None
        sub_94: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_145, mul_308);  convert_element_type_145 = mul_308 = None
        sub_95: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_301);  sub_94 = unsqueeze_301 = None
        mul_309: "f32[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_307);  sub_95 = unsqueeze_307 = None
        mul_310: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_39, squeeze_4);  sum_39 = squeeze_4 = None
        convert_element_type_147: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_309, torch.bfloat16);  mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:92 in forward, code: out = self.conv1(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(convert_element_type_147, getitem_2, convert_element_type_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_147 = getitem_2 = convert_element_type_4 = None
        getitem_96: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = convolution_backward_18[0]
        getitem_97: "bf16[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        add_115: "bf16[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.aten.add.Tensor(where_14, getitem_96);  where_14 = getitem_96 = None
        convert_element_type_148: "f32[64, 64, 3, 3][576, 1, 192, 64]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        full_default_16: "f32[1024, 12544][12544, 1]cuda:0" = torch.ops.aten.full.default([1024, 12544], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_3: "bf16[1024, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(add_115, [1024, 3136]);  add_115 = None
        _low_memory_max_pool_offsets_to_indices: "i64[16, 64, 56, 56][200704, 1, 3584, 64]cuda:0" = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(getitem_3, [3, 3], [112, 112], [2, 2], [1, 1], [1, 1]);  getitem_3 = None
        view_4: "i64[1024, 3136][3136, 1]cuda:0" = torch.ops.aten.reshape.default(_low_memory_max_pool_offsets_to_indices, [1024, 3136]);  _low_memory_max_pool_offsets_to_indices = None
        convert_element_type_149: "f32[1024, 3136][3136, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        scatter_add: "f32[1024, 12544][12544, 1]cuda:0" = torch.ops.aten.scatter_add.default(full_default_16, 1, view_4, convert_element_type_149);  full_default_16 = view_4 = convert_element_type_149 = None
        view_5: "f32[16, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.aten.reshape.default(scatter_add, [16, 64, 112, 112]);  scatter_add = None
        convert_element_type_150: "bf16[16, 64, 112, 112][802816, 12544, 112, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        sub: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None
        convert_element_type_3: "bf16[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu: "bf16[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.relu.default(convert_element_type_3);  convert_element_type_3 = None
        le_16: "b8[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        where_16: "bf16[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.where.self(le_16, full_default, convert_element_type_150);  le_16 = full_default = convert_element_type_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        convert_element_type_151: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(where_16, torch.float32);  where_16 = None
        squeeze: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_308: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_309: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
        unsqueeze_310: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
        sum_40: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_151, [0, 2, 3])
        convert_element_type_2: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(convolution, torch.float32);  convolution = None
        sub_96: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_2, unsqueeze_310);  convert_element_type_2 = unsqueeze_310 = None
        mul_311: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_151, sub_96)
        sum_41: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 2, 3]);  mul_311 = None
        mul_312: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_40, 4.982461734693877e-06)
        unsqueeze_311: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_312, 0);  mul_312 = None
        unsqueeze_312: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_313: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, 4.982461734693877e-06)
        squeeze_1: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_314: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_315: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_313, mul_314);  mul_313 = mul_314 = None
        unsqueeze_314: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
        unsqueeze_315: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        mul_316: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_317: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_316, 0);  mul_316 = None
        unsqueeze_318: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_317: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_316);  sub_96 = unsqueeze_316 = None
        sub_98: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_151, mul_317);  convert_element_type_151 = mul_317 = None
        sub_99: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_313);  sub_98 = unsqueeze_313 = None
        mul_318: "f32[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_319);  sub_99 = unsqueeze_319 = None
        mul_319: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_41, squeeze_1);  sum_41 = squeeze_1 = None
        convert_element_type_153: "bf16[16, 64, 112, 112][802816, 1, 7168, 64]cuda:0" = torch.ops.prims.convert_element_type.default(mul_318, torch.bfloat16);  mul_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:268 in _forward_impl, code: x = self.conv1(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(convert_element_type_153, convert_element_type_1, convert_element_type, [0], [2, 2], [3, 3], [1, 1], False, [0, 0], 1, [False, True, False]);  convert_element_type_153 = convert_element_type_1 = convert_element_type = None
        getitem_100: "bf16[64, 3, 7, 7][147, 1, 21, 3]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_154: "f32[64, 3, 7, 7][147, 1, 21, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_100, torch.float32);  getitem_100 = None
        return (convert_element_type_154, None, None, None, None, mul_319, sum_40, convert_element_type_148, None, None, None, mul_310, sum_38, convert_element_type_144, None, None, None, mul_301, sum_36, convert_element_type_140, None, None, None, mul_292, sum_34, convert_element_type_136, None, None, None, mul_283, sum_32, convert_element_type_132, None, None, None, mul_274, sum_30, convert_element_type_128, None, None, None, mul_265, sum_28, convert_element_type_124, None, None, None, mul_256, sum_26, convert_element_type_120, None, None, None, mul_247, sum_24, convert_element_type_116, None, None, None, mul_238, sum_22, convert_element_type_112, None, None, None, mul_229, sum_20, convert_element_type_108, None, None, None, mul_220, sum_18, convert_element_type_104, None, None, None, mul_211, sum_16, convert_element_type_100, None, None, None, mul_202, sum_14, convert_element_type_96, None, None, None, mul_193, sum_12, convert_element_type_92, None, None, None, mul_184, sum_10, convert_element_type_88, None, None, None, mul_175, sum_8, convert_element_type_84, None, None, None, mul_166, sum_6, convert_element_type_80, None, None, None, mul_157, sum_4, convert_element_type_76, None, None, None, mul_148, sum_2, convert_element_type_71, convert_element_type_72)
