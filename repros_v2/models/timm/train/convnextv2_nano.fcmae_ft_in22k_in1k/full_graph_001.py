class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[80][1]cuda:0", primals_8: "f32[80][1]cuda:0", primals_13: "f32[320][1]cuda:0", primals_18: "f32[80][1]cuda:0", primals_23: "f32[320][1]cuda:0", primals_26: "f32[80][1]cuda:0", primals_32: "f32[160][1]cuda:0", primals_37: "f32[640][1]cuda:0", primals_42: "f32[160][1]cuda:0", primals_47: "f32[640][1]cuda:0", primals_50: "f32[160][1]cuda:0", primals_56: "f32[320][1]cuda:0", primals_61: "f32[1280][1]cuda:0", primals_66: "f32[320][1]cuda:0", primals_71: "f32[1280][1]cuda:0", primals_76: "f32[320][1]cuda:0", primals_81: "f32[1280][1]cuda:0", primals_86: "f32[320][1]cuda:0", primals_91: "f32[1280][1]cuda:0", primals_96: "f32[320][1]cuda:0", primals_101: "f32[1280][1]cuda:0", primals_106: "f32[320][1]cuda:0", primals_111: "f32[1280][1]cuda:0", primals_116: "f32[320][1]cuda:0", primals_121: "f32[1280][1]cuda:0", primals_126: "f32[320][1]cuda:0", primals_131: "f32[1280][1]cuda:0", primals_134: "f32[320][1]cuda:0", primals_140: "f32[640][1]cuda:0", primals_145: "f32[2560][1]cuda:0", primals_150: "f32[640][1]cuda:0", primals_155: "f32[2560][1]cuda:0", primals_158: "f32[640][1]cuda:0", convert_element_type_1: "bf16[80, 3, 4, 4][48, 1, 12, 3]cuda:0", convert_element_type_2: "bf16[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", convolution: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", getitem_1: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", rsqrt: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", convert_element_type_5: "bf16[80, 1, 7, 7][49, 1, 7, 1]cuda:0", convert_element_type_6: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", convolution_1: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", getitem_3: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", rsqrt_1: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", convert_element_type_9: "bf16[320, 80, 1, 1][80, 1, 80, 80]cuda:0", convert_element_type_10: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", convolution_2: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0", pow_2: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0", add_5: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_15: "bf16[80, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_16: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0", convert_element_type_18: "bf16[80, 1, 7, 7][49, 1, 7, 1]cuda:0", convert_element_type_19: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", convolution_4: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", getitem_5: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", rsqrt_2: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", convert_element_type_22: "bf16[320, 80, 1, 1][80, 1, 80, 80]cuda:0", convert_element_type_23: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", convolution_5: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0", pow_4: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0", add_11: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_28: "bf16[80, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_29: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0", mul_14: "f32[128, 56, 56, 80][250880, 1, 4480, 56]cuda:0", convert_element_type_31: "bf16[160, 80, 2, 2][320, 1, 160, 80]cuda:0", convert_element_type_32: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", convolution_7: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", convert_element_type_34: "bf16[160, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_8: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", getitem_9: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", rsqrt_4: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", convert_element_type_37: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0", convert_element_type_38: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", convolution_9: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0", pow_6: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0", add_19: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_43: "bf16[160, 640, 1, 1][640, 1, 640, 640]cuda:0", convert_element_type_44: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0", add_21: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", convert_element_type_46: "bf16[160, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_11: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", getitem_11: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", rsqrt_5: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", convert_element_type_49: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0", convert_element_type_50: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", convolution_12: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0", pow_8: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0", add_25: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_55: "bf16[160, 640, 1, 1][640, 1, 640, 640]cuda:0", convert_element_type_56: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0", permute_12: "bf16[128, 28, 28, 160][125440, 1, 4480, 28]cuda:0", getitem_13: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", rsqrt_6: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", convert_element_type_59: "bf16[320, 160, 2, 2][640, 1, 320, 160]cuda:0", convert_element_type_60: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", convolution_14: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convert_element_type_62: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_15: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_15: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_7: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", convert_element_type_65: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_66: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_16: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_10: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_33: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_71: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convert_element_type_72: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_35: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convert_element_type_74: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_18: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_17: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_8: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", convert_element_type_77: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_78: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_19: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_12: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_39: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_83: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convert_element_type_84: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_41: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convert_element_type_86: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_21: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_19: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_9: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", convert_element_type_89: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_90: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_22: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_14: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_45: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_95: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convert_element_type_96: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_47: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convert_element_type_98: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_24: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_21: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_10: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", convert_element_type_101: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_102: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_25: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_16: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_51: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_107: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convert_element_type_108: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_53: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convert_element_type_110: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_27: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_23: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_11: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", convert_element_type_113: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_114: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_28: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_18: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_57: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_119: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convert_element_type_120: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_59: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convert_element_type_122: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_30: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_25: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_12: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", convert_element_type_125: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_126: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_31: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_20: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_63: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_131: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convert_element_type_132: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_65: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convert_element_type_134: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_33: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_27: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_13: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", convert_element_type_137: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_138: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_34: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_22: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_69: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_143: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convert_element_type_144: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_71: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convert_element_type_146: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_36: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_29: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_14: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", convert_element_type_149: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", convert_element_type_150: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_37: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_24: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_75: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_155: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", convert_element_type_156: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", permute_30: "bf16[128, 14, 14, 320][62720, 1, 4480, 14]cuda:0", getitem_31: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_15: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", convert_element_type_159: "bf16[640, 320, 2, 2][1280, 1, 640, 320]cuda:0", convert_element_type_160: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_39: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", convert_element_type_162: "bf16[640, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_40: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", getitem_33: "f32[128, 7, 7, 1][49, 1, 7, 7]cuda:0", rsqrt_16: "f32[128, 7, 7, 1][49, 1, 7, 7]cuda:0", convert_element_type_165: "bf16[2560, 640, 1, 1][640, 1, 640, 640]cuda:0", convert_element_type_166: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", convolution_41: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0", pow_26: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", add_83: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_171: "bf16[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", convert_element_type_172: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0", add_85: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", convert_element_type_174: "bf16[640, 1, 7, 7][49, 1, 7, 1]cuda:0", convolution_43: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", getitem_35: "f32[128, 7, 7, 1][49, 1, 7, 7]cuda:0", rsqrt_17: "f32[128, 7, 7, 1][49, 1, 7, 7]cuda:0", convert_element_type_177: "bf16[2560, 640, 1, 1][640, 1, 640, 640]cuda:0", convert_element_type_178: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", convolution_44: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0", pow_28: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", add_89: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_183: "bf16[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", convert_element_type_184: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0", permute_36: "bf16[128, 1, 1, 640][640, 640, 640, 1]cuda:0", getitem_37: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", rsqrt_18: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", convert_element_type_188: "bf16[128, 640][640, 1]cuda:0", permute_39: "bf16[1000, 640][640, 1]cuda:0", div_90: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", tangents_1: "bf16[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:219 in forward, code: x = self.fc(x)
        mm: "bf16[128, 640][640, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_39);  permute_39 = None
        permute_40: "bf16[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "bf16[1000, 640][640, 1]cuda:0" = torch.ops.aten.mm.default(permute_40, convert_element_type_188);  permute_40 = convert_element_type_188 = None
        sum_15: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True, dtype = torch.float32);  tangents_1 = None
        view_29: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_15, [1000]);  sum_15 = None
        convert_element_type_196: "bf16[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        convert_element_type_197: "f32[128, 640][640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_198: "f32[1000, 640][640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_199: "f32[1000][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_196, torch.float32);  convert_element_type_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:214 in forward, code: x = self.flatten(x)
        view_30: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_197, [128, 640, 1, 1]);  convert_element_type_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_43: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 3, 1]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_95: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_43, primals_158);  primals_158 = None
        mul_96: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, 640)
        sum_16: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_95, [3], True)
        convert_element_type_185: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_36, torch.float32);  permute_36 = None
        sub_18: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_185, getitem_37);  convert_element_type_185 = getitem_37 = None
        mul_92: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
        mul_97: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, mul_92);  mul_95 = None
        sum_17: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_97, [3], True);  mul_97 = None
        mul_98: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, sum_17);  sum_17 = None
        sub_20: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_96, sum_16);  mul_96 = sum_16 = None
        sub_21: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_20, mul_98);  sub_20 = mul_98 = None
        div_14: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 640);  rsqrt_18 = None
        mul_99: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_21);  div_14 = sub_21 = None
        mul_100: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_43, mul_92);  mul_92 = None
        sum_18: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_100, [0, 1, 2]);  mul_100 = None
        sum_19: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_43, [0, 1, 2]);  permute_43 = None
        convert_element_type_200: "bf16[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_99, torch.bfloat16);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_44: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_200, [0, 3, 1, 2]);  convert_element_type_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze: "bf16[128, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_44, 3);  permute_44 = None
        squeeze_1: "bf16[128, 640][640, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
        full: "bf16[81920][1]cuda:0" = torch.ops.aten.full.default([81920], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "bf16[81920][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full, squeeze_1, [128, 640], [640, 1], 0);  full = squeeze_1 = None
        as_strided_5: "bf16[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 640, 1, 1], [640, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "bf16[128, 640, 7, 7][640, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [128, 640, 7, 7]);  as_strided_5 = None
        div_15: "bf16[128, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_20: "bf16[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_15, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(div_15, convert_element_type_184, convert_element_type_183, [640], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_184 = convert_element_type_183 = None
        getitem_38: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = convolution_backward[0]
        getitem_39: "bf16[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_201: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_38, torch.float32);  getitem_38 = None
        convert_element_type_202: "f32[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_39, torch.float32);  getitem_39 = None
        convert_element_type_203: "f32[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_20, torch.float32);  sum_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_204: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_201, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_179: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_44, torch.float32);  convolution_44 = None
        mul_88: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_179, 0.5)
        mul_89: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_179, 0.7071067811865476)
        erf_13: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
        add_88: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_90: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, add_88);  mul_88 = None
        convert_element_type_180: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_13: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.div.Tensor(pow_28, add_89)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_91: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_180, div_13)
        mul_101: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Scalar(mul_91, 1);  mul_91 = None
        mul_102: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_201, mul_101);  mul_101 = None
        view_27: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_155, [1, -1, 1, 1]);  primals_155 = None
        mul_103: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_27, 1);  view_27 = None
        mul_104: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_201, mul_103);  mul_103 = None
        sum_21: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_201, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_201 = None
        sum_22: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_102, [0, 2, 3], True, dtype = torch.float32);  mul_102 = None
        mul_105: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, convert_element_type_180)
        mul_106: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, div_13);  mul_104 = None
        convert_element_type_205: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(mul_106, torch.bfloat16);  mul_106 = None
        sum_23: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_105, [2, 3], True, dtype = torch.float32);  mul_105 = None
        add_94: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_204, convert_element_type_205);  convert_element_type_204 = convert_element_type_205 = None
        view_31: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_22, [2560]);  sum_22 = None
        view_32: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_21, [2560]);  sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_17: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.div.Tensor(div_13, add_89);  div_13 = None
        neg: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_23)
        mul_107: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.mul.Tensor(neg, div_17);  neg = div_17 = None
        div_18: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_23, add_89);  sum_23 = add_89 = None
        sum_24: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_107, [1], True, dtype = torch.float32);  mul_107 = None
        expand_2: "f32[128, 2560, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_24, [128, 2560, 1, 1]);  sum_24 = None
        div_19: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_2, 2560);  expand_2 = None
        add_95: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_18, div_19);  div_18 = div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_20: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_180, pow_28);  convert_element_type_180 = None
        eq: "b8[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.eq.Scalar(pow_28, 0);  pow_28 = None
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_20);  eq = div_20 = None
        clone_29: "f32[128, 2560, 7, 7][125440, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(where, memory_format = torch.contiguous_format);  where = None
        mul_108: "f32[128, 2560, 7, 7][125440, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_95, clone_29);  add_95 = clone_29 = None
        convert_element_type_206: "bf16[128, 2560, 7, 7][125440, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_108, torch.bfloat16);  mul_108 = None
        add_96: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(add_94, convert_element_type_206);  add_94 = convert_element_type_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_207: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(add_96, torch.float32);  add_96 = None
        mul_110: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
        mul_111: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_179, convert_element_type_179)
        mul_112: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, -0.5);  mul_111 = None
        exp: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.exp.default(mul_112);  mul_112 = None
        mul_113: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_114: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_179, mul_113);  convert_element_type_179 = mul_113 = None
        add_98: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(mul_110, mul_114);  mul_110 = mul_114 = None
        mul_115: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_207, add_98);  convert_element_type_207 = add_98 = None
        convert_element_type_209: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(mul_115, torch.bfloat16);  mul_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_25: "bf16[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_209, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(convert_element_type_209, convert_element_type_178, convert_element_type_177, [2560], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_209 = convert_element_type_178 = convert_element_type_177 = None
        getitem_41: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = convolution_backward_1[0]
        getitem_42: "bf16[2560, 640, 1, 1][640, 1, 640, 640]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None
        convert_element_type_210: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
        convert_element_type_211: "f32[2560, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_42, torch.float32);  getitem_42 = None
        convert_element_type_212: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_25, torch.float32);  sum_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_45: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_210, [0, 2, 3, 1]);  convert_element_type_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_117: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_45, primals_150);  primals_150 = None
        mul_118: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, 640)
        sum_26: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_117, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_34: "bf16[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(convolution_43, [0, 2, 3, 1]);  convolution_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_175: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_34, torch.float32);  permute_34 = None
        sub_17: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_175, getitem_35);  convert_element_type_175 = getitem_35 = None
        mul_86: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
        mul_119: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, mul_86);  mul_117 = None
        sum_27: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_119, [3], True);  mul_119 = None
        mul_120: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_86, sum_27);  sum_27 = None
        sub_23: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_118, sum_26);  mul_118 = sum_26 = None
        sub_24: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_23, mul_120);  sub_23 = mul_120 = None
        div_21: "f32[128, 7, 7, 1][49, 1, 7, 7]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 640);  rsqrt_17 = None
        mul_121: "f32[128, 7, 7, 640][31360, 640, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_21, sub_24);  div_21 = sub_24 = None
        mul_122: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_45, mul_86);  mul_86 = None
        sum_28: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_122, [0, 1, 2]);  mul_122 = None
        sum_29: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_45, [0, 1, 2]);  permute_45 = None
        convert_element_type_213: "bf16[128, 7, 7, 640][31360, 640, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_121, torch.bfloat16);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_46: "bf16[128, 640, 7, 7][31360, 1, 640, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_213, [0, 3, 1, 2]);  convert_element_type_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_30: "bf16[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_46, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(permute_46, add_85, convert_element_type_174, [640], [1, 1], [3, 3], [1, 1], False, [0, 0], 640, [True, True, False]);  permute_46 = add_85 = convert_element_type_174 = None
        getitem_44: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = convolution_backward_2[0]
        getitem_45: "bf16[640, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        add_99: "bf16[128, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(div_15, getitem_44);  div_15 = getitem_44 = None
        convert_element_type_214: "f32[640, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_45, torch.float32);  getitem_45 = None
        convert_element_type_215: "f32[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_30, torch.float32);  sum_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_31: "bf16[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_99, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_99, convert_element_type_172, convert_element_type_171, [640], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_172 = convert_element_type_171 = None
        getitem_47: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = convolution_backward_3[0]
        getitem_48: "bf16[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None
        convert_element_type_216: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
        convert_element_type_217: "f32[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_48, torch.float32);  getitem_48 = None
        convert_element_type_218: "f32[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_31, torch.float32);  sum_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_219: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_216, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_167: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_41, torch.float32);  convolution_41 = None
        mul_82: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_167, 0.5)
        mul_83: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_167, 0.7071067811865476)
        erf_12: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_82: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_84: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, add_82);  mul_82 = None
        convert_element_type_168: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16);  mul_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_12: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.div.Tensor(pow_26, add_83)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_85: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_168, div_12)
        mul_123: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Scalar(mul_85, 1);  mul_85 = None
        mul_124: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_216, mul_123);  mul_123 = None
        view_25: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_145, [1, -1, 1, 1]);  primals_145 = None
        mul_125: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_25, 1);  view_25 = None
        mul_126: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_216, mul_125);  mul_125 = None
        sum_32: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_216, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_216 = None
        sum_33: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 2, 3], True, dtype = torch.float32);  mul_124 = None
        mul_127: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, convert_element_type_168)
        mul_128: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, div_12);  mul_126 = None
        convert_element_type_220: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(mul_128, torch.bfloat16);  mul_128 = None
        sum_34: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_127, [2, 3], True, dtype = torch.float32);  mul_127 = None
        add_100: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_219, convert_element_type_220);  convert_element_type_219 = convert_element_type_220 = None
        view_33: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [2560]);  sum_33 = None
        view_34: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_32, [2560]);  sum_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_23: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.div.Tensor(div_12, add_83);  div_12 = None
        neg_1: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_34)
        mul_129: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.mul.Tensor(neg_1, div_23);  neg_1 = div_23 = None
        div_24: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_34, add_83);  sum_34 = add_83 = None
        sum_35: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_129, [1], True, dtype = torch.float32);  mul_129 = None
        expand_3: "f32[128, 2560, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_35, [128, 2560, 1, 1]);  sum_35 = None
        div_25: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_3, 2560);  expand_3 = None
        add_101: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_24, div_25);  div_24 = div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_26: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_168, pow_26);  convert_element_type_168 = None
        eq_1: "b8[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.eq.Scalar(pow_26, 0);  pow_26 = None
        where_1: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.where.self(eq_1, full_default, div_26);  eq_1 = div_26 = None
        clone_30: "f32[128, 2560, 7, 7][125440, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(where_1, memory_format = torch.contiguous_format);  where_1 = None
        mul_130: "f32[128, 2560, 7, 7][125440, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_101, clone_30);  add_101 = clone_30 = None
        convert_element_type_221: "bf16[128, 2560, 7, 7][125440, 49, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_130, torch.bfloat16);  mul_130 = None
        add_102: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(add_100, convert_element_type_221);  add_100 = convert_element_type_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_222: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.float32);  add_102 = None
        mul_132: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
        mul_133: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_167, convert_element_type_167)
        mul_134: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, -0.5);  mul_133 = None
        exp_1: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.exp.default(mul_134);  mul_134 = None
        mul_135: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_136: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_167, mul_135);  convert_element_type_167 = mul_135 = None
        add_104: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(mul_132, mul_136);  mul_132 = mul_136 = None
        mul_137: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_222, add_104);  convert_element_type_222 = add_104 = None
        convert_element_type_224: "bf16[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_36: "bf16[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_224, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(convert_element_type_224, convert_element_type_166, convert_element_type_165, [2560], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_224 = convert_element_type_166 = convert_element_type_165 = None
        getitem_50: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = convolution_backward_4[0]
        getitem_51: "bf16[2560, 640, 1, 1][640, 1, 640, 640]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        convert_element_type_225: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_50, torch.float32);  getitem_50 = None
        convert_element_type_226: "f32[2560, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
        convert_element_type_227: "f32[2560][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_36, torch.float32);  sum_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_47: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_225, [0, 2, 3, 1]);  convert_element_type_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_139: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_47, primals_140);  primals_140 = None
        mul_140: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, 640)
        sum_37: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_139, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_32: "bf16[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(convolution_40, [0, 2, 3, 1]);  convolution_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_163: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_32, torch.float32);  permute_32 = None
        sub_16: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_163, getitem_33);  convert_element_type_163 = getitem_33 = None
        mul_80: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        mul_141: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, mul_80);  mul_139 = None
        sum_38: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_141, [3], True);  mul_141 = None
        mul_142: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, sum_38);  sum_38 = None
        sub_26: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_140, sum_37);  mul_140 = sum_37 = None
        sub_27: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_26, mul_142);  sub_26 = mul_142 = None
        div_27: "f32[128, 7, 7, 1][49, 1, 7, 7]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 640);  rsqrt_16 = None
        mul_143: "f32[128, 7, 7, 640][31360, 640, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_27, sub_27);  div_27 = sub_27 = None
        mul_144: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_47, mul_80);  mul_80 = None
        sum_39: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_144, [0, 1, 2]);  mul_144 = None
        sum_40: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_47, [0, 1, 2]);  permute_47 = None
        convert_element_type_228: "bf16[128, 7, 7, 640][31360, 640, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_48: "bf16[128, 640, 7, 7][31360, 1, 640, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_228, [0, 3, 1, 2]);  convert_element_type_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_41: "bf16[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_48, [0, 2, 3])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(permute_48, convolution_39, convert_element_type_162, [640], [1, 1], [3, 3], [1, 1], False, [0, 0], 640, [True, True, False]);  permute_48 = convolution_39 = convert_element_type_162 = None
        getitem_53: "bf16[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = convolution_backward_5[0]
        getitem_54: "bf16[640, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        add_105: "bf16[128, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_99, getitem_53);  add_99 = getitem_53 = None
        convert_element_type_229: "f32[640, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_54, torch.float32);  getitem_54 = None
        convert_element_type_230: "f32[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_41, torch.float32);  sum_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        sum_42: "bf16[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_105, [0, 2, 3])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(add_105, convert_element_type_160, convert_element_type_159, [640], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_105 = convert_element_type_160 = convert_element_type_159 = None
        getitem_56: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_6[0]
        getitem_57: "bf16[640, 320, 2, 2][1280, 1, 640, 320]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None
        convert_element_type_231: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_56, torch.float32);  getitem_56 = None
        convert_element_type_232: "f32[640, 320, 2, 2][1280, 1, 640, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
        convert_element_type_233: "f32[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_42, torch.float32);  sum_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_49: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_231, [0, 2, 3, 1]);  convert_element_type_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_146: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_49, primals_134);  primals_134 = None
        mul_147: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, 320)
        sum_43: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_146, [3], True)
        convert_element_type_157: "f32[128, 14, 14, 320][62720, 1, 4480, 14]cuda:0" = torch.ops.prims.convert_element_type.default(permute_30, torch.float32);  permute_30 = None
        sub_15: "f32[128, 14, 14, 320][62720, 1, 4480, 14]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_157, getitem_31);  convert_element_type_157 = getitem_31 = None
        mul_78: "f32[128, 14, 14, 320][62720, 1, 4480, 14]cuda:0" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
        mul_148: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, mul_78);  mul_146 = None
        sum_44: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_148, [3], True);  mul_148 = None
        mul_149: "f32[128, 14, 14, 320][62720, 1, 4480, 14]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, sum_44);  sum_44 = None
        sub_29: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_147, sum_43);  mul_147 = sum_43 = None
        sub_30: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_29, mul_149);  sub_29 = mul_149 = None
        div_28: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 320);  rsqrt_15 = None
        mul_150: "f32[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_28, sub_30);  div_28 = sub_30 = None
        mul_151: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_49, mul_78);  mul_78 = None
        sum_45: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_151, [0, 1, 2]);  mul_151 = None
        sum_46: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_49, [0, 1, 2]);  permute_49 = None
        convert_element_type_234: "bf16[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_50: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_234, [0, 3, 1, 2]);  convert_element_type_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_47: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_50, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(permute_50, convert_element_type_156, convert_element_type_155, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_156 = convert_element_type_155 = None
        getitem_59: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_7[0]
        getitem_60: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None
        convert_element_type_235: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_59, torch.float32);  getitem_59 = None
        convert_element_type_236: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_60, torch.float32);  getitem_60 = None
        convert_element_type_237: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_47, torch.float32);  sum_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_238: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_235, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_151: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_37, torch.float32);  convolution_37 = None
        mul_74: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_151, 0.5)
        mul_75: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_151, 0.7071067811865476)
        erf_11: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_74: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_76: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, add_74);  mul_74 = None
        convert_element_type_152: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_76, torch.bfloat16);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_11: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_24, add_75)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_77: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_152, div_11)
        mul_152: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_77, 1);  mul_77 = None
        mul_153: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, mul_152);  mul_152 = None
        view_23: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_131, [1, -1, 1, 1]);  primals_131 = None
        mul_154: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_23, 1);  view_23 = None
        mul_155: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_235, mul_154);  mul_154 = None
        sum_48: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_235, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_235 = None
        sum_49: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 2, 3], True, dtype = torch.float32);  mul_153 = None
        mul_156: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, convert_element_type_152)
        mul_157: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, div_11);  mul_155 = None
        convert_element_type_239: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_157, torch.bfloat16);  mul_157 = None
        sum_50: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_156, [2, 3], True, dtype = torch.float32);  mul_156 = None
        add_106: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_238, convert_element_type_239);  convert_element_type_238 = convert_element_type_239 = None
        view_35: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [1280]);  sum_49 = None
        view_36: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [1280]);  sum_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_30: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_11, add_75);  div_11 = None
        neg_2: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_50)
        mul_158: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_2, div_30);  neg_2 = div_30 = None
        div_31: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_50, add_75);  sum_50 = add_75 = None
        sum_51: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_158, [1], True, dtype = torch.float32);  mul_158 = None
        expand_4: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_51, [128, 1280, 1, 1]);  sum_51 = None
        div_32: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_4, 1280);  expand_4 = None
        add_107: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_31, div_32);  div_31 = div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_33: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_152, pow_24);  convert_element_type_152 = None
        eq_2: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_24, 0);  pow_24 = None
        where_2: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_2, full_default, div_33);  eq_2 = div_33 = None
        clone_31: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_2, memory_format = torch.contiguous_format);  where_2 = None
        mul_159: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_107, clone_31);  add_107 = clone_31 = None
        convert_element_type_240: "bf16[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_159, torch.bfloat16);  mul_159 = None
        add_108: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_106, convert_element_type_240);  add_106 = convert_element_type_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_241: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_108, torch.float32);  add_108 = None
        mul_161: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_162: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_151, convert_element_type_151)
        mul_163: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, -0.5);  mul_162 = None
        exp_2: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_163);  mul_163 = None
        mul_164: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_165: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_151, mul_164);  convert_element_type_151 = mul_164 = None
        add_110: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_161, mul_165);  mul_161 = mul_165 = None
        mul_166: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_241, add_110);  convert_element_type_241 = add_110 = None
        convert_element_type_243: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_166, torch.bfloat16);  mul_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_52: "bf16[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_243, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(convert_element_type_243, convert_element_type_150, convert_element_type_149, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_243 = convert_element_type_150 = convert_element_type_149 = None
        getitem_62: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_8[0]
        getitem_63: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None
        convert_element_type_244: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_62, torch.float32);  getitem_62 = None
        convert_element_type_245: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_63, torch.float32);  getitem_63 = None
        convert_element_type_246: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_52, torch.float32);  sum_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_51: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_244, [0, 2, 3, 1]);  convert_element_type_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_168: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_51, primals_126);  primals_126 = None
        mul_169: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, 320)
        sum_53: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_168, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_28: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_36, [0, 2, 3, 1]);  convolution_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_147: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_28, torch.float32);  permute_28 = None
        sub_14: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_147, getitem_29);  convert_element_type_147 = getitem_29 = None
        mul_72: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        mul_170: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, mul_72);  mul_168 = None
        sum_54: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_170, [3], True);  mul_170 = None
        mul_171: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_72, sum_54);  sum_54 = None
        sub_32: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_169, sum_53);  mul_169 = sum_53 = None
        sub_33: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_32, mul_171);  sub_32 = mul_171 = None
        div_34: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 320);  rsqrt_14 = None
        mul_172: "f32[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_34, sub_33);  div_34 = sub_33 = None
        mul_173: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_51, mul_72);  mul_72 = None
        sum_55: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_173, [0, 1, 2]);  mul_173 = None
        sum_56: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_51, [0, 1, 2]);  permute_51 = None
        convert_element_type_247: "bf16[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_172, torch.bfloat16);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_52: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_247, [0, 3, 1, 2]);  convert_element_type_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_57: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_52, [0, 2, 3])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(permute_52, add_71, convert_element_type_146, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_52 = add_71 = convert_element_type_146 = None
        getitem_65: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_9[0]
        getitem_66: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        add_111: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(permute_50, getitem_65);  permute_50 = getitem_65 = None
        convert_element_type_248: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_66, torch.float32);  getitem_66 = None
        convert_element_type_249: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_57, torch.float32);  sum_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_58: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_111, [0, 2, 3])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(add_111, convert_element_type_144, convert_element_type_143, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_144 = convert_element_type_143 = None
        getitem_68: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_10[0]
        getitem_69: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None
        convert_element_type_250: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_68, torch.float32);  getitem_68 = None
        convert_element_type_251: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_69, torch.float32);  getitem_69 = None
        convert_element_type_252: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_58, torch.float32);  sum_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_253: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_250, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_139: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_34, torch.float32);  convolution_34 = None
        mul_68: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_69: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476)
        erf_10: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_68: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_70: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_68);  mul_68 = None
        convert_element_type_140: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_70, torch.bfloat16);  mul_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_10: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_22, add_69)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_71: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_140, div_10)
        mul_174: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_71, 1);  mul_71 = None
        mul_175: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_250, mul_174);  mul_174 = None
        view_21: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_121, [1, -1, 1, 1]);  primals_121 = None
        mul_176: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_21, 1);  view_21 = None
        mul_177: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_250, mul_176);  mul_176 = None
        sum_59: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_250, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_250 = None
        sum_60: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 2, 3], True, dtype = torch.float32);  mul_175 = None
        mul_178: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, convert_element_type_140)
        mul_179: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, div_10);  mul_177 = None
        convert_element_type_254: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_179, torch.bfloat16);  mul_179 = None
        sum_61: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_178, [2, 3], True, dtype = torch.float32);  mul_178 = None
        add_112: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_253, convert_element_type_254);  convert_element_type_253 = convert_element_type_254 = None
        view_37: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_60, [1280]);  sum_60 = None
        view_38: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_59, [1280]);  sum_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_36: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_10, add_69);  div_10 = None
        neg_3: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_61)
        mul_180: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_3, div_36);  neg_3 = div_36 = None
        div_37: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_61, add_69);  sum_61 = add_69 = None
        sum_62: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_180, [1], True, dtype = torch.float32);  mul_180 = None
        expand_5: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_62, [128, 1280, 1, 1]);  sum_62 = None
        div_38: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_5, 1280);  expand_5 = None
        add_113: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_37, div_38);  div_37 = div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_39: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_140, pow_22);  convert_element_type_140 = None
        eq_3: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_22, 0);  pow_22 = None
        where_3: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_3, full_default, div_39);  eq_3 = div_39 = None
        clone_32: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_3, memory_format = torch.contiguous_format);  where_3 = None
        mul_181: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, clone_32);  add_113 = clone_32 = None
        convert_element_type_255: "bf16[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        add_114: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_112, convert_element_type_255);  add_112 = convert_element_type_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_256: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_114, torch.float32);  add_114 = None
        mul_183: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_68, 0.5);  add_68 = None
        mul_184: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, convert_element_type_139)
        mul_185: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, -0.5);  mul_184 = None
        exp_3: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_185);  mul_185 = None
        mul_186: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_187: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_139, mul_186);  convert_element_type_139 = mul_186 = None
        add_116: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_183, mul_187);  mul_183 = mul_187 = None
        mul_188: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_256, add_116);  convert_element_type_256 = add_116 = None
        convert_element_type_258: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_188, torch.bfloat16);  mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_63: "bf16[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_258, [0, 2, 3])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(convert_element_type_258, convert_element_type_138, convert_element_type_137, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_258 = convert_element_type_138 = convert_element_type_137 = None
        getitem_71: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_11[0]
        getitem_72: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        convert_element_type_259: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
        convert_element_type_260: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_72, torch.float32);  getitem_72 = None
        convert_element_type_261: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_63, torch.float32);  sum_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_53: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_259, [0, 2, 3, 1]);  convert_element_type_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_190: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_53, primals_116);  primals_116 = None
        mul_191: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, 320)
        sum_64: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_190, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_26: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1]);  convolution_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_135: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_26, torch.float32);  permute_26 = None
        sub_13: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_135, getitem_27);  convert_element_type_135 = getitem_27 = None
        mul_66: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        mul_192: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, mul_66);  mul_190 = None
        sum_65: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_192, [3], True);  mul_192 = None
        mul_193: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, sum_65);  sum_65 = None
        sub_35: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_191, sum_64);  mul_191 = sum_64 = None
        sub_36: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_35, mul_193);  sub_35 = mul_193 = None
        div_40: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 320);  rsqrt_13 = None
        mul_194: "f32[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_40, sub_36);  div_40 = sub_36 = None
        mul_195: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_53, mul_66);  mul_66 = None
        sum_66: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 1, 2]);  mul_195 = None
        sum_67: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_53, [0, 1, 2]);  permute_53 = None
        convert_element_type_262: "bf16[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_194, torch.bfloat16);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_54: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_262, [0, 3, 1, 2]);  convert_element_type_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_68: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_54, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(permute_54, add_65, convert_element_type_134, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_54 = add_65 = convert_element_type_134 = None
        getitem_74: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_12[0]
        getitem_75: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        add_117: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_111, getitem_74);  add_111 = getitem_74 = None
        convert_element_type_263: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_75, torch.float32);  getitem_75 = None
        convert_element_type_264: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_68, torch.float32);  sum_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_69: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_117, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(add_117, convert_element_type_132, convert_element_type_131, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_132 = convert_element_type_131 = None
        getitem_77: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_13[0]
        getitem_78: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None
        convert_element_type_265: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
        convert_element_type_266: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_78, torch.float32);  getitem_78 = None
        convert_element_type_267: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_69, torch.float32);  sum_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_268: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_265, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_127: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_31, torch.float32);  convolution_31 = None
        mul_62: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_127, 0.5)
        mul_63: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_127, 0.7071067811865476)
        erf_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_62: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_64: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, add_62);  mul_62 = None
        convert_element_type_128: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_9: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_20, add_63)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_65: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_128, div_9)
        mul_196: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_65, 1);  mul_65 = None
        mul_197: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, mul_196);  mul_196 = None
        view_19: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_111, [1, -1, 1, 1]);  primals_111 = None
        mul_198: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_19, 1);  view_19 = None
        mul_199: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_265, mul_198);  mul_198 = None
        sum_70: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_265, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_265 = None
        sum_71: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 2, 3], True, dtype = torch.float32);  mul_197 = None
        mul_200: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, convert_element_type_128)
        mul_201: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, div_9);  mul_199 = None
        convert_element_type_269: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16);  mul_201 = None
        sum_72: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_200, [2, 3], True, dtype = torch.float32);  mul_200 = None
        add_118: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_268, convert_element_type_269);  convert_element_type_268 = convert_element_type_269 = None
        view_39: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_71, [1280]);  sum_71 = None
        view_40: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [1280]);  sum_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_42: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_9, add_63);  div_9 = None
        neg_4: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_72)
        mul_202: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_4, div_42);  neg_4 = div_42 = None
        div_43: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_72, add_63);  sum_72 = add_63 = None
        sum_73: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_202, [1], True, dtype = torch.float32);  mul_202 = None
        expand_6: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_73, [128, 1280, 1, 1]);  sum_73 = None
        div_44: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_6, 1280);  expand_6 = None
        add_119: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_43, div_44);  div_43 = div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_45: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_128, pow_20);  convert_element_type_128 = None
        eq_4: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_20, 0);  pow_20 = None
        where_4: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_4, full_default, div_45);  eq_4 = div_45 = None
        clone_33: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_4, memory_format = torch.contiguous_format);  where_4 = None
        mul_203: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_119, clone_33);  add_119 = clone_33 = None
        convert_element_type_270: "bf16[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_203, torch.bfloat16);  mul_203 = None
        add_120: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_118, convert_element_type_270);  add_118 = convert_element_type_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_271: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.float32);  add_120 = None
        mul_205: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_206: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_127, convert_element_type_127)
        mul_207: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, -0.5);  mul_206 = None
        exp_4: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_207);  mul_207 = None
        mul_208: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_209: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_127, mul_208);  convert_element_type_127 = mul_208 = None
        add_122: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_205, mul_209);  mul_205 = mul_209 = None
        mul_210: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_271, add_122);  convert_element_type_271 = add_122 = None
        convert_element_type_273: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_210, torch.bfloat16);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_74: "bf16[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_273, [0, 2, 3])
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(convert_element_type_273, convert_element_type_126, convert_element_type_125, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_273 = convert_element_type_126 = convert_element_type_125 = None
        getitem_80: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_14[0]
        getitem_81: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None
        convert_element_type_274: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_80, torch.float32);  getitem_80 = None
        convert_element_type_275: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
        convert_element_type_276: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_74, torch.float32);  sum_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_55: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_274, [0, 2, 3, 1]);  convert_element_type_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_212: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_55, primals_106);  primals_106 = None
        mul_213: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_212, 320)
        sum_75: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_212, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_24: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_30, [0, 2, 3, 1]);  convolution_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_123: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_24, torch.float32);  permute_24 = None
        sub_12: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_123, getitem_25);  convert_element_type_123 = getitem_25 = None
        mul_60: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
        mul_214: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_212, mul_60);  mul_212 = None
        sum_76: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_214, [3], True);  mul_214 = None
        mul_215: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_60, sum_76);  sum_76 = None
        sub_38: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_213, sum_75);  mul_213 = sum_75 = None
        sub_39: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_38, mul_215);  sub_38 = mul_215 = None
        div_46: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 320);  rsqrt_12 = None
        mul_216: "f32[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_46, sub_39);  div_46 = sub_39 = None
        mul_217: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_55, mul_60);  mul_60 = None
        sum_77: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_217, [0, 1, 2]);  mul_217 = None
        sum_78: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_55, [0, 1, 2]);  permute_55 = None
        convert_element_type_277: "bf16[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_216, torch.bfloat16);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_56: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_277, [0, 3, 1, 2]);  convert_element_type_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_79: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_56, [0, 2, 3])
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(permute_56, add_59, convert_element_type_122, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_56 = add_59 = convert_element_type_122 = None
        getitem_83: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_15[0]
        getitem_84: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        add_123: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_117, getitem_83);  add_117 = getitem_83 = None
        convert_element_type_278: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_84, torch.float32);  getitem_84 = None
        convert_element_type_279: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_79, torch.float32);  sum_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_80: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_123, [0, 2, 3])
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(add_123, convert_element_type_120, convert_element_type_119, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_120 = convert_element_type_119 = None
        getitem_86: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_16[0]
        getitem_87: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None
        convert_element_type_280: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_86, torch.float32);  getitem_86 = None
        convert_element_type_281: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
        convert_element_type_282: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_80, torch.float32);  sum_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_283: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_280, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_115: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_28, torch.float32);  convolution_28 = None
        mul_56: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.5)
        mul_57: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, 0.7071067811865476)
        erf_8: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
        add_56: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_58: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, add_56);  mul_56 = None
        convert_element_type_116: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_8: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_18, add_57)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_59: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, div_8)
        mul_218: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_59, 1);  mul_59 = None
        mul_219: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, mul_218);  mul_218 = None
        view_17: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_101, [1, -1, 1, 1]);  primals_101 = None
        mul_220: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_17, 1);  view_17 = None
        mul_221: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, mul_220);  mul_220 = None
        sum_81: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_280, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_280 = None
        sum_82: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_219, [0, 2, 3], True, dtype = torch.float32);  mul_219 = None
        mul_222: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, convert_element_type_116)
        mul_223: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, div_8);  mul_221 = None
        convert_element_type_284: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_223, torch.bfloat16);  mul_223 = None
        sum_83: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_222, [2, 3], True, dtype = torch.float32);  mul_222 = None
        add_124: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_283, convert_element_type_284);  convert_element_type_283 = convert_element_type_284 = None
        view_41: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [1280]);  sum_82 = None
        view_42: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [1280]);  sum_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_48: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_8, add_57);  div_8 = None
        neg_5: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_83)
        mul_224: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_5, div_48);  neg_5 = div_48 = None
        div_49: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_83, add_57);  sum_83 = add_57 = None
        sum_84: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_224, [1], True, dtype = torch.float32);  mul_224 = None
        expand_7: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_84, [128, 1280, 1, 1]);  sum_84 = None
        div_50: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_7, 1280);  expand_7 = None
        add_125: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_49, div_50);  div_49 = div_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_51: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_116, pow_18);  convert_element_type_116 = None
        eq_5: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_18, 0);  pow_18 = None
        where_5: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_5, full_default, div_51);  eq_5 = div_51 = None
        clone_34: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_5, memory_format = torch.contiguous_format);  where_5 = None
        mul_225: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, clone_34);  add_125 = clone_34 = None
        convert_element_type_285: "bf16[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_225, torch.bfloat16);  mul_225 = None
        add_126: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_124, convert_element_type_285);  add_124 = convert_element_type_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_286: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_126, torch.float32);  add_126 = None
        mul_227: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_228: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, convert_element_type_115)
        mul_229: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_228, -0.5);  mul_228 = None
        exp_5: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_229);  mul_229 = None
        mul_230: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_231: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_115, mul_230);  convert_element_type_115 = mul_230 = None
        add_128: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_227, mul_231);  mul_227 = mul_231 = None
        mul_232: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_286, add_128);  convert_element_type_286 = add_128 = None
        convert_element_type_288: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_232, torch.bfloat16);  mul_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_85: "bf16[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_288, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(convert_element_type_288, convert_element_type_114, convert_element_type_113, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_288 = convert_element_type_114 = convert_element_type_113 = None
        getitem_89: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_17[0]
        getitem_90: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None
        convert_element_type_289: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_89, torch.float32);  getitem_89 = None
        convert_element_type_290: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_90, torch.float32);  getitem_90 = None
        convert_element_type_291: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_85, torch.float32);  sum_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_57: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_289, [0, 2, 3, 1]);  convert_element_type_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_234: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_57, primals_96);  primals_96 = None
        mul_235: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, 320)
        sum_86: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_234, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_22: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1]);  convolution_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_111: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_22, torch.float32);  permute_22 = None
        sub_11: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_111, getitem_23);  convert_element_type_111 = getitem_23 = None
        mul_54: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        mul_236: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, mul_54);  mul_234 = None
        sum_87: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_236, [3], True);  mul_236 = None
        mul_237: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, sum_87);  sum_87 = None
        sub_41: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_235, sum_86);  mul_235 = sum_86 = None
        sub_42: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_41, mul_237);  sub_41 = mul_237 = None
        div_52: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 320);  rsqrt_11 = None
        mul_238: "f32[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, sub_42);  div_52 = sub_42 = None
        mul_239: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_57, mul_54);  mul_54 = None
        sum_88: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1, 2]);  mul_239 = None
        sum_89: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_57, [0, 1, 2]);  permute_57 = None
        convert_element_type_292: "bf16[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_238, torch.bfloat16);  mul_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_58: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_292, [0, 3, 1, 2]);  convert_element_type_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_90: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_58, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(permute_58, add_53, convert_element_type_110, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_58 = add_53 = convert_element_type_110 = None
        getitem_92: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_18[0]
        getitem_93: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        add_129: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_123, getitem_92);  add_123 = getitem_92 = None
        convert_element_type_293: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_93, torch.float32);  getitem_93 = None
        convert_element_type_294: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_90, torch.float32);  sum_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_91: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_129, [0, 2, 3])
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(add_129, convert_element_type_108, convert_element_type_107, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_108 = convert_element_type_107 = None
        getitem_95: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_19[0]
        getitem_96: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None
        convert_element_type_295: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_95, torch.float32);  getitem_95 = None
        convert_element_type_296: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_96, torch.float32);  getitem_96 = None
        convert_element_type_297: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_91, torch.float32);  sum_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_298: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_295, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_103: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_25, torch.float32);  convolution_25 = None
        mul_50: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_103, 0.5)
        mul_51: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_103, 0.7071067811865476)
        erf_7: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_50: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_52: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_50);  mul_50 = None
        convert_element_type_104: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_52, torch.bfloat16);  mul_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_7: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_16, add_51)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_53: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_104, div_7)
        mul_240: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_53, 1);  mul_53 = None
        mul_241: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_295, mul_240);  mul_240 = None
        view_15: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_91, [1, -1, 1, 1]);  primals_91 = None
        mul_242: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_15, 1);  view_15 = None
        mul_243: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_295, mul_242);  mul_242 = None
        sum_92: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_295, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_295 = None
        sum_93: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3], True, dtype = torch.float32);  mul_241 = None
        mul_244: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, convert_element_type_104)
        mul_245: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, div_7);  mul_243 = None
        convert_element_type_299: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_245, torch.bfloat16);  mul_245 = None
        sum_94: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_244, [2, 3], True, dtype = torch.float32);  mul_244 = None
        add_130: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_298, convert_element_type_299);  convert_element_type_298 = convert_element_type_299 = None
        view_43: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [1280]);  sum_93 = None
        view_44: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_92, [1280]);  sum_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_54: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_7, add_51);  div_7 = None
        neg_6: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_94)
        mul_246: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_6, div_54);  neg_6 = div_54 = None
        div_55: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_94, add_51);  sum_94 = add_51 = None
        sum_95: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_246, [1], True, dtype = torch.float32);  mul_246 = None
        expand_8: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_95, [128, 1280, 1, 1]);  sum_95 = None
        div_56: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_8, 1280);  expand_8 = None
        add_131: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_55, div_56);  div_55 = div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_57: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_104, pow_16);  convert_element_type_104 = None
        eq_6: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_16, 0);  pow_16 = None
        where_6: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_6, full_default, div_57);  eq_6 = div_57 = None
        clone_35: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_6, memory_format = torch.contiguous_format);  where_6 = None
        mul_247: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_131, clone_35);  add_131 = clone_35 = None
        convert_element_type_300: "bf16[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_247, torch.bfloat16);  mul_247 = None
        add_132: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_130, convert_element_type_300);  add_130 = convert_element_type_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_301: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_132, torch.float32);  add_132 = None
        mul_249: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
        mul_250: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_103, convert_element_type_103)
        mul_251: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, -0.5);  mul_250 = None
        exp_6: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_251);  mul_251 = None
        mul_252: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_253: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_103, mul_252);  convert_element_type_103 = mul_252 = None
        add_134: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_249, mul_253);  mul_249 = mul_253 = None
        mul_254: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_301, add_134);  convert_element_type_301 = add_134 = None
        convert_element_type_303: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_254, torch.bfloat16);  mul_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_96: "bf16[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_303, [0, 2, 3])
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(convert_element_type_303, convert_element_type_102, convert_element_type_101, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_303 = convert_element_type_102 = convert_element_type_101 = None
        getitem_98: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_20[0]
        getitem_99: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None
        convert_element_type_304: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_98, torch.float32);  getitem_98 = None
        convert_element_type_305: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_99, torch.float32);  getitem_99 = None
        convert_element_type_306: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_96, torch.float32);  sum_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_59: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_304, [0, 2, 3, 1]);  convert_element_type_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_256: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_59, primals_86);  primals_86 = None
        mul_257: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, 320)
        sum_97: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_256, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_20: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1]);  convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_99: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_20, torch.float32);  permute_20 = None
        sub_10: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_99, getitem_21);  convert_element_type_99 = getitem_21 = None
        mul_48: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        mul_258: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, mul_48);  mul_256 = None
        sum_98: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_258, [3], True);  mul_258 = None
        mul_259: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, sum_98);  sum_98 = None
        sub_44: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_257, sum_97);  mul_257 = sum_97 = None
        sub_45: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_44, mul_259);  sub_44 = mul_259 = None
        div_58: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 320);  rsqrt_10 = None
        mul_260: "f32[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_58, sub_45);  div_58 = sub_45 = None
        mul_261: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_59, mul_48);  mul_48 = None
        sum_99: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 1, 2]);  mul_261 = None
        sum_100: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_59, [0, 1, 2]);  permute_59 = None
        convert_element_type_307: "bf16[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_260, torch.bfloat16);  mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_60: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_307, [0, 3, 1, 2]);  convert_element_type_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_101: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_60, [0, 2, 3])
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(permute_60, add_47, convert_element_type_98, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_60 = add_47 = convert_element_type_98 = None
        getitem_101: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_21[0]
        getitem_102: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        add_135: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_129, getitem_101);  add_129 = getitem_101 = None
        convert_element_type_308: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_102, torch.float32);  getitem_102 = None
        convert_element_type_309: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_101, torch.float32);  sum_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_102: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_135, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_135, convert_element_type_96, convert_element_type_95, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_96 = convert_element_type_95 = None
        getitem_104: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_22[0]
        getitem_105: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None
        convert_element_type_310: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_104, torch.float32);  getitem_104 = None
        convert_element_type_311: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_105, torch.float32);  getitem_105 = None
        convert_element_type_312: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_102, torch.float32);  sum_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_313: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_310, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_91: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_22, torch.float32);  convolution_22 = None
        mul_44: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.5)
        mul_45: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, 0.7071067811865476)
        erf_6: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_44: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_46: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, add_44);  mul_44 = None
        convert_element_type_92: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_46, torch.bfloat16);  mul_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_6: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_14, add_45)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_47: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_92, div_6)
        mul_262: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_47, 1);  mul_47 = None
        mul_263: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_310, mul_262);  mul_262 = None
        view_13: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_81, [1, -1, 1, 1]);  primals_81 = None
        mul_264: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_13, 1);  view_13 = None
        mul_265: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_310, mul_264);  mul_264 = None
        sum_103: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_310, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_310 = None
        sum_104: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 2, 3], True, dtype = torch.float32);  mul_263 = None
        mul_266: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_265, convert_element_type_92)
        mul_267: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_265, div_6);  mul_265 = None
        convert_element_type_314: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_267, torch.bfloat16);  mul_267 = None
        sum_105: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [2, 3], True, dtype = torch.float32);  mul_266 = None
        add_136: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_313, convert_element_type_314);  convert_element_type_313 = convert_element_type_314 = None
        view_45: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_104, [1280]);  sum_104 = None
        view_46: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_103, [1280]);  sum_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_60: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_6, add_45);  div_6 = None
        neg_7: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_105)
        mul_268: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_7, div_60);  neg_7 = div_60 = None
        div_61: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_105, add_45);  sum_105 = add_45 = None
        sum_106: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_268, [1], True, dtype = torch.float32);  mul_268 = None
        expand_9: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_106, [128, 1280, 1, 1]);  sum_106 = None
        div_62: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_9, 1280);  expand_9 = None
        add_137: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_61, div_62);  div_61 = div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_63: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_92, pow_14);  convert_element_type_92 = None
        eq_7: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_14, 0);  pow_14 = None
        where_7: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_7, full_default, div_63);  eq_7 = div_63 = None
        clone_36: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_7, memory_format = torch.contiguous_format);  where_7 = None
        mul_269: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_137, clone_36);  add_137 = clone_36 = None
        convert_element_type_315: "bf16[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_269, torch.bfloat16);  mul_269 = None
        add_138: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_136, convert_element_type_315);  add_136 = convert_element_type_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_316: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_138, torch.float32);  add_138 = None
        mul_271: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
        mul_272: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, convert_element_type_91)
        mul_273: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_272, -0.5);  mul_272 = None
        exp_7: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_273);  mul_273 = None
        mul_274: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_275: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_91, mul_274);  convert_element_type_91 = mul_274 = None
        add_140: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_271, mul_275);  mul_271 = mul_275 = None
        mul_276: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_316, add_140);  convert_element_type_316 = add_140 = None
        convert_element_type_318: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_276, torch.bfloat16);  mul_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_107: "bf16[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_318, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(convert_element_type_318, convert_element_type_90, convert_element_type_89, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_318 = convert_element_type_90 = convert_element_type_89 = None
        getitem_107: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_23[0]
        getitem_108: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None
        convert_element_type_319: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
        convert_element_type_320: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_108, torch.float32);  getitem_108 = None
        convert_element_type_321: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_107, torch.float32);  sum_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_61: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_319, [0, 2, 3, 1]);  convert_element_type_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_278: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_61, primals_76);  primals_76 = None
        mul_279: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, 320)
        sum_108: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_278, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_18: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_87: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_18, torch.float32);  permute_18 = None
        sub_9: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_87, getitem_19);  convert_element_type_87 = getitem_19 = None
        mul_42: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
        mul_280: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, mul_42);  mul_278 = None
        sum_109: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_280, [3], True);  mul_280 = None
        mul_281: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, sum_109);  sum_109 = None
        sub_47: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_279, sum_108);  mul_279 = sum_108 = None
        sub_48: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_47, mul_281);  sub_47 = mul_281 = None
        div_64: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 320);  rsqrt_9 = None
        mul_282: "f32[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_64, sub_48);  div_64 = sub_48 = None
        mul_283: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_61, mul_42);  mul_42 = None
        sum_110: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1, 2]);  mul_283 = None
        sum_111: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_61, [0, 1, 2]);  permute_61 = None
        convert_element_type_322: "bf16[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_282, torch.bfloat16);  mul_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_62: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_322, [0, 3, 1, 2]);  convert_element_type_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_112: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_62, [0, 2, 3])
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(permute_62, add_41, convert_element_type_86, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_62 = add_41 = convert_element_type_86 = None
        getitem_110: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_24[0]
        getitem_111: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        add_141: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_135, getitem_110);  add_135 = getitem_110 = None
        convert_element_type_323: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
        convert_element_type_324: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_112, torch.float32);  sum_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_113: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_141, [0, 2, 3])
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(add_141, convert_element_type_84, convert_element_type_83, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_84 = convert_element_type_83 = None
        getitem_113: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_25[0]
        getitem_114: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None
        convert_element_type_325: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32);  getitem_113 = None
        convert_element_type_326: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_114, torch.float32);  getitem_114 = None
        convert_element_type_327: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_113, torch.float32);  sum_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_328: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_325, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_79: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_19, torch.float32);  convolution_19 = None
        mul_38: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, 0.5)
        mul_39: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, 0.7071067811865476)
        erf_5: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_38: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_40: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, add_38);  mul_38 = None
        convert_element_type_80: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_5: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_12, add_39)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_41: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_80, div_5)
        mul_284: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_41, 1);  mul_41 = None
        mul_285: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_325, mul_284);  mul_284 = None
        view_11: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_71, [1, -1, 1, 1]);  primals_71 = None
        mul_286: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_11, 1);  view_11 = None
        mul_287: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_325, mul_286);  mul_286 = None
        sum_114: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_325, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_325 = None
        sum_115: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 2, 3], True, dtype = torch.float32);  mul_285 = None
        mul_288: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, convert_element_type_80)
        mul_289: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, div_5);  mul_287 = None
        convert_element_type_329: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_289, torch.bfloat16);  mul_289 = None
        sum_116: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_288, [2, 3], True, dtype = torch.float32);  mul_288 = None
        add_142: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_328, convert_element_type_329);  convert_element_type_328 = convert_element_type_329 = None
        view_47: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_115, [1280]);  sum_115 = None
        view_48: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_114, [1280]);  sum_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_66: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_5, add_39);  div_5 = None
        neg_8: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_116)
        mul_290: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_8, div_66);  neg_8 = div_66 = None
        div_67: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_116, add_39);  sum_116 = add_39 = None
        sum_117: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_290, [1], True, dtype = torch.float32);  mul_290 = None
        expand_10: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_117, [128, 1280, 1, 1]);  sum_117 = None
        div_68: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_10, 1280);  expand_10 = None
        add_143: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_67, div_68);  div_67 = div_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_69: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_80, pow_12);  convert_element_type_80 = None
        eq_8: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_12, 0);  pow_12 = None
        where_8: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_8, full_default, div_69);  eq_8 = div_69 = None
        clone_37: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_8, memory_format = torch.contiguous_format);  where_8 = None
        mul_291: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_143, clone_37);  add_143 = clone_37 = None
        convert_element_type_330: "bf16[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_291, torch.bfloat16);  mul_291 = None
        add_144: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_142, convert_element_type_330);  add_142 = convert_element_type_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_331: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.float32);  add_144 = None
        mul_293: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_38, 0.5);  add_38 = None
        mul_294: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, convert_element_type_79)
        mul_295: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, -0.5);  mul_294 = None
        exp_8: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_295);  mul_295 = None
        mul_296: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_297: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, mul_296);  convert_element_type_79 = mul_296 = None
        add_146: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_293, mul_297);  mul_293 = mul_297 = None
        mul_298: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_331, add_146);  convert_element_type_331 = add_146 = None
        convert_element_type_333: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_298, torch.bfloat16);  mul_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_118: "bf16[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_333, [0, 2, 3])
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(convert_element_type_333, convert_element_type_78, convert_element_type_77, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_333 = convert_element_type_78 = convert_element_type_77 = None
        getitem_116: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_26[0]
        getitem_117: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None
        convert_element_type_334: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_116, torch.float32);  getitem_116 = None
        convert_element_type_335: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
        convert_element_type_336: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_118, torch.float32);  sum_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_63: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_334, [0, 2, 3, 1]);  convert_element_type_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_300: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_63, primals_66);  primals_66 = None
        mul_301: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_300, 320)
        sum_119: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_300, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_16: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_18, [0, 2, 3, 1]);  convolution_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_75: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_16, torch.float32);  permute_16 = None
        sub_8: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_75, getitem_17);  convert_element_type_75 = getitem_17 = None
        mul_36: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        mul_302: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_300, mul_36);  mul_300 = None
        sum_120: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_302, [3], True);  mul_302 = None
        mul_303: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, sum_120);  sum_120 = None
        sub_50: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_301, sum_119);  mul_301 = sum_119 = None
        sub_51: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_50, mul_303);  sub_50 = mul_303 = None
        div_70: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 320);  rsqrt_8 = None
        mul_304: "f32[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, sub_51);  div_70 = sub_51 = None
        mul_305: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_63, mul_36);  mul_36 = None
        sum_121: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 1, 2]);  mul_305 = None
        sum_122: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_63, [0, 1, 2]);  permute_63 = None
        convert_element_type_337: "bf16[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_304, torch.bfloat16);  mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_64: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_337, [0, 3, 1, 2]);  convert_element_type_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_123: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_64, [0, 2, 3])
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(permute_64, add_35, convert_element_type_74, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_64 = add_35 = convert_element_type_74 = None
        getitem_119: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_27[0]
        getitem_120: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        add_147: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_141, getitem_119);  add_141 = getitem_119 = None
        convert_element_type_338: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_120, torch.float32);  getitem_120 = None
        convert_element_type_339: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_123, torch.float32);  sum_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_124: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_147, [0, 2, 3])
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_147, convert_element_type_72, convert_element_type_71, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_72 = convert_element_type_71 = None
        getitem_122: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_28[0]
        getitem_123: "bf16[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None
        convert_element_type_340: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_122, torch.float32);  getitem_122 = None
        convert_element_type_341: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_123, torch.float32);  getitem_123 = None
        convert_element_type_342: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_124, torch.float32);  sum_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_343: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_340, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_67: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_16, torch.float32);  convolution_16 = None
        mul_32: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_33: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476)
        erf_4: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_32: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, add_32);  mul_32 = None
        convert_element_type_68: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_34, torch.bfloat16);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_4: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_10, add_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_35: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_68, div_4)
        mul_306: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_35, 1);  mul_35 = None
        mul_307: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_340, mul_306);  mul_306 = None
        view_9: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_61, [1, -1, 1, 1]);  primals_61 = None
        mul_308: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_9, 1);  view_9 = None
        mul_309: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_340, mul_308);  mul_308 = None
        sum_125: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_340, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_340 = None
        sum_126: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 2, 3], True, dtype = torch.float32);  mul_307 = None
        mul_310: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, convert_element_type_68)
        mul_311: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, div_4);  mul_309 = None
        convert_element_type_344: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_311, torch.bfloat16);  mul_311 = None
        sum_127: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_310, [2, 3], True, dtype = torch.float32);  mul_310 = None
        add_148: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_343, convert_element_type_344);  convert_element_type_343 = convert_element_type_344 = None
        view_49: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_126, [1280]);  sum_126 = None
        view_50: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_125, [1280]);  sum_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_72: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_4, add_33);  div_4 = None
        neg_9: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_127)
        mul_312: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_9, div_72);  neg_9 = div_72 = None
        div_73: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_127, add_33);  sum_127 = add_33 = None
        sum_128: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_312, [1], True, dtype = torch.float32);  mul_312 = None
        expand_11: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_128, [128, 1280, 1, 1]);  sum_128 = None
        div_74: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_11, 1280);  expand_11 = None
        add_149: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_73, div_74);  div_73 = div_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_75: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_68, pow_10);  convert_element_type_68 = None
        eq_9: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_10, 0);  pow_10 = None
        where_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_9, full_default, div_75);  eq_9 = div_75 = None
        clone_38: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_9, memory_format = torch.contiguous_format);  where_9 = None
        mul_313: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_149, clone_38);  add_149 = clone_38 = None
        convert_element_type_345: "bf16[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_313, torch.bfloat16);  mul_313 = None
        add_150: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_148, convert_element_type_345);  add_148 = convert_element_type_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_346: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(add_150, torch.float32);  add_150 = None
        mul_315: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
        mul_316: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, convert_element_type_67)
        mul_317: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_316, -0.5);  mul_316 = None
        exp_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_317);  mul_317 = None
        mul_318: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_319: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_67, mul_318);  convert_element_type_67 = mul_318 = None
        add_152: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_315, mul_319);  mul_315 = mul_319 = None
        mul_320: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_346, add_152);  convert_element_type_346 = add_152 = None
        convert_element_type_348: "bf16[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.prims.convert_element_type.default(mul_320, torch.bfloat16);  mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_129: "bf16[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_348, [0, 2, 3])
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(convert_element_type_348, convert_element_type_66, convert_element_type_65, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_348 = convert_element_type_66 = convert_element_type_65 = None
        getitem_125: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_29[0]
        getitem_126: "bf16[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None
        convert_element_type_349: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_125, torch.float32);  getitem_125 = None
        convert_element_type_350: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_126, torch.float32);  getitem_126 = None
        convert_element_type_351: "f32[1280][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_129, torch.float32);  sum_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_65: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_349, [0, 2, 3, 1]);  convert_element_type_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_322: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_65, primals_56);  primals_56 = None
        mul_323: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, 320)
        sum_130: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_322, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_14: "bf16[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_63: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_14, torch.float32);  permute_14 = None
        sub_7: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_63, getitem_15);  convert_element_type_63 = getitem_15 = None
        mul_30: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        mul_324: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, mul_30);  mul_322 = None
        sum_131: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_324, [3], True);  mul_324 = None
        mul_325: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_30, sum_131);  sum_131 = None
        sub_53: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_323, sum_130);  mul_323 = sum_130 = None
        sub_54: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_53, mul_325);  sub_53 = mul_325 = None
        div_76: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 320);  rsqrt_7 = None
        mul_326: "f32[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_76, sub_54);  div_76 = sub_54 = None
        mul_327: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_65, mul_30);  mul_30 = None
        sum_132: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_327, [0, 1, 2]);  mul_327 = None
        sum_133: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_65, [0, 1, 2]);  permute_65 = None
        convert_element_type_352: "bf16[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_326, torch.bfloat16);  mul_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_66: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_352, [0, 3, 1, 2]);  convert_element_type_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_134: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_66, [0, 2, 3])
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(permute_66, convolution_14, convert_element_type_62, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_66 = convolution_14 = convert_element_type_62 = None
        getitem_128: "bf16[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_30[0]
        getitem_129: "bf16[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        add_153: "bf16[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_147, getitem_128);  add_147 = getitem_128 = None
        convert_element_type_353: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_129, torch.float32);  getitem_129 = None
        convert_element_type_354: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_134, torch.float32);  sum_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        sum_135: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_153, [0, 2, 3])
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(add_153, convert_element_type_60, convert_element_type_59, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_153 = convert_element_type_60 = convert_element_type_59 = None
        getitem_131: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = convolution_backward_31[0]
        getitem_132: "bf16[320, 160, 2, 2][640, 1, 320, 160]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None
        convert_element_type_355: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_131, torch.float32);  getitem_131 = None
        convert_element_type_356: "f32[320, 160, 2, 2][640, 1, 320, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_132, torch.float32);  getitem_132 = None
        convert_element_type_357: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_135, torch.float32);  sum_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_67: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_355, [0, 2, 3, 1]);  convert_element_type_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_329: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_67, primals_50);  primals_50 = None
        mul_330: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, 160)
        sum_136: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_329, [3], True)
        convert_element_type_57: "f32[128, 28, 28, 160][125440, 1, 4480, 28]cuda:0" = torch.ops.prims.convert_element_type.default(permute_12, torch.float32);  permute_12 = None
        sub_6: "f32[128, 28, 28, 160][125440, 1, 4480, 28]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_57, getitem_13);  convert_element_type_57 = getitem_13 = None
        mul_28: "f32[128, 28, 28, 160][125440, 1, 4480, 28]cuda:0" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
        mul_331: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, mul_28);  mul_329 = None
        sum_137: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_331, [3], True);  mul_331 = None
        mul_332: "f32[128, 28, 28, 160][125440, 1, 4480, 28]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, sum_137);  sum_137 = None
        sub_56: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_330, sum_136);  mul_330 = sum_136 = None
        sub_57: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, mul_332);  sub_56 = mul_332 = None
        div_77: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 160);  rsqrt_6 = None
        mul_333: "f32[128, 28, 28, 160][125440, 160, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_77, sub_57);  div_77 = sub_57 = None
        mul_334: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_67, mul_28);  mul_28 = None
        sum_138: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1, 2]);  mul_334 = None
        sum_139: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_67, [0, 1, 2]);  permute_67 = None
        convert_element_type_358: "bf16[128, 28, 28, 160][125440, 160, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_333, torch.bfloat16);  mul_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_68: "bf16[128, 160, 28, 28][125440, 1, 160, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_358, [0, 3, 1, 2]);  convert_element_type_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_140: "bf16[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_68, [0, 2, 3])
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(permute_68, convert_element_type_56, convert_element_type_55, [160], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_56 = convert_element_type_55 = None
        getitem_134: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = convolution_backward_32[0]
        getitem_135: "bf16[160, 640, 1, 1][640, 1, 640, 640]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None
        convert_element_type_359: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_134, torch.float32);  getitem_134 = None
        convert_element_type_360: "f32[160, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_135, torch.float32);  getitem_135 = None
        convert_element_type_361: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_140, torch.float32);  sum_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_362: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_359, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_51: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_12, torch.float32);  convolution_12 = None
        mul_24: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, 0.5)
        mul_25: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, 0.7071067811865476)
        erf_3: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_24: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_26: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, add_24);  mul_24 = None
        convert_element_type_52: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(mul_26, torch.bfloat16);  mul_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_3: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.div.Tensor(pow_8, add_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_27: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_52, div_3)
        mul_335: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Scalar(mul_27, 1);  mul_27 = None
        mul_336: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_359, mul_335);  mul_335 = None
        view_7: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_47, [1, -1, 1, 1]);  primals_47 = None
        mul_337: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_7, 1);  view_7 = None
        mul_338: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_359, mul_337);  mul_337 = None
        sum_141: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_359, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_359 = None
        sum_142: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2, 3], True, dtype = torch.float32);  mul_336 = None
        mul_339: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, convert_element_type_52)
        mul_340: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, div_3);  mul_338 = None
        convert_element_type_363: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(mul_340, torch.bfloat16);  mul_340 = None
        sum_143: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [2, 3], True, dtype = torch.float32);  mul_339 = None
        add_154: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_362, convert_element_type_363);  convert_element_type_362 = convert_element_type_363 = None
        view_51: "f32[640][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [640]);  sum_142 = None
        view_52: "f32[640][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [640]);  sum_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_79: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.div.Tensor(div_3, add_25);  div_3 = None
        neg_10: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_143)
        mul_341: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.mul.Tensor(neg_10, div_79);  neg_10 = div_79 = None
        div_80: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_143, add_25);  sum_143 = add_25 = None
        sum_144: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_341, [1], True, dtype = torch.float32);  mul_341 = None
        expand_12: "f32[128, 640, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_144, [128, 640, 1, 1]);  sum_144 = None
        div_81: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_12, 640);  expand_12 = None
        add_155: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_80, div_81);  div_80 = div_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_82: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_52, pow_8);  convert_element_type_52 = None
        eq_10: "b8[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.eq.Scalar(pow_8, 0);  pow_8 = None
        where_10: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.where.self(eq_10, full_default, div_82);  eq_10 = div_82 = None
        clone_39: "f32[128, 640, 28, 28][501760, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(where_10, memory_format = torch.contiguous_format);  where_10 = None
        mul_342: "f32[128, 640, 28, 28][501760, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_155, clone_39);  add_155 = clone_39 = None
        convert_element_type_364: "bf16[128, 640, 28, 28][501760, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_342, torch.bfloat16);  mul_342 = None
        add_156: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(add_154, convert_element_type_364);  add_154 = convert_element_type_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_365: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(add_156, torch.float32);  add_156 = None
        mul_344: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
        mul_345: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, convert_element_type_51)
        mul_346: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_345, -0.5);  mul_345 = None
        exp_10: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.exp.default(mul_346);  mul_346 = None
        mul_347: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_348: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_51, mul_347);  convert_element_type_51 = mul_347 = None
        add_158: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_344, mul_348);  mul_344 = mul_348 = None
        mul_349: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_365, add_158);  convert_element_type_365 = add_158 = None
        convert_element_type_367: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(mul_349, torch.bfloat16);  mul_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_145: "bf16[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_367, [0, 2, 3])
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(convert_element_type_367, convert_element_type_50, convert_element_type_49, [640], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_367 = convert_element_type_50 = convert_element_type_49 = None
        getitem_137: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = convolution_backward_33[0]
        getitem_138: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None
        convert_element_type_368: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_137, torch.float32);  getitem_137 = None
        convert_element_type_369: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_138, torch.float32);  getitem_138 = None
        convert_element_type_370: "f32[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_145, torch.float32);  sum_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_69: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_368, [0, 2, 3, 1]);  convert_element_type_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_351: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_69, primals_42);  primals_42 = None
        mul_352: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, 160)
        sum_146: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_351, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_10: "bf16[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1]);  convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_47: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_10, torch.float32);  permute_10 = None
        sub_5: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_47, getitem_11);  convert_element_type_47 = getitem_11 = None
        mul_22: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        mul_353: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, mul_22);  mul_351 = None
        sum_147: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_353, [3], True);  mul_353 = None
        mul_354: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, sum_147);  sum_147 = None
        sub_59: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_352, sum_146);  mul_352 = sum_146 = None
        sub_60: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_59, mul_354);  sub_59 = mul_354 = None
        div_83: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 160);  rsqrt_5 = None
        mul_355: "f32[128, 28, 28, 160][125440, 160, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_83, sub_60);  div_83 = sub_60 = None
        mul_356: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_69, mul_22);  mul_22 = None
        sum_148: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_356, [0, 1, 2]);  mul_356 = None
        sum_149: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_69, [0, 1, 2]);  permute_69 = None
        convert_element_type_371: "bf16[128, 28, 28, 160][125440, 160, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_355, torch.bfloat16);  mul_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_70: "bf16[128, 160, 28, 28][125440, 1, 160, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_371, [0, 3, 1, 2]);  convert_element_type_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_150: "bf16[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_70, [0, 2, 3])
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(permute_70, add_21, convert_element_type_46, [160], [1, 1], [3, 3], [1, 1], False, [0, 0], 160, [True, True, False]);  permute_70 = add_21 = convert_element_type_46 = None
        getitem_140: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = convolution_backward_34[0]
        getitem_141: "bf16[160, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        add_159: "bf16[128, 160, 28, 28][125440, 1, 160, 4480]cuda:0" = torch.ops.aten.add.Tensor(permute_68, getitem_140);  permute_68 = getitem_140 = None
        convert_element_type_372: "f32[160, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_141, torch.float32);  getitem_141 = None
        convert_element_type_373: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_150, torch.float32);  sum_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_151: "bf16[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_159, [0, 2, 3])
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(add_159, convert_element_type_44, convert_element_type_43, [160], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_44 = convert_element_type_43 = None
        getitem_143: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = convolution_backward_35[0]
        getitem_144: "bf16[160, 640, 1, 1][640, 1, 640, 640]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None
        convert_element_type_374: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_143, torch.float32);  getitem_143 = None
        convert_element_type_375: "f32[160, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_144, torch.float32);  getitem_144 = None
        convert_element_type_376: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_151, torch.float32);  sum_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_377: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_374, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_39: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_9, torch.float32);  convolution_9 = None
        mul_18: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, 0.5)
        mul_19: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, 0.7071067811865476)
        erf_2: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_18: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, add_18);  mul_18 = None
        convert_element_type_40: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_2: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.div.Tensor(pow_6, add_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_21: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_40, div_2)
        mul_357: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Scalar(mul_21, 1);  mul_21 = None
        mul_358: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_374, mul_357);  mul_357 = None
        view_5: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_37, [1, -1, 1, 1]);  primals_37 = None
        mul_359: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_5, 1);  view_5 = None
        mul_360: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_374, mul_359);  mul_359 = None
        sum_152: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_374, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_374 = None
        sum_153: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 2, 3], True, dtype = torch.float32);  mul_358 = None
        mul_361: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, convert_element_type_40)
        mul_362: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, div_2);  mul_360 = None
        convert_element_type_378: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None
        sum_154: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_361, [2, 3], True, dtype = torch.float32);  mul_361 = None
        add_160: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_377, convert_element_type_378);  convert_element_type_377 = convert_element_type_378 = None
        view_53: "f32[640][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [640]);  sum_153 = None
        view_54: "f32[640][1]cuda:0" = torch.ops.aten.reshape.default(sum_152, [640]);  sum_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_85: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.div.Tensor(div_2, add_19);  div_2 = None
        neg_11: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_154)
        mul_363: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.mul.Tensor(neg_11, div_85);  neg_11 = div_85 = None
        div_86: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_154, add_19);  sum_154 = add_19 = None
        sum_155: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_363, [1], True, dtype = torch.float32);  mul_363 = None
        expand_13: "f32[128, 640, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_155, [128, 640, 1, 1]);  sum_155 = None
        div_87: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_13, 640);  expand_13 = None
        add_161: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_86, div_87);  div_86 = div_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_88: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_40, pow_6);  convert_element_type_40 = None
        eq_11: "b8[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.eq.Scalar(pow_6, 0);  pow_6 = None
        where_11: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.where.self(eq_11, full_default, div_88);  eq_11 = div_88 = None
        clone_40: "f32[128, 640, 28, 28][501760, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(where_11, memory_format = torch.contiguous_format);  where_11 = None
        mul_364: "f32[128, 640, 28, 28][501760, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_161, clone_40);  add_161 = clone_40 = None
        convert_element_type_379: "bf16[128, 640, 28, 28][501760, 784, 28, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_364, torch.bfloat16);  mul_364 = None
        add_162: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(add_160, convert_element_type_379);  add_160 = convert_element_type_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_380: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(add_162, torch.float32);  add_162 = None
        mul_366: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
        mul_367: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, convert_element_type_39)
        mul_368: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, -0.5);  mul_367 = None
        exp_11: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.exp.default(mul_368);  mul_368 = None
        mul_369: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_370: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_39, mul_369);  convert_element_type_39 = mul_369 = None
        add_164: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_366, mul_370);  mul_366 = mul_370 = None
        mul_371: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, add_164);  convert_element_type_380 = add_164 = None
        convert_element_type_382: "bf16[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.prims.convert_element_type.default(mul_371, torch.bfloat16);  mul_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_156: "bf16[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_382, [0, 2, 3])
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(convert_element_type_382, convert_element_type_38, convert_element_type_37, [640], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_382 = convert_element_type_38 = convert_element_type_37 = None
        getitem_146: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = convolution_backward_36[0]
        getitem_147: "bf16[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None
        convert_element_type_383: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_146, torch.float32);  getitem_146 = None
        convert_element_type_384: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_147, torch.float32);  getitem_147 = None
        convert_element_type_385: "f32[640][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_156, torch.float32);  sum_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_71: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_383, [0, 2, 3, 1]);  convert_element_type_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_373: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_71, primals_32);  primals_32 = None
        mul_374: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_373, 160)
        sum_157: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_373, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_8: "bf16[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convolution_8, [0, 2, 3, 1]);  convolution_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_35: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_8, torch.float32);  permute_8 = None
        sub_4: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_35, getitem_9);  convert_element_type_35 = getitem_9 = None
        mul_16: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        mul_375: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_373, mul_16);  mul_373 = None
        sum_158: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_375, [3], True);  mul_375 = None
        mul_376: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, sum_158);  sum_158 = None
        sub_62: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_374, sum_157);  mul_374 = sum_157 = None
        sub_63: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_62, mul_376);  sub_62 = mul_376 = None
        div_89: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 160);  rsqrt_4 = None
        mul_377: "f32[128, 28, 28, 160][125440, 160, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_89, sub_63);  div_89 = sub_63 = None
        mul_378: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_71, mul_16);  mul_16 = None
        sum_159: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_378, [0, 1, 2]);  mul_378 = None
        sum_160: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_71, [0, 1, 2]);  permute_71 = None
        convert_element_type_386: "bf16[128, 28, 28, 160][125440, 160, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_377, torch.bfloat16);  mul_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_72: "bf16[128, 160, 28, 28][125440, 1, 160, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_386, [0, 3, 1, 2]);  convert_element_type_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_161: "bf16[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_72, [0, 2, 3])
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(permute_72, convolution_7, convert_element_type_34, [160], [1, 1], [3, 3], [1, 1], False, [0, 0], 160, [True, True, False]);  permute_72 = convolution_7 = convert_element_type_34 = None
        getitem_149: "bf16[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = convolution_backward_37[0]
        getitem_150: "bf16[160, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        add_165: "bf16[128, 160, 28, 28][125440, 1, 160, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_159, getitem_149);  add_159 = getitem_149 = None
        convert_element_type_387: "f32[160, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_150, torch.float32);  getitem_150 = None
        convert_element_type_388: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_161, torch.float32);  sum_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        sum_162: "bf16[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_165, [0, 2, 3])
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(add_165, convert_element_type_32, convert_element_type_31, [160], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_165 = convert_element_type_32 = convert_element_type_31 = None
        getitem_152: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = convolution_backward_38[0]
        getitem_153: "bf16[160, 80, 2, 2][320, 1, 160, 80]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None
        convert_element_type_389: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_152, torch.float32);  getitem_152 = None
        convert_element_type_390: "f32[160, 80, 2, 2][320, 1, 160, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_153, torch.float32);  getitem_153 = None
        convert_element_type_391: "f32[160][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_162, torch.float32);  sum_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_73: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_389, [0, 2, 3, 1]);  convert_element_type_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_380: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_73, primals_26);  primals_26 = None
        mul_381: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_380, 80)
        sum_163: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_380, [3], True)
        mul_382: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_380, mul_14);  mul_380 = None
        sum_164: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_382, [3], True);  mul_382 = None
        mul_383: "f32[128, 56, 56, 80][250880, 1, 4480, 56]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, sum_164);  sum_164 = None
        sub_65: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_381, sum_163);  mul_381 = sum_163 = None
        sub_66: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_65, mul_383);  sub_65 = mul_383 = None
        mul_384: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_90, sub_66);  div_90 = sub_66 = None
        mul_385: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_73, mul_14);  mul_14 = None
        sum_165: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_385, [0, 1, 2]);  mul_385 = None
        sum_166: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_73, [0, 1, 2]);  permute_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_74: "f32[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.permute.default(mul_384, [0, 3, 1, 2]);  mul_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        convert_element_type_392: "bf16[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.prims.convert_element_type.default(permute_74, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_167: "bf16[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_392, [0, 2, 3])
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(convert_element_type_392, convert_element_type_29, convert_element_type_28, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_392 = convert_element_type_29 = convert_element_type_28 = None
        getitem_155: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = convolution_backward_39[0]
        getitem_156: "bf16[80, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None
        convert_element_type_393: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_155, torch.float32);  getitem_155 = None
        convert_element_type_394: "f32[80, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_156, torch.float32);  getitem_156 = None
        convert_element_type_395: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_167, torch.float32);  sum_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_396: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_393, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_24: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_5, torch.float32);  convolution_5 = None
        mul_10: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5)
        mul_11: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.7071067811865476)
        erf_1: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_10: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, add_10);  mul_10 = None
        convert_element_type_25: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_1: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.div.Tensor(pow_4, add_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_13: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_25, div_1)
        mul_386: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Scalar(mul_13, 1);  mul_13 = None
        mul_387: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_393, mul_386);  mul_386 = None
        view_3: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_23, [1, -1, 1, 1]);  primals_23 = None
        mul_388: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_3, 1);  view_3 = None
        mul_389: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_393, mul_388);  mul_388 = None
        sum_168: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_393, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_393 = None
        sum_169: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 2, 3], True, dtype = torch.float32);  mul_387 = None
        mul_390: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, convert_element_type_25)
        mul_391: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, div_1);  mul_389 = None
        convert_element_type_397: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_391, torch.bfloat16);  mul_391 = None
        sum_170: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_390, [2, 3], True, dtype = torch.float32);  mul_390 = None
        add_166: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_396, convert_element_type_397);  convert_element_type_396 = convert_element_type_397 = None
        view_55: "f32[320][1]cuda:0" = torch.ops.aten.reshape.default(sum_169, [320]);  sum_169 = None
        view_56: "f32[320][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [320]);  sum_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_92: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.div.Tensor(div_1, add_11);  div_1 = None
        neg_12: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_170)
        mul_392: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.mul.Tensor(neg_12, div_92);  neg_12 = div_92 = None
        div_93: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_170, add_11);  sum_170 = add_11 = None
        sum_171: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [1], True, dtype = torch.float32);  mul_392 = None
        expand_14: "f32[128, 320, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_171, [128, 320, 1, 1]);  sum_171 = None
        div_94: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_14, 320);  expand_14 = None
        add_167: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_93, div_94);  div_93 = div_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_95: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_25, pow_4);  convert_element_type_25 = None
        eq_12: "b8[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.eq.Scalar(pow_4, 0);  pow_4 = None
        where_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.where.self(eq_12, full_default, div_95);  eq_12 = div_95 = None
        clone_41: "f32[128, 320, 56, 56][1003520, 3136, 56, 1]cuda:0" = torch.ops.aten.clone.default(where_12, memory_format = torch.contiguous_format);  where_12 = None
        mul_393: "f32[128, 320, 56, 56][1003520, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_167, clone_41);  add_167 = clone_41 = None
        convert_element_type_398: "bf16[128, 320, 56, 56][1003520, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_393, torch.bfloat16);  mul_393 = None
        add_168: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(add_166, convert_element_type_398);  add_166 = convert_element_type_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_399: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_168, torch.float32);  add_168 = None
        mul_395: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_396: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, convert_element_type_24)
        mul_397: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
        exp_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
        mul_398: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_399: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_24, mul_398);  convert_element_type_24 = mul_398 = None
        add_170: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
        mul_400: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_399, add_170);  convert_element_type_399 = add_170 = None
        convert_element_type_401: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_400, torch.bfloat16);  mul_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_172: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_401, [0, 2, 3])
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(convert_element_type_401, convert_element_type_23, convert_element_type_22, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_401 = convert_element_type_23 = convert_element_type_22 = None
        getitem_158: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = convolution_backward_40[0]
        getitem_159: "bf16[320, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None
        convert_element_type_402: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_158, torch.float32);  getitem_158 = None
        convert_element_type_403: "f32[320, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_159, torch.float32);  getitem_159 = None
        convert_element_type_404: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_172, torch.float32);  sum_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_75: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_402, [0, 2, 3, 1]);  convert_element_type_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_402: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_75, primals_18);  primals_18 = None
        mul_403: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, 80)
        sum_173: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_402, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_4: "bf16[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1]);  convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_20: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_4, torch.float32);  permute_4 = None
        sub_2: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_20, getitem_5);  convert_element_type_20 = getitem_5 = None
        mul_8: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        mul_404: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, mul_8);  mul_402 = None
        sum_174: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_404, [3], True);  mul_404 = None
        mul_405: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, sum_174);  sum_174 = None
        sub_68: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_403, sum_173);  mul_403 = sum_173 = None
        sub_69: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_68, mul_405);  sub_68 = mul_405 = None
        div_96: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 80);  rsqrt_2 = None
        mul_406: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_96, sub_69);  div_96 = sub_69 = None
        mul_407: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_75, mul_8);  mul_8 = None
        sum_175: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1, 2]);  mul_407 = None
        sum_176: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_75, [0, 1, 2]);  permute_75 = None
        convert_element_type_405: "bf16[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_406, torch.bfloat16);  mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_76: "bf16[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_405, [0, 3, 1, 2]);  convert_element_type_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_177: "bf16[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_76, [0, 2, 3])
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(permute_76, convert_element_type_19, convert_element_type_18, [80], [1, 1], [3, 3], [1, 1], False, [0, 0], 80, [True, True, False]);  permute_76 = convert_element_type_19 = convert_element_type_18 = None
        getitem_161: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = convolution_backward_41[0]
        getitem_162: "bf16[80, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        convert_element_type_406: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_161, torch.float32);  getitem_161 = None
        add_171: "f32[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.add.Tensor(permute_74, convert_element_type_406);  permute_74 = convert_element_type_406 = None
        convert_element_type_407: "f32[80, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_162, torch.float32);  getitem_162 = None
        convert_element_type_408: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_177, torch.float32);  sum_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:212 in forward, code: x = self.drop_path(x) + self.shortcut(shortcut)
        convert_element_type_409: "bf16[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_178: "bf16[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_409, [0, 2, 3])
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(convert_element_type_409, convert_element_type_16, convert_element_type_15, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_409 = convert_element_type_16 = convert_element_type_15 = None
        getitem_164: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = convolution_backward_42[0]
        getitem_165: "bf16[80, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None
        convert_element_type_410: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_164, torch.float32);  getitem_164 = None
        convert_element_type_411: "f32[80, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_165, torch.float32);  getitem_165 = None
        convert_element_type_412: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_178, torch.float32);  sum_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        convert_element_type_413: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_410, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_11: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(convolution_2, torch.float32);  convolution_2 = None
        mul_4: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.5)
        mul_5: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_11, 0.7071067811865476)
        erf: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = None
        convert_element_type_12: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.div.Tensor(pow_2, add_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_7: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_12, div)
        mul_408: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Scalar(mul_7, 1);  mul_7 = None
        mul_409: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_410, mul_408);  mul_408 = None
        view_1: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_13, [1, -1, 1, 1]);  primals_13 = None
        mul_410: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_1, 1);  view_1 = None
        mul_411: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_410, mul_410);  mul_410 = None
        sum_179: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_410, [0, 2, 3], True, dtype = torch.float32);  convert_element_type_410 = None
        sum_180: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3], True, dtype = torch.float32);  mul_409 = None
        mul_412: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, convert_element_type_12)
        mul_413: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, div);  mul_411 = None
        convert_element_type_414: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_413, torch.bfloat16);  mul_413 = None
        sum_181: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_412, [2, 3], True, dtype = torch.float32);  mul_412 = None
        add_172: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_413, convert_element_type_414);  convert_element_type_413 = convert_element_type_414 = None
        view_57: "f32[320][1]cuda:0" = torch.ops.aten.reshape.default(sum_180, [320]);  sum_180 = None
        view_58: "f32[320][1]cuda:0" = torch.ops.aten.reshape.default(sum_179, [320]);  sum_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_98: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.div.Tensor(div, add_5);  div = None
        neg_13: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_181)
        mul_414: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.mul.Tensor(neg_13, div_98);  neg_13 = div_98 = None
        div_99: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_181, add_5);  sum_181 = add_5 = None
        sum_182: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_414, [1], True, dtype = torch.float32);  mul_414 = None
        expand_15: "f32[128, 320, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_182, [128, 320, 1, 1]);  sum_182 = None
        div_100: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_15, 320);  expand_15 = None
        add_173: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_99, div_100);  div_99 = div_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_101: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type_12, pow_2);  convert_element_type_12 = None
        eq_13: "b8[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.eq.Scalar(pow_2, 0);  pow_2 = None
        where_13: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.where.self(eq_13, full_default, div_101);  eq_13 = full_default = div_101 = None
        clone_42: "f32[128, 320, 56, 56][1003520, 3136, 56, 1]cuda:0" = torch.ops.aten.clone.default(where_13, memory_format = torch.contiguous_format);  where_13 = None
        mul_415: "f32[128, 320, 56, 56][1003520, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_173, clone_42);  add_173 = clone_42 = None
        convert_element_type_415: "bf16[128, 320, 56, 56][1003520, 3136, 56, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_415, torch.bfloat16);  mul_415 = None
        add_174: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(add_172, convert_element_type_415);  add_172 = convert_element_type_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        convert_element_type_416: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(add_174, torch.float32);  add_174 = None
        mul_417: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
        mul_418: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_11, convert_element_type_11)
        mul_419: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_418, -0.5);  mul_418 = None
        exp_13: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.exp.default(mul_419);  mul_419 = None
        mul_420: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_421: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_11, mul_420);  convert_element_type_11 = mul_420 = None
        add_176: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_417, mul_421);  mul_417 = mul_421 = None
        mul_422: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_416, add_176);  convert_element_type_416 = add_176 = None
        convert_element_type_418: "bf16[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.prims.convert_element_type.default(mul_422, torch.bfloat16);  mul_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_183: "bf16[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_418, [0, 2, 3])
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(convert_element_type_418, convert_element_type_10, convert_element_type_9, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_418 = convert_element_type_10 = convert_element_type_9 = None
        getitem_167: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = convolution_backward_43[0]
        getitem_168: "bf16[320, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None
        convert_element_type_419: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_167, torch.float32);  getitem_167 = None
        convert_element_type_420: "f32[320, 80, 1, 1][80, 1, 80, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_168, torch.float32);  getitem_168 = None
        convert_element_type_421: "f32[320][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_183, torch.float32);  sum_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_77: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_419, [0, 2, 3, 1]);  convert_element_type_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_424: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_77, primals_8);  primals_8 = None
        mul_425: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_424, 80)
        sum_184: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_424, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_2: "bf16[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_7: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_2, torch.float32);  permute_2 = None
        sub_1: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_7, getitem_3);  convert_element_type_7 = getitem_3 = None
        mul_2: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_426: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_424, mul_2);  mul_424 = None
        sum_185: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_426, [3], True);  mul_426 = None
        mul_427: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, sum_185);  sum_185 = None
        sub_71: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_425, sum_184);  mul_425 = sum_184 = None
        sub_72: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_71, mul_427);  sub_71 = mul_427 = None
        div_102: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 80);  rsqrt_1 = None
        mul_428: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_102, sub_72);  div_102 = sub_72 = None
        mul_429: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_77, mul_2);  mul_2 = None
        sum_186: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_429, [0, 1, 2]);  mul_429 = None
        sum_187: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_77, [0, 1, 2]);  permute_77 = None
        convert_element_type_422: "bf16[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_428, torch.bfloat16);  mul_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_78: "bf16[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_422, [0, 3, 1, 2]);  convert_element_type_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_188: "bf16[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_78, [0, 2, 3])
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(permute_78, convert_element_type_6, convert_element_type_5, [80], [1, 1], [3, 3], [1, 1], False, [0, 0], 80, [True, True, False]);  permute_78 = convert_element_type_6 = convert_element_type_5 = None
        getitem_170: "bf16[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = convolution_backward_44[0]
        getitem_171: "bf16[80, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        convert_element_type_423: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_170, torch.float32);  getitem_170 = None
        add_177: "f32[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_171, convert_element_type_423);  add_171 = convert_element_type_423 = None
        convert_element_type_424: "f32[80, 1, 7, 7][49, 1, 7, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_171, torch.float32);  getitem_171 = None
        convert_element_type_425: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_188, torch.float32);  sum_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_79: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.permute.default(add_177, [0, 2, 3, 1]);  add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_431: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_79, primals_4);  primals_4 = None
        mul_432: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_431, 80)
        sum_189: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_431, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute: "bf16[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_3: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute, torch.float32);  permute = None
        sub: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_3, getitem_1);  convert_element_type_3 = getitem_1 = None
        mul: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_433: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_431, mul);  mul_431 = None
        sum_190: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_433, [3], True);  mul_433 = None
        mul_434: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_190);  sum_190 = None
        sub_74: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_432, sum_189);  mul_432 = sum_189 = None
        sub_75: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_74, mul_434);  sub_74 = mul_434 = None
        div_103: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 80);  rsqrt = None
        mul_435: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_103, sub_75);  div_103 = sub_75 = None
        mul_436: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_79, mul);  mul = None
        sum_191: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 1, 2]);  mul_436 = None
        sum_192: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_79, [0, 1, 2]);  permute_79 = None
        convert_element_type_426: "bf16[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_435, torch.bfloat16);  mul_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_80: "bf16[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.permute.default(convert_element_type_426, [0, 3, 1, 2]);  convert_element_type_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:608 in forward_features, code: x = self.stem(x)
        sum_193: "bf16[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_80, [0, 2, 3])
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(permute_80, convert_element_type_2, convert_element_type_1, [80], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  permute_80 = convert_element_type_2 = convert_element_type_1 = None
        getitem_174: "bf16[80, 3, 4, 4][48, 1, 12, 3]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        convert_element_type_427: "f32[80, 3, 4, 4][48, 1, 12, 3]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_174, torch.float32);  getitem_174 = None
        convert_element_type_428: "f32[80][1]cuda:0" = torch.ops.prims.convert_element_type.default(sum_193, torch.float32);  sum_193 = None
        return (convert_element_type_427, convert_element_type_428, None, sum_191, sum_192, convert_element_type_424, convert_element_type_425, sum_186, sum_187, convert_element_type_420, convert_element_type_421, view_58, view_57, convert_element_type_411, convert_element_type_412, convert_element_type_407, convert_element_type_408, sum_175, sum_176, convert_element_type_403, convert_element_type_404, view_56, view_55, convert_element_type_394, convert_element_type_395, sum_165, sum_166, convert_element_type_390, convert_element_type_391, convert_element_type_387, convert_element_type_388, sum_159, sum_160, convert_element_type_384, convert_element_type_385, view_54, view_53, convert_element_type_375, convert_element_type_376, convert_element_type_372, convert_element_type_373, sum_148, sum_149, convert_element_type_369, convert_element_type_370, view_52, view_51, convert_element_type_360, convert_element_type_361, sum_138, sum_139, convert_element_type_356, convert_element_type_357, convert_element_type_353, convert_element_type_354, sum_132, sum_133, convert_element_type_350, convert_element_type_351, view_50, view_49, convert_element_type_341, convert_element_type_342, convert_element_type_338, convert_element_type_339, sum_121, sum_122, convert_element_type_335, convert_element_type_336, view_48, view_47, convert_element_type_326, convert_element_type_327, convert_element_type_323, convert_element_type_324, sum_110, sum_111, convert_element_type_320, convert_element_type_321, view_46, view_45, convert_element_type_311, convert_element_type_312, convert_element_type_308, convert_element_type_309, sum_99, sum_100, convert_element_type_305, convert_element_type_306, view_44, view_43, convert_element_type_296, convert_element_type_297, convert_element_type_293, convert_element_type_294, sum_88, sum_89, convert_element_type_290, convert_element_type_291, view_42, view_41, convert_element_type_281, convert_element_type_282, convert_element_type_278, convert_element_type_279, sum_77, sum_78, convert_element_type_275, convert_element_type_276, view_40, view_39, convert_element_type_266, convert_element_type_267, convert_element_type_263, convert_element_type_264, sum_66, sum_67, convert_element_type_260, convert_element_type_261, view_38, view_37, convert_element_type_251, convert_element_type_252, convert_element_type_248, convert_element_type_249, sum_55, sum_56, convert_element_type_245, convert_element_type_246, view_36, view_35, convert_element_type_236, convert_element_type_237, sum_45, sum_46, convert_element_type_232, convert_element_type_233, convert_element_type_229, convert_element_type_230, sum_39, sum_40, convert_element_type_226, convert_element_type_227, view_34, view_33, convert_element_type_217, convert_element_type_218, convert_element_type_214, convert_element_type_215, sum_28, sum_29, convert_element_type_211, convert_element_type_212, view_32, view_31, convert_element_type_202, convert_element_type_203, sum_18, sum_19, convert_element_type_198, convert_element_type_199)
