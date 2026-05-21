class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[80, 3, 4, 4][48, 1, 12, 3]cuda:0", primals_3: "f32[128, 3, 224, 224][150528, 1, 672, 3]cuda:0", primals_4: "f32[80][1]cuda:0", primals_6: "f32[80, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_8: "f32[80][1]cuda:0", primals_10: "f32[320, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_13: "f32[320][1]cuda:0", primals_14: "f32[80, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_16: "f32[80, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_18: "f32[80][1]cuda:0", primals_20: "f32[320, 80, 1, 1][80, 1, 80, 80]cuda:0", primals_23: "f32[320][1]cuda:0", primals_24: "f32[80, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_26: "f32[80][1]cuda:0", primals_28: "f32[160, 80, 2, 2][320, 1, 160, 80]cuda:0", primals_30: "f32[160, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_32: "f32[160][1]cuda:0", primals_34: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_37: "f32[640][1]cuda:0", primals_38: "f32[160, 640, 1, 1][640, 1, 640, 640]cuda:0", primals_40: "f32[160, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_42: "f32[160][1]cuda:0", primals_44: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_47: "f32[640][1]cuda:0", primals_48: "f32[160, 640, 1, 1][640, 1, 640, 640]cuda:0", primals_50: "f32[160][1]cuda:0", primals_52: "f32[320, 160, 2, 2][640, 1, 320, 160]cuda:0", primals_54: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_56: "f32[320][1]cuda:0", primals_58: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_61: "f32[1280][1]cuda:0", primals_62: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_64: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_66: "f32[320][1]cuda:0", primals_68: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_71: "f32[1280][1]cuda:0", primals_72: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_74: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_76: "f32[320][1]cuda:0", primals_78: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_81: "f32[1280][1]cuda:0", primals_82: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_84: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_86: "f32[320][1]cuda:0", primals_88: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_91: "f32[1280][1]cuda:0", primals_92: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_94: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_96: "f32[320][1]cuda:0", primals_98: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_101: "f32[1280][1]cuda:0", primals_102: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_104: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_106: "f32[320][1]cuda:0", primals_108: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_111: "f32[1280][1]cuda:0", primals_112: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_114: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_116: "f32[320][1]cuda:0", primals_118: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_121: "f32[1280][1]cuda:0", primals_122: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_124: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_126: "f32[320][1]cuda:0", primals_128: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0", primals_131: "f32[1280][1]cuda:0", primals_132: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", primals_134: "f32[320][1]cuda:0", primals_136: "f32[640, 320, 2, 2][1280, 1, 640, 320]cuda:0", primals_138: "f32[640, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_140: "f32[640][1]cuda:0", primals_142: "f32[2560, 640, 1, 1][640, 1, 640, 640]cuda:0", primals_145: "f32[2560][1]cuda:0", primals_146: "f32[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", primals_148: "f32[640, 1, 7, 7][49, 1, 7, 1]cuda:0", primals_150: "f32[640][1]cuda:0", primals_152: "f32[2560, 640, 1, 1][640, 1, 640, 640]cuda:0", primals_155: "f32[2560][1]cuda:0", primals_156: "f32[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", primals_158: "f32[640][1]cuda:0", primals_160: "f32[1000, 640][640, 1]cuda:0", convolution: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", getitem_1: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", rsqrt: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", permute_1: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", convolution_1: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", getitem_3: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", rsqrt_1: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", permute_3: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", convolution_2: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0", pow_2: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0", add_5: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_6: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0", add_7: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", convolution_4: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", getitem_5: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", rsqrt_2: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", permute_5: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", convolution_5: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0", pow_4: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0", add_11: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0", mul_14: "f32[128, 56, 56, 80][250880, 1, 4480, 56]cuda:0", permute_7: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0", convolution_7: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", convolution_8: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", getitem_9: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", rsqrt_4: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", permute_9: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", convolution_9: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0", pow_6: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0", add_19: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_20: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0", add_21: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", convolution_11: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", getitem_11: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", rsqrt_5: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", permute_11: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", convolution_12: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0", pow_8: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0", add_25: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_26: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0", mul_28: "f32[128, 28, 28, 160][125440, 1, 4480, 28]cuda:0", permute_13: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0", convolution_14: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_15: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_15: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_7: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", permute_15: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_16: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_10: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_33: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_34: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_35: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_18: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_17: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_8: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", permute_17: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_19: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_12: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_39: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_40: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_41: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_21: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_19: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_9: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", permute_19: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_22: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_14: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_45: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_46: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_47: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_24: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_21: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_10: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", permute_21: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_25: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_16: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_51: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_52: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_53: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_27: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_23: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_11: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", permute_23: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_28: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_18: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_57: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_58: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_59: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_30: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_25: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_12: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", permute_25: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_31: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_20: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_63: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_64: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_65: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_33: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_27: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_13: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", permute_27: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_34: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_22: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_69: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_70: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", add_71: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_36: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", getitem_29: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", rsqrt_14: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", permute_29: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_37: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", pow_24: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0", add_75: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_76: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0", mul_78: "f32[128, 14, 14, 320][62720, 1, 4480, 14]cuda:0", permute_31: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0", convolution_39: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", convolution_40: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", getitem_33: "f32[128, 7, 7, 1][49, 1, 7, 7]cuda:0", rsqrt_16: "f32[128, 7, 7, 1][49, 1, 7, 7]cuda:0", permute_33: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", convolution_41: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0", pow_26: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", add_83: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_84: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0", add_85: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", convolution_43: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", getitem_35: "f32[128, 7, 7, 1][49, 1, 7, 7]cuda:0", rsqrt_17: "f32[128, 7, 7, 1][49, 1, 7, 7]cuda:0", permute_35: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0", convolution_44: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0", pow_28: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0", add_89: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", add_90: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0", mul_92: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0", view_28: "f32[128, 640][640, 1]cuda:0", div_14: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0", div_28: "f32[128, 14, 14, 1][196, 1, 14, 14]cuda:0", div_77: "f32[128, 28, 28, 1][784, 1, 28, 28]cuda:0", div_90: "f32[128, 56, 56, 1][3136, 1, 56, 56]cuda:0", tangents_1: "f32[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:219 in forward, code: x = self.fc(x)
        permute_38: "f32[640, 1000][1, 640]cuda:0" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        permute_39: "f32[1000, 640][640, 1]cuda:0" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        mm: "f32[128, 640][640, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_39);  permute_39 = None
        permute_40: "f32[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 640][640, 1]cuda:0" = torch.ops.aten.mm.default(permute_40, view_28);  permute_40 = view_28 = None
        sum_15: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_29: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_15, [1000]);  sum_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:214 in forward, code: x = self.flatten(x)
        view_30: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 640, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_43: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 3, 1]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_95: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_43, primals_158);  primals_158 = None
        mul_96: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, 640)
        sum_16: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_95, [3], True)
        mul_97: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_95, mul_92);  mul_95 = None
        sum_17: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_97, [3], True);  mul_97 = None
        mul_98: "f32[128, 1, 1, 640][640, 640, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_92, sum_17);  sum_17 = None
        sub_20: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_96, sum_16);  mul_96 = sum_16 = None
        sub_21: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_20, mul_98);  sub_20 = mul_98 = None
        mul_99: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_21);  div_14 = sub_21 = None
        mul_100: "f32[128, 1, 1, 640][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_43, mul_92);  mul_92 = None
        sum_18: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_100, [0, 1, 2]);  mul_100 = None
        sum_19: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_43, [0, 1, 2]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_44: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.permute.default(mul_99, [0, 3, 1, 2]);  mul_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze: "f32[128, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_44, 3);  permute_44 = None
        squeeze_1: "f32[128, 640][640, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze, 2);  squeeze = None
        full: "f32[81920][1]cuda:0" = torch.ops.aten.full.default([81920], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "f32[81920][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full, squeeze_1, [128, 640], [640, 1], 0);  full = squeeze_1 = None
        as_strided_5: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 640, 1, 1], [640, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "f32[128, 640, 7, 7][640, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [128, 640, 7, 7]);  as_strided_5 = None
        div_15: "f32[128, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_20: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_15, [0, 2, 3])
        convolution_backward = torch.ops.aten.convolution_backward.default(div_15, add_90, primals_156, [640], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_90 = primals_156 = None
        getitem_38: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = convolution_backward[0]
        getitem_39: "f32[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = convolution_backward[1];  convolution_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_88: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_44, 0.5)
        mul_89: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_44, 0.7071067811865476)
        erf_13: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
        add_88: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_90: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_88, add_88);  mul_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_13: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.div.Tensor(pow_28, add_89)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_91: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_90, div_13)
        mul_101: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Scalar(mul_91, 1);  mul_91 = None
        mul_102: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(getitem_38, mul_101);  mul_101 = None
        view_27: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_155, [1, -1, 1, 1]);  primals_155 = None
        mul_103: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_27, 1);  view_27 = None
        mul_104: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(getitem_38, mul_103);  mul_103 = None
        sum_21: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_38, [0, 2, 3], True)
        sum_22: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_102, [0, 2, 3], True);  mul_102 = None
        mul_105: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, mul_90)
        mul_106: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_104, div_13);  mul_104 = None
        sum_23: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_105, [2, 3], True);  mul_105 = None
        add_94: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, mul_106);  getitem_38 = mul_106 = None
        view_31: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_22, [2560]);  sum_22 = None
        view_32: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_21, [2560]);  sum_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_17: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.div.Tensor(div_13, add_89);  div_13 = None
        neg: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_23)
        mul_107: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.mul.Tensor(neg, div_17);  neg = div_17 = None
        div_18: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_23, add_89);  sum_23 = add_89 = None
        sum_24: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_107, [1], True);  mul_107 = None
        expand_2: "f32[128, 2560, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_24, [128, 2560, 1, 1]);  sum_24 = None
        div_19: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_2, 2560);  expand_2 = None
        add_95: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_18, div_19);  div_18 = div_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_20: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.div.Tensor(mul_90, pow_28);  mul_90 = None
        eq: "b8[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.eq.Scalar(pow_28, 0);  pow_28 = None
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_20);  eq = div_20 = None
        clone_29: "f32[128, 2560, 7, 7][125440, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(where, memory_format = torch.contiguous_format);  where = None
        mul_108: "f32[128, 2560, 7, 7][125440, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_95, clone_29);  add_95 = clone_29 = None
        add_96: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(add_94, mul_108);  add_94 = mul_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_110: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
        mul_111: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_44, convolution_44)
        mul_112: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_111, -0.5);  mul_111 = None
        exp: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.exp.default(mul_112);  mul_112 = None
        mul_113: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
        mul_114: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_44, mul_113);  convolution_44 = mul_113 = None
        add_98: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(mul_110, mul_114);  mul_110 = mul_114 = None
        mul_115: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(add_96, add_98);  add_96 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_25: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_115, [0, 2, 3])
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_115, permute_35, primals_152, [2560], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_115 = permute_35 = primals_152 = None
        getitem_41: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = convolution_backward_1[0]
        getitem_42: "f32[2560, 640, 1, 1][640, 1, 640, 640]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_45: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(getitem_41, [0, 2, 3, 1]);  getitem_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_117: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_45, primals_150);  primals_150 = None
        mul_118: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, 640)
        sum_26: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_117, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_34: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(convolution_43, [0, 2, 3, 1]);  convolution_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_17: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_34, getitem_35);  permute_34 = getitem_35 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_46: "f32[128, 640, 7, 7][31360, 1, 640, 4480]cuda:0" = torch.ops.aten.permute.default(mul_121, [0, 3, 1, 2]);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_30: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_46, [0, 2, 3])
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(permute_46, add_85, primals_148, [640], [1, 1], [3, 3], [1, 1], False, [0, 0], 640, [True, True, False]);  permute_46 = add_85 = primals_148 = None
        getitem_44: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = convolution_backward_2[0]
        getitem_45: "f32[640, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None
        add_99: "f32[128, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(div_15, getitem_44);  div_15 = getitem_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_31: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_99, [0, 2, 3])
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(add_99, add_84, primals_146, [640], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_84 = primals_146 = None
        getitem_47: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = convolution_backward_3[0]
        getitem_48: "f32[640, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_82: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_41, 0.5)
        mul_83: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_41, 0.7071067811865476)
        erf_12: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.erf.default(mul_83);  mul_83 = None
        add_82: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_84: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, add_82);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_12: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.div.Tensor(pow_26, add_83)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_85: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, div_12)
        mul_123: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Scalar(mul_85, 1);  mul_85 = None
        mul_124: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(getitem_47, mul_123);  mul_123 = None
        view_25: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_145, [1, -1, 1, 1]);  primals_145 = None
        mul_125: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_25, 1);  view_25 = None
        mul_126: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(getitem_47, mul_125);  mul_125 = None
        sum_32: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_47, [0, 2, 3], True)
        sum_33: "f32[1, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 2, 3], True);  mul_124 = None
        mul_127: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, mul_84)
        mul_128: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, div_12);  mul_126 = None
        sum_34: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_127, [2, 3], True);  mul_127 = None
        add_100: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(getitem_47, mul_128);  getitem_47 = mul_128 = None
        view_33: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [2560]);  sum_33 = None
        view_34: "f32[2560][1]cuda:0" = torch.ops.aten.reshape.default(sum_32, [2560]);  sum_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_23: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.div.Tensor(div_12, add_83);  div_12 = None
        neg_1: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_34)
        mul_129: "f32[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.mul.Tensor(neg_1, div_23);  neg_1 = div_23 = None
        div_24: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_34, add_83);  sum_34 = add_83 = None
        sum_35: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_129, [1], True);  mul_129 = None
        expand_3: "f32[128, 2560, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_35, [128, 2560, 1, 1]);  sum_35 = None
        div_25: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_3, 2560);  expand_3 = None
        add_101: "f32[128, 2560, 1, 1][2560, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_24, div_25);  div_24 = div_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_26: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.div.Tensor(mul_84, pow_26);  mul_84 = None
        eq_1: "b8[128, 2560, 1, 1][2560, 1, 2560, 2560]cuda:0" = torch.ops.aten.eq.Scalar(pow_26, 0);  pow_26 = None
        where_1: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.where.self(eq_1, full_default, div_26);  eq_1 = div_26 = None
        clone_30: "f32[128, 2560, 7, 7][125440, 49, 7, 1]cuda:0" = torch.ops.aten.clone.default(where_1, memory_format = torch.contiguous_format);  where_1 = None
        mul_130: "f32[128, 2560, 7, 7][125440, 49, 7, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_101, clone_30);  add_101 = clone_30 = None
        add_102: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(add_100, mul_130);  add_100 = mul_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_132: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(add_82, 0.5);  add_82 = None
        mul_133: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_41, convolution_41)
        mul_134: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, -0.5);  mul_133 = None
        exp_1: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.exp.default(mul_134);  mul_134 = None
        mul_135: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
        mul_136: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(convolution_41, mul_135);  convolution_41 = mul_135 = None
        add_104: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.add.Tensor(mul_132, mul_136);  mul_132 = mul_136 = None
        mul_137: "f32[128, 2560, 7, 7][125440, 1, 17920, 2560]cuda:0" = torch.ops.aten.mul.Tensor(add_102, add_104);  add_102 = add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_36: "f32[2560][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_137, [0, 2, 3])
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_137, permute_33, primals_142, [2560], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_137 = permute_33 = primals_142 = None
        getitem_50: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = convolution_backward_4[0]
        getitem_51: "f32[2560, 640, 1, 1][640, 1, 640, 640]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_47: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(getitem_50, [0, 2, 3, 1]);  getitem_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_139: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_47, primals_140);  primals_140 = None
        mul_140: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, 640)
        sum_37: "f32[128, 7, 7, 1][49, 7, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_139, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_32: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.permute.default(convolution_40, [0, 2, 3, 1]);  convolution_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_16: "f32[128, 7, 7, 640][31360, 4480, 640, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_32, getitem_33);  permute_32 = getitem_33 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_48: "f32[128, 640, 7, 7][31360, 1, 640, 4480]cuda:0" = torch.ops.aten.permute.default(mul_143, [0, 3, 1, 2]);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_41: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_48, [0, 2, 3])
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(permute_48, convolution_39, primals_138, [640], [1, 1], [3, 3], [1, 1], False, [0, 0], 640, [True, True, False]);  permute_48 = convolution_39 = primals_138 = None
        getitem_53: "f32[128, 640, 7, 7][31360, 1, 4480, 640]cuda:0" = convolution_backward_5[0]
        getitem_54: "f32[640, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None
        add_105: "f32[128, 640, 7, 7][31360, 49, 7, 1]cuda:0" = torch.ops.aten.add.Tensor(add_99, getitem_53);  add_99 = getitem_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        sum_42: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_105, [0, 2, 3])
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(add_105, permute_31, primals_136, [640], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_105 = permute_31 = primals_136 = None
        getitem_56: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_6[0]
        getitem_57: "f32[640, 320, 2, 2][1280, 1, 640, 320]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_49: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(getitem_56, [0, 2, 3, 1]);  getitem_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_146: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_49, primals_134);  primals_134 = None
        mul_147: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, 320)
        sum_43: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_146, [3], True)
        mul_148: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, mul_78);  mul_146 = None
        sum_44: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_148, [3], True);  mul_148 = None
        mul_149: "f32[128, 14, 14, 320][62720, 1, 4480, 14]cuda:0" = torch.ops.aten.mul.Tensor(mul_78, sum_44);  sum_44 = None
        sub_29: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_147, sum_43);  mul_147 = sum_43 = None
        sub_30: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_29, mul_149);  sub_29 = mul_149 = None
        mul_150: "f32[128, 14, 14, 320][62720, 320, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_28, sub_30);  div_28 = sub_30 = None
        mul_151: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_49, mul_78);  mul_78 = None
        sum_45: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_151, [0, 1, 2]);  mul_151 = None
        sum_46: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_49, [0, 1, 2]);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_50: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(mul_150, [0, 3, 1, 2]);  mul_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_47: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_50, [0, 2, 3])
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(permute_50, add_76, primals_132, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_76 = primals_132 = None
        getitem_59: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_7[0]
        getitem_60: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_74: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_37, 0.5)
        mul_75: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_37, 0.7071067811865476)
        erf_11: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_74: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_76: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, add_74);  mul_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_11: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_24, add_75)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_77: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_76, div_11)
        mul_152: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_77, 1);  mul_77 = None
        mul_153: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_59, mul_152);  mul_152 = None
        view_23: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_131, [1, -1, 1, 1]);  primals_131 = None
        mul_154: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_23, 1);  view_23 = None
        mul_155: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_59, mul_154);  mul_154 = None
        sum_48: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_59, [0, 2, 3], True)
        sum_49: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 2, 3], True);  mul_153 = None
        mul_156: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, mul_76)
        mul_157: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, div_11);  mul_155 = None
        sum_50: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_156, [2, 3], True);  mul_156 = None
        add_106: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(getitem_59, mul_157);  getitem_59 = mul_157 = None
        view_35: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [1280]);  sum_49 = None
        view_36: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [1280]);  sum_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_30: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_11, add_75);  div_11 = None
        neg_2: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_50)
        mul_158: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_2, div_30);  neg_2 = div_30 = None
        div_31: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_50, add_75);  sum_50 = add_75 = None
        sum_51: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_158, [1], True);  mul_158 = None
        expand_4: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_51, [128, 1280, 1, 1]);  sum_51 = None
        div_32: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_4, 1280);  expand_4 = None
        add_107: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_31, div_32);  div_31 = div_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_33: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(mul_76, pow_24);  mul_76 = None
        eq_2: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_24, 0);  pow_24 = None
        where_2: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_2, full_default, div_33);  eq_2 = div_33 = None
        clone_31: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_2, memory_format = torch.contiguous_format);  where_2 = None
        mul_159: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_107, clone_31);  add_107 = clone_31 = None
        add_108: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_106, mul_159);  add_106 = mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_161: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_74, 0.5);  add_74 = None
        mul_162: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_37, convolution_37)
        mul_163: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, -0.5);  mul_162 = None
        exp_2: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_163);  mul_163 = None
        mul_164: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
        mul_165: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_37, mul_164);  convolution_37 = mul_164 = None
        add_110: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_161, mul_165);  mul_161 = mul_165 = None
        mul_166: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_108, add_110);  add_108 = add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_52: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_166, [0, 2, 3])
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_166, permute_29, primals_128, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_166 = permute_29 = primals_128 = None
        getitem_62: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_8[0]
        getitem_63: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_51: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(getitem_62, [0, 2, 3, 1]);  getitem_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_168: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_51, primals_126);  primals_126 = None
        mul_169: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_168, 320)
        sum_53: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_168, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_28: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_36, [0, 2, 3, 1]);  convolution_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_14: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_28, getitem_29);  permute_28 = getitem_29 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_52: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(mul_172, [0, 3, 1, 2]);  mul_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_57: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_52, [0, 2, 3])
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(permute_52, add_71, primals_124, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_52 = add_71 = primals_124 = None
        getitem_65: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_9[0]
        getitem_66: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None
        add_111: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(permute_50, getitem_65);  permute_50 = getitem_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_58: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_111, [0, 2, 3])
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(add_111, add_70, primals_122, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_70 = primals_122 = None
        getitem_68: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_10[0]
        getitem_69: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_68: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_34, 0.5)
        mul_69: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_34, 0.7071067811865476)
        erf_10: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
        add_68: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_70: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, add_68);  mul_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_10: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_22, add_69)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_71: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, div_10)
        mul_174: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_71, 1);  mul_71 = None
        mul_175: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_68, mul_174);  mul_174 = None
        view_21: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_121, [1, -1, 1, 1]);  primals_121 = None
        mul_176: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_21, 1);  view_21 = None
        mul_177: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_68, mul_176);  mul_176 = None
        sum_59: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_68, [0, 2, 3], True)
        sum_60: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 2, 3], True);  mul_175 = None
        mul_178: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, mul_70)
        mul_179: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, div_10);  mul_177 = None
        sum_61: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_178, [2, 3], True);  mul_178 = None
        add_112: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, mul_179);  getitem_68 = mul_179 = None
        view_37: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_60, [1280]);  sum_60 = None
        view_38: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_59, [1280]);  sum_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_36: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_10, add_69);  div_10 = None
        neg_3: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_61)
        mul_180: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_3, div_36);  neg_3 = div_36 = None
        div_37: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_61, add_69);  sum_61 = add_69 = None
        sum_62: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_180, [1], True);  mul_180 = None
        expand_5: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_62, [128, 1280, 1, 1]);  sum_62 = None
        div_38: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_5, 1280);  expand_5 = None
        add_113: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_37, div_38);  div_37 = div_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_39: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(mul_70, pow_22);  mul_70 = None
        eq_3: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_22, 0);  pow_22 = None
        where_3: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_3, full_default, div_39);  eq_3 = div_39 = None
        clone_32: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_3, memory_format = torch.contiguous_format);  where_3 = None
        mul_181: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, clone_32);  add_113 = clone_32 = None
        add_114: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_112, mul_181);  add_112 = mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_183: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_68, 0.5);  add_68 = None
        mul_184: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_34, convolution_34)
        mul_185: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_184, -0.5);  mul_184 = None
        exp_3: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_185);  mul_185 = None
        mul_186: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
        mul_187: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_34, mul_186);  convolution_34 = mul_186 = None
        add_116: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_183, mul_187);  mul_183 = mul_187 = None
        mul_188: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_114, add_116);  add_114 = add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_63: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_188, [0, 2, 3])
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_188, permute_27, primals_118, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_188 = permute_27 = primals_118 = None
        getitem_71: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_11[0]
        getitem_72: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_53: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(getitem_71, [0, 2, 3, 1]);  getitem_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_190: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_53, primals_116);  primals_116 = None
        mul_191: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_190, 320)
        sum_64: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_190, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_26: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_33, [0, 2, 3, 1]);  convolution_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_13: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_26, getitem_27);  permute_26 = getitem_27 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_54: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(mul_194, [0, 3, 1, 2]);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_68: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_54, [0, 2, 3])
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(permute_54, add_65, primals_114, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_54 = add_65 = primals_114 = None
        getitem_74: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_12[0]
        getitem_75: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None
        add_117: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_111, getitem_74);  add_111 = getitem_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_69: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_117, [0, 2, 3])
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(add_117, add_64, primals_112, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_64 = primals_112 = None
        getitem_77: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_13[0]
        getitem_78: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_62: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_31, 0.5)
        mul_63: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_31, 0.7071067811865476)
        erf_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
        add_62: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_64: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, add_62);  mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_9: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_20, add_63)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_65: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, div_9)
        mul_196: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_65, 1);  mul_65 = None
        mul_197: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_77, mul_196);  mul_196 = None
        view_19: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_111, [1, -1, 1, 1]);  primals_111 = None
        mul_198: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_19, 1);  view_19 = None
        mul_199: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_77, mul_198);  mul_198 = None
        sum_70: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_77, [0, 2, 3], True)
        sum_71: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 2, 3], True);  mul_197 = None
        mul_200: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, mul_64)
        mul_201: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, div_9);  mul_199 = None
        sum_72: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_200, [2, 3], True);  mul_200 = None
        add_118: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(getitem_77, mul_201);  getitem_77 = mul_201 = None
        view_39: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_71, [1280]);  sum_71 = None
        view_40: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [1280]);  sum_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_42: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_9, add_63);  div_9 = None
        neg_4: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_72)
        mul_202: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_4, div_42);  neg_4 = div_42 = None
        div_43: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_72, add_63);  sum_72 = add_63 = None
        sum_73: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_202, [1], True);  mul_202 = None
        expand_6: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_73, [128, 1280, 1, 1]);  sum_73 = None
        div_44: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_6, 1280);  expand_6 = None
        add_119: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_43, div_44);  div_43 = div_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_45: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(mul_64, pow_20);  mul_64 = None
        eq_4: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_20, 0);  pow_20 = None
        where_4: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_4, full_default, div_45);  eq_4 = div_45 = None
        clone_33: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_4, memory_format = torch.contiguous_format);  where_4 = None
        mul_203: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_119, clone_33);  add_119 = clone_33 = None
        add_120: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_118, mul_203);  add_118 = mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_205: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
        mul_206: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_31, convolution_31)
        mul_207: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_206, -0.5);  mul_206 = None
        exp_4: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_207);  mul_207 = None
        mul_208: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
        mul_209: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_31, mul_208);  convolution_31 = mul_208 = None
        add_122: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_205, mul_209);  mul_205 = mul_209 = None
        mul_210: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_120, add_122);  add_120 = add_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_74: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 2, 3])
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_210, permute_25, primals_108, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_210 = permute_25 = primals_108 = None
        getitem_80: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_14[0]
        getitem_81: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_55: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(getitem_80, [0, 2, 3, 1]);  getitem_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_212: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_55, primals_106);  primals_106 = None
        mul_213: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_212, 320)
        sum_75: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_212, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_24: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_30, [0, 2, 3, 1]);  convolution_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_12: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_24, getitem_25);  permute_24 = getitem_25 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_56: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(mul_216, [0, 3, 1, 2]);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_79: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_56, [0, 2, 3])
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(permute_56, add_59, primals_104, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_56 = add_59 = primals_104 = None
        getitem_83: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_15[0]
        getitem_84: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None
        add_123: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_117, getitem_83);  add_117 = getitem_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_80: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_123, [0, 2, 3])
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(add_123, add_58, primals_102, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_58 = primals_102 = None
        getitem_86: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_16[0]
        getitem_87: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_56: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_28, 0.5)
        mul_57: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_28, 0.7071067811865476)
        erf_8: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_57);  mul_57 = None
        add_56: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_58: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, add_56);  mul_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_8: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_18, add_57)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_59: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, div_8)
        mul_218: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_59, 1);  mul_59 = None
        mul_219: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_86, mul_218);  mul_218 = None
        view_17: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_101, [1, -1, 1, 1]);  primals_101 = None
        mul_220: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_17, 1);  view_17 = None
        mul_221: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_86, mul_220);  mul_220 = None
        sum_81: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_86, [0, 2, 3], True)
        sum_82: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_219, [0, 2, 3], True);  mul_219 = None
        mul_222: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, mul_58)
        mul_223: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, div_8);  mul_221 = None
        sum_83: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_222, [2, 3], True);  mul_222 = None
        add_124: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, mul_223);  getitem_86 = mul_223 = None
        view_41: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [1280]);  sum_82 = None
        view_42: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [1280]);  sum_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_48: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_8, add_57);  div_8 = None
        neg_5: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_83)
        mul_224: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_5, div_48);  neg_5 = div_48 = None
        div_49: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_83, add_57);  sum_83 = add_57 = None
        sum_84: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_224, [1], True);  mul_224 = None
        expand_7: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_84, [128, 1280, 1, 1]);  sum_84 = None
        div_50: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_7, 1280);  expand_7 = None
        add_125: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_49, div_50);  div_49 = div_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_51: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(mul_58, pow_18);  mul_58 = None
        eq_5: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_18, 0);  pow_18 = None
        where_5: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_5, full_default, div_51);  eq_5 = div_51 = None
        clone_34: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_5, memory_format = torch.contiguous_format);  where_5 = None
        mul_225: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_125, clone_34);  add_125 = clone_34 = None
        add_126: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_124, mul_225);  add_124 = mul_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_227: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_228: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_28, convolution_28)
        mul_229: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_228, -0.5);  mul_228 = None
        exp_5: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_229);  mul_229 = None
        mul_230: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
        mul_231: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_28, mul_230);  convolution_28 = mul_230 = None
        add_128: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_227, mul_231);  mul_227 = mul_231 = None
        mul_232: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_126, add_128);  add_126 = add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_85: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 2, 3])
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_232, permute_23, primals_98, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_232 = permute_23 = primals_98 = None
        getitem_89: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_17[0]
        getitem_90: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_57: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(getitem_89, [0, 2, 3, 1]);  getitem_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_234: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_57, primals_96);  primals_96 = None
        mul_235: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_234, 320)
        sum_86: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_234, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_22: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_27, [0, 2, 3, 1]);  convolution_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_11: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_22, getitem_23);  permute_22 = getitem_23 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_58: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(mul_238, [0, 3, 1, 2]);  mul_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_90: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_58, [0, 2, 3])
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(permute_58, add_53, primals_94, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_58 = add_53 = primals_94 = None
        getitem_92: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_18[0]
        getitem_93: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        add_129: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_123, getitem_92);  add_123 = getitem_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_91: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_129, [0, 2, 3])
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(add_129, add_52, primals_92, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_52 = primals_92 = None
        getitem_95: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_19[0]
        getitem_96: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_50: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_25, 0.5)
        mul_51: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_25, 0.7071067811865476)
        erf_7: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_50: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_52: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, add_50);  mul_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_7: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_16, add_51)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_53: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, div_7)
        mul_240: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_53, 1);  mul_53 = None
        mul_241: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_95, mul_240);  mul_240 = None
        view_15: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_91, [1, -1, 1, 1]);  primals_91 = None
        mul_242: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_15, 1);  view_15 = None
        mul_243: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_95, mul_242);  mul_242 = None
        sum_92: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_95, [0, 2, 3], True)
        sum_93: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3], True);  mul_241 = None
        mul_244: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, mul_52)
        mul_245: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, div_7);  mul_243 = None
        sum_94: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_244, [2, 3], True);  mul_244 = None
        add_130: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(getitem_95, mul_245);  getitem_95 = mul_245 = None
        view_43: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [1280]);  sum_93 = None
        view_44: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_92, [1280]);  sum_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_54: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_7, add_51);  div_7 = None
        neg_6: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_94)
        mul_246: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_6, div_54);  neg_6 = div_54 = None
        div_55: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_94, add_51);  sum_94 = add_51 = None
        sum_95: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_246, [1], True);  mul_246 = None
        expand_8: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_95, [128, 1280, 1, 1]);  sum_95 = None
        div_56: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_8, 1280);  expand_8 = None
        add_131: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_55, div_56);  div_55 = div_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_57: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(mul_52, pow_16);  mul_52 = None
        eq_6: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_16, 0);  pow_16 = None
        where_6: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_6, full_default, div_57);  eq_6 = div_57 = None
        clone_35: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_6, memory_format = torch.contiguous_format);  where_6 = None
        mul_247: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_131, clone_35);  add_131 = clone_35 = None
        add_132: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_130, mul_247);  add_130 = mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_249: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_50, 0.5);  add_50 = None
        mul_250: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_25, convolution_25)
        mul_251: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, -0.5);  mul_250 = None
        exp_6: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_251);  mul_251 = None
        mul_252: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
        mul_253: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_25, mul_252);  convolution_25 = mul_252 = None
        add_134: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_249, mul_253);  mul_249 = mul_253 = None
        mul_254: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_132, add_134);  add_132 = add_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_96: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 2, 3])
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_254, permute_21, primals_88, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_254 = permute_21 = primals_88 = None
        getitem_98: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_20[0]
        getitem_99: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_59: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(getitem_98, [0, 2, 3, 1]);  getitem_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_256: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_59, primals_86);  primals_86 = None
        mul_257: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, 320)
        sum_97: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_256, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_20: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_24, [0, 2, 3, 1]);  convolution_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_10: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_20, getitem_21);  permute_20 = getitem_21 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_60: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(mul_260, [0, 3, 1, 2]);  mul_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_101: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_60, [0, 2, 3])
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(permute_60, add_47, primals_84, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_60 = add_47 = primals_84 = None
        getitem_101: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_21[0]
        getitem_102: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None
        add_135: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_129, getitem_101);  add_129 = getitem_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_102: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_135, [0, 2, 3])
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(add_135, add_46, primals_82, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_46 = primals_82 = None
        getitem_104: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_22[0]
        getitem_105: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_44: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_22, 0.5)
        mul_45: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_22, 0.7071067811865476)
        erf_6: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_45);  mul_45 = None
        add_44: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_46: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, add_44);  mul_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_6: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_14, add_45)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_47: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, div_6)
        mul_262: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_47, 1);  mul_47 = None
        mul_263: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_104, mul_262);  mul_262 = None
        view_13: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_81, [1, -1, 1, 1]);  primals_81 = None
        mul_264: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_13, 1);  view_13 = None
        mul_265: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_104, mul_264);  mul_264 = None
        sum_103: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_104, [0, 2, 3], True)
        sum_104: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 2, 3], True);  mul_263 = None
        mul_266: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_265, mul_46)
        mul_267: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_265, div_6);  mul_265 = None
        sum_105: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [2, 3], True);  mul_266 = None
        add_136: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(getitem_104, mul_267);  getitem_104 = mul_267 = None
        view_45: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_104, [1280]);  sum_104 = None
        view_46: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_103, [1280]);  sum_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_60: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_6, add_45);  div_6 = None
        neg_7: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_105)
        mul_268: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_7, div_60);  neg_7 = div_60 = None
        div_61: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_105, add_45);  sum_105 = add_45 = None
        sum_106: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_268, [1], True);  mul_268 = None
        expand_9: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_106, [128, 1280, 1, 1]);  sum_106 = None
        div_62: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_9, 1280);  expand_9 = None
        add_137: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_61, div_62);  div_61 = div_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_63: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(mul_46, pow_14);  mul_46 = None
        eq_7: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_14, 0);  pow_14 = None
        where_7: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_7, full_default, div_63);  eq_7 = div_63 = None
        clone_36: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_7, memory_format = torch.contiguous_format);  where_7 = None
        mul_269: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_137, clone_36);  add_137 = clone_36 = None
        add_138: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_136, mul_269);  add_136 = mul_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_271: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_44, 0.5);  add_44 = None
        mul_272: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_22, convolution_22)
        mul_273: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_272, -0.5);  mul_272 = None
        exp_7: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_273);  mul_273 = None
        mul_274: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
        mul_275: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_22, mul_274);  convolution_22 = mul_274 = None
        add_140: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_271, mul_275);  mul_271 = mul_275 = None
        mul_276: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_138, add_140);  add_138 = add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_107: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 2, 3])
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_276, permute_19, primals_78, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_276 = permute_19 = primals_78 = None
        getitem_107: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_23[0]
        getitem_108: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_61: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(getitem_107, [0, 2, 3, 1]);  getitem_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_278: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_61, primals_76);  primals_76 = None
        mul_279: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_278, 320)
        sum_108: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_278, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_18: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_21, [0, 2, 3, 1]);  convolution_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_9: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_18, getitem_19);  permute_18 = getitem_19 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_62: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(mul_282, [0, 3, 1, 2]);  mul_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_112: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_62, [0, 2, 3])
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(permute_62, add_41, primals_74, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_62 = add_41 = primals_74 = None
        getitem_110: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_24[0]
        getitem_111: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        add_141: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_135, getitem_110);  add_135 = getitem_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_113: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_141, [0, 2, 3])
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(add_141, add_40, primals_72, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_40 = primals_72 = None
        getitem_113: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_25[0]
        getitem_114: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_38: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_19, 0.5)
        mul_39: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_19, 0.7071067811865476)
        erf_5: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
        add_38: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_40: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, add_38);  mul_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_5: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_12, add_39)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_41: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, div_5)
        mul_284: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_41, 1);  mul_41 = None
        mul_285: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_113, mul_284);  mul_284 = None
        view_11: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_71, [1, -1, 1, 1]);  primals_71 = None
        mul_286: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_11, 1);  view_11 = None
        mul_287: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_113, mul_286);  mul_286 = None
        sum_114: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3], True)
        sum_115: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 2, 3], True);  mul_285 = None
        mul_288: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, mul_40)
        mul_289: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, div_5);  mul_287 = None
        sum_116: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_288, [2, 3], True);  mul_288 = None
        add_142: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(getitem_113, mul_289);  getitem_113 = mul_289 = None
        view_47: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_115, [1280]);  sum_115 = None
        view_48: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_114, [1280]);  sum_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_66: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_5, add_39);  div_5 = None
        neg_8: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_116)
        mul_290: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_8, div_66);  neg_8 = div_66 = None
        div_67: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_116, add_39);  sum_116 = add_39 = None
        sum_117: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_290, [1], True);  mul_290 = None
        expand_10: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_117, [128, 1280, 1, 1]);  sum_117 = None
        div_68: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_10, 1280);  expand_10 = None
        add_143: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_67, div_68);  div_67 = div_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_69: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(mul_40, pow_12);  mul_40 = None
        eq_8: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_12, 0);  pow_12 = None
        where_8: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_8, full_default, div_69);  eq_8 = div_69 = None
        clone_37: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_8, memory_format = torch.contiguous_format);  where_8 = None
        mul_291: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_143, clone_37);  add_143 = clone_37 = None
        add_144: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_142, mul_291);  add_142 = mul_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_293: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_38, 0.5);  add_38 = None
        mul_294: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_19, convolution_19)
        mul_295: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_294, -0.5);  mul_294 = None
        exp_8: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_295);  mul_295 = None
        mul_296: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
        mul_297: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_19, mul_296);  convolution_19 = mul_296 = None
        add_146: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_293, mul_297);  mul_293 = mul_297 = None
        mul_298: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_144, add_146);  add_144 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_118: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 2, 3])
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_298, permute_17, primals_68, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_298 = permute_17 = primals_68 = None
        getitem_116: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_26[0]
        getitem_117: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_63: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(getitem_116, [0, 2, 3, 1]);  getitem_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_300: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_63, primals_66);  primals_66 = None
        mul_301: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_300, 320)
        sum_119: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_300, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_16: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_18, [0, 2, 3, 1]);  convolution_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_8: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_16, getitem_17);  permute_16 = getitem_17 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_64: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(mul_304, [0, 3, 1, 2]);  mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_123: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_64, [0, 2, 3])
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(permute_64, add_35, primals_64, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_64 = add_35 = primals_64 = None
        getitem_119: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_27[0]
        getitem_120: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        add_147: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_141, getitem_119);  add_141 = getitem_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_124: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_147, [0, 2, 3])
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(add_147, add_34, primals_62, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_34 = primals_62 = None
        getitem_122: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = convolution_backward_28[0]
        getitem_123: "f32[320, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_32: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_16, 0.5)
        mul_33: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_16, 0.7071067811865476)
        erf_4: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_32: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, add_32);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_4: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(pow_10, add_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_35: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, div_4)
        mul_306: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Scalar(mul_35, 1);  mul_35 = None
        mul_307: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_122, mul_306);  mul_306 = None
        view_9: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_61, [1, -1, 1, 1]);  primals_61 = None
        mul_308: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_9, 1);  view_9 = None
        mul_309: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(getitem_122, mul_308);  mul_308 = None
        sum_125: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_122, [0, 2, 3], True)
        sum_126: "f32[1, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 2, 3], True);  mul_307 = None
        mul_310: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, mul_34)
        mul_311: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, div_4);  mul_309 = None
        sum_127: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_310, [2, 3], True);  mul_310 = None
        add_148: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(getitem_122, mul_311);  getitem_122 = mul_311 = None
        view_49: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_126, [1280]);  sum_126 = None
        view_50: "f32[1280][1]cuda:0" = torch.ops.aten.reshape.default(sum_125, [1280]);  sum_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_72: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.div.Tensor(div_4, add_33);  div_4 = None
        neg_9: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_127)
        mul_312: "f32[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.mul.Tensor(neg_9, div_72);  neg_9 = div_72 = None
        div_73: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_127, add_33);  sum_127 = add_33 = None
        sum_128: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_312, [1], True);  mul_312 = None
        expand_11: "f32[128, 1280, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_128, [128, 1280, 1, 1]);  sum_128 = None
        div_74: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_11, 1280);  expand_11 = None
        add_149: "f32[128, 1280, 1, 1][1280, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_73, div_74);  div_73 = div_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_75: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.div.Tensor(mul_34, pow_10);  mul_34 = None
        eq_9: "b8[128, 1280, 1, 1][1280, 1, 1280, 1280]cuda:0" = torch.ops.aten.eq.Scalar(pow_10, 0);  pow_10 = None
        where_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.where.self(eq_9, full_default, div_75);  eq_9 = div_75 = None
        clone_38: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.clone.default(where_9, memory_format = torch.contiguous_format);  where_9 = None
        mul_313: "f32[128, 1280, 14, 14][250880, 196, 14, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_149, clone_38);  add_149 = clone_38 = None
        add_150: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(add_148, mul_313);  add_148 = mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_315: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
        mul_316: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_16, convolution_16)
        mul_317: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(mul_316, -0.5);  mul_316 = None
        exp_9: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.exp.default(mul_317);  mul_317 = None
        mul_318: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
        mul_319: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(convolution_16, mul_318);  convolution_16 = mul_318 = None
        add_152: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.add.Tensor(mul_315, mul_319);  mul_315 = mul_319 = None
        mul_320: "f32[128, 1280, 14, 14][250880, 1, 17920, 1280]cuda:0" = torch.ops.aten.mul.Tensor(add_150, add_152);  add_150 = add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_129: "f32[1280][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_320, [0, 2, 3])
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_320, permute_15, primals_58, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_320 = permute_15 = primals_58 = None
        getitem_125: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_29[0]
        getitem_126: "f32[1280, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_65: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(getitem_125, [0, 2, 3, 1]);  getitem_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_322: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_65, primals_56);  primals_56 = None
        mul_323: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, 320)
        sum_130: "f32[128, 14, 14, 1][196, 14, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_322, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_14: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.permute.default(convolution_15, [0, 2, 3, 1]);  convolution_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_7: "f32[128, 14, 14, 320][62720, 4480, 320, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_14, getitem_15);  permute_14 = getitem_15 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_66: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.permute.default(mul_326, [0, 3, 1, 2]);  mul_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_134: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_66, [0, 2, 3])
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(permute_66, convolution_14, primals_54, [320], [1, 1], [3, 3], [1, 1], False, [0, 0], 320, [True, True, False]);  permute_66 = convolution_14 = primals_54 = None
        getitem_128: "f32[128, 320, 14, 14][62720, 1, 4480, 320]cuda:0" = convolution_backward_30[0]
        getitem_129: "f32[320, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None
        add_153: "f32[128, 320, 14, 14][62720, 1, 320, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_147, getitem_128);  add_147 = getitem_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        sum_135: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_153, [0, 2, 3])
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(add_153, permute_13, primals_52, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_153 = permute_13 = primals_52 = None
        getitem_131: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = convolution_backward_31[0]
        getitem_132: "f32[320, 160, 2, 2][640, 1, 320, 160]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_67: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(getitem_131, [0, 2, 3, 1]);  getitem_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_329: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_67, primals_50);  primals_50 = None
        mul_330: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, 160)
        sum_136: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_329, [3], True)
        mul_331: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, mul_28);  mul_329 = None
        sum_137: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_331, [3], True);  mul_331 = None
        mul_332: "f32[128, 28, 28, 160][125440, 1, 4480, 28]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, sum_137);  sum_137 = None
        sub_56: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_330, sum_136);  mul_330 = sum_136 = None
        sub_57: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, mul_332);  sub_56 = mul_332 = None
        mul_333: "f32[128, 28, 28, 160][125440, 160, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_77, sub_57);  div_77 = sub_57 = None
        mul_334: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_67, mul_28);  mul_28 = None
        sum_138: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1, 2]);  mul_334 = None
        sum_139: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_67, [0, 1, 2]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_68: "f32[128, 160, 28, 28][125440, 1, 160, 4480]cuda:0" = torch.ops.aten.permute.default(mul_333, [0, 3, 1, 2]);  mul_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_140: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_68, [0, 2, 3])
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(permute_68, add_26, primals_48, [160], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_26 = primals_48 = None
        getitem_134: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = convolution_backward_32[0]
        getitem_135: "f32[160, 640, 1, 1][640, 1, 640, 640]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_24: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_12, 0.5)
        mul_25: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_12, 0.7071067811865476)
        erf_3: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_24: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_26: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_24, add_24);  mul_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_3: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.div.Tensor(pow_8, add_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_27: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, div_3)
        mul_335: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Scalar(mul_27, 1);  mul_27 = None
        mul_336: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(getitem_134, mul_335);  mul_335 = None
        view_7: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_47, [1, -1, 1, 1]);  primals_47 = None
        mul_337: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_7, 1);  view_7 = None
        mul_338: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(getitem_134, mul_337);  mul_337 = None
        sum_141: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_134, [0, 2, 3], True)
        sum_142: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_336, [0, 2, 3], True);  mul_336 = None
        mul_339: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, mul_26)
        mul_340: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, div_3);  mul_338 = None
        sum_143: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [2, 3], True);  mul_339 = None
        add_154: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(getitem_134, mul_340);  getitem_134 = mul_340 = None
        view_51: "f32[640][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [640]);  sum_142 = None
        view_52: "f32[640][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [640]);  sum_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_79: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.div.Tensor(div_3, add_25);  div_3 = None
        neg_10: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_143)
        mul_341: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.mul.Tensor(neg_10, div_79);  neg_10 = div_79 = None
        div_80: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_143, add_25);  sum_143 = add_25 = None
        sum_144: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_341, [1], True);  mul_341 = None
        expand_12: "f32[128, 640, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_144, [128, 640, 1, 1]);  sum_144 = None
        div_81: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_12, 640);  expand_12 = None
        add_155: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_80, div_81);  div_80 = div_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_82: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.div.Tensor(mul_26, pow_8);  mul_26 = None
        eq_10: "b8[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.eq.Scalar(pow_8, 0);  pow_8 = None
        where_10: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.where.self(eq_10, full_default, div_82);  eq_10 = div_82 = None
        clone_39: "f32[128, 640, 28, 28][501760, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(where_10, memory_format = torch.contiguous_format);  where_10 = None
        mul_342: "f32[128, 640, 28, 28][501760, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_155, clone_39);  add_155 = clone_39 = None
        add_156: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(add_154, mul_342);  add_154 = mul_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_344: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
        mul_345: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_12, convolution_12)
        mul_346: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_345, -0.5);  mul_345 = None
        exp_10: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.exp.default(mul_346);  mul_346 = None
        mul_347: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
        mul_348: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_12, mul_347);  convolution_12 = mul_347 = None
        add_158: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_344, mul_348);  mul_344 = mul_348 = None
        mul_349: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(add_156, add_158);  add_156 = add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_145: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 2, 3])
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_349, permute_11, primals_44, [640], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_349 = permute_11 = primals_44 = None
        getitem_137: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = convolution_backward_33[0]
        getitem_138: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_69: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(getitem_137, [0, 2, 3, 1]);  getitem_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_351: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_69, primals_42);  primals_42 = None
        mul_352: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_351, 160)
        sum_146: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_351, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_10: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convolution_11, [0, 2, 3, 1]);  convolution_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_5: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_10, getitem_11);  permute_10 = getitem_11 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_70: "f32[128, 160, 28, 28][125440, 1, 160, 4480]cuda:0" = torch.ops.aten.permute.default(mul_355, [0, 3, 1, 2]);  mul_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_150: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_70, [0, 2, 3])
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(permute_70, add_21, primals_40, [160], [1, 1], [3, 3], [1, 1], False, [0, 0], 160, [True, True, False]);  permute_70 = add_21 = primals_40 = None
        getitem_140: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = convolution_backward_34[0]
        getitem_141: "f32[160, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        add_159: "f32[128, 160, 28, 28][125440, 1, 160, 4480]cuda:0" = torch.ops.aten.add.Tensor(permute_68, getitem_140);  permute_68 = getitem_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_151: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_159, [0, 2, 3])
        convolution_backward_35 = torch.ops.aten.convolution_backward.default(add_159, add_20, primals_38, [160], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_20 = primals_38 = None
        getitem_143: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = convolution_backward_35[0]
        getitem_144: "f32[160, 640, 1, 1][640, 1, 640, 640]cuda:0" = convolution_backward_35[1];  convolution_backward_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_18: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_9, 0.5)
        mul_19: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_9, 0.7071067811865476)
        erf_2: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_18: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, add_18);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_2: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.div.Tensor(pow_6, add_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_21: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, div_2)
        mul_357: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Scalar(mul_21, 1);  mul_21 = None
        mul_358: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(getitem_143, mul_357);  mul_357 = None
        view_5: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_37, [1, -1, 1, 1]);  primals_37 = None
        mul_359: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_5, 1);  view_5 = None
        mul_360: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(getitem_143, mul_359);  mul_359 = None
        sum_152: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_143, [0, 2, 3], True)
        sum_153: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 2, 3], True);  mul_358 = None
        mul_361: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, mul_20)
        mul_362: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, div_2);  mul_360 = None
        sum_154: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_361, [2, 3], True);  mul_361 = None
        add_160: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(getitem_143, mul_362);  getitem_143 = mul_362 = None
        view_53: "f32[640][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [640]);  sum_153 = None
        view_54: "f32[640][1]cuda:0" = torch.ops.aten.reshape.default(sum_152, [640]);  sum_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_85: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.div.Tensor(div_2, add_19);  div_2 = None
        neg_11: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_154)
        mul_363: "f32[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.mul.Tensor(neg_11, div_85);  neg_11 = div_85 = None
        div_86: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_154, add_19);  sum_154 = add_19 = None
        sum_155: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_363, [1], True);  mul_363 = None
        expand_13: "f32[128, 640, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_155, [128, 640, 1, 1]);  sum_155 = None
        div_87: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_13, 640);  expand_13 = None
        add_161: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_86, div_87);  div_86 = div_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_88: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.div.Tensor(mul_20, pow_6);  mul_20 = None
        eq_11: "b8[128, 640, 1, 1][640, 1, 640, 640]cuda:0" = torch.ops.aten.eq.Scalar(pow_6, 0);  pow_6 = None
        where_11: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.where.self(eq_11, full_default, div_88);  eq_11 = div_88 = None
        clone_40: "f32[128, 640, 28, 28][501760, 784, 28, 1]cuda:0" = torch.ops.aten.clone.default(where_11, memory_format = torch.contiguous_format);  where_11 = None
        mul_364: "f32[128, 640, 28, 28][501760, 784, 28, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_161, clone_40);  add_161 = clone_40 = None
        add_162: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(add_160, mul_364);  add_160 = mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_366: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
        mul_367: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_9, convolution_9)
        mul_368: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, -0.5);  mul_367 = None
        exp_11: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.exp.default(mul_368);  mul_368 = None
        mul_369: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
        mul_370: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(convolution_9, mul_369);  convolution_9 = mul_369 = None
        add_164: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_366, mul_370);  mul_366 = mul_370 = None
        mul_371: "f32[128, 640, 28, 28][501760, 1, 17920, 640]cuda:0" = torch.ops.aten.mul.Tensor(add_162, add_164);  add_162 = add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_156: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_371, [0, 2, 3])
        convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_371, permute_9, primals_34, [640], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_371 = permute_9 = primals_34 = None
        getitem_146: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = convolution_backward_36[0]
        getitem_147: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_36[1];  convolution_backward_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_71: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(getitem_146, [0, 2, 3, 1]);  getitem_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_373: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_71, primals_32);  primals_32 = None
        mul_374: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_373, 160)
        sum_157: "f32[128, 28, 28, 1][784, 28, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_373, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_8: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.permute.default(convolution_8, [0, 2, 3, 1]);  convolution_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_4: "f32[128, 28, 28, 160][125440, 4480, 160, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_8, getitem_9);  permute_8 = getitem_9 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_72: "f32[128, 160, 28, 28][125440, 1, 160, 4480]cuda:0" = torch.ops.aten.permute.default(mul_377, [0, 3, 1, 2]);  mul_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_161: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_72, [0, 2, 3])
        convolution_backward_37 = torch.ops.aten.convolution_backward.default(permute_72, convolution_7, primals_30, [160], [1, 1], [3, 3], [1, 1], False, [0, 0], 160, [True, True, False]);  permute_72 = convolution_7 = primals_30 = None
        getitem_149: "f32[128, 160, 28, 28][125440, 1, 4480, 160]cuda:0" = convolution_backward_37[0]
        getitem_150: "f32[160, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_37[1];  convolution_backward_37 = None
        add_165: "f32[128, 160, 28, 28][125440, 1, 160, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_159, getitem_149);  add_159 = getitem_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:302 in forward, code: x = self.downsample(x)
        sum_162: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_165, [0, 2, 3])
        convolution_backward_38 = torch.ops.aten.convolution_backward.default(add_165, permute_7, primals_28, [160], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_165 = permute_7 = primals_28 = None
        getitem_152: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = convolution_backward_38[0]
        getitem_153: "f32[160, 80, 2, 2][320, 1, 160, 80]cuda:0" = convolution_backward_38[1];  convolution_backward_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_73: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_152, [0, 2, 3, 1]);  getitem_152 = None

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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_167: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_74, [0, 2, 3])
        convolution_backward_39 = torch.ops.aten.convolution_backward.default(permute_74, add_12, primals_24, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_12 = primals_24 = None
        getitem_155: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = convolution_backward_39[0]
        getitem_156: "f32[80, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_39[1];  convolution_backward_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_10: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_5, 0.5)
        mul_11: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_5, 0.7071067811865476)
        erf_1: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.erf.default(mul_11);  mul_11 = None
        add_10: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, add_10);  mul_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_1: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.div.Tensor(pow_4, add_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_13: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, div_1)
        mul_386: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Scalar(mul_13, 1);  mul_13 = None
        mul_387: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(getitem_155, mul_386);  mul_386 = None
        view_3: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_23, [1, -1, 1, 1]);  primals_23 = None
        mul_388: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_3, 1);  view_3 = None
        mul_389: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(getitem_155, mul_388);  mul_388 = None
        sum_168: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_155, [0, 2, 3], True)
        sum_169: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_387, [0, 2, 3], True);  mul_387 = None
        mul_390: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, mul_12)
        mul_391: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, div_1);  mul_389 = None
        sum_170: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_390, [2, 3], True);  mul_390 = None
        add_166: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(getitem_155, mul_391);  getitem_155 = mul_391 = None
        view_55: "f32[320][1]cuda:0" = torch.ops.aten.reshape.default(sum_169, [320]);  sum_169 = None
        view_56: "f32[320][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [320]);  sum_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_92: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.div.Tensor(div_1, add_11);  div_1 = None
        neg_12: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_170)
        mul_392: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.mul.Tensor(neg_12, div_92);  neg_12 = div_92 = None
        div_93: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_170, add_11);  sum_170 = add_11 = None
        sum_171: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [1], True);  mul_392 = None
        expand_14: "f32[128, 320, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_171, [128, 320, 1, 1]);  sum_171 = None
        div_94: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_14, 320);  expand_14 = None
        add_167: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_93, div_94);  div_93 = div_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_95: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.div.Tensor(mul_12, pow_4);  mul_12 = None
        eq_12: "b8[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.eq.Scalar(pow_4, 0);  pow_4 = None
        where_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.where.self(eq_12, full_default, div_95);  eq_12 = div_95 = None
        clone_41: "f32[128, 320, 56, 56][1003520, 3136, 56, 1]cuda:0" = torch.ops.aten.clone.default(where_12, memory_format = torch.contiguous_format);  where_12 = None
        mul_393: "f32[128, 320, 56, 56][1003520, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_167, clone_41);  add_167 = clone_41 = None
        add_168: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(add_166, mul_393);  add_166 = mul_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_395: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(add_10, 0.5);  add_10 = None
        mul_396: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_5, convolution_5)
        mul_397: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_396, -0.5);  mul_396 = None
        exp_12: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.exp.default(mul_397);  mul_397 = None
        mul_398: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_399: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_5, mul_398);  convolution_5 = mul_398 = None
        add_170: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_395, mul_399);  mul_395 = mul_399 = None
        mul_400: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(add_168, add_170);  add_168 = add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_172: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3])
        convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_400, permute_5, primals_20, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_400 = permute_5 = primals_20 = None
        getitem_158: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = convolution_backward_40[0]
        getitem_159: "f32[320, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_40[1];  convolution_backward_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_75: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_158, [0, 2, 3, 1]);  getitem_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_402: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_75, primals_18);  primals_18 = None
        mul_403: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, 80)
        sum_173: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_402, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_4: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution_4, [0, 2, 3, 1]);  convolution_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_2: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_4, getitem_5);  permute_4 = getitem_5 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_76: "f32[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.permute.default(mul_406, [0, 3, 1, 2]);  mul_406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_177: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_76, [0, 2, 3])
        convolution_backward_41 = torch.ops.aten.convolution_backward.default(permute_76, add_7, primals_16, [80], [1, 1], [3, 3], [1, 1], False, [0, 0], 80, [True, True, False]);  permute_76 = add_7 = primals_16 = None
        getitem_161: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = convolution_backward_41[0]
        getitem_162: "f32[80, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_41[1];  convolution_backward_41 = None
        add_171: "f32[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.add.Tensor(permute_74, getitem_161);  permute_74 = getitem_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:288 in forward, code: x = self.fc2(x)
        sum_178: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_171, [0, 2, 3])
        convolution_backward_42 = torch.ops.aten.convolution_backward.default(add_171, add_6, primals_14, [80], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  add_6 = primals_14 = None
        getitem_164: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = convolution_backward_42[0]
        getitem_165: "f32[80, 320, 1, 1][320, 1, 320, 320]cuda:0" = convolution_backward_42[1];  convolution_backward_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_4: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_2, 0.5)
        mul_5: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_2, 0.7071067811865476)
        erf: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_4: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, add_4);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.div.Tensor(pow_2, add_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:47 in forward, code: return x + torch.addcmul(self.bias.view(self.wb_shape), self.weight.view(self.wb_shape), x * x_n)
        mul_7: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_6, div)
        mul_408: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Scalar(mul_7, 1);  mul_7 = None
        mul_409: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(getitem_164, mul_408);  mul_408 = None
        view_1: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(primals_13, [1, -1, 1, 1]);  primals_13 = None
        mul_410: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(view_1, 1);  view_1 = None
        mul_411: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(getitem_164, mul_410);  mul_410 = None
        sum_179: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_164, [0, 2, 3], True)
        sum_180: "f32[1, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3], True);  mul_409 = None
        mul_412: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, mul_6)
        mul_413: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, div);  mul_411 = None
        sum_181: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_412, [2, 3], True);  mul_412 = None
        add_172: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(getitem_164, mul_413);  getitem_164 = mul_413 = None
        view_57: "f32[320][1]cuda:0" = torch.ops.aten.reshape.default(sum_180, [320]);  sum_180 = None
        view_58: "f32[320][1]cuda:0" = torch.ops.aten.reshape.default(sum_179, [320]);  sum_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:46 in forward, code: x_n = x_g / (x_g.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        div_98: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.div.Tensor(div, add_5);  div = None
        neg_13: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.neg.default(sum_181)
        mul_414: "f32[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.mul.Tensor(neg_13, div_98);  neg_13 = div_98 = None
        div_99: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(sum_181, add_5);  sum_181 = add_5 = None
        sum_182: "f32[128, 1, 1, 1][1, 1, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_414, [1], True);  mul_414 = None
        expand_15: "f32[128, 320, 1, 1][1, 0, 1, 1]cuda:0" = torch.ops.aten.expand.default(sum_182, [128, 320, 1, 1]);  sum_182 = None
        div_100: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_15, 320);  expand_15 = None
        add_173: "f32[128, 320, 1, 1][320, 1, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(div_99, div_100);  div_99 = div_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/grn.py:45 in forward, code: x_g = x.norm(p=2, dim=self.spatial_dim, keepdim=True)
        div_101: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.div.Tensor(mul_6, pow_2);  mul_6 = None
        eq_13: "b8[128, 320, 1, 1][320, 1, 320, 320]cuda:0" = torch.ops.aten.eq.Scalar(pow_2, 0);  pow_2 = None
        where_13: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.where.self(eq_13, full_default, div_101);  eq_13 = full_default = div_101 = None
        clone_42: "f32[128, 320, 56, 56][1003520, 3136, 56, 1]cuda:0" = torch.ops.aten.clone.default(where_13, memory_format = torch.contiguous_format);  where_13 = None
        mul_415: "f32[128, 320, 56, 56][1003520, 3136, 56, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_173, clone_42);  add_173 = clone_42 = None
        add_174: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(add_172, mul_415);  add_172 = mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/activations.py:145 in forward, code: return F.gelu(input)
        mul_417: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(add_4, 0.5);  add_4 = None
        mul_418: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_2, convolution_2)
        mul_419: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(mul_418, -0.5);  mul_418 = None
        exp_13: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.exp.default(mul_419);  mul_419 = None
        mul_420: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_421: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(convolution_2, mul_420);  convolution_2 = mul_420 = None
        add_176: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.add.Tensor(mul_417, mul_421);  mul_417 = mul_421 = None
        mul_422: "f32[128, 320, 56, 56][1003520, 1, 17920, 320]cuda:0" = torch.ops.aten.mul.Tensor(add_174, add_176);  add_174 = add_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:284 in forward, code: x = self.fc1(x)
        sum_183: "f32[320][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 2, 3])
        convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_422, permute_3, primals_10, [320], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_422 = permute_3 = primals_10 = None
        getitem_167: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = convolution_backward_43[0]
        getitem_168: "f32[320, 80, 1, 1][80, 1, 80, 80]cuda:0" = convolution_backward_43[1];  convolution_backward_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_77: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(getitem_167, [0, 2, 3, 1]);  getitem_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_424: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_77, primals_8);  primals_8 = None
        mul_425: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_424, 80)
        sum_184: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_424, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_2: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution_1, [0, 2, 3, 1]);  convolution_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub_1: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_2, getitem_3);  permute_2 = getitem_3 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_78: "f32[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.permute.default(mul_428, [0, 3, 1, 2]);  mul_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:200 in forward, code: x = self.conv_dw(x)
        sum_188: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_78, [0, 2, 3])
        convolution_backward_44 = torch.ops.aten.convolution_backward.default(permute_78, permute_1, primals_6, [80], [1, 1], [3, 3], [1, 1], False, [0, 0], 80, [True, True, False]);  permute_78 = permute_1 = primals_6 = None
        getitem_170: "f32[128, 80, 56, 56][250880, 1, 4480, 80]cuda:0" = convolution_backward_44[0]
        getitem_171: "f32[80, 1, 7, 7][49, 1, 7, 1]cuda:0" = convolution_backward_44[1];  convolution_backward_44 = None
        add_177: "f32[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.add.Tensor(add_171, getitem_170);  add_171 = getitem_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:133 in forward, code: x = x.permute(0, 3, 1, 2)
        permute_79: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.permute.default(add_177, [0, 2, 3, 1]);  add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mul_431: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_79, primals_4);  primals_4 = None
        mul_432: "f32[128, 56, 56, 80][250880, 80, 4480, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_431, 80)
        sum_189: "f32[128, 56, 56, 1][3136, 56, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_431, [3], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.permute.default(convolution, [0, 2, 3, 1]);  convolution = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:132 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        sub: "f32[128, 56, 56, 80][250880, 4480, 80, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute, getitem_1);  permute = getitem_1 = None
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

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:128 in forward, code: x = x.permute(0, 2, 3, 1)
        permute_80: "f32[128, 80, 56, 56][250880, 1, 80, 4480]cuda:0" = torch.ops.aten.permute.default(mul_435, [0, 3, 1, 2]);  mul_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/convnext.py:608 in forward_features, code: x = self.stem(x)
        sum_193: "f32[80][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_80, [0, 2, 3])
        convolution_backward_45 = torch.ops.aten.convolution_backward.default(permute_80, primals_3, primals_1, [80], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  permute_80 = primals_3 = primals_1 = None
        getitem_174: "f32[80, 3, 4, 4][48, 1, 12, 3]cuda:0" = convolution_backward_45[1];  convolution_backward_45 = None
        return (getitem_174, sum_193, None, sum_191, sum_192, getitem_171, sum_188, sum_186, sum_187, getitem_168, sum_183, view_58, view_57, getitem_165, sum_178, getitem_162, sum_177, sum_175, sum_176, getitem_159, sum_172, view_56, view_55, getitem_156, sum_167, sum_165, sum_166, getitem_153, sum_162, getitem_150, sum_161, sum_159, sum_160, getitem_147, sum_156, view_54, view_53, getitem_144, sum_151, getitem_141, sum_150, sum_148, sum_149, getitem_138, sum_145, view_52, view_51, getitem_135, sum_140, sum_138, sum_139, getitem_132, sum_135, getitem_129, sum_134, sum_132, sum_133, getitem_126, sum_129, view_50, view_49, getitem_123, sum_124, getitem_120, sum_123, sum_121, sum_122, getitem_117, sum_118, view_48, view_47, getitem_114, sum_113, getitem_111, sum_112, sum_110, sum_111, getitem_108, sum_107, view_46, view_45, getitem_105, sum_102, getitem_102, sum_101, sum_99, sum_100, getitem_99, sum_96, view_44, view_43, getitem_96, sum_91, getitem_93, sum_90, sum_88, sum_89, getitem_90, sum_85, view_42, view_41, getitem_87, sum_80, getitem_84, sum_79, sum_77, sum_78, getitem_81, sum_74, view_40, view_39, getitem_78, sum_69, getitem_75, sum_68, sum_66, sum_67, getitem_72, sum_63, view_38, view_37, getitem_69, sum_58, getitem_66, sum_57, sum_55, sum_56, getitem_63, sum_52, view_36, view_35, getitem_60, sum_47, sum_45, sum_46, getitem_57, sum_42, getitem_54, sum_41, sum_39, sum_40, getitem_51, sum_36, view_34, view_33, getitem_48, sum_31, getitem_45, sum_30, sum_28, sum_29, getitem_42, sum_25, view_32, view_31, getitem_39, sum_20, sum_18, sum_19, mm_1, view_29)
