class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[8, 1024][1024, 1]cuda:0", primals_3: "f32[512][1]cuda:0", primals_9: "f32[512][1]cuda:0", primals_12: "f32[512][1]cuda:0", primals_17: "f32[512][1]cuda:0", primals_20: "f32[512][1]cuda:0", primals_25: "f32[512][1]cuda:0", primals_28: "f32[512][1]cuda:0", primals_33: "f32[512][1]cuda:0", primals_36: "f32[512][1]cuda:0", primals_41: "f32[512][1]cuda:0", primals_44: "f32[512][1]cuda:0", primals_49: "f32[512][1]cuda:0", primals_52: "f32[512][1]cuda:0", primals_53: "i64[8, 1024][1024, 1]cuda:0", primals_54: "f32[512][1]cuda:0", primals_60: "f32[512][1]cuda:0", primals_65: "f32[512][1]cuda:0", primals_68: "f32[512][1]cuda:0", primals_73: "f32[512][1]cuda:0", primals_78: "f32[512][1]cuda:0", primals_81: "f32[512][1]cuda:0", primals_86: "f32[512][1]cuda:0", primals_91: "f32[512][1]cuda:0", primals_94: "f32[512][1]cuda:0", primals_99: "f32[512][1]cuda:0", primals_104: "f32[512][1]cuda:0", primals_107: "f32[512][1]cuda:0", primals_112: "f32[512][1]cuda:0", primals_117: "f32[512][1]cuda:0", primals_120: "f32[512][1]cuda:0", primals_125: "f32[512][1]cuda:0", primals_130: "f32[512][1]cuda:0", primals_133: "f32[512][1]cuda:0", embedding: "f32[8, 1024, 512][524288, 512, 1]cuda:0", gt: "b8[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_1: "bf16[8192, 512][512, 1]cuda:0", bmm: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0", add_6: "i64[1024, 1024][1024, 1]cuda:0", embedding_1: "f32[1024, 1024, 8][8192, 8, 1]cuda:0", amax: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_1: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_2: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_20: "bf16[8192, 512][512, 1]cuda:0", gt_3: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_9: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_1: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_22: "bf16[8192, 512][512, 1]cuda:0", gt_4: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_24: "bf16[8192, 2048][2048, 1]cuda:0", gt_5: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_11: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_2: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_26: "bf16[8192, 512][512, 1]cuda:0", convert_element_type_48: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", amax_1: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_2: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_6: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_45: "bf16[8192, 512][512, 1]cuda:0", gt_7: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_14: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_3: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_47: "bf16[8192, 512][512, 1]cuda:0", gt_8: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_49: "bf16[8192, 2048][2048, 1]cuda:0", gt_9: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_16: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_4: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_51: "bf16[8192, 512][512, 1]cuda:0", convert_element_type_79: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", amax_2: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_3: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_10: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_70: "bf16[8192, 512][512, 1]cuda:0", gt_11: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_19: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_5: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_72: "bf16[8192, 512][512, 1]cuda:0", gt_12: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_74: "bf16[8192, 2048][2048, 1]cuda:0", gt_13: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_21: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_6: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_76: "bf16[8192, 512][512, 1]cuda:0", convert_element_type_110: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", amax_3: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_4: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_14: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_95: "bf16[8192, 512][512, 1]cuda:0", gt_15: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_24: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_7: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_97: "bf16[8192, 512][512, 1]cuda:0", gt_16: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_99: "bf16[8192, 2048][2048, 1]cuda:0", gt_17: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_26: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_8: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_101: "bf16[8192, 512][512, 1]cuda:0", convert_element_type_141: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", amax_4: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_5: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_18: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_120: "bf16[8192, 512][512, 1]cuda:0", gt_19: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_29: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_9: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_122: "bf16[8192, 512][512, 1]cuda:0", gt_20: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_124: "bf16[8192, 2048][2048, 1]cuda:0", gt_21: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_31: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_10: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_126: "bf16[8192, 512][512, 1]cuda:0", convert_element_type_172: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", amax_5: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_6: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_22: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_145: "bf16[8192, 512][512, 1]cuda:0", gt_23: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_34: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_11: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_147: "bf16[8192, 512][512, 1]cuda:0", gt_24: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_149: "bf16[8192, 2048][2048, 1]cuda:0", gt_25: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_36: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_12: "f32[8, 1024, 1][1024, 1, 1]cuda:0", gt_26: "b8[8, 1024, 512][524288, 512, 1]cuda:0", unsqueeze_7: "i64[1, 1, 1024][1024, 1024, 1]cuda:0", gt_27: "b8[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_152: "bf16[8192, 512][512, 1]cuda:0", bmm_12: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0", add_45: "i64[1024, 1024][1024, 1]cuda:0", embedding_3: "f32[1024, 1024, 8][8192, 8, 1]cuda:0", amax_6: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_7: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_28: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_171: "bf16[8192, 512][512, 1]cuda:0", gt_29: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_48: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_173: "bf16[8192, 512][512, 1]cuda:0", view_176: "bf16[8192, 512][512, 1]cuda:0", bmm_14: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0", amax_7: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_8: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_30: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_192: "bf16[8192, 512][512, 1]cuda:0", gt_31: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_52: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_15: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_194: "bf16[8192, 512][512, 1]cuda:0", gt_32: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_196: "bf16[8192, 2048][2048, 1]cuda:0", gt_33: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_54: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_16: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_198: "bf16[8192, 512][512, 1]cuda:0", convert_element_type_258: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", amax_8: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_9: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_34: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_217: "bf16[8192, 512][512, 1]cuda:0", gt_35: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_57: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_17: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_219: "bf16[8192, 512][512, 1]cuda:0", bmm_18: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0", amax_9: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_10: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_36: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_238: "bf16[8192, 512][512, 1]cuda:0", gt_37: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_60: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_18: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_240: "bf16[8192, 512][512, 1]cuda:0", gt_38: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_242: "bf16[8192, 2048][2048, 1]cuda:0", gt_39: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_62: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_244: "bf16[8192, 512][512, 1]cuda:0", convert_element_type_311: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", amax_10: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_11: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_40: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_263: "bf16[8192, 512][512, 1]cuda:0", gt_41: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_65: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_265: "bf16[8192, 512][512, 1]cuda:0", bmm_22: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0", amax_11: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_12: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_42: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_284: "bf16[8192, 512][512, 1]cuda:0", gt_43: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_68: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_21: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_286: "bf16[8192, 512][512, 1]cuda:0", gt_44: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_288: "bf16[8192, 2048][2048, 1]cuda:0", gt_45: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_70: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_22: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_290: "bf16[8192, 512][512, 1]cuda:0", convert_element_type_364: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", amax_12: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_13: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_46: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_309: "bf16[8192, 512][512, 1]cuda:0", gt_47: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_73: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_23: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_311: "bf16[8192, 512][512, 1]cuda:0", bmm_26: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0", amax_13: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_14: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_48: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_330: "bf16[8192, 512][512, 1]cuda:0", gt_49: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_76: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_332: "bf16[8192, 512][512, 1]cuda:0", gt_50: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_334: "bf16[8192, 2048][2048, 1]cuda:0", gt_51: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_78: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_25: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_336: "bf16[8192, 512][512, 1]cuda:0", convert_element_type_417: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", amax_14: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_15: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_52: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_355: "bf16[8192, 512][512, 1]cuda:0", gt_53: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_81: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_26: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_357: "bf16[8192, 512][512, 1]cuda:0", bmm_30: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0", amax_15: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_16: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_54: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_376: "bf16[8192, 512][512, 1]cuda:0", gt_55: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_84: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_27: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_378: "bf16[8192, 512][512, 1]cuda:0", gt_56: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_380: "bf16[8192, 2048][2048, 1]cuda:0", gt_57: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_86: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_28: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_382: "bf16[8192, 512][512, 1]cuda:0", convert_element_type_470: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", amax_16: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_17: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_58: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_401: "bf16[8192, 512][512, 1]cuda:0", gt_59: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_89: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_29: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_403: "bf16[8192, 512][512, 1]cuda:0", bmm_34: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0", amax_17: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", sum_18: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0", gt_60: "b8[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0", view_422: "bf16[8192, 512][512, 1]cuda:0", gt_61: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_92: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_30: "f32[8, 1024, 1][1024, 1, 1]cuda:0", view_424: "bf16[8192, 512][512, 1]cuda:0", gt_62: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", view_426: "bf16[8192, 2048][2048, 1]cuda:0", gt_63: "b8[8, 1024, 512][524288, 512, 1]cuda:0", add_94: "f32[8, 1024, 512][524288, 512, 1]cuda:0", rsqrt_31: "f32[8, 1024, 1][1024, 1, 1]cuda:0", gt_64: "b8[8, 1024, 512][524288, 512, 1]cuda:0", view_428: "bf16[8192, 512][512, 1]cuda:0", view_429: "bf16[8, 1024, 32128][32899072, 32128, 1]cuda:0", amax_18: "f32[8192, 1][1, 1]cuda:0", log_2: "f32[8192, 1][1, 1]cuda:0", convert_element_type_516: "f32[][]cuda:0", permute_191: "bf16[32128, 512][512, 1]cuda:0", permute_195: "bf16[512, 2048][2048, 1]cuda:0", le_1: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_199: "bf16[2048, 512][512, 1]cuda:0", permute_203: "bf16[512, 512][512, 1]cuda:0", permute_206: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_207: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_208: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_209: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_214: "bf16[512, 512][512, 1]cuda:0", permute_219: "bf16[512, 512][512, 1]cuda:0", permute_224: "bf16[512, 512][512, 1]cuda:0", permute_228: "bf16[512, 512][512, 1]cuda:0", permute_231: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_232: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_233: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_234: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_239: "bf16[512, 512][512, 1]cuda:0", permute_244: "bf16[512, 512][512, 1]cuda:0", permute_249: "bf16[512, 512][512, 1]cuda:0", permute_253: "bf16[512, 2048][2048, 1]cuda:0", le_2: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_257: "bf16[2048, 512][512, 1]cuda:0", permute_261: "bf16[512, 512][512, 1]cuda:0", permute_264: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_265: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_266: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_267: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_272: "bf16[512, 512][512, 1]cuda:0", permute_277: "bf16[512, 512][512, 1]cuda:0", permute_282: "bf16[512, 512][512, 1]cuda:0", permute_286: "bf16[512, 512][512, 1]cuda:0", permute_289: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_290: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_291: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_292: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_297: "bf16[512, 512][512, 1]cuda:0", permute_302: "bf16[512, 512][512, 1]cuda:0", permute_307: "bf16[512, 512][512, 1]cuda:0", permute_311: "bf16[512, 2048][2048, 1]cuda:0", le_3: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_315: "bf16[2048, 512][512, 1]cuda:0", permute_319: "bf16[512, 512][512, 1]cuda:0", permute_322: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_323: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_324: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_325: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_330: "bf16[512, 512][512, 1]cuda:0", permute_335: "bf16[512, 512][512, 1]cuda:0", permute_340: "bf16[512, 512][512, 1]cuda:0", permute_344: "bf16[512, 512][512, 1]cuda:0", permute_347: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_348: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_349: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_350: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_355: "bf16[512, 512][512, 1]cuda:0", permute_360: "bf16[512, 512][512, 1]cuda:0", permute_365: "bf16[512, 512][512, 1]cuda:0", permute_369: "bf16[512, 2048][2048, 1]cuda:0", le_4: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_373: "bf16[2048, 512][512, 1]cuda:0", permute_377: "bf16[512, 512][512, 1]cuda:0", permute_380: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_381: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_382: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_383: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_388: "bf16[512, 512][512, 1]cuda:0", permute_393: "bf16[512, 512][512, 1]cuda:0", permute_398: "bf16[512, 512][512, 1]cuda:0", permute_402: "bf16[512, 512][512, 1]cuda:0", permute_405: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_406: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_407: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_408: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_413: "bf16[512, 512][512, 1]cuda:0", permute_418: "bf16[512, 512][512, 1]cuda:0", permute_423: "bf16[512, 512][512, 1]cuda:0", permute_427: "bf16[512, 2048][2048, 1]cuda:0", le_5: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_431: "bf16[2048, 512][512, 1]cuda:0", permute_435: "bf16[512, 512][512, 1]cuda:0", permute_438: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_439: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_440: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_441: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_446: "bf16[512, 512][512, 1]cuda:0", permute_451: "bf16[512, 512][512, 1]cuda:0", permute_456: "bf16[512, 512][512, 1]cuda:0", permute_460: "bf16[512, 512][512, 1]cuda:0", permute_463: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_464: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_465: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_466: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_471: "bf16[512, 512][512, 1]cuda:0", permute_476: "bf16[512, 512][512, 1]cuda:0", permute_481: "bf16[512, 512][512, 1]cuda:0", permute_485: "bf16[512, 2048][2048, 1]cuda:0", le_6: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_489: "bf16[2048, 512][512, 1]cuda:0", permute_493: "bf16[512, 512][512, 1]cuda:0", permute_496: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_497: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_498: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_499: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_504: "bf16[512, 512][512, 1]cuda:0", permute_509: "bf16[512, 512][512, 1]cuda:0", permute_514: "bf16[512, 512][512, 1]cuda:0", permute_518: "bf16[512, 512][512, 1]cuda:0", permute_521: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_522: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_524: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_525: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_530: "bf16[512, 512][512, 1]cuda:0", permute_535: "bf16[512, 512][512, 1]cuda:0", permute_540: "bf16[512, 512][512, 1]cuda:0", permute_544: "bf16[512, 2048][2048, 1]cuda:0", le_7: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_548: "bf16[2048, 512][512, 1]cuda:0", permute_552: "bf16[512, 512][512, 1]cuda:0", permute_555: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_556: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_557: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_558: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_563: "bf16[512, 512][512, 1]cuda:0", permute_568: "bf16[512, 512][512, 1]cuda:0", permute_573: "bf16[512, 512][512, 1]cuda:0", permute_577: "bf16[512, 2048][2048, 1]cuda:0", le_8: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_581: "bf16[2048, 512][512, 1]cuda:0", permute_585: "bf16[512, 512][512, 1]cuda:0", permute_588: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_589: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_590: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_591: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_596: "bf16[512, 512][512, 1]cuda:0", permute_601: "bf16[512, 512][512, 1]cuda:0", permute_606: "bf16[512, 512][512, 1]cuda:0", permute_610: "bf16[512, 2048][2048, 1]cuda:0", le_9: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_614: "bf16[2048, 512][512, 1]cuda:0", permute_618: "bf16[512, 512][512, 1]cuda:0", permute_621: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_622: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_623: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_624: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_629: "bf16[512, 512][512, 1]cuda:0", permute_634: "bf16[512, 512][512, 1]cuda:0", permute_639: "bf16[512, 512][512, 1]cuda:0", permute_643: "bf16[512, 2048][2048, 1]cuda:0", le_10: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_647: "bf16[2048, 512][512, 1]cuda:0", permute_651: "bf16[512, 512][512, 1]cuda:0", permute_654: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_655: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_656: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_657: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_662: "bf16[512, 512][512, 1]cuda:0", permute_667: "bf16[512, 512][512, 1]cuda:0", permute_672: "bf16[512, 512][512, 1]cuda:0", permute_676: "bf16[512, 2048][2048, 1]cuda:0", le_11: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_680: "bf16[2048, 512][512, 1]cuda:0", permute_684: "bf16[512, 512][512, 1]cuda:0", permute_687: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_688: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_689: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_690: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_695: "bf16[512, 512][512, 1]cuda:0", permute_700: "bf16[512, 512][512, 1]cuda:0", permute_705: "bf16[512, 512][512, 1]cuda:0", permute_709: "bf16[512, 2048][2048, 1]cuda:0", le_12: "b8[8, 1024, 2048][2097152, 2048, 1]cuda:0", permute_713: "bf16[2048, 512][512, 1]cuda:0", permute_717: "bf16[512, 512][512, 1]cuda:0", permute_720: "bf16[64, 1024, 1024][1048576, 1, 1024]cuda:0", permute_721: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_723: "bf16[64, 64, 1024][65536, 1, 64]cuda:0", permute_724: "bf16[64, 1024, 64][65536, 1, 1024]cuda:0", permute_729: "bf16[512, 512][512, 1]cuda:0", permute_734: "bf16[512, 512][512, 1]cuda:0", permute_739: "bf16[512, 512][512, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[8, 1024, 32128][32899072, 32128, 1]cuda:0", tangents_3: "f32[8, 1024, 512][524288, 512, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        div_23: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_516);  tangents_1 = convert_element_type_516 = None
        view_431: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(primals_53, [-1]);  primals_53 = None
        unsqueeze_19: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_431, 1);  view_431 = None
        ne_3: "b8[8192, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_19, -100)
        full_default_4: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_19, full_default_4);  unsqueeze_19 = full_default_4 = None

        # No stacktrace found for following nodes
        iota_default: "i64[32128][1]cuda:0" = torch.ops.prims.iota.default(32128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 32128][32128, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 32128]);  iota_default = None
        expand_default: "i64[8192, 32128][1, 0]cuda:0" = torch.ops.aten.expand.default(where_7, [8192, 32128]);  where_7 = None
        eq_tensor: "b8[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_self: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1104 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_8: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_23, full_default_1);  ne_3 = div_23 = None
        mul_196: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_8);  where_self = where_8 = None
        convert_element_type_517: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_196, torch.bfloat16);  mul_196 = None
        convert_element_type_518: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_517, torch.float32);  convert_element_type_517 = None
        view_430: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.reshape.default(view_429, [-1, 32128]);  view_429 = None
        convert_element_type_513: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_430, torch.float32);  view_430 = None
        sub_20: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_513, amax_18);  convert_element_type_513 = amax_18 = None
        sub_21: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_20, log_2);  sub_20 = log_2 = None
        convert_element_type_514: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_21, torch.bfloat16);  sub_21 = None
        convert_element_type_515: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_514, torch.float32);  convert_element_type_514 = None
        exp_19: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.exp.default(convert_element_type_515);  convert_element_type_515 = None
        sum_22: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_518, [1], True)
        mul_197: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_19, sum_22);  exp_19 = sum_22 = None
        sub_22: "f32[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_518, mul_197);  convert_element_type_518 = mul_197 = None
        convert_element_type_520: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_22, torch.bfloat16);  sub_22 = None
        view_432: "bf16[8, 1024, 32128][32899072, 32128, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_520, [8, 1024, 32128]);  convert_element_type_520 = None
        add_96: "bf16[8, 1024, 32128][32899072, 32128, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, view_432);  tangents_2 = view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1097 in forward, code: lm_logits = self.lm_head(sequence_output)
        view_433: "bf16[8192, 32128][32128, 1]cuda:0" = torch.ops.aten.reshape.default(add_96, [8192, 32128]);  add_96 = None
        permute_189: "bf16[32128, 8192][1, 32128]cuda:0" = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_97: "bf16[32128, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_189, view_428);  permute_189 = view_428 = None
        mm_98: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_433, permute_191);  view_433 = permute_191 = None
        view_434: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [8, 1024, 512]);  mm_98 = None
        convert_element_type_525: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_434, torch.float32);  view_434 = None
        convert_element_type_526: "f32[32128, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:1095 in forward, code: sequence_output = sequence_output * (self.model_dim**-0.5)
        mul_198: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_525, 0.04419417382415922);  convert_element_type_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_527: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_64, torch.float32);  gt_64 = None
        mul_199: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_527, 1.1111111111111112);  convert_element_type_527 = None
        mul_200: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, mul_199);  mul_198 = mul_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_201: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, primals_133);  primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_191: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_94, rsqrt_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_202: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_200, mul_191);  mul_200 = mul_191 = None
        sum_23: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1], True, dtype = torch.float32);  mul_202 = None
        view_435: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_23, [512]);  sum_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_203: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_201, add_94)
        mul_204: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_201, rsqrt_31);  mul_201 = None
        sum_24: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True, dtype = torch.float32);  mul_203 = None
        pow_33: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_31, 3);  rsqrt_31 = None
        mul_205: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_24, -0.5);  sum_24 = None
        mul_206: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, pow_33);  mul_205 = pow_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_75: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_206, [8, 1024, 512]);  mul_206 = None
        div_24: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_75, 512);  expand_75 = None
        pow_34: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_94, 1.0);  add_94 = None
        mul_207: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_34, 2.0);  pow_34 = None
        mul_208: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_24, mul_207);  div_24 = mul_207 = None
        add_97: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_204, mul_208);  mul_204 = mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_528: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16)
        convert_element_type_529: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_63, torch.bfloat16);  gt_63 = None
        mul_209: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_529, 1.1111111111111112);  convert_element_type_529 = None
        mul_210: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_528, mul_209);  convert_element_type_528 = mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_436: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_210, [8192, 512]);  mul_210 = None
        permute_193: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_99: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_193, view_426);  permute_193 = view_426 = None
        mm_100: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_436, permute_195);  view_436 = permute_195 = None
        view_437: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [8, 1024, 2048]);  mm_100 = None
        convert_element_type_535: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_17: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_437, torch.bfloat16);  view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_537: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_62, torch.bfloat16);  gt_62 = None
        mul_211: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_537, 1.1111111111111112);  convert_element_type_537 = None
        mul_212: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_17, mul_211);  convert_element_type_default_17 = mul_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        full_default_9: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_1, full_default_9, mul_212);  le_1 = mul_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_438: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_9, [8192, 2048]);  where_9 = None
        permute_197: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_101: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_197, view_424);  permute_197 = view_424 = None
        mm_102: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_438, permute_199);  view_438 = permute_199 = None
        view_439: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [8, 1024, 512]);  mm_102 = None
        convert_element_type_542: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_439, torch.float32);  view_439 = None
        convert_element_type_543: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_213: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_542, primals_130);  primals_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_185: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_92, rsqrt_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_214: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_542, mul_185);  convert_element_type_542 = mul_185 = None
        sum_25: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1], True, dtype = torch.float32);  mul_214 = None
        view_440: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_25, [512]);  sum_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_215: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, add_92)
        mul_216: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, rsqrt_30);  mul_213 = None
        sum_26: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True, dtype = torch.float32);  mul_215 = None
        add_98: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_97, mul_216);  add_97 = mul_216 = None
        pow_35: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_30, 3);  rsqrt_30 = None
        mul_217: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_26, -0.5);  sum_26 = None
        mul_218: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, pow_35);  mul_217 = pow_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_76: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_218, [8, 1024, 512]);  mul_218 = None
        div_25: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_76, 512);  expand_76 = None
        pow_36: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_92, 1.0);  add_92 = None
        mul_219: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_36, 2.0);  pow_36 = None
        mul_220: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_25, mul_219);  div_25 = mul_219 = None
        add_99: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_98, mul_220);  add_98 = mul_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_544: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16)
        convert_element_type_545: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_61, torch.bfloat16);  gt_61 = None
        mul_221: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_545, 1.1111111111111112);  convert_element_type_545 = None
        mul_222: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_544, mul_221);  convert_element_type_544 = mul_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_441: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_222, [8192, 512]);  mul_222 = None
        permute_201: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_441, [1, 0])
        mm_103: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_201, view_422);  permute_201 = view_422 = None
        mm_104: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_441, permute_203);  view_441 = permute_203 = None
        view_442: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [8, 1024, 512]);  mm_104 = None
        convert_element_type_550: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_443: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_442, [8, 1024, 8, 64]);  view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_205: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_76: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_444: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [64, 1024, 64]);  clone_76 = None
        bmm_36: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_206, view_444);  permute_206 = None
        bmm_37: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_444, permute_207);  view_444 = permute_207 = None
        view_445: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [8, 8, 1024, 64]);  bmm_36 = None
        view_446: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [8, 8, 1024, 1024]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_555: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_60, torch.bfloat16);  gt_60 = None
        mul_223: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_555, 1.1111111111111112);  convert_element_type_555 = None
        mul_224: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_446, mul_223);  view_446 = mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_556: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_224, torch.float32);  mul_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_414: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [8, 8, 1024, 1024]);  bmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_19: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_414, torch.bfloat16);  view_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_493: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_19, torch.float32);  convert_element_type_default_19 = None
        sub_19: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_493, amax_17);  convert_element_type_493 = amax_17 = None
        exp_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        div_21: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        mul_225: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_556, div_21);  convert_element_type_556 = None
        sum_27: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_225, [-1], True)
        neg_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_27, mul_225);  neg_2 = sum_27 = mul_225 = None
        convert_element_type_557: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_447: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_557, [64, 1024, 1024]);  convert_element_type_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_38: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_208, view_447);  permute_208 = None
        bmm_39: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_447, permute_209);  view_447 = permute_209 = None
        view_452: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [8, 8, 64, 1024]);  bmm_38 = None
        view_453: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [8, 8, 1024, 64]);  bmm_39 = None
        permute_210: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_452, [0, 1, 3, 2]);  view_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_211: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
        clone_79: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
        view_454: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [8, 1024, 512]);  clone_79 = None
        view_455: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_454, [8192, 512]);  view_454 = None
        permute_212: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_455, [1, 0])
        mm_105: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_212, view_176);  permute_212 = None
        mm_106: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_455, permute_214);  view_455 = permute_214 = None
        view_456: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [8, 1024, 512]);  mm_106 = None
        convert_element_type_566: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_456, torch.float32);  view_456 = None
        add_100: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_3, convert_element_type_566);  tangents_3 = convert_element_type_566 = None
        convert_element_type_567: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_216: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_210, [0, 2, 1, 3]);  permute_210 = None
        view_457: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_216, [8, 1024, 512]);  permute_216 = None
        clone_80: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_457, memory_format = torch.contiguous_format);  view_457 = None
        view_458: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [8192, 512]);  clone_80 = None
        permute_217: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_458, [1, 0])
        mm_107: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_217, view_176);  permute_217 = None
        mm_108: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_458, permute_219);  view_458 = permute_219 = None
        view_459: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [8, 1024, 512]);  mm_108 = None
        convert_element_type_572: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        add_101: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_100, convert_element_type_572);  add_100 = convert_element_type_572 = None
        convert_element_type_573: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_221: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        clone_81: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
        view_460: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [8, 1024, 512]);  clone_81 = None
        view_461: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_460, [8192, 512]);  view_460 = None
        permute_222: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_461, [1, 0])
        mm_109: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_222, view_403);  permute_222 = view_403 = None
        mm_110: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_461, permute_224);  view_461 = permute_224 = None
        view_462: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [8, 1024, 512]);  mm_110 = None
        convert_element_type_578: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_462, torch.float32);  view_462 = None
        convert_element_type_579: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_226: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_578, primals_125);  primals_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_179: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_89, rsqrt_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_227: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_578, mul_179);  convert_element_type_578 = mul_179 = None
        sum_28: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1], True, dtype = torch.float32);  mul_227 = None
        view_463: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [512]);  sum_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_228: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_226, add_89)
        mul_229: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_226, rsqrt_29);  mul_226 = None
        sum_29: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_228, [2], True, dtype = torch.float32);  mul_228 = None
        add_102: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_99, mul_229);  add_99 = mul_229 = None
        pow_37: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_29, 3);  rsqrt_29 = None
        mul_230: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_29, -0.5);  sum_29 = None
        mul_231: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, pow_37);  mul_230 = pow_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_77: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_231, [8, 1024, 512]);  mul_231 = None
        div_26: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_77, 512);  expand_77 = None
        pow_38: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_89, 1.0);  add_89 = None
        mul_232: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
        mul_233: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_26, mul_232);  div_26 = mul_232 = None
        add_103: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_102, mul_233);  add_102 = mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_580: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16)
        convert_element_type_581: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_59, torch.bfloat16);  gt_59 = None
        mul_234: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_581, 1.1111111111111112);  convert_element_type_581 = None
        mul_235: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_580, mul_234);  convert_element_type_580 = mul_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_464: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_235, [8192, 512]);  mul_235 = None
        permute_226: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_111: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_226, view_401);  permute_226 = view_401 = None
        mm_112: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_464, permute_228);  view_464 = permute_228 = None
        view_465: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [8, 1024, 512]);  mm_112 = None
        convert_element_type_586: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_466: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_465, [8, 1024, 8, 64]);  view_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_230: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_83: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
        view_467: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [64, 1024, 64]);  clone_83 = None
        bmm_40: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_231, view_467);  permute_231 = None
        bmm_41: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_467, permute_232);  view_467 = permute_232 = None
        view_468: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [8, 8, 1024, 64]);  bmm_40 = None
        view_469: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [8, 8, 1024, 1024]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_591: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_58, torch.bfloat16);  gt_58 = None
        mul_236: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_591, 1.1111111111111112);  convert_element_type_591 = None
        mul_237: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_469, mul_236);  view_469 = mul_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_592: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_237, torch.float32);  mul_237 = None
        convert_element_type_471: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_470, torch.float32);  convert_element_type_470 = None
        sub_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_471, amax_16);  convert_element_type_471 = amax_16 = None
        exp_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        div_20: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        mul_238: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_592, div_20);  convert_element_type_592 = None
        sum_30: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_238, [-1], True)
        neg_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_20);  div_20 = None
        fma_1: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_30, mul_238);  neg_3 = sum_30 = mul_238 = None
        convert_element_type_593: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_470: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_593, [64, 1024, 1024]);  convert_element_type_593 = None
        view_472: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_470, [8, 8, 1024, 1024]);  view_470 = None
        view_473: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_472, [64, 1024, 1024])
        convert_element_type_594: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_472, torch.float32);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_42: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_233, view_473);  permute_233 = None
        bmm_43: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_473, permute_234);  view_473 = permute_234 = None
        view_475: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [8, 8, 64, 1024]);  bmm_42 = None
        view_476: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [8, 8, 1024, 64]);  bmm_43 = None
        permute_235: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_475, [0, 1, 3, 2]);  view_475 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_236: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
        clone_86: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
        view_477: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [8, 1024, 512]);  clone_86 = None
        view_478: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_477, [8192, 512]);  view_477 = None
        permute_237: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_478, [1, 0])
        mm_113: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_237, view_382);  permute_237 = None
        mm_114: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_478, permute_239);  view_478 = permute_239 = None
        view_479: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [8, 1024, 512]);  mm_114 = None
        convert_element_type_603: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_479, torch.float32);  view_479 = None
        convert_element_type_604: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_241: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_235, [0, 2, 1, 3]);  permute_235 = None
        view_480: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_241, [8, 1024, 512]);  permute_241 = None
        clone_87: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_480, memory_format = torch.contiguous_format);  view_480 = None
        view_481: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [8192, 512]);  clone_87 = None
        permute_242: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_481, [1, 0])
        mm_115: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_242, view_382);  permute_242 = None
        mm_116: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_481, permute_244);  view_481 = permute_244 = None
        view_482: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [8, 1024, 512]);  mm_116 = None
        convert_element_type_609: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_482, torch.float32);  view_482 = None
        add_104: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_603, convert_element_type_609);  convert_element_type_603 = convert_element_type_609 = None
        convert_element_type_610: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_246: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
        clone_88: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        view_483: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [8, 1024, 512]);  clone_88 = None
        view_484: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_483, [8192, 512]);  view_483 = None
        permute_247: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_484, [1, 0])
        mm_117: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_247, view_382);  permute_247 = view_382 = None
        mm_118: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_484, permute_249);  view_484 = permute_249 = None
        view_485: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [8, 1024, 512]);  mm_118 = None
        convert_element_type_615: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_485, torch.float32);  view_485 = None
        add_105: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_104, convert_element_type_615);  add_104 = convert_element_type_615 = None
        convert_element_type_616: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_239: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_105, primals_120);  primals_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_173: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_86, rsqrt_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_240: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_105, mul_173);  add_105 = mul_173 = None
        sum_31: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_240, [0, 1], True, dtype = torch.float32);  mul_240 = None
        view_486: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_31, [512]);  sum_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_241: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_239, add_86)
        mul_242: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_239, rsqrt_28);  mul_239 = None
        sum_32: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True, dtype = torch.float32);  mul_241 = None
        add_106: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_103, mul_242);  add_103 = mul_242 = None
        pow_39: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_28, 3);  rsqrt_28 = None
        mul_243: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_32, -0.5);  sum_32 = None
        mul_244: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, pow_39);  mul_243 = pow_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_78: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_244, [8, 1024, 512]);  mul_244 = None
        div_27: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_78, 512);  expand_78 = None
        pow_40: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_86, 1.0);  add_86 = None
        mul_245: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
        mul_246: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_27, mul_245);  div_27 = mul_245 = None
        add_107: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_106, mul_246);  add_106 = mul_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_617: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16)
        convert_element_type_618: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_57, torch.bfloat16);  gt_57 = None
        mul_247: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_618, 1.1111111111111112);  convert_element_type_618 = None
        mul_248: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_617, mul_247);  convert_element_type_617 = mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_487: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_248, [8192, 512]);  mul_248 = None
        permute_251: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_487, [1, 0])
        mm_119: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_251, view_380);  permute_251 = view_380 = None
        mm_120: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_487, permute_253);  view_487 = permute_253 = None
        view_488: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [8, 1024, 2048]);  mm_120 = None
        convert_element_type_624: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_16: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_488, torch.bfloat16);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_626: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_56, torch.bfloat16);  gt_56 = None
        mul_249: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_626, 1.1111111111111112);  convert_element_type_626 = None
        mul_250: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_16, mul_249);  convert_element_type_default_16 = mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_10: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_2, full_default_9, mul_250);  le_2 = mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_489: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_10, [8192, 2048]);  where_10 = None
        permute_255: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_489, [1, 0])
        mm_121: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_255, view_378);  permute_255 = view_378 = None
        mm_122: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_489, permute_257);  view_489 = permute_257 = None
        view_490: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [8, 1024, 512]);  mm_122 = None
        convert_element_type_631: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_490, torch.float32);  view_490 = None
        convert_element_type_632: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_251: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_631, primals_117);  primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_167: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_84, rsqrt_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_252: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_631, mul_167);  convert_element_type_631 = mul_167 = None
        sum_33: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_252, [0, 1], True, dtype = torch.float32);  mul_252 = None
        view_491: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [512]);  sum_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_253: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_251, add_84)
        mul_254: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_251, rsqrt_27);  mul_251 = None
        sum_34: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True, dtype = torch.float32);  mul_253 = None
        add_108: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_107, mul_254);  add_107 = mul_254 = None
        pow_41: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_27, 3);  rsqrt_27 = None
        mul_255: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_34, -0.5);  sum_34 = None
        mul_256: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_255, pow_41);  mul_255 = pow_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_79: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_256, [8, 1024, 512]);  mul_256 = None
        div_28: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_79, 512);  expand_79 = None
        pow_42: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_84, 1.0);  add_84 = None
        mul_257: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
        mul_258: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_28, mul_257);  div_28 = mul_257 = None
        add_109: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_108, mul_258);  add_108 = mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_633: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16)
        convert_element_type_634: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_55, torch.bfloat16);  gt_55 = None
        mul_259: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_634, 1.1111111111111112);  convert_element_type_634 = None
        mul_260: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_633, mul_259);  convert_element_type_633 = mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_492: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_260, [8192, 512]);  mul_260 = None
        permute_259: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_492, [1, 0])
        mm_123: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_259, view_376);  permute_259 = view_376 = None
        mm_124: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_492, permute_261);  view_492 = permute_261 = None
        view_493: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [8, 1024, 512]);  mm_124 = None
        convert_element_type_639: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_494: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_493, [8, 1024, 8, 64]);  view_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_263: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_92: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
        view_495: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_92, [64, 1024, 64]);  clone_92 = None
        bmm_44: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_264, view_495);  permute_264 = None
        bmm_45: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_495, permute_265);  view_495 = permute_265 = None
        view_496: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [8, 8, 1024, 64]);  bmm_44 = None
        view_497: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [8, 8, 1024, 1024]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_644: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_54, torch.bfloat16);  gt_54 = None
        mul_261: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_644, 1.1111111111111112);  convert_element_type_644 = None
        mul_262: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_497, mul_261);  view_497 = mul_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_645: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_262, torch.float32);  mul_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_368: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [8, 8, 1024, 1024]);  bmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_21: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_368, torch.bfloat16);  view_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_440: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_21, torch.float32);  convert_element_type_default_21 = None
        sub_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_440, amax_15);  convert_element_type_440 = amax_15 = None
        exp_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        div_19: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        mul_263: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_645, div_19);  convert_element_type_645 = None
        sum_35: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_263, [-1], True)
        neg_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_35, mul_263);  neg_4 = sum_35 = mul_263 = None
        convert_element_type_646: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_498: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_646, [64, 1024, 1024]);  convert_element_type_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_46: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_266, view_498);  permute_266 = None
        bmm_47: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_498, permute_267);  view_498 = permute_267 = None
        view_503: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [8, 8, 64, 1024]);  bmm_46 = None
        view_504: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [8, 8, 1024, 64]);  bmm_47 = None
        permute_268: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_503, [0, 1, 3, 2]);  view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_269: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        clone_95: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_505: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_95, [8, 1024, 512]);  clone_95 = None
        view_506: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_505, [8192, 512]);  view_505 = None
        permute_270: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_506, [1, 0])
        mm_125: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_270, view_176);  permute_270 = None
        mm_126: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_506, permute_272);  view_506 = permute_272 = None
        view_507: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [8, 1024, 512]);  mm_126 = None
        convert_element_type_655: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_507, torch.float32);  view_507 = None
        add_110: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, convert_element_type_655);  add_101 = convert_element_type_655 = None
        convert_element_type_656: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_274: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_268, [0, 2, 1, 3]);  permute_268 = None
        view_508: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_274, [8, 1024, 512]);  permute_274 = None
        clone_96: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_508, memory_format = torch.contiguous_format);  view_508 = None
        view_509: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [8192, 512]);  clone_96 = None
        permute_275: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_509, [1, 0])
        mm_127: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_275, view_176);  permute_275 = None
        mm_128: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_509, permute_277);  view_509 = permute_277 = None
        view_510: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [8, 1024, 512]);  mm_128 = None
        convert_element_type_661: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_510, torch.float32);  view_510 = None
        add_111: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_110, convert_element_type_661);  add_110 = convert_element_type_661 = None
        convert_element_type_662: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_279: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_504, [0, 2, 1, 3]);  view_504 = None
        clone_97: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_279, memory_format = torch.contiguous_format);  permute_279 = None
        view_511: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_97, [8, 1024, 512]);  clone_97 = None
        view_512: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_511, [8192, 512]);  view_511 = None
        permute_280: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_512, [1, 0])
        mm_129: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_280, view_357);  permute_280 = view_357 = None
        mm_130: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_512, permute_282);  view_512 = permute_282 = None
        view_513: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [8, 1024, 512]);  mm_130 = None
        convert_element_type_667: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_513, torch.float32);  view_513 = None
        convert_element_type_668: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_264: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_667, primals_112);  primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_161: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_81, rsqrt_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_265: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_667, mul_161);  convert_element_type_667 = mul_161 = None
        sum_36: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_265, [0, 1], True, dtype = torch.float32);  mul_265 = None
        view_514: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_36, [512]);  sum_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_266: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_264, add_81)
        mul_267: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_264, rsqrt_26);  mul_264 = None
        sum_37: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_266, [2], True, dtype = torch.float32);  mul_266 = None
        add_112: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_109, mul_267);  add_109 = mul_267 = None
        pow_43: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_26, 3);  rsqrt_26 = None
        mul_268: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_37, -0.5);  sum_37 = None
        mul_269: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_268, pow_43);  mul_268 = pow_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_80: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_269, [8, 1024, 512]);  mul_269 = None
        div_29: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_80, 512);  expand_80 = None
        pow_44: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_81, 1.0);  add_81 = None
        mul_270: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_44, 2.0);  pow_44 = None
        mul_271: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_29, mul_270);  div_29 = mul_270 = None
        add_113: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_112, mul_271);  add_112 = mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_669: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.bfloat16)
        convert_element_type_670: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_53, torch.bfloat16);  gt_53 = None
        mul_272: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_670, 1.1111111111111112);  convert_element_type_670 = None
        mul_273: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_669, mul_272);  convert_element_type_669 = mul_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_515: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_273, [8192, 512]);  mul_273 = None
        permute_284: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_515, [1, 0])
        mm_131: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_284, view_355);  permute_284 = view_355 = None
        mm_132: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_515, permute_286);  view_515 = permute_286 = None
        view_516: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [8, 1024, 512]);  mm_132 = None
        convert_element_type_675: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_517: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_516, [8, 1024, 8, 64]);  view_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_288: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_517, [0, 2, 1, 3]);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_99: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_288, memory_format = torch.contiguous_format);  permute_288 = None
        view_518: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_99, [64, 1024, 64]);  clone_99 = None
        bmm_48: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_289, view_518);  permute_289 = None
        bmm_49: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_518, permute_290);  view_518 = permute_290 = None
        view_519: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [8, 8, 1024, 64]);  bmm_48 = None
        view_520: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [8, 8, 1024, 1024]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_680: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_52, torch.bfloat16);  gt_52 = None
        mul_274: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_680, 1.1111111111111112);  convert_element_type_680 = None
        mul_275: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_520, mul_274);  view_520 = mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_681: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_275, torch.float32);  mul_275 = None
        convert_element_type_418: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_417, torch.float32);  convert_element_type_417 = None
        sub_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_418, amax_14);  convert_element_type_418 = amax_14 = None
        exp_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        div_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        mul_276: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_681, div_18);  convert_element_type_681 = None
        sum_38: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
        neg_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_38, mul_276);  neg_5 = sum_38 = mul_276 = None
        convert_element_type_682: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_521: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_682, [64, 1024, 1024]);  convert_element_type_682 = None
        view_523: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_521, [8, 8, 1024, 1024]);  view_521 = None
        view_524: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_523, [64, 1024, 1024])
        convert_element_type_683: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_523, torch.float32);  view_523 = None
        add_114: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_594, convert_element_type_683);  convert_element_type_594 = convert_element_type_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_50: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_291, view_524);  permute_291 = None
        bmm_51: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_524, permute_292);  view_524 = permute_292 = None
        view_526: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [8, 8, 64, 1024]);  bmm_50 = None
        view_527: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [8, 8, 1024, 64]);  bmm_51 = None
        permute_293: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_526, [0, 1, 3, 2]);  view_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_294: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_519, [0, 2, 1, 3]);  view_519 = None
        clone_102: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
        view_528: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [8, 1024, 512]);  clone_102 = None
        view_529: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_528, [8192, 512]);  view_528 = None
        permute_295: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_133: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_295, view_336);  permute_295 = None
        mm_134: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_529, permute_297);  view_529 = permute_297 = None
        view_530: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [8, 1024, 512]);  mm_134 = None
        convert_element_type_692: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_530, torch.float32);  view_530 = None
        convert_element_type_693: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_299: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_293, [0, 2, 1, 3]);  permute_293 = None
        view_531: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_299, [8, 1024, 512]);  permute_299 = None
        clone_103: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_531, memory_format = torch.contiguous_format);  view_531 = None
        view_532: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [8192, 512]);  clone_103 = None
        permute_300: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_532, [1, 0])
        mm_135: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_300, view_336);  permute_300 = None
        mm_136: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_532, permute_302);  view_532 = permute_302 = None
        view_533: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [8, 1024, 512]);  mm_136 = None
        convert_element_type_698: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_533, torch.float32);  view_533 = None
        add_115: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_692, convert_element_type_698);  convert_element_type_692 = convert_element_type_698 = None
        convert_element_type_699: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_304: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None
        clone_104: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
        view_534: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_104, [8, 1024, 512]);  clone_104 = None
        view_535: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_534, [8192, 512]);  view_534 = None
        permute_305: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_137: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_305, view_336);  permute_305 = view_336 = None
        mm_138: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_535, permute_307);  view_535 = permute_307 = None
        view_536: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [8, 1024, 512]);  mm_138 = None
        convert_element_type_704: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_536, torch.float32);  view_536 = None
        add_116: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_115, convert_element_type_704);  add_115 = convert_element_type_704 = None
        convert_element_type_705: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_277: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, primals_107);  primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_155: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_78, rsqrt_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_278: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, mul_155);  add_116 = mul_155 = None
        sum_39: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_278, [0, 1], True, dtype = torch.float32);  mul_278 = None
        view_537: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [512]);  sum_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_279: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_277, add_78)
        mul_280: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_277, rsqrt_25);  mul_277 = None
        sum_40: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True, dtype = torch.float32);  mul_279 = None
        add_117: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_113, mul_280);  add_113 = mul_280 = None
        pow_45: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_25, 3);  rsqrt_25 = None
        mul_281: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_40, -0.5);  sum_40 = None
        mul_282: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, pow_45);  mul_281 = pow_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_81: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_282, [8, 1024, 512]);  mul_282 = None
        div_30: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_81, 512);  expand_81 = None
        pow_46: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_78, 1.0);  add_78 = None
        mul_283: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
        mul_284: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_30, mul_283);  div_30 = mul_283 = None
        add_118: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_117, mul_284);  add_117 = mul_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_706: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_118, torch.bfloat16)
        convert_element_type_707: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_51, torch.bfloat16);  gt_51 = None
        mul_285: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_707, 1.1111111111111112);  convert_element_type_707 = None
        mul_286: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_706, mul_285);  convert_element_type_706 = mul_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_538: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_286, [8192, 512]);  mul_286 = None
        permute_309: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_538, [1, 0])
        mm_139: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_309, view_334);  permute_309 = view_334 = None
        mm_140: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_538, permute_311);  view_538 = permute_311 = None
        view_539: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [8, 1024, 2048]);  mm_140 = None
        convert_element_type_713: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_15: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_539, torch.bfloat16);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_715: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_50, torch.bfloat16);  gt_50 = None
        mul_287: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_715, 1.1111111111111112);  convert_element_type_715 = None
        mul_288: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_15, mul_287);  convert_element_type_default_15 = mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_11: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_3, full_default_9, mul_288);  le_3 = mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_540: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_11, [8192, 2048]);  where_11 = None
        permute_313: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_540, [1, 0])
        mm_141: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_313, view_332);  permute_313 = view_332 = None
        mm_142: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_540, permute_315);  view_540 = permute_315 = None
        view_541: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [8, 1024, 512]);  mm_142 = None
        convert_element_type_720: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_541, torch.float32);  view_541 = None
        convert_element_type_721: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_289: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_720, primals_104);  primals_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_149: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_76, rsqrt_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_290: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_720, mul_149);  convert_element_type_720 = mul_149 = None
        sum_41: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_290, [0, 1], True, dtype = torch.float32);  mul_290 = None
        view_542: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_41, [512]);  sum_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_291: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_289, add_76)
        mul_292: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_289, rsqrt_24);  mul_289 = None
        sum_42: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_291, [2], True, dtype = torch.float32);  mul_291 = None
        add_119: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_118, mul_292);  add_118 = mul_292 = None
        pow_47: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_24, 3);  rsqrt_24 = None
        mul_293: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_42, -0.5);  sum_42 = None
        mul_294: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_293, pow_47);  mul_293 = pow_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_82: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_294, [8, 1024, 512]);  mul_294 = None
        div_31: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_82, 512);  expand_82 = None
        pow_48: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_76, 1.0);  add_76 = None
        mul_295: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
        mul_296: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_31, mul_295);  div_31 = mul_295 = None
        add_120: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_119, mul_296);  add_119 = mul_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_722: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16)
        convert_element_type_723: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_49, torch.bfloat16);  gt_49 = None
        mul_297: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_723, 1.1111111111111112);  convert_element_type_723 = None
        mul_298: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_722, mul_297);  convert_element_type_722 = mul_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_543: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_298, [8192, 512]);  mul_298 = None
        permute_317: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_543, [1, 0])
        mm_143: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_317, view_330);  permute_317 = view_330 = None
        mm_144: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_543, permute_319);  view_543 = permute_319 = None
        view_544: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [8, 1024, 512]);  mm_144 = None
        convert_element_type_728: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_545: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_544, [8, 1024, 8, 64]);  view_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_321: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_545, [0, 2, 1, 3]);  view_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_108: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_321, memory_format = torch.contiguous_format);  permute_321 = None
        view_546: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [64, 1024, 64]);  clone_108 = None
        bmm_52: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_322, view_546);  permute_322 = None
        bmm_53: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_546, permute_323);  view_546 = permute_323 = None
        view_547: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [8, 8, 1024, 64]);  bmm_52 = None
        view_548: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [8, 8, 1024, 1024]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_733: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_48, torch.bfloat16);  gt_48 = None
        mul_299: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_733, 1.1111111111111112);  convert_element_type_733 = None
        mul_300: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_548, mul_299);  view_548 = mul_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_734: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_300, torch.float32);  mul_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_322: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [8, 8, 1024, 1024]);  bmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_23: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_322, torch.bfloat16);  view_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_387: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_23, torch.float32);  convert_element_type_default_23 = None
        sub_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_387, amax_13);  convert_element_type_387 = amax_13 = None
        exp_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        div_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        mul_301: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_734, div_17);  convert_element_type_734 = None
        sum_43: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
        neg_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_43, mul_301);  neg_6 = sum_43 = mul_301 = None
        convert_element_type_735: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_549: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_735, [64, 1024, 1024]);  convert_element_type_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_54: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_324, view_549);  permute_324 = None
        bmm_55: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_549, permute_325);  view_549 = permute_325 = None
        view_554: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [8, 8, 64, 1024]);  bmm_54 = None
        view_555: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [8, 8, 1024, 64]);  bmm_55 = None
        permute_326: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_554, [0, 1, 3, 2]);  view_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_327: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_547, [0, 2, 1, 3]);  view_547 = None
        clone_111: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_327, memory_format = torch.contiguous_format);  permute_327 = None
        view_556: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_111, [8, 1024, 512]);  clone_111 = None
        view_557: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_556, [8192, 512]);  view_556 = None
        permute_328: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_557, [1, 0])
        mm_145: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_328, view_176);  permute_328 = None
        mm_146: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_557, permute_330);  view_557 = permute_330 = None
        view_558: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [8, 1024, 512]);  mm_146 = None
        convert_element_type_744: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_558, torch.float32);  view_558 = None
        add_121: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_111, convert_element_type_744);  add_111 = convert_element_type_744 = None
        convert_element_type_745: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_332: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_326, [0, 2, 1, 3]);  permute_326 = None
        view_559: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_332, [8, 1024, 512]);  permute_332 = None
        clone_112: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_559, memory_format = torch.contiguous_format);  view_559 = None
        view_560: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_112, [8192, 512]);  clone_112 = None
        permute_333: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_560, [1, 0])
        mm_147: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_333, view_176);  permute_333 = None
        mm_148: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_560, permute_335);  view_560 = permute_335 = None
        view_561: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_148, [8, 1024, 512]);  mm_148 = None
        convert_element_type_750: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_561, torch.float32);  view_561 = None
        add_122: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_121, convert_element_type_750);  add_121 = convert_element_type_750 = None
        convert_element_type_751: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_337: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_555, [0, 2, 1, 3]);  view_555 = None
        clone_113: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
        view_562: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_113, [8, 1024, 512]);  clone_113 = None
        view_563: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_562, [8192, 512]);  view_562 = None
        permute_338: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_563, [1, 0])
        mm_149: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_338, view_311);  permute_338 = view_311 = None
        mm_150: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_563, permute_340);  view_563 = permute_340 = None
        view_564: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_150, [8, 1024, 512]);  mm_150 = None
        convert_element_type_756: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_564, torch.float32);  view_564 = None
        convert_element_type_757: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_302: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_756, primals_99);  primals_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_143: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_73, rsqrt_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_303: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_756, mul_143);  convert_element_type_756 = mul_143 = None
        sum_44: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1], True, dtype = torch.float32);  mul_303 = None
        view_565: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_44, [512]);  sum_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_304: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, add_73)
        mul_305: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, rsqrt_23);  mul_302 = None
        sum_45: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True, dtype = torch.float32);  mul_304 = None
        add_123: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_120, mul_305);  add_120 = mul_305 = None
        pow_49: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_23, 3);  rsqrt_23 = None
        mul_306: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_45, -0.5);  sum_45 = None
        mul_307: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, pow_49);  mul_306 = pow_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_83: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_307, [8, 1024, 512]);  mul_307 = None
        div_32: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_83, 512);  expand_83 = None
        pow_50: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_73, 1.0);  add_73 = None
        mul_308: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
        mul_309: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_32, mul_308);  div_32 = mul_308 = None
        add_124: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_123, mul_309);  add_123 = mul_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_758: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16)
        convert_element_type_759: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_47, torch.bfloat16);  gt_47 = None
        mul_310: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_759, 1.1111111111111112);  convert_element_type_759 = None
        mul_311: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_758, mul_310);  convert_element_type_758 = mul_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_566: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_311, [8192, 512]);  mul_311 = None
        permute_342: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_151: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_342, view_309);  permute_342 = view_309 = None
        mm_152: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_566, permute_344);  view_566 = permute_344 = None
        view_567: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_152, [8, 1024, 512]);  mm_152 = None
        convert_element_type_764: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_568: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_567, [8, 1024, 8, 64]);  view_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_346: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_115: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_346, memory_format = torch.contiguous_format);  permute_346 = None
        view_569: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [64, 1024, 64]);  clone_115 = None
        bmm_56: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_347, view_569);  permute_347 = None
        bmm_57: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_569, permute_348);  view_569 = permute_348 = None
        view_570: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [8, 8, 1024, 64]);  bmm_56 = None
        view_571: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [8, 8, 1024, 1024]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_769: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_46, torch.bfloat16);  gt_46 = None
        mul_312: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_769, 1.1111111111111112);  convert_element_type_769 = None
        mul_313: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_571, mul_312);  view_571 = mul_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_770: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_313, torch.float32);  mul_313 = None
        convert_element_type_365: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_364, torch.float32);  convert_element_type_364 = None
        sub_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_365, amax_12);  convert_element_type_365 = amax_12 = None
        exp_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        div_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        mul_314: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_770, div_16);  convert_element_type_770 = None
        sum_46: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_314, [-1], True)
        neg_7: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_46, mul_314);  neg_7 = sum_46 = mul_314 = None
        convert_element_type_771: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_572: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_771, [64, 1024, 1024]);  convert_element_type_771 = None
        view_574: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_572, [8, 8, 1024, 1024]);  view_572 = None
        view_575: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_574, [64, 1024, 1024])
        convert_element_type_772: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_574, torch.float32);  view_574 = None
        add_125: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_114, convert_element_type_772);  add_114 = convert_element_type_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_58: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_349, view_575);  permute_349 = None
        bmm_59: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_575, permute_350);  view_575 = permute_350 = None
        view_577: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [8, 8, 64, 1024]);  bmm_58 = None
        view_578: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [8, 8, 1024, 64]);  bmm_59 = None
        permute_351: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_577, [0, 1, 3, 2]);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_352: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
        clone_118: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None
        view_579: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_118, [8, 1024, 512]);  clone_118 = None
        view_580: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_579, [8192, 512]);  view_579 = None
        permute_353: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_580, [1, 0])
        mm_153: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_353, view_290);  permute_353 = None
        mm_154: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_580, permute_355);  view_580 = permute_355 = None
        view_581: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_154, [8, 1024, 512]);  mm_154 = None
        convert_element_type_781: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_581, torch.float32);  view_581 = None
        convert_element_type_782: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_357: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_351, [0, 2, 1, 3]);  permute_351 = None
        view_582: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_357, [8, 1024, 512]);  permute_357 = None
        clone_119: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_582, memory_format = torch.contiguous_format);  view_582 = None
        view_583: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_119, [8192, 512]);  clone_119 = None
        permute_358: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_583, [1, 0])
        mm_155: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_358, view_290);  permute_358 = None
        mm_156: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_583, permute_360);  view_583 = permute_360 = None
        view_584: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_156, [8, 1024, 512]);  mm_156 = None
        convert_element_type_787: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_584, torch.float32);  view_584 = None
        add_126: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_781, convert_element_type_787);  convert_element_type_781 = convert_element_type_787 = None
        convert_element_type_788: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_155, torch.float32);  mm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_362: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
        clone_120: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_362, memory_format = torch.contiguous_format);  permute_362 = None
        view_585: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_120, [8, 1024, 512]);  clone_120 = None
        view_586: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_585, [8192, 512]);  view_585 = None
        permute_363: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_586, [1, 0])
        mm_157: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_363, view_290);  permute_363 = view_290 = None
        mm_158: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_586, permute_365);  view_586 = permute_365 = None
        view_587: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_158, [8, 1024, 512]);  mm_158 = None
        convert_element_type_793: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_587, torch.float32);  view_587 = None
        add_127: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_126, convert_element_type_793);  add_126 = convert_element_type_793 = None
        convert_element_type_794: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_315: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_127, primals_94);  primals_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_137: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_70, rsqrt_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_316: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_127, mul_137);  add_127 = mul_137 = None
        sum_47: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1], True, dtype = torch.float32);  mul_316 = None
        view_588: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_47, [512]);  sum_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_317: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, add_70)
        mul_318: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, rsqrt_22);  mul_315 = None
        sum_48: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True, dtype = torch.float32);  mul_317 = None
        add_128: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_124, mul_318);  add_124 = mul_318 = None
        pow_51: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_22, 3);  rsqrt_22 = None
        mul_319: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_48, -0.5);  sum_48 = None
        mul_320: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_319, pow_51);  mul_319 = pow_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_84: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_320, [8, 1024, 512]);  mul_320 = None
        div_33: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_84, 512);  expand_84 = None
        pow_52: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_70, 1.0);  add_70 = None
        mul_321: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_52, 2.0);  pow_52 = None
        mul_322: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_33, mul_321);  div_33 = mul_321 = None
        add_129: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_128, mul_322);  add_128 = mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_795: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_129, torch.bfloat16)
        convert_element_type_796: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_45, torch.bfloat16);  gt_45 = None
        mul_323: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_796, 1.1111111111111112);  convert_element_type_796 = None
        mul_324: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_795, mul_323);  convert_element_type_795 = mul_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_589: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_324, [8192, 512]);  mul_324 = None
        permute_367: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_589, [1, 0])
        mm_159: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_367, view_288);  permute_367 = view_288 = None
        mm_160: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_589, permute_369);  view_589 = permute_369 = None
        view_590: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_160, [8, 1024, 2048]);  mm_160 = None
        convert_element_type_802: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_14: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_590, torch.bfloat16);  view_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_804: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_44, torch.bfloat16);  gt_44 = None
        mul_325: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_804, 1.1111111111111112);  convert_element_type_804 = None
        mul_326: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_14, mul_325);  convert_element_type_default_14 = mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_12: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_4, full_default_9, mul_326);  le_4 = mul_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_591: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_12, [8192, 2048]);  where_12 = None
        permute_371: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_591, [1, 0])
        mm_161: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_371, view_286);  permute_371 = view_286 = None
        mm_162: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_591, permute_373);  view_591 = permute_373 = None
        view_592: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_162, [8, 1024, 512]);  mm_162 = None
        convert_element_type_809: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_592, torch.float32);  view_592 = None
        convert_element_type_810: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_327: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_809, primals_91);  primals_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_131: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_68, rsqrt_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_328: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_809, mul_131);  convert_element_type_809 = mul_131 = None
        sum_49: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1], True, dtype = torch.float32);  mul_328 = None
        view_593: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [512]);  sum_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_329: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, add_68)
        mul_330: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, rsqrt_21);  mul_327 = None
        sum_50: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True, dtype = torch.float32);  mul_329 = None
        add_130: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_129, mul_330);  add_129 = mul_330 = None
        pow_53: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_21, 3);  rsqrt_21 = None
        mul_331: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_50, -0.5);  sum_50 = None
        mul_332: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_331, pow_53);  mul_331 = pow_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_85: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_332, [8, 1024, 512]);  mul_332 = None
        div_34: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_85, 512);  expand_85 = None
        pow_54: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_68, 1.0);  add_68 = None
        mul_333: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_54, 2.0);  pow_54 = None
        mul_334: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_34, mul_333);  div_34 = mul_333 = None
        add_131: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_130, mul_334);  add_130 = mul_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_811: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16)
        convert_element_type_812: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_43, torch.bfloat16);  gt_43 = None
        mul_335: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_812, 1.1111111111111112);  convert_element_type_812 = None
        mul_336: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_811, mul_335);  convert_element_type_811 = mul_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_594: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_336, [8192, 512]);  mul_336 = None
        permute_375: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_594, [1, 0])
        mm_163: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_375, view_284);  permute_375 = view_284 = None
        mm_164: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_594, permute_377);  view_594 = permute_377 = None
        view_595: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_164, [8, 1024, 512]);  mm_164 = None
        convert_element_type_817: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_596: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_595, [8, 1024, 8, 64]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_379: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_124: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_379, memory_format = torch.contiguous_format);  permute_379 = None
        view_597: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [64, 1024, 64]);  clone_124 = None
        bmm_60: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_380, view_597);  permute_380 = None
        bmm_61: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_597, permute_381);  view_597 = permute_381 = None
        view_598: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [8, 8, 1024, 64]);  bmm_60 = None
        view_599: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [8, 8, 1024, 1024]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_822: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_42, torch.bfloat16);  gt_42 = None
        mul_337: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_822, 1.1111111111111112);  convert_element_type_822 = None
        mul_338: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_599, mul_337);  view_599 = mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_823: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_338, torch.float32);  mul_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_276: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [8, 8, 1024, 1024]);  bmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_25: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_276, torch.bfloat16);  view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_334: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_25, torch.float32);  convert_element_type_default_25 = None
        sub_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_334, amax_11);  convert_element_type_334 = amax_11 = None
        exp_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        div_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        mul_339: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_823, div_15);  convert_element_type_823 = None
        sum_51: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [-1], True)
        neg_8: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_8, sum_51, mul_339);  neg_8 = sum_51 = mul_339 = None
        convert_element_type_824: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_600: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_824, [64, 1024, 1024]);  convert_element_type_824 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_62: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_382, view_600);  permute_382 = None
        bmm_63: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_600, permute_383);  view_600 = permute_383 = None
        view_605: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [8, 8, 64, 1024]);  bmm_62 = None
        view_606: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [8, 8, 1024, 64]);  bmm_63 = None
        permute_384: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_605, [0, 1, 3, 2]);  view_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_385: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
        clone_127: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
        view_607: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_127, [8, 1024, 512]);  clone_127 = None
        view_608: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_607, [8192, 512]);  view_607 = None
        permute_386: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_608, [1, 0])
        mm_165: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_386, view_176);  permute_386 = None
        mm_166: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_608, permute_388);  view_608 = permute_388 = None
        view_609: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_166, [8, 1024, 512]);  mm_166 = None
        convert_element_type_833: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_609, torch.float32);  view_609 = None
        add_132: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_122, convert_element_type_833);  add_122 = convert_element_type_833 = None
        convert_element_type_834: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_390: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_384, [0, 2, 1, 3]);  permute_384 = None
        view_610: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_390, [8, 1024, 512]);  permute_390 = None
        clone_128: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_610, memory_format = torch.contiguous_format);  view_610 = None
        view_611: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_128, [8192, 512]);  clone_128 = None
        permute_391: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_611, [1, 0])
        mm_167: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_391, view_176);  permute_391 = None
        mm_168: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_611, permute_393);  view_611 = permute_393 = None
        view_612: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_168, [8, 1024, 512]);  mm_168 = None
        convert_element_type_839: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_612, torch.float32);  view_612 = None
        add_133: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_132, convert_element_type_839);  add_132 = convert_element_type_839 = None
        convert_element_type_840: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_395: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None
        clone_129: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_395, memory_format = torch.contiguous_format);  permute_395 = None
        view_613: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [8, 1024, 512]);  clone_129 = None
        view_614: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_613, [8192, 512]);  view_613 = None
        permute_396: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_614, [1, 0])
        mm_169: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_396, view_265);  permute_396 = view_265 = None
        mm_170: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_614, permute_398);  view_614 = permute_398 = None
        view_615: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_170, [8, 1024, 512]);  mm_170 = None
        convert_element_type_845: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_615, torch.float32);  view_615 = None
        convert_element_type_846: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_169, torch.float32);  mm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_340: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_845, primals_86);  primals_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_125: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_65, rsqrt_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_341: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_845, mul_125);  convert_element_type_845 = mul_125 = None
        sum_52: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 1], True, dtype = torch.float32);  mul_341 = None
        view_616: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_52, [512]);  sum_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_342: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_340, add_65)
        mul_343: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_340, rsqrt_20);  mul_340 = None
        sum_53: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True, dtype = torch.float32);  mul_342 = None
        add_134: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_131, mul_343);  add_131 = mul_343 = None
        pow_55: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_20, 3);  rsqrt_20 = None
        mul_344: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_53, -0.5);  sum_53 = None
        mul_345: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_344, pow_55);  mul_344 = pow_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_86: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_345, [8, 1024, 512]);  mul_345 = None
        div_35: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_86, 512);  expand_86 = None
        pow_56: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_65, 1.0);  add_65 = None
        mul_346: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_56, 2.0);  pow_56 = None
        mul_347: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_35, mul_346);  div_35 = mul_346 = None
        add_135: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, mul_347);  add_134 = mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_847: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.bfloat16)
        convert_element_type_848: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_41, torch.bfloat16);  gt_41 = None
        mul_348: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_848, 1.1111111111111112);  convert_element_type_848 = None
        mul_349: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_847, mul_348);  convert_element_type_847 = mul_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_617: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_349, [8192, 512]);  mul_349 = None
        permute_400: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_617, [1, 0])
        mm_171: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_400, view_263);  permute_400 = view_263 = None
        mm_172: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_617, permute_402);  view_617 = permute_402 = None
        view_618: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_172, [8, 1024, 512]);  mm_172 = None
        convert_element_type_853: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_619: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_618, [8, 1024, 8, 64]);  view_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_404: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_131: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_404, memory_format = torch.contiguous_format);  permute_404 = None
        view_620: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [64, 1024, 64]);  clone_131 = None
        bmm_64: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_405, view_620);  permute_405 = None
        bmm_65: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_620, permute_406);  view_620 = permute_406 = None
        view_621: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [8, 8, 1024, 64]);  bmm_64 = None
        view_622: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [8, 8, 1024, 1024]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_858: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_40, torch.bfloat16);  gt_40 = None
        mul_350: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_858, 1.1111111111111112);  convert_element_type_858 = None
        mul_351: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_622, mul_350);  view_622 = mul_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_859: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_351, torch.float32);  mul_351 = None
        convert_element_type_312: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_311, torch.float32);  convert_element_type_311 = None
        sub_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_312, amax_10);  convert_element_type_312 = amax_10 = None
        exp_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        div_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        mul_352: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_859, div_14);  convert_element_type_859 = None
        sum_54: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_352, [-1], True)
        neg_9: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_14);  div_14 = None
        fma_7: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_9, sum_54, mul_352);  neg_9 = sum_54 = mul_352 = None
        convert_element_type_860: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_623: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_860, [64, 1024, 1024]);  convert_element_type_860 = None
        view_625: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_623, [8, 8, 1024, 1024]);  view_623 = None
        view_626: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_625, [64, 1024, 1024])
        convert_element_type_861: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_625, torch.float32);  view_625 = None
        add_136: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_125, convert_element_type_861);  add_125 = convert_element_type_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_66: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_407, view_626);  permute_407 = None
        bmm_67: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_626, permute_408);  view_626 = permute_408 = None
        view_628: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [8, 8, 64, 1024]);  bmm_66 = None
        view_629: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [8, 8, 1024, 64]);  bmm_67 = None
        permute_409: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_628, [0, 1, 3, 2]);  view_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_410: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_621, [0, 2, 1, 3]);  view_621 = None
        clone_134: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
        view_630: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_134, [8, 1024, 512]);  clone_134 = None
        view_631: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_630, [8192, 512]);  view_630 = None
        permute_411: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_173: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_411, view_244);  permute_411 = None
        mm_174: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_631, permute_413);  view_631 = permute_413 = None
        view_632: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_174, [8, 1024, 512]);  mm_174 = None
        convert_element_type_870: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_632, torch.float32);  view_632 = None
        convert_element_type_871: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_415: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_409, [0, 2, 1, 3]);  permute_409 = None
        view_633: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_415, [8, 1024, 512]);  permute_415 = None
        clone_135: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_633, memory_format = torch.contiguous_format);  view_633 = None
        view_634: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_135, [8192, 512]);  clone_135 = None
        permute_416: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_634, [1, 0])
        mm_175: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_416, view_244);  permute_416 = None
        mm_176: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_634, permute_418);  view_634 = permute_418 = None
        view_635: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_176, [8, 1024, 512]);  mm_176 = None
        convert_element_type_876: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_635, torch.float32);  view_635 = None
        add_137: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_870, convert_element_type_876);  convert_element_type_870 = convert_element_type_876 = None
        convert_element_type_877: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_420: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None
        clone_136: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_420, memory_format = torch.contiguous_format);  permute_420 = None
        view_636: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [8, 1024, 512]);  clone_136 = None
        view_637: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_636, [8192, 512]);  view_636 = None
        permute_421: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_637, [1, 0])
        mm_177: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_421, view_244);  permute_421 = view_244 = None
        mm_178: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_637, permute_423);  view_637 = permute_423 = None
        view_638: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_178, [8, 1024, 512]);  mm_178 = None
        convert_element_type_882: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_638, torch.float32);  view_638 = None
        add_138: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_137, convert_element_type_882);  add_137 = convert_element_type_882 = None
        convert_element_type_883: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_353: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_138, primals_81);  primals_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_119: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_62, rsqrt_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_354: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_138, mul_119);  add_138 = mul_119 = None
        sum_55: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 1], True, dtype = torch.float32);  mul_354 = None
        view_639: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_55, [512]);  sum_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_355: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, add_62)
        mul_356: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, rsqrt_19);  mul_353 = None
        sum_56: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True, dtype = torch.float32);  mul_355 = None
        add_139: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_135, mul_356);  add_135 = mul_356 = None
        pow_57: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_19, 3);  rsqrt_19 = None
        mul_357: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_56, -0.5);  sum_56 = None
        mul_358: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, pow_57);  mul_357 = pow_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_87: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_358, [8, 1024, 512]);  mul_358 = None
        div_36: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_87, 512);  expand_87 = None
        pow_58: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_62, 1.0);  add_62 = None
        mul_359: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_58, 2.0);  pow_58 = None
        mul_360: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_36, mul_359);  div_36 = mul_359 = None
        add_140: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_139, mul_360);  add_139 = mul_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_884: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_140, torch.bfloat16)
        convert_element_type_885: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_39, torch.bfloat16);  gt_39 = None
        mul_361: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_885, 1.1111111111111112);  convert_element_type_885 = None
        mul_362: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_884, mul_361);  convert_element_type_884 = mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_640: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_362, [8192, 512]);  mul_362 = None
        permute_425: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_640, [1, 0])
        mm_179: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_425, view_242);  permute_425 = view_242 = None
        mm_180: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_640, permute_427);  view_640 = permute_427 = None
        view_641: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_180, [8, 1024, 2048]);  mm_180 = None
        convert_element_type_891: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_179, torch.float32);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_13: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_641, torch.bfloat16);  view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_893: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_38, torch.bfloat16);  gt_38 = None
        mul_363: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_893, 1.1111111111111112);  convert_element_type_893 = None
        mul_364: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_13, mul_363);  convert_element_type_default_13 = mul_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_13: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_5, full_default_9, mul_364);  le_5 = mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_642: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_13, [8192, 2048]);  where_13 = None
        permute_429: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_642, [1, 0])
        mm_181: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_429, view_240);  permute_429 = view_240 = None
        mm_182: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_642, permute_431);  view_642 = permute_431 = None
        view_643: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_182, [8, 1024, 512]);  mm_182 = None
        convert_element_type_898: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_643, torch.float32);  view_643 = None
        convert_element_type_899: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_365: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_898, primals_78);  primals_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_113: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_60, rsqrt_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_366: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_898, mul_113);  convert_element_type_898 = mul_113 = None
        sum_57: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 1], True, dtype = torch.float32);  mul_366 = None
        view_644: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [512]);  sum_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_367: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, add_60)
        mul_368: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, rsqrt_18);  mul_365 = None
        sum_58: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True, dtype = torch.float32);  mul_367 = None
        add_141: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_140, mul_368);  add_140 = mul_368 = None
        pow_59: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_18, 3);  rsqrt_18 = None
        mul_369: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_58, -0.5);  sum_58 = None
        mul_370: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_369, pow_59);  mul_369 = pow_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_88: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_370, [8, 1024, 512]);  mul_370 = None
        div_37: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_88, 512);  expand_88 = None
        pow_60: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_60, 1.0);  add_60 = None
        mul_371: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
        mul_372: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_37, mul_371);  div_37 = mul_371 = None
        add_142: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_141, mul_372);  add_141 = mul_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_900: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16)
        convert_element_type_901: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_37, torch.bfloat16);  gt_37 = None
        mul_373: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_901, 1.1111111111111112);  convert_element_type_901 = None
        mul_374: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_900, mul_373);  convert_element_type_900 = mul_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_645: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_374, [8192, 512]);  mul_374 = None
        permute_433: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_645, [1, 0])
        mm_183: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_433, view_238);  permute_433 = view_238 = None
        mm_184: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_645, permute_435);  view_645 = permute_435 = None
        view_646: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_184, [8, 1024, 512]);  mm_184 = None
        convert_element_type_906: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_647: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_646, [8, 1024, 8, 64]);  view_646 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_437: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_647, [0, 2, 1, 3]);  view_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_140: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_437, memory_format = torch.contiguous_format);  permute_437 = None
        view_648: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_140, [64, 1024, 64]);  clone_140 = None
        bmm_68: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_438, view_648);  permute_438 = None
        bmm_69: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_648, permute_439);  view_648 = permute_439 = None
        view_649: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [8, 8, 1024, 64]);  bmm_68 = None
        view_650: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [8, 8, 1024, 1024]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_911: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_36, torch.bfloat16);  gt_36 = None
        mul_375: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_911, 1.1111111111111112);  convert_element_type_911 = None
        mul_376: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_650, mul_375);  view_650 = mul_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_912: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_376, torch.float32);  mul_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_230: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_18, [8, 8, 1024, 1024]);  bmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_27: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_230, torch.bfloat16);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_281: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_27, torch.float32);  convert_element_type_default_27 = None
        sub_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_281, amax_9);  convert_element_type_281 = amax_9 = None
        exp_9: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        div_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        mul_377: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_912, div_13);  convert_element_type_912 = None
        sum_59: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_377, [-1], True)
        neg_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_8: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_10, sum_59, mul_377);  neg_10 = sum_59 = mul_377 = None
        convert_element_type_913: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_651: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_913, [64, 1024, 1024]);  convert_element_type_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_70: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_440, view_651);  permute_440 = None
        bmm_71: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_651, permute_441);  view_651 = permute_441 = None
        view_656: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [8, 8, 64, 1024]);  bmm_70 = None
        view_657: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [8, 8, 1024, 64]);  bmm_71 = None
        permute_442: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_656, [0, 1, 3, 2]);  view_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_443: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_649, [0, 2, 1, 3]);  view_649 = None
        clone_143: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_443, memory_format = torch.contiguous_format);  permute_443 = None
        view_658: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_143, [8, 1024, 512]);  clone_143 = None
        view_659: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_658, [8192, 512]);  view_658 = None
        permute_444: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_659, [1, 0])
        mm_185: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_444, view_176);  permute_444 = None
        mm_186: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_659, permute_446);  view_659 = permute_446 = None
        view_660: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_186, [8, 1024, 512]);  mm_186 = None
        convert_element_type_922: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_660, torch.float32);  view_660 = None
        add_143: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_133, convert_element_type_922);  add_133 = convert_element_type_922 = None
        convert_element_type_923: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_448: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_442, [0, 2, 1, 3]);  permute_442 = None
        view_661: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_448, [8, 1024, 512]);  permute_448 = None
        clone_144: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_661, memory_format = torch.contiguous_format);  view_661 = None
        view_662: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_144, [8192, 512]);  clone_144 = None
        permute_449: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_662, [1, 0])
        mm_187: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_449, view_176);  permute_449 = None
        mm_188: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_662, permute_451);  view_662 = permute_451 = None
        view_663: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_188, [8, 1024, 512]);  mm_188 = None
        convert_element_type_928: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_663, torch.float32);  view_663 = None
        add_144: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_143, convert_element_type_928);  add_143 = convert_element_type_928 = None
        convert_element_type_929: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_453: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_657, [0, 2, 1, 3]);  view_657 = None
        clone_145: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_453, memory_format = torch.contiguous_format);  permute_453 = None
        view_664: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [8, 1024, 512]);  clone_145 = None
        view_665: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_664, [8192, 512]);  view_664 = None
        permute_454: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_665, [1, 0])
        mm_189: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_454, view_219);  permute_454 = view_219 = None
        mm_190: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_665, permute_456);  view_665 = permute_456 = None
        view_666: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_190, [8, 1024, 512]);  mm_190 = None
        convert_element_type_934: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_666, torch.float32);  view_666 = None
        convert_element_type_935: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_378: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_934, primals_73);  primals_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_107: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_57, rsqrt_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_379: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_934, mul_107);  convert_element_type_934 = mul_107 = None
        sum_60: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1], True, dtype = torch.float32);  mul_379 = None
        view_667: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_60, [512]);  sum_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_380: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_378, add_57)
        mul_381: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_378, rsqrt_17);  mul_378 = None
        sum_61: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_380, [2], True, dtype = torch.float32);  mul_380 = None
        add_145: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_142, mul_381);  add_142 = mul_381 = None
        pow_61: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_17, 3);  rsqrt_17 = None
        mul_382: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_61, -0.5);  sum_61 = None
        mul_383: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_382, pow_61);  mul_382 = pow_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_89: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_383, [8, 1024, 512]);  mul_383 = None
        div_38: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_89, 512);  expand_89 = None
        pow_62: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_57, 1.0);  add_57 = None
        mul_384: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_62, 2.0);  pow_62 = None
        mul_385: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, mul_384);  div_38 = mul_384 = None
        add_146: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, mul_385);  add_145 = mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_936: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_146, torch.bfloat16)
        convert_element_type_937: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_35, torch.bfloat16);  gt_35 = None
        mul_386: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_937, 1.1111111111111112);  convert_element_type_937 = None
        mul_387: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_936, mul_386);  convert_element_type_936 = mul_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_668: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_387, [8192, 512]);  mul_387 = None
        permute_458: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_668, [1, 0])
        mm_191: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_458, view_217);  permute_458 = view_217 = None
        mm_192: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_668, permute_460);  view_668 = permute_460 = None
        view_669: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_192, [8, 1024, 512]);  mm_192 = None
        convert_element_type_942: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_191, torch.float32);  mm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_670: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_669, [8, 1024, 8, 64]);  view_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_462: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_670, [0, 2, 1, 3]);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_147: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_462, memory_format = torch.contiguous_format);  permute_462 = None
        view_671: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_147, [64, 1024, 64]);  clone_147 = None
        bmm_72: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_463, view_671);  permute_463 = None
        bmm_73: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_671, permute_464);  view_671 = permute_464 = None
        view_672: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_72, [8, 8, 1024, 64]);  bmm_72 = None
        view_673: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_73, [8, 8, 1024, 1024]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_947: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_34, torch.bfloat16);  gt_34 = None
        mul_388: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_947, 1.1111111111111112);  convert_element_type_947 = None
        mul_389: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_673, mul_388);  view_673 = mul_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_948: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_389, torch.float32);  mul_389 = None
        convert_element_type_259: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_258, torch.float32);  convert_element_type_258 = None
        sub_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_259, amax_8);  convert_element_type_259 = amax_8 = None
        exp_8: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        div_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        mul_390: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_948, div_12);  convert_element_type_948 = None
        sum_62: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_390, [-1], True)
        neg_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_9: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_11, sum_62, mul_390);  neg_11 = sum_62 = mul_390 = None
        convert_element_type_949: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_674: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_949, [64, 1024, 1024]);  convert_element_type_949 = None
        view_676: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_674, [8, 8, 1024, 1024]);  view_674 = None
        view_677: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_676, [64, 1024, 1024])
        convert_element_type_950: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_676, torch.float32);  view_676 = None
        add_147: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_136, convert_element_type_950);  add_136 = convert_element_type_950 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_74: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_465, view_677);  permute_465 = None
        bmm_75: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_677, permute_466);  view_677 = permute_466 = None
        view_679: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_74, [8, 8, 64, 1024]);  bmm_74 = None
        view_680: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_75, [8, 8, 1024, 64]);  bmm_75 = None
        permute_467: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_679, [0, 1, 3, 2]);  view_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_468: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None
        clone_150: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_468, memory_format = torch.contiguous_format);  permute_468 = None
        view_681: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_150, [8, 1024, 512]);  clone_150 = None
        view_682: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_681, [8192, 512]);  view_681 = None
        permute_469: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_682, [1, 0])
        mm_193: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_469, view_198);  permute_469 = None
        mm_194: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_682, permute_471);  view_682 = permute_471 = None
        view_683: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_194, [8, 1024, 512]);  mm_194 = None
        convert_element_type_959: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_683, torch.float32);  view_683 = None
        convert_element_type_960: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_193, torch.float32);  mm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_473: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_467, [0, 2, 1, 3]);  permute_467 = None
        view_684: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_473, [8, 1024, 512]);  permute_473 = None
        clone_151: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_684, memory_format = torch.contiguous_format);  view_684 = None
        view_685: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [8192, 512]);  clone_151 = None
        permute_474: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_685, [1, 0])
        mm_195: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_474, view_198);  permute_474 = None
        mm_196: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_685, permute_476);  view_685 = permute_476 = None
        view_686: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_196, [8, 1024, 512]);  mm_196 = None
        convert_element_type_965: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_686, torch.float32);  view_686 = None
        add_148: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_959, convert_element_type_965);  convert_element_type_959 = convert_element_type_965 = None
        convert_element_type_966: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_478: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_680, [0, 2, 1, 3]);  view_680 = None
        clone_152: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_478, memory_format = torch.contiguous_format);  permute_478 = None
        view_687: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_152, [8, 1024, 512]);  clone_152 = None
        view_688: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_687, [8192, 512]);  view_687 = None
        permute_479: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_688, [1, 0])
        mm_197: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_479, view_198);  permute_479 = view_198 = None
        mm_198: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_688, permute_481);  view_688 = permute_481 = None
        view_689: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_198, [8, 1024, 512]);  mm_198 = None
        convert_element_type_971: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_689, torch.float32);  view_689 = None
        add_149: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_148, convert_element_type_971);  add_148 = convert_element_type_971 = None
        convert_element_type_972: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_391: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_149, primals_68);  primals_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_101: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_54, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_392: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_149, mul_101);  add_149 = mul_101 = None
        sum_63: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 1], True, dtype = torch.float32);  mul_392 = None
        view_690: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [512]);  sum_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_393: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_391, add_54)
        mul_394: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_391, rsqrt_16);  mul_391 = None
        sum_64: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True, dtype = torch.float32);  mul_393 = None
        add_150: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_146, mul_394);  add_146 = mul_394 = None
        pow_63: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_16, 3);  rsqrt_16 = None
        mul_395: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_64, -0.5);  sum_64 = None
        mul_396: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_395, pow_63);  mul_395 = pow_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_90: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_396, [8, 1024, 512]);  mul_396 = None
        div_39: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_90, 512);  expand_90 = None
        pow_64: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_54, 1.0);  add_54 = None
        mul_397: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
        mul_398: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_39, mul_397);  div_39 = mul_397 = None
        add_151: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_150, mul_398);  add_150 = mul_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_973: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_151, torch.bfloat16)
        convert_element_type_974: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_33, torch.bfloat16);  gt_33 = None
        mul_399: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_974, 1.1111111111111112);  convert_element_type_974 = None
        mul_400: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_973, mul_399);  convert_element_type_973 = mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_691: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_400, [8192, 512]);  mul_400 = None
        permute_483: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_691, [1, 0])
        mm_199: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_483, view_196);  permute_483 = view_196 = None
        mm_200: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_691, permute_485);  view_691 = permute_485 = None
        view_692: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_200, [8, 1024, 2048]);  mm_200 = None
        convert_element_type_980: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_12: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_692, torch.bfloat16);  view_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_982: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_32, torch.bfloat16);  gt_32 = None
        mul_401: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_982, 1.1111111111111112);  convert_element_type_982 = None
        mul_402: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_12, mul_401);  convert_element_type_default_12 = mul_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_14: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_6, full_default_9, mul_402);  le_6 = mul_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_693: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_14, [8192, 2048]);  where_14 = None
        permute_487: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_693, [1, 0])
        mm_201: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_487, view_194);  permute_487 = view_194 = None
        mm_202: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_693, permute_489);  view_693 = permute_489 = None
        view_694: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_202, [8, 1024, 512]);  mm_202 = None
        convert_element_type_987: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_694, torch.float32);  view_694 = None
        convert_element_type_988: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_201, torch.float32);  mm_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_403: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_987, primals_65);  primals_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_95: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_52, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_404: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_987, mul_95);  convert_element_type_987 = mul_95 = None
        sum_65: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 1], True, dtype = torch.float32);  mul_404 = None
        view_695: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_65, [512]);  sum_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_405: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_403, add_52)
        mul_406: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_403, rsqrt_15);  mul_403 = None
        sum_66: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_405, [2], True, dtype = torch.float32);  mul_405 = None
        add_152: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_151, mul_406);  add_151 = mul_406 = None
        pow_65: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_15, 3);  rsqrt_15 = None
        mul_407: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_66, -0.5);  sum_66 = None
        mul_408: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_407, pow_65);  mul_407 = pow_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_91: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_408, [8, 1024, 512]);  mul_408 = None
        div_40: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_91, 512);  expand_91 = None
        pow_66: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_52, 1.0);  add_52 = None
        mul_409: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_66, 2.0);  pow_66 = None
        mul_410: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_40, mul_409);  div_40 = mul_409 = None
        add_153: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_152, mul_410);  add_152 = mul_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:393 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_989: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16)
        convert_element_type_990: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_31, torch.bfloat16);  gt_31 = None
        mul_411: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_990, 1.1111111111111112);  convert_element_type_990 = None
        mul_412: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_989, mul_411);  convert_element_type_989 = mul_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_696: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_412, [8192, 512]);  mul_412 = None
        permute_491: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_696, [1, 0])
        mm_203: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_491, view_192);  permute_491 = view_192 = None
        mm_204: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_696, permute_493);  view_696 = permute_493 = None
        view_697: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_204, [8, 1024, 512]);  mm_204 = None
        convert_element_type_995: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_203, torch.float32);  mm_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_698: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_697, [8, 1024, 8, 64]);  view_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_495: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_156: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_495, memory_format = torch.contiguous_format);  permute_495 = None
        view_699: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_156, [64, 1024, 64]);  clone_156 = None
        bmm_76: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_496, view_699);  permute_496 = None
        bmm_77: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_699, permute_497);  view_699 = permute_497 = None
        view_700: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_76, [8, 8, 1024, 64]);  bmm_76 = None
        view_701: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_77, [8, 8, 1024, 1024]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_1000: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_30, torch.bfloat16);  gt_30 = None
        mul_413: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1000, 1.1111111111111112);  convert_element_type_1000 = None
        mul_414: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_701, mul_413);  view_701 = mul_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_1001: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_414, torch.float32);  mul_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_184: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [8, 8, 1024, 1024]);  bmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        convert_element_type_default_29: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_184, torch.bfloat16);  view_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_228: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_default_29, torch.float32);  convert_element_type_default_29 = None
        sub_9: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_228, amax_7);  convert_element_type_228 = amax_7 = None
        exp_7: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        div_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        mul_415: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1001, div_11);  convert_element_type_1001 = None
        sum_67: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_415, [-1], True)
        neg_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_11);  div_11 = None
        fma_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_12, sum_67, mul_415);  neg_12 = sum_67 = mul_415 = None
        convert_element_type_1002: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_702: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1002, [64, 1024, 1024]);  convert_element_type_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_78: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_498, view_702);  permute_498 = None
        bmm_79: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_702, permute_499);  view_702 = permute_499 = None
        view_707: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_78, [8, 8, 64, 1024]);  bmm_78 = None
        view_708: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_79, [8, 8, 1024, 64]);  bmm_79 = None
        permute_500: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_707, [0, 1, 3, 2]);  view_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_501: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_700, [0, 2, 1, 3]);  view_700 = None
        clone_159: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_501, memory_format = torch.contiguous_format);  permute_501 = None
        view_709: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_159, [8, 1024, 512]);  clone_159 = None
        view_710: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_709, [8192, 512]);  view_709 = None
        permute_502: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_710, [1, 0])
        mm_205: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_502, view_176);  permute_502 = None
        mm_206: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_710, permute_504);  view_710 = permute_504 = None
        view_711: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_206, [8, 1024, 512]);  mm_206 = None
        convert_element_type_1011: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_711, torch.float32);  view_711 = None
        add_154: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_144, convert_element_type_1011);  add_144 = convert_element_type_1011 = None
        convert_element_type_1012: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_205, torch.float32);  mm_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_506: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_500, [0, 2, 1, 3]);  permute_500 = None
        view_712: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_506, [8, 1024, 512]);  permute_506 = None
        clone_160: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_712, memory_format = torch.contiguous_format);  view_712 = None
        view_713: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_160, [8192, 512]);  clone_160 = None
        permute_507: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_713, [1, 0])
        mm_207: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_507, view_176);  permute_507 = view_176 = None
        mm_208: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_713, permute_509);  view_713 = permute_509 = None
        view_714: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_208, [8, 1024, 512]);  mm_208 = None
        convert_element_type_1017: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_714, torch.float32);  view_714 = None
        add_155: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_154, convert_element_type_1017);  add_154 = convert_element_type_1017 = None
        convert_element_type_1018: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_511: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_708, [0, 2, 1, 3]);  view_708 = None
        clone_161: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_511, memory_format = torch.contiguous_format);  permute_511 = None
        view_715: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [8, 1024, 512]);  clone_161 = None
        view_716: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_715, [8192, 512]);  view_715 = None
        permute_512: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_716, [1, 0])
        mm_209: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_512, view_173);  permute_512 = view_173 = None
        mm_210: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_716, permute_514);  view_716 = permute_514 = None
        view_717: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_210, [8, 1024, 512]);  mm_210 = None
        convert_element_type_1023: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_717, torch.float32);  view_717 = None
        convert_element_type_1024: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_209, torch.float32);  mm_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_416: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1023, primals_60);  primals_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_89: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_417: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1023, mul_89);  convert_element_type_1023 = mul_89 = None
        sum_68: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1], True, dtype = torch.float32);  mul_417 = None
        view_718: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_68, [512]);  sum_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_418: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, add_48)
        mul_419: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_416, rsqrt_14);  mul_416 = None
        sum_69: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True, dtype = torch.float32);  mul_418 = None
        add_156: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_153, mul_419);  add_153 = mul_419 = None
        pow_67: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_14, 3);  rsqrt_14 = None
        mul_420: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_69, -0.5);  sum_69 = None
        mul_421: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, pow_67);  mul_420 = pow_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_92: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_421, [8, 1024, 512]);  mul_421 = None
        div_41: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_92, 512);  expand_92 = None
        pow_68: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_48, 1.0);  add_48 = None
        mul_422: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_68, 2.0);  pow_68 = None
        mul_423: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_41, mul_422);  div_41 = mul_422 = None
        add_157: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_156, mul_423);  add_156 = mul_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_1025: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16)
        convert_element_type_1026: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_29, torch.bfloat16);  gt_29 = None
        mul_424: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1026, 1.1111111111111112);  convert_element_type_1026 = None
        mul_425: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1025, mul_424);  convert_element_type_1025 = mul_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_719: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_425, [8192, 512]);  mul_425 = None
        permute_516: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_719, [1, 0])
        mm_211: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_516, view_171);  permute_516 = view_171 = None
        mm_212: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_719, permute_518);  view_719 = permute_518 = None
        view_720: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_212, [8, 1024, 512]);  mm_212 = None
        convert_element_type_1031: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_211, torch.float32);  mm_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_721: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_720, [8, 1024, 8, 64]);  view_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_520: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_721, [0, 2, 1, 3]);  view_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_163: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_520, memory_format = torch.contiguous_format);  permute_520 = None
        view_722: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_163, [64, 1024, 64]);  clone_163 = None
        bmm_80: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_521, view_722);  permute_521 = None
        bmm_81: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_722, permute_522);  view_722 = permute_522 = None
        view_723: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_80, [8, 8, 1024, 64]);  bmm_80 = None
        view_724: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_81, [8, 8, 1024, 1024]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_1036: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_28, torch.bfloat16);  gt_28 = None
        mul_426: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1036, 1.1111111111111112);  convert_element_type_1036 = None
        mul_427: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_724, mul_426);  view_724 = mul_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_1037: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_427, torch.float32);  mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_8: "i64[1, 1, 1024, 1][1024, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_11: "i64[1, 1, 1, 1024][1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.le.Tensor(unsqueeze_11, unsqueeze_8);  unsqueeze_11 = unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_25: "b8[8, 1, 1024, 1024][0, 1048576, 1024, 1]cuda:0" = torch.ops.aten.expand.default(le, [8, -1, 1024, 1024]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_2: "f32[][]cuda:0" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.where.self(expand_25, full_default_1, full_default_2);  expand_25 = full_default_1 = full_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_163: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [8, 8, 1024, 1024]);  bmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_74: "f32[8, 1024, 1024][1, 8192, 8]cuda:0" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
        unsqueeze_17: "f32[1, 8, 1024, 1024][8, 1, 8192, 8]cuda:0" = torch.ops.aten.unsqueeze.default(permute_74, 0);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_46: "f32[8, 8, 1024, 1024][8388608, 1, 8192, 8]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_17, where_2);  unsqueeze_17 = where_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_47: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_163, add_46);  view_163 = add_46 = None
        convert_element_type_205: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.bfloat16);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_206: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_205, torch.float32);  convert_element_type_205 = None
        sub_8: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_206, amax_6);  convert_element_type_206 = amax_6 = None
        exp_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        div_10: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        mul_428: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1037, div_10);  convert_element_type_1037 = None
        sum_70: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_428, [-1], True)
        neg_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_11: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_13, sum_70, mul_428);  neg_13 = sum_70 = mul_428 = None
        convert_element_type_1038: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_725: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1038, [64, 1024, 1024]);  convert_element_type_1038 = None
        view_727: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_725, [8, 8, 1024, 1024]);  view_725 = None
        view_728: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_727, [64, 1024, 1024])
        convert_element_type_1039: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_727, torch.float32);  view_727 = None
        add_158: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_147, convert_element_type_1039);  add_147 = convert_element_type_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_71: "f32[1, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_158, [0], True, dtype = torch.float32);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_1: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_71, 0);  sum_71 = None
        permute_523: "f32[1024, 1024, 8][1024, 1, 1048576]cuda:0" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        full_default_15: "b8[1024, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.full.default([1024, 1024, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[32, 8][8, 1]cuda:0" = torch.ops.aten.full.default([32, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[32, 8][8, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_16, full_default_15, [add_45], permute_523);  add_45 = permute_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_82: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_524, view_728);  permute_524 = None
        bmm_83: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_728, permute_525);  view_728 = permute_525 = None
        view_730: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_82, [8, 8, 64, 1024]);  bmm_82 = None
        view_731: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_83, [8, 8, 1024, 64]);  bmm_83 = None
        permute_526: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_730, [0, 1, 3, 2]);  view_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_527: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_723, [0, 2, 1, 3]);  view_723 = None
        clone_166: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_527, memory_format = torch.contiguous_format);  permute_527 = None
        view_732: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_166, [8, 1024, 512]);  clone_166 = None
        view_733: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_732, [8192, 512]);  view_732 = None
        permute_528: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_733, [1, 0])
        mm_213: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_528, view_152);  permute_528 = None
        mm_214: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_733, permute_530);  view_733 = permute_530 = None
        view_734: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_214, [8, 1024, 512]);  mm_214 = None
        convert_element_type_1048: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_734, torch.float32);  view_734 = None
        convert_element_type_1049: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_213, torch.float32);  mm_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_532: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_526, [0, 2, 1, 3]);  permute_526 = None
        view_735: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_532, [8, 1024, 512]);  permute_532 = None
        clone_167: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_735, memory_format = torch.contiguous_format);  view_735 = None
        view_736: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_167, [8192, 512]);  clone_167 = None
        permute_533: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_736, [1, 0])
        mm_215: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_533, view_152);  permute_533 = None
        mm_216: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_736, permute_535);  view_736 = permute_535 = None
        view_737: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_216, [8, 1024, 512]);  mm_216 = None
        convert_element_type_1054: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_737, torch.float32);  view_737 = None
        add_159: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1048, convert_element_type_1054);  convert_element_type_1048 = convert_element_type_1054 = None
        convert_element_type_1055: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_215, torch.float32);  mm_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_537: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_731, [0, 2, 1, 3]);  view_731 = None
        clone_168: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_537, memory_format = torch.contiguous_format);  permute_537 = None
        view_738: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_168, [8, 1024, 512]);  clone_168 = None
        view_739: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_738, [8192, 512]);  view_738 = None
        permute_538: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_739, [1, 0])
        mm_217: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_538, view_152);  permute_538 = view_152 = None
        mm_218: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_739, permute_540);  view_739 = permute_540 = None
        view_740: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_218, [8, 1024, 512]);  mm_218 = None
        convert_element_type_1060: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_740, torch.float32);  view_740 = None
        add_160: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_159, convert_element_type_1060);  add_159 = convert_element_type_1060 = None
        convert_element_type_1061: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_217, torch.float32);  mm_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_429: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_160, primals_54);  primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_80: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_27, embedding)
        mul_81: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 1.1111111111111112);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_82: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_430: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_160, mul_82);  add_160 = mul_82 = None
        sum_72: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1], True, dtype = torch.float32);  mul_430 = None
        view_741: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_72, [512]);  sum_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_431: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_429, mul_81)
        mul_432: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_429, rsqrt_13);  mul_429 = None
        sum_73: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_431, [2], True, dtype = torch.float32);  mul_431 = None
        add_161: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_157, mul_432);  add_157 = mul_432 = None
        pow_69: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
        mul_433: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_73, -0.5);  sum_73 = None
        mul_434: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_433, pow_69);  mul_433 = pow_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_93: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_434, [8, 1024, 512]);  mul_434 = None
        div_42: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_93, 512);  expand_93 = None
        pow_70: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_81, 1.0);  mul_81 = None
        mul_435: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
        mul_436: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_42, mul_435);  div_42 = mul_435 = None
        add_162: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_161, mul_436);  add_161 = mul_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_1062: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_437: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1062, 1.1111111111111112);  convert_element_type_1062 = None
        mul_438: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_162, mul_437);  add_162 = mul_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        ge_3: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_1, 0)
        lt_3: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_1, 32128)
        bitwise_and_2: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_3, lt_3);  ge_3 = lt_3 = None
        ne_6: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_1, -1)
        bitwise_and_3: "b8[8, 1024][1024, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_6);  bitwise_and_2 = ne_6 = None
        unsqueeze_21: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_default_17: "f32[32128, 512][512, 1]cuda:0" = torch.ops.aten.full.default([32128, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[32128, 512][512, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_17, unsqueeze_21, [primals_1], mul_438);  mul_438 = None
        add_163: "f32[32128, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_526, _unsafe_masked_index_put_accumulate_1);  convert_element_type_526 = _unsafe_masked_index_put_accumulate_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:755 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1063: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_439: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1063, 1.1111111111111112);  convert_element_type_1063 = None
        mul_440: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_155, mul_439);  add_155 = mul_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_441: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, primals_52);  primals_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_76: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_36, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_442: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, mul_76);  mul_440 = mul_76 = None
        sum_74: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 1], True, dtype = torch.float32);  mul_442 = None
        view_742: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_74, [512]);  sum_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_443: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, add_36)
        mul_444: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, rsqrt_12);  mul_441 = None
        sum_75: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True, dtype = torch.float32);  mul_443 = None
        pow_71: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
        mul_445: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_75, -0.5);  sum_75 = None
        mul_446: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_445, pow_71);  mul_445 = pow_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_94: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_446, [8, 1024, 512]);  mul_446 = None
        div_43: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_94, 512);  expand_94 = None
        pow_72: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_36, 1.0);  add_36 = None
        mul_447: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
        mul_448: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_43, mul_447);  div_43 = mul_447 = None
        add_164: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_444, mul_448);  mul_444 = mul_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_1064: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16)
        convert_element_type_1065: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_25, torch.bfloat16);  gt_25 = None
        mul_449: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1065, 1.1111111111111112);  convert_element_type_1065 = None
        mul_450: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1064, mul_449);  convert_element_type_1064 = mul_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_743: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_450, [8192, 512]);  mul_450 = None
        permute_542: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_743, [1, 0])
        mm_219: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_542, view_149);  permute_542 = view_149 = None
        mm_220: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_743, permute_544);  view_743 = permute_544 = None
        view_744: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_220, [8, 1024, 2048]);  mm_220 = None
        convert_element_type_1071: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_219, torch.float32);  mm_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_11: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_744, torch.bfloat16);  view_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1073: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_24, torch.bfloat16);  gt_24 = None
        mul_451: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1073, 1.1111111111111112);  convert_element_type_1073 = None
        mul_452: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_11, mul_451);  convert_element_type_default_11 = mul_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_15: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_7, full_default_9, mul_452);  le_7 = mul_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_745: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_15, [8192, 2048]);  where_15 = None
        permute_546: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_745, [1, 0])
        mm_221: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_546, view_147);  permute_546 = view_147 = None
        mm_222: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_745, permute_548);  view_745 = permute_548 = None
        view_746: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_222, [8, 1024, 512]);  mm_222 = None
        convert_element_type_1078: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_746, torch.float32);  view_746 = None
        convert_element_type_1079: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_221, torch.float32);  mm_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_453: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1078, primals_49);  primals_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_70: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_34, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_454: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1078, mul_70);  convert_element_type_1078 = mul_70 = None
        sum_76: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 1], True, dtype = torch.float32);  mul_454 = None
        view_747: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_76, [512]);  sum_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_455: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, add_34)
        mul_456: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_453, rsqrt_11);  mul_453 = None
        sum_77: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True, dtype = torch.float32);  mul_455 = None
        add_165: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_164, mul_456);  add_164 = mul_456 = None
        pow_73: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_11, 3);  rsqrt_11 = None
        mul_457: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_77, -0.5);  sum_77 = None
        mul_458: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_457, pow_73);  mul_457 = pow_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_95: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_458, [8, 1024, 512]);  mul_458 = None
        div_44: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_95, 512);  expand_95 = None
        pow_74: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_34, 1.0);  add_34 = None
        mul_459: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
        mul_460: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_44, mul_459);  div_44 = mul_459 = None
        add_166: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_165, mul_460);  add_165 = mul_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_1080: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_166, torch.bfloat16)
        convert_element_type_1081: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_23, torch.bfloat16);  gt_23 = None
        mul_461: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1081, 1.1111111111111112);  convert_element_type_1081 = None
        mul_462: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1080, mul_461);  convert_element_type_1080 = mul_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_748: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_462, [8192, 512]);  mul_462 = None
        permute_550: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_748, [1, 0])
        mm_223: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_550, view_145);  permute_550 = view_145 = None
        mm_224: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_748, permute_552);  view_748 = permute_552 = None
        view_749: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_224, [8, 1024, 512]);  mm_224 = None
        convert_element_type_1086: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_223, torch.float32);  mm_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_750: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_749, [8, 1024, 8, 64]);  view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_554: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_750, [0, 2, 1, 3]);  view_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_174: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_554, memory_format = torch.contiguous_format);  permute_554 = None
        view_751: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_174, [64, 1024, 64]);  clone_174 = None
        bmm_84: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_555, view_751);  permute_555 = None
        bmm_85: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_751, permute_556);  view_751 = permute_556 = None
        view_752: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_84, [8, 8, 1024, 64]);  bmm_84 = None
        view_753: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_85, [8, 8, 1024, 1024]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_1091: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_22, torch.bfloat16);  gt_22 = None
        mul_463: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1091, 1.1111111111111112);  convert_element_type_1091 = None
        mul_464: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_753, mul_463);  view_753 = mul_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_1092: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_464, torch.float32);  mul_464 = None
        convert_element_type_173: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_172, torch.float32);  convert_element_type_172 = None
        sub_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_173, amax_5);  convert_element_type_173 = amax_5 = None
        exp_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        div_7: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        mul_465: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1092, div_7);  convert_element_type_1092 = None
        sum_78: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_465, [-1], True)
        neg_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_12: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_14, sum_78, mul_465);  neg_14 = sum_78 = mul_465 = None
        convert_element_type_1093: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_754: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1093, [64, 1024, 1024]);  convert_element_type_1093 = None
        view_756: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_754, [8, 8, 1024, 1024]);  view_754 = None
        view_757: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_756, [64, 1024, 1024])
        convert_element_type_1094: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_756, torch.float32);  view_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_86: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_557, view_757);  permute_557 = None
        bmm_87: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_757, permute_558);  view_757 = permute_558 = None
        view_759: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_86, [8, 8, 64, 1024]);  bmm_86 = None
        view_760: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_87, [8, 8, 1024, 64]);  bmm_87 = None
        permute_559: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_759, [0, 1, 3, 2]);  view_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_560: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_752, [0, 2, 1, 3]);  view_752 = None
        clone_177: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format);  permute_560 = None
        view_761: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [8, 1024, 512]);  clone_177 = None
        view_762: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_761, [8192, 512]);  view_761 = None
        permute_561: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_762, [1, 0])
        mm_225: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_561, view_126);  permute_561 = None
        mm_226: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_762, permute_563);  view_762 = permute_563 = None
        view_763: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_226, [8, 1024, 512]);  mm_226 = None
        convert_element_type_1103: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_763, torch.float32);  view_763 = None
        convert_element_type_1104: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_225, torch.float32);  mm_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_565: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_559, [0, 2, 1, 3]);  permute_559 = None
        view_764: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_565, [8, 1024, 512]);  permute_565 = None
        clone_178: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_764, memory_format = torch.contiguous_format);  view_764 = None
        view_765: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [8192, 512]);  clone_178 = None
        permute_566: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_765, [1, 0])
        mm_227: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_566, view_126);  permute_566 = None
        mm_228: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_765, permute_568);  view_765 = permute_568 = None
        view_766: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_228, [8, 1024, 512]);  mm_228 = None
        convert_element_type_1109: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_766, torch.float32);  view_766 = None
        add_167: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1103, convert_element_type_1109);  convert_element_type_1103 = convert_element_type_1109 = None
        convert_element_type_1110: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_227, torch.float32);  mm_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_570: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None
        clone_179: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_570, memory_format = torch.contiguous_format);  permute_570 = None
        view_767: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [8, 1024, 512]);  clone_179 = None
        view_768: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_767, [8192, 512]);  view_767 = None
        permute_571: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_768, [1, 0])
        mm_229: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_571, view_126);  permute_571 = view_126 = None
        mm_230: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_768, permute_573);  view_768 = permute_573 = None
        view_769: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_230, [8, 1024, 512]);  mm_230 = None
        convert_element_type_1115: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_769, torch.float32);  view_769 = None
        add_168: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_167, convert_element_type_1115);  add_167 = convert_element_type_1115 = None
        convert_element_type_1116: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_229, torch.float32);  mm_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_466: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_168, primals_44);  primals_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_64: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_31, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_467: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_168, mul_64);  add_168 = mul_64 = None
        sum_79: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 1], True, dtype = torch.float32);  mul_467 = None
        view_770: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_79, [512]);  sum_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_468: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_466, add_31)
        mul_469: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_466, rsqrt_10);  mul_466 = None
        sum_80: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_468, [2], True, dtype = torch.float32);  mul_468 = None
        add_169: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_166, mul_469);  add_166 = mul_469 = None
        pow_75: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_10, 3);  rsqrt_10 = None
        mul_470: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_80, -0.5);  sum_80 = None
        mul_471: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_470, pow_75);  mul_470 = pow_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_96: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_471, [8, 1024, 512]);  mul_471 = None
        div_45: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_96, 512);  expand_96 = None
        pow_76: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_31, 1.0);  add_31 = None
        mul_472: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_76, 2.0);  pow_76 = None
        mul_473: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_45, mul_472);  div_45 = mul_472 = None
        add_170: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_169, mul_473);  add_169 = mul_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_1117: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_170, torch.bfloat16)
        convert_element_type_1118: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_21, torch.bfloat16);  gt_21 = None
        mul_474: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1118, 1.1111111111111112);  convert_element_type_1118 = None
        mul_475: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1117, mul_474);  convert_element_type_1117 = mul_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_771: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_475, [8192, 512]);  mul_475 = None
        permute_575: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_771, [1, 0])
        mm_231: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_575, view_124);  permute_575 = view_124 = None
        mm_232: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_771, permute_577);  view_771 = permute_577 = None
        view_772: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_232, [8, 1024, 2048]);  mm_232 = None
        convert_element_type_1124: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_231, torch.float32);  mm_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_10: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_772, torch.bfloat16);  view_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1126: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_20, torch.bfloat16);  gt_20 = None
        mul_476: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1126, 1.1111111111111112);  convert_element_type_1126 = None
        mul_477: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_10, mul_476);  convert_element_type_default_10 = mul_476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_16: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_8, full_default_9, mul_477);  le_8 = mul_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_773: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_16, [8192, 2048]);  where_16 = None
        permute_579: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_773, [1, 0])
        mm_233: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_579, view_122);  permute_579 = view_122 = None
        mm_234: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_773, permute_581);  view_773 = permute_581 = None
        view_774: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_234, [8, 1024, 512]);  mm_234 = None
        convert_element_type_1131: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_774, torch.float32);  view_774 = None
        convert_element_type_1132: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_233, torch.float32);  mm_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_478: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1131, primals_41);  primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_58: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_29, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_479: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1131, mul_58);  convert_element_type_1131 = mul_58 = None
        sum_81: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_479, [0, 1], True, dtype = torch.float32);  mul_479 = None
        view_775: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [512]);  sum_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_480: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_478, add_29)
        mul_481: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_478, rsqrt_9);  mul_478 = None
        sum_82: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_480, [2], True, dtype = torch.float32);  mul_480 = None
        add_171: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_170, mul_481);  add_170 = mul_481 = None
        pow_77: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_9, 3);  rsqrt_9 = None
        mul_482: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_82, -0.5);  sum_82 = None
        mul_483: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_482, pow_77);  mul_482 = pow_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_97: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_483, [8, 1024, 512]);  mul_483 = None
        div_46: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_97, 512);  expand_97 = None
        pow_78: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_29, 1.0);  add_29 = None
        mul_484: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_78, 2.0);  pow_78 = None
        mul_485: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_46, mul_484);  div_46 = mul_484 = None
        add_172: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_171, mul_485);  add_171 = mul_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_1133: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_172, torch.bfloat16)
        convert_element_type_1134: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_19, torch.bfloat16);  gt_19 = None
        mul_486: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1134, 1.1111111111111112);  convert_element_type_1134 = None
        mul_487: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1133, mul_486);  convert_element_type_1133 = mul_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_776: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_487, [8192, 512]);  mul_487 = None
        permute_583: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_776, [1, 0])
        mm_235: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_583, view_120);  permute_583 = view_120 = None
        mm_236: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_776, permute_585);  view_776 = permute_585 = None
        view_777: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_236, [8, 1024, 512]);  mm_236 = None
        convert_element_type_1139: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_778: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_777, [8, 1024, 8, 64]);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_587: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_778, [0, 2, 1, 3]);  view_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_183: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
        view_779: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_183, [64, 1024, 64]);  clone_183 = None
        bmm_88: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_588, view_779);  permute_588 = None
        bmm_89: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_779, permute_589);  view_779 = permute_589 = None
        view_780: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_88, [8, 8, 1024, 64]);  bmm_88 = None
        view_781: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_89, [8, 8, 1024, 1024]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_1144: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_18, torch.bfloat16);  gt_18 = None
        mul_488: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1144, 1.1111111111111112);  convert_element_type_1144 = None
        mul_489: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_781, mul_488);  view_781 = mul_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_1145: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_489, torch.float32);  mul_489 = None
        convert_element_type_142: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_141, torch.float32);  convert_element_type_141 = None
        sub_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_142, amax_4);  convert_element_type_142 = amax_4 = None
        exp_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        div_6: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        mul_490: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1145, div_6);  convert_element_type_1145 = None
        sum_83: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_490, [-1], True)
        neg_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_13: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_15, sum_83, mul_490);  neg_15 = sum_83 = mul_490 = None
        convert_element_type_1146: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_782: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1146, [64, 1024, 1024]);  convert_element_type_1146 = None
        view_784: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_782, [8, 8, 1024, 1024]);  view_782 = None
        view_785: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_784, [64, 1024, 1024])
        convert_element_type_1147: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_784, torch.float32);  view_784 = None
        add_173: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1094, convert_element_type_1147);  convert_element_type_1094 = convert_element_type_1147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_90: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_590, view_785);  permute_590 = None
        bmm_91: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_785, permute_591);  view_785 = permute_591 = None
        view_787: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_90, [8, 8, 64, 1024]);  bmm_90 = None
        view_788: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_91, [8, 8, 1024, 64]);  bmm_91 = None
        permute_592: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_787, [0, 1, 3, 2]);  view_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_593: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_780, [0, 2, 1, 3]);  view_780 = None
        clone_186: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_593, memory_format = torch.contiguous_format);  permute_593 = None
        view_789: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [8, 1024, 512]);  clone_186 = None
        view_790: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_789, [8192, 512]);  view_789 = None
        permute_594: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_237: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_594, view_101);  permute_594 = None
        mm_238: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_790, permute_596);  view_790 = permute_596 = None
        view_791: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_238, [8, 1024, 512]);  mm_238 = None
        convert_element_type_1156: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_791, torch.float32);  view_791 = None
        convert_element_type_1157: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_598: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_592, [0, 2, 1, 3]);  permute_592 = None
        view_792: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_598, [8, 1024, 512]);  permute_598 = None
        clone_187: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_792, memory_format = torch.contiguous_format);  view_792 = None
        view_793: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_187, [8192, 512]);  clone_187 = None
        permute_599: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_793, [1, 0])
        mm_239: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_599, view_101);  permute_599 = None
        mm_240: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_793, permute_601);  view_793 = permute_601 = None
        view_794: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_240, [8, 1024, 512]);  mm_240 = None
        convert_element_type_1162: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_794, torch.float32);  view_794 = None
        add_174: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1156, convert_element_type_1162);  convert_element_type_1156 = convert_element_type_1162 = None
        convert_element_type_1163: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_239, torch.float32);  mm_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_603: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_788, [0, 2, 1, 3]);  view_788 = None
        clone_188: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
        view_795: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_188, [8, 1024, 512]);  clone_188 = None
        view_796: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_795, [8192, 512]);  view_795 = None
        permute_604: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_796, [1, 0])
        mm_241: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_604, view_101);  permute_604 = view_101 = None
        mm_242: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_796, permute_606);  view_796 = permute_606 = None
        view_797: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_242, [8, 1024, 512]);  mm_242 = None
        convert_element_type_1168: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_797, torch.float32);  view_797 = None
        add_175: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_174, convert_element_type_1168);  add_174 = convert_element_type_1168 = None
        convert_element_type_1169: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_241, torch.float32);  mm_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_491: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_175, primals_36);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_52: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_26, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_492: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_175, mul_52);  add_175 = mul_52 = None
        sum_84: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_492, [0, 1], True, dtype = torch.float32);  mul_492 = None
        view_798: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_84, [512]);  sum_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_493: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_491, add_26)
        mul_494: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_491, rsqrt_8);  mul_491 = None
        sum_85: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_493, [2], True, dtype = torch.float32);  mul_493 = None
        add_176: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_172, mul_494);  add_172 = mul_494 = None
        pow_79: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_8, 3);  rsqrt_8 = None
        mul_495: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_85, -0.5);  sum_85 = None
        mul_496: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, pow_79);  mul_495 = pow_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_98: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_496, [8, 1024, 512]);  mul_496 = None
        div_47: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_98, 512);  expand_98 = None
        pow_80: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_26, 1.0);  add_26 = None
        mul_497: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_80, 2.0);  pow_80 = None
        mul_498: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_47, mul_497);  div_47 = mul_497 = None
        add_177: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_176, mul_498);  add_176 = mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_1170: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_177, torch.bfloat16)
        convert_element_type_1171: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_17, torch.bfloat16);  gt_17 = None
        mul_499: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1171, 1.1111111111111112);  convert_element_type_1171 = None
        mul_500: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1170, mul_499);  convert_element_type_1170 = mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_799: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_500, [8192, 512]);  mul_500 = None
        permute_608: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_799, [1, 0])
        mm_243: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_608, view_99);  permute_608 = view_99 = None
        mm_244: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_799, permute_610);  view_799 = permute_610 = None
        view_800: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_244, [8, 1024, 2048]);  mm_244 = None
        convert_element_type_1177: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_243, torch.float32);  mm_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_9: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_800, torch.bfloat16);  view_800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1179: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_16, torch.bfloat16);  gt_16 = None
        mul_501: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1179, 1.1111111111111112);  convert_element_type_1179 = None
        mul_502: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_9, mul_501);  convert_element_type_default_9 = mul_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_17: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_9, full_default_9, mul_502);  le_9 = mul_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_801: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_17, [8192, 2048]);  where_17 = None
        permute_612: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_801, [1, 0])
        mm_245: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_612, view_97);  permute_612 = view_97 = None
        mm_246: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_801, permute_614);  view_801 = permute_614 = None
        view_802: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_246, [8, 1024, 512]);  mm_246 = None
        convert_element_type_1184: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_802, torch.float32);  view_802 = None
        convert_element_type_1185: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_245, torch.float32);  mm_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_503: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1184, primals_33);  primals_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_46: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_24, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_504: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1184, mul_46);  convert_element_type_1184 = mul_46 = None
        sum_86: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_504, [0, 1], True, dtype = torch.float32);  mul_504 = None
        view_803: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_86, [512]);  sum_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_505: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_503, add_24)
        mul_506: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_503, rsqrt_7);  mul_503 = None
        sum_87: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_505, [2], True, dtype = torch.float32);  mul_505 = None
        add_178: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_177, mul_506);  add_177 = mul_506 = None
        pow_81: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_7, 3);  rsqrt_7 = None
        mul_507: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_87, -0.5);  sum_87 = None
        mul_508: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_507, pow_81);  mul_507 = pow_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_99: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_508, [8, 1024, 512]);  mul_508 = None
        div_48: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_99, 512);  expand_99 = None
        pow_82: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_24, 1.0);  add_24 = None
        mul_509: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_82, 2.0);  pow_82 = None
        mul_510: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_48, mul_509);  div_48 = mul_509 = None
        add_179: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_178, mul_510);  add_178 = mul_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_1186: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16)
        convert_element_type_1187: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_15, torch.bfloat16);  gt_15 = None
        mul_511: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1187, 1.1111111111111112);  convert_element_type_1187 = None
        mul_512: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1186, mul_511);  convert_element_type_1186 = mul_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_804: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_512, [8192, 512]);  mul_512 = None
        permute_616: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_804, [1, 0])
        mm_247: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_616, view_95);  permute_616 = view_95 = None
        mm_248: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_804, permute_618);  view_804 = permute_618 = None
        view_805: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_248, [8, 1024, 512]);  mm_248 = None
        convert_element_type_1192: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_247, torch.float32);  mm_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_806: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_805, [8, 1024, 8, 64]);  view_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_620: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_806, [0, 2, 1, 3]);  view_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_192: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_620, memory_format = torch.contiguous_format);  permute_620 = None
        view_807: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_192, [64, 1024, 64]);  clone_192 = None
        bmm_92: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_621, view_807);  permute_621 = None
        bmm_93: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_807, permute_622);  view_807 = permute_622 = None
        view_808: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_92, [8, 8, 1024, 64]);  bmm_92 = None
        view_809: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_93, [8, 8, 1024, 1024]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_1197: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_14, torch.bfloat16);  gt_14 = None
        mul_513: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1197, 1.1111111111111112);  convert_element_type_1197 = None
        mul_514: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_809, mul_513);  view_809 = mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_1198: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_514, torch.float32);  mul_514 = None
        convert_element_type_111: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_110, torch.float32);  convert_element_type_110 = None
        sub_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_111, amax_3);  convert_element_type_111 = amax_3 = None
        exp_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div_5: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        mul_515: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1198, div_5);  convert_element_type_1198 = None
        sum_88: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_515, [-1], True)
        neg_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_14: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_16, sum_88, mul_515);  neg_16 = sum_88 = mul_515 = None
        convert_element_type_1199: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_810: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1199, [64, 1024, 1024]);  convert_element_type_1199 = None
        view_812: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_810, [8, 8, 1024, 1024]);  view_810 = None
        view_813: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_812, [64, 1024, 1024])
        convert_element_type_1200: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_812, torch.float32);  view_812 = None
        add_180: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_173, convert_element_type_1200);  add_173 = convert_element_type_1200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_94: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_623, view_813);  permute_623 = None
        bmm_95: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_813, permute_624);  view_813 = permute_624 = None
        view_815: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_94, [8, 8, 64, 1024]);  bmm_94 = None
        view_816: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_95, [8, 8, 1024, 64]);  bmm_95 = None
        permute_625: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_815, [0, 1, 3, 2]);  view_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_626: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_808, [0, 2, 1, 3]);  view_808 = None
        clone_195: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_626, memory_format = torch.contiguous_format);  permute_626 = None
        view_817: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_195, [8, 1024, 512]);  clone_195 = None
        view_818: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_817, [8192, 512]);  view_817 = None
        permute_627: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_249: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_627, view_76);  permute_627 = None
        mm_250: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_818, permute_629);  view_818 = permute_629 = None
        view_819: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_250, [8, 1024, 512]);  mm_250 = None
        convert_element_type_1209: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_819, torch.float32);  view_819 = None
        convert_element_type_1210: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_631: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_625, [0, 2, 1, 3]);  permute_625 = None
        view_820: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_631, [8, 1024, 512]);  permute_631 = None
        clone_196: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_820, memory_format = torch.contiguous_format);  view_820 = None
        view_821: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_196, [8192, 512]);  clone_196 = None
        permute_632: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_821, [1, 0])
        mm_251: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_632, view_76);  permute_632 = None
        mm_252: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_821, permute_634);  view_821 = permute_634 = None
        view_822: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_252, [8, 1024, 512]);  mm_252 = None
        convert_element_type_1215: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_822, torch.float32);  view_822 = None
        add_181: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1209, convert_element_type_1215);  convert_element_type_1209 = convert_element_type_1215 = None
        convert_element_type_1216: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_251, torch.float32);  mm_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_636: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_816, [0, 2, 1, 3]);  view_816 = None
        clone_197: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_636, memory_format = torch.contiguous_format);  permute_636 = None
        view_823: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_197, [8, 1024, 512]);  clone_197 = None
        view_824: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_823, [8192, 512]);  view_823 = None
        permute_637: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_824, [1, 0])
        mm_253: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_637, view_76);  permute_637 = view_76 = None
        mm_254: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_824, permute_639);  view_824 = permute_639 = None
        view_825: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_254, [8, 1024, 512]);  mm_254 = None
        convert_element_type_1221: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_825, torch.float32);  view_825 = None
        add_182: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_181, convert_element_type_1221);  add_181 = convert_element_type_1221 = None
        convert_element_type_1222: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_253, torch.float32);  mm_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_516: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_182, primals_28);  primals_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_40: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_21, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_517: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_182, mul_40);  add_182 = mul_40 = None
        sum_89: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 1], True, dtype = torch.float32);  mul_517 = None
        view_826: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_89, [512]);  sum_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_518: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_516, add_21)
        mul_519: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_516, rsqrt_6);  mul_516 = None
        sum_90: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_518, [2], True, dtype = torch.float32);  mul_518 = None
        add_183: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_179, mul_519);  add_179 = mul_519 = None
        pow_83: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_6, 3);  rsqrt_6 = None
        mul_520: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_90, -0.5);  sum_90 = None
        mul_521: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_520, pow_83);  mul_520 = pow_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_100: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_521, [8, 1024, 512]);  mul_521 = None
        div_49: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_100, 512);  expand_100 = None
        pow_84: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_21, 1.0);  add_21 = None
        mul_522: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_84, 2.0);  pow_84 = None
        mul_523: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_49, mul_522);  div_49 = mul_522 = None
        add_184: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_183, mul_523);  add_183 = mul_523 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_1223: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16)
        convert_element_type_1224: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_13, torch.bfloat16);  gt_13 = None
        mul_524: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1224, 1.1111111111111112);  convert_element_type_1224 = None
        mul_525: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1223, mul_524);  convert_element_type_1223 = mul_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_827: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_525, [8192, 512]);  mul_525 = None
        permute_641: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_827, [1, 0])
        mm_255: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_641, view_74);  permute_641 = view_74 = None
        mm_256: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_827, permute_643);  view_827 = permute_643 = None
        view_828: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_256, [8, 1024, 2048]);  mm_256 = None
        convert_element_type_1230: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_255, torch.float32);  mm_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_8: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_828, torch.bfloat16);  view_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1232: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_12, torch.bfloat16);  gt_12 = None
        mul_526: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1232, 1.1111111111111112);  convert_element_type_1232 = None
        mul_527: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_8, mul_526);  convert_element_type_default_8 = mul_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_18: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_10, full_default_9, mul_527);  le_10 = mul_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_829: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_18, [8192, 2048]);  where_18 = None
        permute_645: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_829, [1, 0])
        mm_257: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_645, view_72);  permute_645 = view_72 = None
        mm_258: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_829, permute_647);  view_829 = permute_647 = None
        view_830: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_258, [8, 1024, 512]);  mm_258 = None
        convert_element_type_1237: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_830, torch.float32);  view_830 = None
        convert_element_type_1238: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_257, torch.float32);  mm_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_528: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1237, primals_25);  primals_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_34: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_19, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_529: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1237, mul_34);  convert_element_type_1237 = mul_34 = None
        sum_91: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 1], True, dtype = torch.float32);  mul_529 = None
        view_831: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_91, [512]);  sum_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_530: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_528, add_19)
        mul_531: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_528, rsqrt_5);  mul_528 = None
        sum_92: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_530, [2], True, dtype = torch.float32);  mul_530 = None
        add_185: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_184, mul_531);  add_184 = mul_531 = None
        pow_85: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_5, 3);  rsqrt_5 = None
        mul_532: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_92, -0.5);  sum_92 = None
        mul_533: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_532, pow_85);  mul_532 = pow_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_101: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_533, [8, 1024, 512]);  mul_533 = None
        div_50: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_101, 512);  expand_101 = None
        pow_86: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_19, 1.0);  add_19 = None
        mul_534: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_86, 2.0);  pow_86 = None
        mul_535: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, mul_534);  div_50 = mul_534 = None
        add_186: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_185, mul_535);  add_185 = mul_535 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_1239: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16)
        convert_element_type_1240: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_11, torch.bfloat16);  gt_11 = None
        mul_536: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1240, 1.1111111111111112);  convert_element_type_1240 = None
        mul_537: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1239, mul_536);  convert_element_type_1239 = mul_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_832: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_537, [8192, 512]);  mul_537 = None
        permute_649: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_832, [1, 0])
        mm_259: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_649, view_70);  permute_649 = view_70 = None
        mm_260: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_832, permute_651);  view_832 = permute_651 = None
        view_833: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_260, [8, 1024, 512]);  mm_260 = None
        convert_element_type_1245: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_259, torch.float32);  mm_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_834: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_833, [8, 1024, 8, 64]);  view_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_653: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_834, [0, 2, 1, 3]);  view_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_201: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_653, memory_format = torch.contiguous_format);  permute_653 = None
        view_835: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [64, 1024, 64]);  clone_201 = None
        bmm_96: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_654, view_835);  permute_654 = None
        bmm_97: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_835, permute_655);  view_835 = permute_655 = None
        view_836: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_96, [8, 8, 1024, 64]);  bmm_96 = None
        view_837: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_97, [8, 8, 1024, 1024]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_1250: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_10, torch.bfloat16);  gt_10 = None
        mul_538: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1250, 1.1111111111111112);  convert_element_type_1250 = None
        mul_539: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_837, mul_538);  view_837 = mul_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_1251: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_539, torch.float32);  mul_539 = None
        convert_element_type_80: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_79, torch.float32);  convert_element_type_79 = None
        sub_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_80, amax_2);  convert_element_type_80 = amax_2 = None
        exp_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        div_4: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        mul_540: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1251, div_4);  convert_element_type_1251 = None
        sum_93: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_540, [-1], True)
        neg_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_15: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_17, sum_93, mul_540);  neg_17 = sum_93 = mul_540 = None
        convert_element_type_1252: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_838: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1252, [64, 1024, 1024]);  convert_element_type_1252 = None
        view_840: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_838, [8, 8, 1024, 1024]);  view_838 = None
        view_841: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_840, [64, 1024, 1024])
        convert_element_type_1253: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_840, torch.float32);  view_840 = None
        add_187: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_180, convert_element_type_1253);  add_180 = convert_element_type_1253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_98: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_656, view_841);  permute_656 = None
        bmm_99: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_841, permute_657);  view_841 = permute_657 = None
        view_843: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_98, [8, 8, 64, 1024]);  bmm_98 = None
        view_844: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_99, [8, 8, 1024, 64]);  bmm_99 = None
        permute_658: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_843, [0, 1, 3, 2]);  view_843 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_659: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_836, [0, 2, 1, 3]);  view_836 = None
        clone_204: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_659, memory_format = torch.contiguous_format);  permute_659 = None
        view_845: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_204, [8, 1024, 512]);  clone_204 = None
        view_846: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_845, [8192, 512]);  view_845 = None
        permute_660: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_846, [1, 0])
        mm_261: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_660, view_51);  permute_660 = None
        mm_262: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_846, permute_662);  view_846 = permute_662 = None
        view_847: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_262, [8, 1024, 512]);  mm_262 = None
        convert_element_type_1262: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_847, torch.float32);  view_847 = None
        convert_element_type_1263: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_261, torch.float32);  mm_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_664: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_658, [0, 2, 1, 3]);  permute_658 = None
        view_848: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_664, [8, 1024, 512]);  permute_664 = None
        clone_205: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_848, memory_format = torch.contiguous_format);  view_848 = None
        view_849: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_205, [8192, 512]);  clone_205 = None
        permute_665: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_849, [1, 0])
        mm_263: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_665, view_51);  permute_665 = None
        mm_264: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_849, permute_667);  view_849 = permute_667 = None
        view_850: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_264, [8, 1024, 512]);  mm_264 = None
        convert_element_type_1268: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_850, torch.float32);  view_850 = None
        add_188: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1262, convert_element_type_1268);  convert_element_type_1262 = convert_element_type_1268 = None
        convert_element_type_1269: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_263, torch.float32);  mm_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_669: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_844, [0, 2, 1, 3]);  view_844 = None
        clone_206: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_669, memory_format = torch.contiguous_format);  permute_669 = None
        view_851: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_206, [8, 1024, 512]);  clone_206 = None
        view_852: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_851, [8192, 512]);  view_851 = None
        permute_670: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_852, [1, 0])
        mm_265: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_670, view_51);  permute_670 = view_51 = None
        mm_266: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_852, permute_672);  view_852 = permute_672 = None
        view_853: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_266, [8, 1024, 512]);  mm_266 = None
        convert_element_type_1274: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_853, torch.float32);  view_853 = None
        add_189: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_188, convert_element_type_1274);  add_188 = convert_element_type_1274 = None
        convert_element_type_1275: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_265, torch.float32);  mm_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_541: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_189, primals_20);  primals_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_28: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_16, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_542: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_189, mul_28);  add_189 = mul_28 = None
        sum_94: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_542, [0, 1], True, dtype = torch.float32);  mul_542 = None
        view_854: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [512]);  sum_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_543: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_541, add_16)
        mul_544: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_541, rsqrt_4);  mul_541 = None
        sum_95: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_543, [2], True, dtype = torch.float32);  mul_543 = None
        add_190: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_186, mul_544);  add_186 = mul_544 = None
        pow_87: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_4, 3);  rsqrt_4 = None
        mul_545: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_95, -0.5);  sum_95 = None
        mul_546: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_545, pow_87);  mul_545 = pow_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_102: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_546, [8, 1024, 512]);  mul_546 = None
        div_51: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_102, 512);  expand_102 = None
        pow_88: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_16, 1.0);  add_16 = None
        mul_547: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_88, 2.0);  pow_88 = None
        mul_548: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_51, mul_547);  div_51 = mul_547 = None
        add_191: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_190, mul_548);  add_190 = mul_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_1276: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16)
        convert_element_type_1277: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_9, torch.bfloat16);  gt_9 = None
        mul_549: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1277, 1.1111111111111112);  convert_element_type_1277 = None
        mul_550: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1276, mul_549);  convert_element_type_1276 = mul_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_855: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_550, [8192, 512]);  mul_550 = None
        permute_674: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_855, [1, 0])
        mm_267: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_674, view_49);  permute_674 = view_49 = None
        mm_268: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_855, permute_676);  view_855 = permute_676 = None
        view_856: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_268, [8, 1024, 2048]);  mm_268 = None
        convert_element_type_1283: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_267, torch.float32);  mm_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_7: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_856, torch.bfloat16);  view_856 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1285: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_8, torch.bfloat16);  gt_8 = None
        mul_551: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1285, 1.1111111111111112);  convert_element_type_1285 = None
        mul_552: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_7, mul_551);  convert_element_type_default_7 = mul_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_19: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_11, full_default_9, mul_552);  le_11 = mul_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_857: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_19, [8192, 2048]);  where_19 = None
        permute_678: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_857, [1, 0])
        mm_269: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_678, view_47);  permute_678 = view_47 = None
        mm_270: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_857, permute_680);  view_857 = permute_680 = None
        view_858: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_270, [8, 1024, 512]);  mm_270 = None
        convert_element_type_1290: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_858, torch.float32);  view_858 = None
        convert_element_type_1291: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_269, torch.float32);  mm_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_553: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1290, primals_17);  primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_22: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_554: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1290, mul_22);  convert_element_type_1290 = mul_22 = None
        sum_96: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 1], True, dtype = torch.float32);  mul_554 = None
        view_859: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_96, [512]);  sum_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_555: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_553, add_14)
        mul_556: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_553, rsqrt_3);  mul_553 = None
        sum_97: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_555, [2], True, dtype = torch.float32);  mul_555 = None
        add_192: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_191, mul_556);  add_191 = mul_556 = None
        pow_89: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_557: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_97, -0.5);  sum_97 = None
        mul_558: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_557, pow_89);  mul_557 = pow_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_103: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_558, [8, 1024, 512]);  mul_558 = None
        div_52: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_103, 512);  expand_103 = None
        pow_90: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_14, 1.0);  add_14 = None
        mul_559: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_90, 2.0);  pow_90 = None
        mul_560: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, mul_559);  div_52 = mul_559 = None
        add_193: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_192, mul_560);  add_192 = mul_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_1292: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16)
        convert_element_type_1293: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_7, torch.bfloat16);  gt_7 = None
        mul_561: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1293, 1.1111111111111112);  convert_element_type_1293 = None
        mul_562: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1292, mul_561);  convert_element_type_1292 = mul_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_860: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_562, [8192, 512]);  mul_562 = None
        permute_682: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_860, [1, 0])
        mm_271: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_682, view_45);  permute_682 = view_45 = None
        mm_272: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_860, permute_684);  view_860 = permute_684 = None
        view_861: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_272, [8, 1024, 512]);  mm_272 = None
        convert_element_type_1298: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_271, torch.float32);  mm_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_862: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_861, [8, 1024, 8, 64]);  view_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_686: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_862, [0, 2, 1, 3]);  view_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_210: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_686, memory_format = torch.contiguous_format);  permute_686 = None
        view_863: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_210, [64, 1024, 64]);  clone_210 = None
        bmm_100: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_687, view_863);  permute_687 = None
        bmm_101: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_863, permute_688);  view_863 = permute_688 = None
        view_864: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_100, [8, 8, 1024, 64]);  bmm_100 = None
        view_865: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_101, [8, 8, 1024, 1024]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_1303: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_6, torch.bfloat16);  gt_6 = None
        mul_563: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1303, 1.1111111111111112);  convert_element_type_1303 = None
        mul_564: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_865, mul_563);  view_865 = mul_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_1304: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_564, torch.float32);  mul_564 = None
        convert_element_type_49: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_48, torch.float32);  convert_element_type_48 = None
        sub_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_49, amax_1);  convert_element_type_49 = amax_1 = None
        exp_1: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        div_3: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        mul_565: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1304, div_3);  convert_element_type_1304 = None
        sum_98: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_565, [-1], True)
        neg_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_16: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_18, sum_98, mul_565);  neg_18 = sum_98 = mul_565 = None
        convert_element_type_1305: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_866: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1305, [64, 1024, 1024]);  convert_element_type_1305 = None
        view_868: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_866, [8, 8, 1024, 1024]);  view_866 = None
        view_869: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_868, [64, 1024, 1024])
        convert_element_type_1306: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_868, torch.float32);  view_868 = None
        add_194: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_187, convert_element_type_1306);  add_187 = convert_element_type_1306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_102: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_689, view_869);  permute_689 = None
        bmm_103: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_869, permute_690);  view_869 = permute_690 = None
        view_871: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_102, [8, 8, 64, 1024]);  bmm_102 = None
        view_872: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_103, [8, 8, 1024, 64]);  bmm_103 = None
        permute_691: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_871, [0, 1, 3, 2]);  view_871 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_692: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_864, [0, 2, 1, 3]);  view_864 = None
        clone_213: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_692, memory_format = torch.contiguous_format);  permute_692 = None
        view_873: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_213, [8, 1024, 512]);  clone_213 = None
        view_874: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_873, [8192, 512]);  view_873 = None
        permute_693: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_874, [1, 0])
        mm_273: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_693, view_26);  permute_693 = None
        mm_274: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_874, permute_695);  view_874 = permute_695 = None
        view_875: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_274, [8, 1024, 512]);  mm_274 = None
        convert_element_type_1315: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_875, torch.float32);  view_875 = None
        convert_element_type_1316: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_273, torch.float32);  mm_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_697: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_691, [0, 2, 1, 3]);  permute_691 = None
        view_876: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_697, [8, 1024, 512]);  permute_697 = None
        clone_214: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_876, memory_format = torch.contiguous_format);  view_876 = None
        view_877: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_214, [8192, 512]);  clone_214 = None
        permute_698: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_877, [1, 0])
        mm_275: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_698, view_26);  permute_698 = None
        mm_276: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_877, permute_700);  view_877 = permute_700 = None
        view_878: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_276, [8, 1024, 512]);  mm_276 = None
        convert_element_type_1321: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_878, torch.float32);  view_878 = None
        add_195: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1315, convert_element_type_1321);  convert_element_type_1315 = convert_element_type_1321 = None
        convert_element_type_1322: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_275, torch.float32);  mm_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_702: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_872, [0, 2, 1, 3]);  view_872 = None
        clone_215: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_702, memory_format = torch.contiguous_format);  permute_702 = None
        view_879: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_215, [8, 1024, 512]);  clone_215 = None
        view_880: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_879, [8192, 512]);  view_879 = None
        permute_703: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_880, [1, 0])
        mm_277: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_703, view_26);  permute_703 = view_26 = None
        mm_278: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_880, permute_705);  view_880 = permute_705 = None
        view_881: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_278, [8, 1024, 512]);  mm_278 = None
        convert_element_type_1327: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_881, torch.float32);  view_881 = None
        add_196: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_195, convert_element_type_1327);  add_195 = convert_element_type_1327 = None
        convert_element_type_1328: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_277, torch.float32);  mm_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_566: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_196, primals_12);  primals_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_16: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_11, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_567: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_196, mul_16);  add_196 = mul_16 = None
        sum_99: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_567, [0, 1], True, dtype = torch.float32);  mul_567 = None
        view_882: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [512]);  sum_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_568: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, add_11)
        mul_569: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_566, rsqrt_2);  mul_566 = None
        sum_100: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_568, [2], True, dtype = torch.float32);  mul_568 = None
        add_197: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_193, mul_569);  add_193 = mul_569 = None
        pow_91: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_570: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_100, -0.5);  sum_100 = None
        mul_571: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_570, pow_91);  mul_570 = pow_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_104: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_571, [8, 1024, 512]);  mul_571 = None
        div_53: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_104, 512);  expand_104 = None
        pow_92: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_11, 1.0);  add_11 = None
        mul_572: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_92, 2.0);  pow_92 = None
        mul_573: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_53, mul_572);  div_53 = mul_572 = None
        add_198: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_197, mul_573);  add_197 = mul_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:136 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_1329: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_198, torch.bfloat16)
        convert_element_type_1330: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_5, torch.bfloat16);  gt_5 = None
        mul_574: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1330, 1.1111111111111112);  convert_element_type_1330 = None
        mul_575: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1329, mul_574);  convert_element_type_1329 = mul_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:89 in forward, code: hidden_states = self.wo(hidden_states)
        view_883: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_575, [8192, 512]);  mul_575 = None
        permute_707: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_279: "bf16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_707, view_24);  permute_707 = view_24 = None
        mm_280: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_883, permute_709);  view_883 = permute_709 = None
        view_884: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_280, [8, 1024, 2048]);  mm_280 = None
        convert_element_type_1336: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:88 in forward, code: hidden_states = hidden_states.to(self.wo.weight.dtype)
        convert_element_type_default_6: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_884, torch.bfloat16);  view_884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:82 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1338: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_4, torch.bfloat16);  gt_4 = None
        mul_576: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1338, 1.1111111111111112);  convert_element_type_1338 = None
        mul_577: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_6, mul_576);  convert_element_type_default_6 = mul_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:81 in forward, code: hidden_states = self.act(hidden_states)
        where_20: "bf16[8, 1024, 2048][2097152, 2048, 1]cuda:0" = torch.ops.aten.where.self(le_12, full_default_9, mul_577);  le_12 = full_default_9 = mul_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:80 in forward, code: hidden_states = self.wi(hidden_states)
        view_885: "bf16[8192, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(where_20, [8192, 2048]);  where_20 = None
        permute_711: "bf16[2048, 8192][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_885, [1, 0])
        mm_281: "bf16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_711, view_22);  permute_711 = view_22 = None
        mm_282: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_885, permute_713);  view_885 = permute_713 = None
        view_886: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_282, [8, 1024, 512]);  mm_282 = None
        convert_element_type_1343: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_886, torch.float32);  view_886 = None
        convert_element_type_1344: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_281, torch.float32);  mm_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_578: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1343, primals_9);  primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_10: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_579: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1343, mul_10);  convert_element_type_1343 = mul_10 = None
        sum_101: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 1], True, dtype = torch.float32);  mul_579 = None
        view_887: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_101, [512]);  sum_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_580: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_578, add_9)
        mul_581: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_578, rsqrt_1);  mul_578 = None
        sum_102: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_580, [2], True, dtype = torch.float32);  mul_580 = None
        add_199: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_198, mul_581);  add_198 = mul_581 = None
        pow_93: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_582: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_102, -0.5);  sum_102 = None
        mul_583: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_582, pow_93);  mul_582 = pow_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_105: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_583, [8, 1024, 512]);  mul_583 = None
        div_54: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_105, 512);  expand_105 = None
        pow_94: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(add_9, 1.0);  add_9 = None
        mul_584: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_94, 2.0);  pow_94 = None
        mul_585: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_54, mul_584);  div_54 = mul_584 = None
        add_200: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_199, mul_585);  add_199 = mul_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:362 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_1345: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_200, torch.bfloat16)
        convert_element_type_1346: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_586: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1346, 1.1111111111111112);  convert_element_type_1346 = None
        mul_587: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1345, mul_586);  convert_element_type_1345 = mul_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:325 in forward, code: attn_output = self.o(attn_output)
        view_888: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(mul_587, [8192, 512]);  mul_587 = None
        permute_715: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_888, [1, 0])
        mm_283: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_715, view_20);  permute_715 = view_20 = None
        mm_284: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_888, permute_717);  view_888 = permute_717 = None
        view_889: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_284, [8, 1024, 512]);  mm_284 = None
        convert_element_type_1351: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_283, torch.float32);  mm_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:324 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_890: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_889, [8, 1024, 8, 64]);  view_889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:323 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_719: "bf16[8, 8, 1024, 64][524288, 64, 512, 1]cuda:0" = torch.ops.aten.permute.default(view_890, [0, 2, 1, 3]);  view_890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:321 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_219: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_719, memory_format = torch.contiguous_format);  permute_719 = None
        view_891: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_219, [64, 1024, 64]);  clone_219 = None
        bmm_104: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_720, view_891);  permute_720 = None
        bmm_105: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_891, permute_721);  view_891 = permute_721 = None
        view_892: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_104, [8, 8, 1024, 64]);  bmm_104 = None
        view_893: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_105, [8, 8, 1024, 1024]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:319 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_1356: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_588: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1356, 1.1111111111111112);  convert_element_type_1356 = None
        mul_589: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_893, mul_588);  view_893 = mul_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_1357: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_589, torch.float32);  mul_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[8, 1, 1024, 1024][1048576, 1048576, 1024, 1]cuda:0" = torch.ops.aten.full.default([8, 1, 1024, 1024], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_12: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm, [8, 8, 1024, 1024]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f32[8, 1024, 1024][1, 8192, 8]cuda:0" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_5: "f32[1, 8, 1024, 1024][8, 1, 8192, 8]cuda:0" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f32[8, 8, 1024, 1024][8388608, 1, 8192, 8]cuda:0" = torch.ops.aten.add.Tensor(unsqueeze_5, full_default);  unsqueeze_5 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        add_8: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = add_7 = None
        convert_element_type_17: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16);  add_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:318 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        convert_element_type_18: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_17, torch.float32);  convert_element_type_17 = None
        sub_1: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax);  convert_element_type_18 = amax = None
        exp: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div_2: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_590: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1357, div_2);  convert_element_type_1357 = None
        sum_103: "f32[8, 8, 1024, 1][8192, 1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_590, [-1], True)
        neg_19: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_17: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.fma.default(neg_19, sum_103, mul_590);  neg_19 = sum_103 = mul_590 = None
        convert_element_type_1358: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:315 in forward, code: scores += position_bias_masked
        view_894: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1358, [64, 1024, 1024]);  convert_element_type_1358 = None
        view_896: "bf16[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_894, [8, 8, 1024, 1024]);  view_894 = None
        view_897: "bf16[64, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_896, [64, 1024, 1024])
        convert_element_type_1359: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_896, torch.float32);  view_896 = None
        add_201: "f32[8, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_194, convert_element_type_1359);  add_194 = convert_element_type_1359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:312 in forward, code: position_bias = position_bias + causal_mask
        sum_104: "f32[1, 8, 1024, 1024][8388608, 1048576, 1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_201, [0], True, dtype = torch.float32);  add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:237 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_2: "f32[8, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(sum_104, 0);  sum_104 = None
        permute_722: "f32[1024, 1024, 8][1024, 1, 1048576]cuda:0" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:236 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        _unsafe_masked_index_put_accumulate_2: "f32[32, 8][8, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_16, full_default_15, [add_6], permute_722);  full_default_16 = full_default_15 = add_6 = permute_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:295 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_106: "bf16[64, 64, 1024][65536, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(permute_723, view_897);  permute_723 = None
        bmm_107: "bf16[64, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_897, permute_724);  view_897 = permute_724 = None
        view_899: "bf16[8, 8, 64, 1024][524288, 65536, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_106, [8, 8, 64, 1024]);  bmm_106 = None
        view_900: "bf16[8, 8, 1024, 64][524288, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_107, [8, 8, 1024, 64]);  bmm_107 = None
        permute_725: "bf16[8, 8, 1024, 64][524288, 65536, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_899, [0, 1, 3, 2]);  view_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:286 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_726: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_892, [0, 2, 1, 3]);  view_892 = None
        clone_222: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_726, memory_format = torch.contiguous_format);  permute_726 = None
        view_901: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_222, [8, 1024, 512]);  clone_222 = None
        view_902: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_901, [8192, 512]);  view_901 = None
        permute_727: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_902, [1, 0])
        mm_285: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_727, view_1);  permute_727 = None
        mm_286: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_902, permute_729);  view_902 = permute_729 = None
        view_903: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_286, [8, 1024, 512]);  mm_286 = None
        convert_element_type_1368: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_903, torch.float32);  view_903 = None
        convert_element_type_1369: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_285, torch.float32);  mm_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:285 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_731: "bf16[8, 1024, 8, 64][524288, 1, 65536, 1024]cuda:0" = torch.ops.aten.permute.default(permute_725, [0, 2, 1, 3]);  permute_725 = None
        view_904: "bf16[8, 1024, 512][524288, 1, 1024]cuda:0" = torch.ops.aten.reshape.default(permute_731, [8, 1024, 512]);  permute_731 = None
        clone_223: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.clone.default(view_904, memory_format = torch.contiguous_format);  view_904 = None
        view_905: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_223, [8192, 512]);  clone_223 = None
        permute_732: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_905, [1, 0])
        mm_287: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_732, view_1);  permute_732 = None
        mm_288: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_905, permute_734);  view_905 = permute_734 = None
        view_906: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_288, [8, 1024, 512]);  mm_288 = None
        convert_element_type_1374: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_906, torch.float32);  view_906 = None
        add_202: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1368, convert_element_type_1374);  convert_element_type_1368 = convert_element_type_1374 = None
        convert_element_type_1375: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_287, torch.float32);  mm_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:264 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_736: "bf16[8, 1024, 8, 64][524288, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_900, [0, 2, 1, 3]);  view_900 = None
        clone_224: "bf16[8, 1024, 8, 64][524288, 512, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_736, memory_format = torch.contiguous_format);  permute_736 = None
        view_907: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_224, [8, 1024, 512]);  clone_224 = None
        view_908: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(view_907, [8192, 512]);  view_907 = None
        permute_737: "bf16[512, 8192][1, 512]cuda:0" = torch.ops.aten.permute.default(view_908, [1, 0])
        mm_289: "bf16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_737, view_1);  permute_737 = view_1 = None
        mm_290: "bf16[8192, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_908, permute_739);  view_908 = permute_739 = None
        view_909: "bf16[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_290, [8, 1024, 512]);  mm_290 = None
        convert_element_type_1380: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_909, torch.float32);  view_909 = None
        add_203: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_202, convert_element_type_1380);  add_202 = convert_element_type_1380 = None
        convert_element_type_1381: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_289, torch.float32);  mm_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_591: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_203, primals_3);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_1: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_2: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:68 in forward, code: return self.weight * hidden_states
        mul_592: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_203, mul_2);  add_203 = mul_2 = None
        sum_105: "f32[1, 1, 512][512, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_592, [0, 1], True, dtype = torch.float32);  mul_592 = None
        view_910: "f32[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [512]);  sum_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:62 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_593: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, mul_1)
        mul_594: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, rsqrt);  mul_591 = None
        sum_106: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_593, [2], True, dtype = torch.float32);  mul_593 = None
        add_204: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_200, mul_594);  add_200 = mul_594 = None
        pow_95: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_595: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Scalar(sum_106, -0.5);  sum_106 = None
        mul_596: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_595, pow_95);  mul_595 = pow_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:61 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_106: "f32[8, 1024, 512][1024, 1, 0]cuda:0" = torch.ops.aten.expand.default(mul_596, [8, 1024, 512]);  mul_596 = None
        div_55: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_106, 512);  expand_106 = None
        pow_96: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 1.0);  mul_1 = None
        mul_597: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Scalar(pow_96, 2.0);  pow_96 = None
        mul_598: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_55, mul_597);  div_55 = mul_597 = None
        add_205: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_204, mul_598);  add_204 = mul_598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:721 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_1382: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_599: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1382, 1.1111111111111112);  convert_element_type_1382 = None
        mul_600: "f32[8, 1024, 512][524288, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_205, mul_599);  add_205 = mul_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/t5/modeling_t5.py:669 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        _unsafe_masked_index_put_accumulate_3: "f32[32128, 512][512, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_17, unsqueeze_21, [primals_1], mul_600);  full_default_17 = unsqueeze_21 = primals_1 = mul_600 = None
        add_206: "f32[32128, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_163, _unsafe_masked_index_put_accumulate_3);  add_163 = _unsafe_masked_index_put_accumulate_3 = None
        return (None, add_206, view_910, convert_element_type_1381, convert_element_type_1375, convert_element_type_1369, _unsafe_masked_index_put_accumulate_2, convert_element_type_1351, view_887, convert_element_type_1344, convert_element_type_1336, view_882, convert_element_type_1328, convert_element_type_1322, convert_element_type_1316, convert_element_type_1298, view_859, convert_element_type_1291, convert_element_type_1283, view_854, convert_element_type_1275, convert_element_type_1269, convert_element_type_1263, convert_element_type_1245, view_831, convert_element_type_1238, convert_element_type_1230, view_826, convert_element_type_1222, convert_element_type_1216, convert_element_type_1210, convert_element_type_1192, view_803, convert_element_type_1185, convert_element_type_1177, view_798, convert_element_type_1169, convert_element_type_1163, convert_element_type_1157, convert_element_type_1139, view_775, convert_element_type_1132, convert_element_type_1124, view_770, convert_element_type_1116, convert_element_type_1110, convert_element_type_1104, convert_element_type_1086, view_747, convert_element_type_1079, convert_element_type_1071, view_742, None, view_741, convert_element_type_1061, convert_element_type_1055, convert_element_type_1049, _unsafe_masked_index_put_accumulate, convert_element_type_1031, view_718, convert_element_type_1024, convert_element_type_1018, convert_element_type_1012, convert_element_type_995, view_695, convert_element_type_988, convert_element_type_980, view_690, convert_element_type_972, convert_element_type_966, convert_element_type_960, convert_element_type_942, view_667, convert_element_type_935, convert_element_type_929, convert_element_type_923, convert_element_type_906, view_644, convert_element_type_899, convert_element_type_891, view_639, convert_element_type_883, convert_element_type_877, convert_element_type_871, convert_element_type_853, view_616, convert_element_type_846, convert_element_type_840, convert_element_type_834, convert_element_type_817, view_593, convert_element_type_810, convert_element_type_802, view_588, convert_element_type_794, convert_element_type_788, convert_element_type_782, convert_element_type_764, view_565, convert_element_type_757, convert_element_type_751, convert_element_type_745, convert_element_type_728, view_542, convert_element_type_721, convert_element_type_713, view_537, convert_element_type_705, convert_element_type_699, convert_element_type_693, convert_element_type_675, view_514, convert_element_type_668, convert_element_type_662, convert_element_type_656, convert_element_type_639, view_491, convert_element_type_632, convert_element_type_624, view_486, convert_element_type_616, convert_element_type_610, convert_element_type_604, convert_element_type_586, view_463, convert_element_type_579, convert_element_type_573, convert_element_type_567, convert_element_type_550, view_440, convert_element_type_543, convert_element_type_535, view_435)
