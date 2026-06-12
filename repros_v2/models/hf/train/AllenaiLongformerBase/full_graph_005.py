class GraphModule(torch.nn.Module):
    def forward(self, primals_9: "b8[8, 1024][1024, 1]cuda:0", primals_12: "f32[768][1]cuda:0", primals_18: "f32[768][1]cuda:0", primals_28: "f32[768][1]cuda:0", primals_34: "f32[768][1]cuda:0", primals_44: "f32[768][1]cuda:0", primals_50: "f32[768][1]cuda:0", primals_60: "f32[768][1]cuda:0", primals_66: "f32[768][1]cuda:0", primals_76: "f32[768][1]cuda:0", primals_82: "f32[768][1]cuda:0", primals_92: "f32[768][1]cuda:0", primals_98: "f32[768][1]cuda:0", primals_108: "f32[768][1]cuda:0", primals_114: "f32[768][1]cuda:0", primals_124: "f32[768][1]cuda:0", primals_130: "f32[768][1]cuda:0", primals_140: "f32[768][1]cuda:0", primals_146: "f32[768][1]cuda:0", primals_156: "f32[768][1]cuda:0", primals_162: "f32[768][1]cuda:0", primals_172: "f32[768][1]cuda:0", primals_178: "f32[768][1]cuda:0", primals_188: "f32[768][1]cuda:0", primals_194: "f32[768][1]cuda:0", view: "bf16[8192, 768][768, 1]cuda:0", unsqueeze_8: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0", rev_1: "bf16[1, 256, 1, 257][65792, 257, 257, 1]cuda:0", slice_scatter_10: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", slice_scatter_21: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0", amax: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_1: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_109: "bf16[8192, 768][768, 1]cuda:0", gt_1: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_5: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_111: "bf16[8192, 768][768, 1]cuda:0", addmm: "bf16[8192, 3072][3072, 1]cuda:0", view_113: "bf16[8192, 3072][3072, 1]cuda:0", gt_2: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_12: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_115: "bf16[8192, 768][768, 1]cuda:0", permute_184: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_1: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_2: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_3: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_224: "bf16[8192, 768][768, 1]cuda:0", gt_4: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_19: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_226: "bf16[8192, 768][768, 1]cuda:0", addmm_2: "bf16[8192, 3072][3072, 1]cuda:0", view_228: "bf16[8192, 3072][3072, 1]cuda:0", gt_5: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_26: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_230: "bf16[8192, 768][768, 1]cuda:0", permute_284: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_2: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_3: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_6: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_339: "bf16[8192, 768][768, 1]cuda:0", gt_7: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_33: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_341: "bf16[8192, 768][768, 1]cuda:0", addmm_4: "bf16[8192, 3072][3072, 1]cuda:0", view_343: "bf16[8192, 3072][3072, 1]cuda:0", gt_8: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_40: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_345: "bf16[8192, 768][768, 1]cuda:0", permute_384: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_3: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_4: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_9: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_454: "bf16[8192, 768][768, 1]cuda:0", gt_10: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_47: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_456: "bf16[8192, 768][768, 1]cuda:0", addmm_6: "bf16[8192, 3072][3072, 1]cuda:0", view_458: "bf16[8192, 3072][3072, 1]cuda:0", gt_11: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_54: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_460: "bf16[8192, 768][768, 1]cuda:0", permute_484: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_4: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_5: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_12: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_569: "bf16[8192, 768][768, 1]cuda:0", gt_13: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_61: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_571: "bf16[8192, 768][768, 1]cuda:0", addmm_8: "bf16[8192, 3072][3072, 1]cuda:0", view_573: "bf16[8192, 3072][3072, 1]cuda:0", gt_14: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_68: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_575: "bf16[8192, 768][768, 1]cuda:0", permute_584: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_5: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_6: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_15: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_684: "bf16[8192, 768][768, 1]cuda:0", gt_16: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_75: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_686: "bf16[8192, 768][768, 1]cuda:0", addmm_10: "bf16[8192, 3072][3072, 1]cuda:0", view_688: "bf16[8192, 3072][3072, 1]cuda:0", gt_17: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_82: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_690: "bf16[8192, 768][768, 1]cuda:0", permute_684: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_6: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_7: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_18: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_799: "bf16[8192, 768][768, 1]cuda:0", gt_19: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_89: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_801: "bf16[8192, 768][768, 1]cuda:0", addmm_12: "bf16[8192, 3072][3072, 1]cuda:0", view_803: "bf16[8192, 3072][3072, 1]cuda:0", gt_20: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_96: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_805: "bf16[8192, 768][768, 1]cuda:0", permute_784: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_7: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_8: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_21: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_914: "bf16[8192, 768][768, 1]cuda:0", gt_22: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_103: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_916: "bf16[8192, 768][768, 1]cuda:0", addmm_14: "bf16[8192, 3072][3072, 1]cuda:0", view_918: "bf16[8192, 3072][3072, 1]cuda:0", gt_23: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_110: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_920: "bf16[8192, 768][768, 1]cuda:0", permute_884: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_8: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_9: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_24: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_1029: "bf16[8192, 768][768, 1]cuda:0", gt_25: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_117: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_1031: "bf16[8192, 768][768, 1]cuda:0", addmm_16: "bf16[8192, 3072][3072, 1]cuda:0", view_1033: "bf16[8192, 3072][3072, 1]cuda:0", gt_26: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_124: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_1035: "bf16[8192, 768][768, 1]cuda:0", permute_984: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_9: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_10: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_27: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_1144: "bf16[8192, 768][768, 1]cuda:0", gt_28: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_131: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_1146: "bf16[8192, 768][768, 1]cuda:0", addmm_18: "bf16[8192, 3072][3072, 1]cuda:0", view_1148: "bf16[8192, 3072][3072, 1]cuda:0", gt_29: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_138: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_1150: "bf16[8192, 768][768, 1]cuda:0", permute_1084: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_10: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_11: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_30: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_1259: "bf16[8192, 768][768, 1]cuda:0", gt_31: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_145: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_1261: "bf16[8192, 768][768, 1]cuda:0", addmm_20: "bf16[8192, 3072][3072, 1]cuda:0", view_1263: "bf16[8192, 3072][3072, 1]cuda:0", gt_32: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_152: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_1265: "bf16[8192, 768][768, 1]cuda:0", permute_1184: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0", amax_11: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", sum_12: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0", gt_33: "b8[8, 1024, 12, 513][6422528, 6272, 513, 1]cuda:0", view_1374: "bf16[8192, 768][768, 1]cuda:0", gt_34: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_159: "f32[8, 1024, 768][786432, 768, 1]cuda:0", view_1376: "bf16[8192, 768][768, 1]cuda:0", addmm_22: "bf16[8192, 3072][3072, 1]cuda:0", view_1378: "bf16[8192, 3072][3072, 1]cuda:0", gt_35: "b8[8, 1024, 768][786432, 768, 1]cuda:0", mul_166: "f32[8, 1024, 768][786432, 768, 1]cuda:0", div_120: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1200: "bf16[768, 3072][3072, 1]cuda:0", permute_1204: "bf16[3072, 768][768, 1]cuda:0", div_121: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1210: "bf16[768, 768][768, 1]cuda:0", permute_1216: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_1217: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_1247: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_1248: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_1267: "bf16[768, 768][768, 1]cuda:0", permute_1278: "bf16[768, 768][768, 1]cuda:0", permute_1282: "bf16[768, 768][768, 1]cuda:0", div_123: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1285: "bf16[768, 3072][3072, 1]cuda:0", permute_1289: "bf16[3072, 768][768, 1]cuda:0", div_124: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1295: "bf16[768, 768][768, 1]cuda:0", permute_1301: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_1302: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_1332: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_1333: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_1352: "bf16[768, 768][768, 1]cuda:0", permute_1363: "bf16[768, 768][768, 1]cuda:0", permute_1367: "bf16[768, 768][768, 1]cuda:0", div_126: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1370: "bf16[768, 3072][3072, 1]cuda:0", permute_1374: "bf16[3072, 768][768, 1]cuda:0", div_127: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1380: "bf16[768, 768][768, 1]cuda:0", permute_1386: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_1387: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_1417: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_1418: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_1437: "bf16[768, 768][768, 1]cuda:0", permute_1448: "bf16[768, 768][768, 1]cuda:0", permute_1452: "bf16[768, 768][768, 1]cuda:0", div_129: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1455: "bf16[768, 3072][3072, 1]cuda:0", permute_1459: "bf16[3072, 768][768, 1]cuda:0", div_130: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1465: "bf16[768, 768][768, 1]cuda:0", permute_1471: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_1472: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_1502: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_1503: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_1522: "bf16[768, 768][768, 1]cuda:0", permute_1533: "bf16[768, 768][768, 1]cuda:0", permute_1537: "bf16[768, 768][768, 1]cuda:0", div_132: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1540: "bf16[768, 3072][3072, 1]cuda:0", permute_1544: "bf16[3072, 768][768, 1]cuda:0", div_133: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1550: "bf16[768, 768][768, 1]cuda:0", permute_1556: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_1557: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_1587: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_1588: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_1607: "bf16[768, 768][768, 1]cuda:0", permute_1618: "bf16[768, 768][768, 1]cuda:0", permute_1622: "bf16[768, 768][768, 1]cuda:0", div_135: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1625: "bf16[768, 3072][3072, 1]cuda:0", permute_1629: "bf16[3072, 768][768, 1]cuda:0", div_136: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1635: "bf16[768, 768][768, 1]cuda:0", permute_1641: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_1642: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_1672: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_1673: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_1692: "bf16[768, 768][768, 1]cuda:0", permute_1703: "bf16[768, 768][768, 1]cuda:0", permute_1707: "bf16[768, 768][768, 1]cuda:0", div_138: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1710: "bf16[768, 3072][3072, 1]cuda:0", permute_1714: "bf16[3072, 768][768, 1]cuda:0", div_139: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1720: "bf16[768, 768][768, 1]cuda:0", permute_1726: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_1727: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_1757: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_1758: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_1777: "bf16[768, 768][768, 1]cuda:0", permute_1788: "bf16[768, 768][768, 1]cuda:0", permute_1792: "bf16[768, 768][768, 1]cuda:0", div_141: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1795: "bf16[768, 3072][3072, 1]cuda:0", permute_1799: "bf16[3072, 768][768, 1]cuda:0", div_142: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1805: "bf16[768, 768][768, 1]cuda:0", permute_1811: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_1812: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_1842: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_1843: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_1862: "bf16[768, 768][768, 1]cuda:0", permute_1873: "bf16[768, 768][768, 1]cuda:0", permute_1877: "bf16[768, 768][768, 1]cuda:0", div_144: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1880: "bf16[768, 3072][3072, 1]cuda:0", permute_1884: "bf16[3072, 768][768, 1]cuda:0", div_145: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1890: "bf16[768, 768][768, 1]cuda:0", permute_1896: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_1897: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_1927: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_1928: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_1947: "bf16[768, 768][768, 1]cuda:0", permute_1958: "bf16[768, 768][768, 1]cuda:0", permute_1962: "bf16[768, 768][768, 1]cuda:0", div_147: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1965: "bf16[768, 3072][3072, 1]cuda:0", permute_1969: "bf16[3072, 768][768, 1]cuda:0", div_148: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_1975: "bf16[768, 768][768, 1]cuda:0", permute_1981: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_1982: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_2012: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_2013: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_2032: "bf16[768, 768][768, 1]cuda:0", permute_2043: "bf16[768, 768][768, 1]cuda:0", permute_2047: "bf16[768, 768][768, 1]cuda:0", div_150: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_2050: "bf16[768, 3072][3072, 1]cuda:0", permute_2054: "bf16[3072, 768][768, 1]cuda:0", div_151: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_2060: "bf16[768, 768][768, 1]cuda:0", permute_2066: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_2067: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_2097: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_2098: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_2117: "bf16[768, 768][768, 1]cuda:0", permute_2128: "bf16[768, 768][768, 1]cuda:0", permute_2132: "bf16[768, 768][768, 1]cuda:0", div_153: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_2135: "bf16[768, 3072][3072, 1]cuda:0", permute_2139: "bf16[3072, 768][768, 1]cuda:0", div_154: "f32[8, 1024, 1][1024, 1, 1]cuda:0", permute_2145: "bf16[768, 768][768, 1]cuda:0", permute_2151: "bf16[384, 768, 256][197120, 1, 769]cuda:0", permute_2152: "bf16[384, 64, 768][49152, 1, 64]cuda:0", permute_2182: "bf16[288, 64, 512][32768, 1, 64]cuda:0", permute_2183: "bf16[288, 512, 64][32768, 1, 512]cuda:0", permute_2202: "bf16[768, 768][768, 1]cuda:0", permute_2213: "bf16[768, 768][768, 1]cuda:0", permute_2217: "bf16[768, 768][768, 1]cuda:0", tangents_1: "f32[8, 1024, 768][786432, 768, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_169: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(tangents_1, primals_194);  primals_194 = None
        mul_170: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, 768)
        sum_13: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True)
        mul_171: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, mul_166);  mul_169 = None
        sum_14: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True);  mul_171 = None
        mul_172: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, sum_14);  sum_14 = None
        sub_97: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_170, sum_13);  mul_170 = sum_13 = None
        sub_98: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_97, mul_172);  sub_97 = mul_172 = None
        mul_173: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_120, sub_98);  div_120 = sub_98 = None
        mul_174: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(tangents_1, mul_166);  mul_166 = None
        sum_15: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 1]);  mul_174 = None
        sum_16: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0, 1]);  tangents_1 = None
        convert_element_type_516: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_517: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_35, torch.bfloat16);  gt_35 = None
        mul_175: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_517, 1.1111111111111112);  convert_element_type_517 = None
        mul_176: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_516, mul_175);  convert_element_type_516 = mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1380: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_176, [8192, 768]);  mul_176 = None
        mm_48: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_1380, permute_1200);  permute_1200 = None
        permute_1201: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1380, [1, 0])
        mm_49: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_1201, view_1378);  permute_1201 = view_1378 = None
        sum_17: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1380, [0], True, dtype = torch.float32);  view_1380 = None
        view_1381: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_17, [768]);  sum_17 = None
        convert_element_type_522: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1381, torch.bfloat16);  view_1381 = None
        view_1382: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [8, 1024, 3072]);  mm_48 = None
        convert_element_type_523: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_524: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_522, torch.float32);  convert_element_type_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_525: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1382, torch.float32);  view_1382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1377: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [8, 1024, 3072]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_509: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1377, torch.float32);  view_1377 = None
        mul_162: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_509, 0.7071067811865476)
        erf_11: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_162);  mul_162 = None
        add_176: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_178: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_176, 0.5);  add_176 = None
        mul_179: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_509, convert_element_type_509)
        mul_180: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, -0.5);  mul_179 = None
        exp_12: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_180);  mul_180 = None
        mul_181: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
        mul_182: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_509, mul_181);  convert_element_type_509 = mul_181 = None
        add_181: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_178, mul_182);  mul_178 = mul_182 = None
        mul_183: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_525, add_181);  convert_element_type_525 = add_181 = None
        convert_element_type_527: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_183, torch.bfloat16);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1383: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_527, [8192, 3072]);  convert_element_type_527 = None
        mm_50: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1383, permute_1204);  permute_1204 = None
        permute_1205: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_1383, [1, 0])
        mm_51: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1205, view_1376);  permute_1205 = view_1376 = None
        sum_18: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1383, [0], True, dtype = torch.float32);  view_1383 = None
        view_1384: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_18, [3072]);  sum_18 = None
        convert_element_type_532: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1384, torch.bfloat16);  view_1384 = None
        view_1385: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [8, 1024, 768]);  mm_50 = None
        convert_element_type_533: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1385, torch.float32);  view_1385 = None
        add_182: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_173, convert_element_type_533);  mul_173 = convert_element_type_533 = None
        convert_element_type_534: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_535: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_532, torch.float32);  convert_element_type_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_185: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_182, primals_188);  primals_188 = None
        mul_186: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, 768)
        sum_19: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True)
        mul_187: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, mul_159);  mul_185 = None
        sum_20: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
        mul_188: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_159, sum_20);  sum_20 = None
        sub_100: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_186, sum_19);  mul_186 = sum_19 = None
        sub_101: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_100, mul_188);  sub_100 = mul_188 = None
        mul_189: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_121, sub_101);  div_121 = sub_101 = None
        mul_190: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_182, mul_159);  mul_159 = None
        sum_21: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1]);  mul_190 = None
        sum_22: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1]);  add_182 = None
        convert_element_type_536: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_537: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_34, torch.bfloat16);  gt_34 = None
        mul_191: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_537, 1.1111111111111112);  convert_element_type_537 = None
        mul_192: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_536, mul_191);  convert_element_type_536 = mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_23: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_192, [0, 1], True, dtype = torch.float32)
        view_1386: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_23, [768]);  sum_23 = None
        convert_element_type_538: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1386, torch.bfloat16);  view_1386 = None
        view_1387: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_192, [8192, 768]);  mul_192 = None
        permute_1208: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1387, [1, 0])
        mm_52: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1208, view_1374);  permute_1208 = view_1374 = None
        mm_53: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1387, permute_1210);  view_1387 = permute_1210 = None
        view_1388: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_53, [8, 1024, 768]);  mm_53 = None
        convert_element_type_543: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_52, torch.float32);  mm_52 = None
        convert_element_type_544: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_538, torch.float32);  convert_element_type_538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1212: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_1388, [1, 0, 2]);  view_1388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1389: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1212, [1024, 8, 12, 64]);  permute_1212 = None
        permute_1213: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1389, [1, 0, 2, 3]);  view_1389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1214: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1213, [0, 2, 1, 3]);  permute_1213 = None
        clone_170: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1214, memory_format = torch.contiguous_format);  permute_1214 = None
        view_1390: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [96, 4, 256, 64]);  clone_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1391: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1390, [96, 4, 256, 64, 1]);  view_1390 = None
        permute_1215: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1391, [0, 1, 2, 4, 3]);  view_1391 = None
        view_1392: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1215, [384, 256, 64]);  permute_1215 = None
        bmm_24: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1216, view_1392);  permute_1216 = None
        bmm_25: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_1392, permute_1217);  view_1392 = permute_1217 = None
        view_1393: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_24, [96, 4, 768, 64, 1]);  bmm_24 = None
        view_1394: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_25, [96, 4, 256, 768, 1]);  bmm_25 = None
        squeeze: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1393, 4);  view_1393 = None
        squeeze_1: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1394, 4);  view_1394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        full_default_48: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 256, 769], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_264: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_1, 3, 0, -1);  squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1395: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_264, [96, 4, 196864]);  slice_scatter_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        full_default_49: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.full.default([96, 4, 197120], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_265: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_1395, 2, 0, -256);  view_1395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1396: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_265, [96, 4, 256, 770]);  slice_scatter_265 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_48: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1396, [0, -257]);  view_1396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        full_default_50: "bf16[9437184][1]cuda:0" = torch.ops.aten.full.default([9437184], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_48: "i64[9437184][1]cuda:0" = torch.ops.prims.iota.default(9437184, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        as_strided_108: "i64[96, 4, 768, 64][98304, 16384, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(iota_48, [96, 4, 768, 64], [98304, 16384, 64, 1], 0);  iota_48 = None
        view_1397: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze, [-1]);  squeeze = None
        clone_171: "i64[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.clone.default(as_strided_108, memory_format = torch.contiguous_format);  as_strided_108 = None
        view_1398: "i64[18874368][1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [18874368]);  clone_171 = None
        index_add: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_1397);  view_1397 = None
        as_strided_110: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add, [96, 1536, 64], [98304, 64, 1], 0);  index_add = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_49: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_110, [0, 0, -256, -256]);  as_strided_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1399: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_49, [8, 12, 1024, 64]);  constant_pad_nd_49 = None
        permute_1222: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_1399, [0, 2, 1, 3]);  view_1399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1400: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_48, [8, 12, 1024, 513]);  constant_pad_nd_48 = None
        permute_1223: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1400, [0, 2, 1, 3]);  view_1400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1224: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1222, [1, 0, 2, 3]);  permute_1222 = None
        clone_172: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1224, memory_format = torch.contiguous_format);  permute_1224 = None
        view_1401: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [1024, 8, 768]);  clone_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_549: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_33, torch.bfloat16);  gt_33 = None
        mul_193: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_549, 1.1111111111111112);  convert_element_type_549 = None
        mul_194: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1223, mul_193);  permute_1223 = mul_193 = None
        clone_173: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_194, memory_format = torch.contiguous_format);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_550: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_173, torch.float32);  clone_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        unsqueeze_17: "b8[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_9, 2);  primals_9 = None
        unsqueeze_18: "b8[8, 1024, 1, 1][1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 3);  unsqueeze_17 = None
        full_default_3: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_96: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_550);  convert_element_type_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_495: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1184, torch.float32);  permute_1184 = None
        clone_163: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_495, memory_format = torch.contiguous_format);  convert_element_type_495 = None
        sub_92: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_163, amax_11);  clone_163 = amax_11 = None
        exp_11: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_92);  sub_92 = None
        div_117: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        mul_195: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_96, div_117);  where_96 = None
        sum_24: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_195, [-1], True)
        neg: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_117);  div_117 = None
        fma: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg, sum_24, mul_195);  neg = sum_24 = mul_195 = None
        convert_element_type_551: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1225: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_551, [0, 2, 1, 3]);  convert_element_type_551 = None
        clone_174: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1225, memory_format = torch.contiguous_format);  permute_1225 = None
        view_1402: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_174, [96, 4, 256, 513]);  clone_174 = None
        view_1405: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1402, [8, 12, 1024, 513]);  view_1402 = None
        permute_1227: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1405, [0, 2, 1, 3]);  view_1405 = None
        clone_175: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1227, memory_format = torch.contiguous_format)
        copy_145: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_1227, clone_175);  permute_1227 = clone_175 = None
        permute_1228: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_145, [0, 2, 1, 3]);  copy_145 = None
        view_1407: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1228, [96, 4, 256, 513]);  permute_1228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1413: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1407, [8, 12, 1024, 513]);  view_1407 = None
        permute_1233: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1413, [0, 2, 1, 3]);  view_1413 = None
        slice_1612: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1233, 1, -256, 9223372036854775807)
        slice_1613: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1612, 3, -257, 9223372036854775807)
        clone_176: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1613, memory_format = torch.contiguous_format)
        full_default_52: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 12, 257], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_147: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1613, full_default_52);  slice_1613 = None
        slice_scatter_266: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1612, copy_147, 3, -257, 9223372036854775807);  slice_1612 = copy_147 = None
        slice_scatter_267: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1233, slice_scatter_266, 1, -256, 9223372036854775807);  permute_1233 = slice_scatter_266 = None
        permute_1235: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_267, [0, 2, 1, 3]);  slice_scatter_267 = None
        view_1415: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1235, [96, 4, 256, 513]);  permute_1235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        full_default_53: "bf16[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:753 in _mask_invalid_locations, code: ending_mask = ending_mask.expand(ending_input.size())
        expand_1: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(rev_1, [8, 256, 12, 257]);  rev_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        convert_element_type_18: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
        where_97: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_176);  clone_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        full_default_54: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 256, 12, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_268: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_97, 3, -257, 9223372036854775807);  where_97 = None
        full_default_55: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.full.default([8, 1024, 12, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_269: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_268, 1, -256, 9223372036854775807);  slice_scatter_268 = None
        permute_1237: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_269, [0, 2, 1, 3]);  slice_scatter_269 = None
        clone_177: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1237, memory_format = torch.contiguous_format);  permute_1237 = None
        view_1417: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [96, 4, 256, 513]);  clone_177 = None
        add_183: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1415, view_1417);  view_1415 = view_1417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1422: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_183, [8, 12, 1024, 513]);  add_183 = None
        permute_1241: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1422, [0, 2, 1, 3]);  view_1422 = None
        slice_1620: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1241, 1, 0, 256)
        slice_1621: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1620, 3, 0, 257)
        clone_178: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1621, memory_format = torch.contiguous_format)
        copy_149: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1621, full_default_52);  slice_1621 = None
        slice_scatter_270: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1620, copy_149, 3, 0, 257);  slice_1620 = copy_149 = None
        slice_scatter_271: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1241, slice_scatter_270, 1, 0, 256);  permute_1241 = slice_scatter_270 = None
        permute_1243: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_271, [0, 2, 1, 3]);  slice_scatter_271 = None
        view_1424: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1243, [96, 4, 256, 513]);  permute_1243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:748 in _mask_invalid_locations, code: beginning_mask = beginning_mask.expand(beginning_input.size())
        expand: "bf16[8, 256, 12, 257][0, 257, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_8, [8, 256, 12, 257]);  unsqueeze_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        convert_element_type_17: "b8[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
        where_98: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_178);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_272: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_98, 3, 0, 257);  where_98 = None
        slice_scatter_273: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_272, 1, 0, 256);  slice_scatter_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1245: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_273, [0, 2, 1, 3]);  slice_scatter_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_179: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1245, memory_format = torch.contiguous_format);  permute_1245 = None
        view_1426: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [96, 4, 256, 513]);  clone_179 = None
        add_184: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1424, view_1426);  view_1424 = view_1426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_375: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_184, 1, 0)
        slice_1628: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_375, 1, 1, 256)
        slice_1629: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1628, 2, 1, 256)
        clone_180: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_1629, memory_format = torch.contiguous_format)
        full_default_60: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.full.default([96, 255, 255], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_151: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1629, full_default_60);  slice_1629 = None
        slice_scatter_274: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1628, copy_151, 2, 1, 256);  slice_1628 = copy_151 = None
        slice_scatter_275: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_375, slice_scatter_274, 1, 1, 256);  select_375 = slice_scatter_274 = None
        select_scatter_48: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_184, slice_scatter_275, 1, 0);  add_184 = slice_scatter_275 = None
        full_default_61: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 255, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_276: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_180, 2, -255, 9223372036854775807);  clone_180 = None
        full_default_62: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 512, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_277: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_276, 1, 0, 255);  slice_scatter_276 = None
        full_default_63: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 3, 512, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_49: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_277, 1, 0);  slice_scatter_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1636: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_48, 1, 1, 9223372036854775807)
        slice_1637: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1636, 3, 0, 256)
        clone_181: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_1637, memory_format = torch.contiguous_format)
        full_default_64: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.full.default([96, 3, 256, 256], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_153: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1637, full_default_64);  slice_1637 = None
        slice_scatter_278: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1636, copy_153, 3, 0, 256);  slice_1636 = copy_153 = None
        slice_scatter_279: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_48, slice_scatter_278, 1, 1, 9223372036854775807);  select_scatter_48 = slice_scatter_278 = None
        full_default_65: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 3, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_280: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_181, 3, 257, 9223372036854775807);  clone_181 = None
        slice_scatter_281: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_280, 2, -257, -1);  slice_scatter_280 = None
        add_185: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_49, slice_scatter_281);  select_scatter_49 = slice_scatter_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_380: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_279, 1, -1)
        slice_1642: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_380, 2, 256, 9223372036854775807)
        clone_182: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1642, memory_format = torch.contiguous_format)
        full_default_67: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.full.default([96, 256, 257], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        copy_155: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1642, full_default_67);  slice_1642 = None
        slice_scatter_282: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_380, copy_155, 2, 256, 9223372036854775807);  select_380 = copy_155 = None
        select_scatter_50: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_279, slice_scatter_282, 1, -1);  slice_scatter_279 = slice_scatter_282 = None
        full_default_68: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.full.default([96, 256, 513], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_283: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_182, 2, 0, 257);  clone_182 = None
        slice_scatter_284: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_283, 1, 256, 9223372036854775807);  slice_scatter_283 = None
        select_scatter_51: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_284, 1, -1);  slice_scatter_284 = None
        add_186: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_185, select_scatter_51);  add_185 = select_scatter_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1647: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_50, 1, 0, -1);  select_scatter_50 = None
        slice_1648: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1647, 3, 256, 9223372036854775807);  slice_1647 = None
        clone_183: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1648, memory_format = torch.contiguous_format);  slice_1648 = None
        slice_scatter_285: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_183, 3, 0, 257);  clone_183 = None
        slice_scatter_286: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_285, 2, 0, 256);  slice_scatter_285 = None
        add_187: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_186, slice_scatter_286);  add_186 = slice_scatter_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1427: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_187, [96, 3, 513, 512]);  add_187 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_50: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1427, [0, 0, 0, -1]);  view_1427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1428: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_50, [96, 3, 512, 512, 1]);  constant_pad_nd_50 = None
        permute_1246: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1428, [0, 1, 2, 4, 3]);  view_1428 = None
        view_1429: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1246, [288, 512, 512]);  permute_1246 = None
        bmm_26: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1247, view_1429);  permute_1247 = None
        bmm_27: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1429, permute_1248);  view_1429 = permute_1248 = None
        view_1430: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_26, [96, 3, 64, 512, 1]);  bmm_26 = None
        permute_1249: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_1430, [0, 1, 4, 3, 2]);  view_1430 = None
        view_1431: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_27, [96, 3, 512, 64, 1]);  bmm_27 = None
        permute_1251: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_1249, [0, 1, 3, 4, 2]);  permute_1249 = None
        squeeze_2: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_1251, 4);  permute_1251 = None
        squeeze_3: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1431, 4);  view_1431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        full_default_73: "bf16[6291456][1]cuda:0" = torch.ops.aten.full.default([6291456], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        iota_49: "i64[6291456][1]cuda:0" = torch.ops.prims.iota.default(6291456, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        as_strided_111: "i64[96, 3, 512, 64][64, 1572864, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(iota_49, [96, 3, 512, 64], [64, 1572864, 6144, 1], 0);  iota_49 = None
        clone_184: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_2, memory_format = torch.contiguous_format);  squeeze_2 = None
        view_1432: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_184, [9437184]);  clone_184 = None
        clone_185: "i64[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(as_strided_111, memory_format = torch.contiguous_format);  as_strided_111 = None
        view_1433: "i64[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [9437184]);  clone_185 = None
        index_add_1: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1432);  view_1432 = None
        view_1435: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_3, [-1]);  squeeze_3 = None
        index_add_2: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1435);  view_1435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_125: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_2, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_2 = None
        view_1456: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_125, [96, 1024, 64]);  as_strided_125 = None
        view_1457: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1456, [8, 12, 1024, 64]);  view_1456 = None
        permute_1263: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1457, [0, 2, 1, 3]);  view_1457 = None
        permute_1264: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1263, [1, 0, 2, 3]);  permute_1263 = None
        view_1458: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1264, [1024, 8, 768]);  permute_1264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_122: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_1458, 8.0);  view_1458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_25: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1401, [0, 1], True, dtype = torch.float32)
        view_1459: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_25, [768]);  sum_25 = None
        convert_element_type_556: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1459, torch.bfloat16);  view_1459 = None
        view_1460: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1401, [8192, 768]);  view_1401 = None
        permute_1265: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1460, [1, 0])
        mm_54: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1265, view_1265);  permute_1265 = None
        mm_55: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1460, permute_1267);  view_1460 = permute_1267 = None
        view_1461: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_55, [1024, 8, 768]);  mm_55 = None
        convert_element_type_561: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1461, torch.float32);  view_1461 = None
        convert_element_type_562: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        convert_element_type_563: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_556, torch.float32);  convert_element_type_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_126: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_1, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_1 = None
        view_1462: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_126, [96, 1024, 64]);  as_strided_126 = None
        view_1463: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1462, [8, 12, 1024, 64]);  view_1462 = None
        permute_1269: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1463, [0, 2, 1, 3]);  view_1463 = None
        permute_1270: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1269, [1, 0, 2, 3]);  permute_1269 = None
        view_1464: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1270, [1024, 8, 768]);  permute_1270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_26: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1464, [0, 1], True, dtype = torch.float32)
        view_1465: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_26, [768]);  sum_26 = None
        convert_element_type_564: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1465, torch.bfloat16);  view_1465 = None
        view_1470: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1464, [8192, 768]);  view_1464 = None
        permute_1276: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1470, [1, 0])
        mm_56: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1276, view_1265);  permute_1276 = None
        mm_57: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1470, permute_1278);  view_1470 = permute_1278 = None
        view_1475: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_57, [1024, 8, 768]);  mm_57 = None
        convert_element_type_569: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1475, torch.float32);  view_1475 = None
        add_188: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_561, convert_element_type_569);  convert_element_type_561 = convert_element_type_569 = None
        convert_element_type_570: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        convert_element_type_571: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_564, torch.float32);  convert_element_type_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_27: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_122, [0, 1], True, dtype = torch.float32)
        view_1476: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [768]);  sum_27 = None
        convert_element_type_572: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1476, torch.bfloat16);  view_1476 = None
        view_1477: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_122, [8192, 768]);  div_122 = None
        permute_1280: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1477, [1, 0])
        mm_58: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1280, view_1265);  permute_1280 = view_1265 = None
        mm_59: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1477, permute_1282);  view_1477 = permute_1282 = None
        view_1478: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_59, [1024, 8, 768]);  mm_59 = None
        convert_element_type_577: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1478, torch.float32);  view_1478 = None
        add_189: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_188, convert_element_type_577);  add_188 = convert_element_type_577 = None
        convert_element_type_578: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        convert_element_type_579: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_572, torch.float32);  convert_element_type_572 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1284: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_189, [1, 0, 2]);  add_189 = None
        add_190: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_189, permute_1284);  mul_189 = permute_1284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_197: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_190, primals_178);  primals_178 = None
        mul_198: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_197, 768)
        sum_28: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True)
        mul_199: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_197, mul_152);  mul_197 = None
        sum_29: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True);  mul_199 = None
        mul_200: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, sum_29);  sum_29 = None
        sub_103: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_198, sum_28);  mul_198 = sum_28 = None
        sub_104: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_103, mul_200);  sub_103 = mul_200 = None
        mul_201: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_123, sub_104);  div_123 = sub_104 = None
        mul_202: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_190, mul_152);  mul_152 = None
        sum_30: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1]);  mul_202 = None
        sum_31: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_190, [0, 1]);  add_190 = None
        convert_element_type_580: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_581: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_32, torch.bfloat16);  gt_32 = None
        mul_203: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_581, 1.1111111111111112);  convert_element_type_581 = None
        mul_204: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_580, mul_203);  convert_element_type_580 = mul_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1479: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_204, [8192, 768]);  mul_204 = None
        mm_60: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_1479, permute_1285);  permute_1285 = None
        permute_1286: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1479, [1, 0])
        mm_61: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_1286, view_1263);  permute_1286 = view_1263 = None
        sum_32: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1479, [0], True, dtype = torch.float32);  view_1479 = None
        view_1480: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_32, [768]);  sum_32 = None
        convert_element_type_586: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1480, torch.bfloat16);  view_1480 = None
        view_1481: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [8, 1024, 3072]);  mm_60 = None
        convert_element_type_587: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_588: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_586, torch.float32);  convert_element_type_586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_589: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1481, torch.float32);  view_1481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1262: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [8, 1024, 3072]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_466: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1262, torch.float32);  view_1262 = None
        mul_148: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_466, 0.7071067811865476)
        erf_10: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
        add_161: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_206: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_161, 0.5);  add_161 = None
        mul_207: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_466, convert_element_type_466)
        mul_208: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_207, -0.5);  mul_207 = None
        exp_13: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_208);  mul_208 = None
        mul_209: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
        mul_210: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_466, mul_209);  convert_element_type_466 = mul_209 = None
        add_192: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_206, mul_210);  mul_206 = mul_210 = None
        mul_211: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_589, add_192);  convert_element_type_589 = add_192 = None
        convert_element_type_591: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_211, torch.bfloat16);  mul_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1482: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_591, [8192, 3072]);  convert_element_type_591 = None
        mm_62: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1482, permute_1289);  permute_1289 = None
        permute_1290: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_1482, [1, 0])
        mm_63: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1290, view_1261);  permute_1290 = view_1261 = None
        sum_33: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1482, [0], True, dtype = torch.float32);  view_1482 = None
        view_1483: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [3072]);  sum_33 = None
        convert_element_type_596: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1483, torch.bfloat16);  view_1483 = None
        view_1484: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [8, 1024, 768]);  mm_62 = None
        convert_element_type_597: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1484, torch.float32);  view_1484 = None
        add_193: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_201, convert_element_type_597);  mul_201 = convert_element_type_597 = None
        convert_element_type_598: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_599: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_596, torch.float32);  convert_element_type_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_213: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_193, primals_172);  primals_172 = None
        mul_214: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, 768)
        sum_34: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
        mul_215: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_213, mul_145);  mul_213 = None
        sum_35: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
        mul_216: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_145, sum_35);  sum_35 = None
        sub_106: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_214, sum_34);  mul_214 = sum_34 = None
        sub_107: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_106, mul_216);  sub_106 = mul_216 = None
        mul_217: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_124, sub_107);  div_124 = sub_107 = None
        mul_218: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_193, mul_145);  mul_145 = None
        sum_36: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
        sum_37: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_193, [0, 1]);  add_193 = None
        convert_element_type_600: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_217, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_601: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_31, torch.bfloat16);  gt_31 = None
        mul_219: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_601, 1.1111111111111112);  convert_element_type_601 = None
        mul_220: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_600, mul_219);  convert_element_type_600 = mul_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_38: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1], True, dtype = torch.float32)
        view_1485: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_38, [768]);  sum_38 = None
        convert_element_type_602: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1485, torch.bfloat16);  view_1485 = None
        view_1486: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_220, [8192, 768]);  mul_220 = None
        permute_1293: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1486, [1, 0])
        mm_64: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1293, view_1259);  permute_1293 = view_1259 = None
        mm_65: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1486, permute_1295);  view_1486 = permute_1295 = None
        view_1487: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_65, [8, 1024, 768]);  mm_65 = None
        convert_element_type_607: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_64, torch.float32);  mm_64 = None
        convert_element_type_608: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_602, torch.float32);  convert_element_type_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1297: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_1487, [1, 0, 2]);  view_1487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1488: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1297, [1024, 8, 12, 64]);  permute_1297 = None
        permute_1298: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1488, [1, 0, 2, 3]);  view_1488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1299: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1298, [0, 2, 1, 3]);  permute_1298 = None
        clone_189: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1299, memory_format = torch.contiguous_format);  permute_1299 = None
        view_1489: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_189, [96, 4, 256, 64]);  clone_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1490: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1489, [96, 4, 256, 64, 1]);  view_1489 = None
        permute_1300: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1490, [0, 1, 2, 4, 3]);  view_1490 = None
        view_1491: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1300, [384, 256, 64]);  permute_1300 = None
        bmm_28: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1301, view_1491);  permute_1301 = None
        bmm_29: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_1491, permute_1302);  view_1491 = permute_1302 = None
        view_1492: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [96, 4, 768, 64, 1]);  bmm_28 = None
        view_1493: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [96, 4, 256, 768, 1]);  bmm_29 = None
        squeeze_4: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1492, 4);  view_1492 = None
        squeeze_5: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1493, 4);  view_1493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_287: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_5, 3, 0, -1);  squeeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1494: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_287, [96, 4, 196864]);  slice_scatter_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_288: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_1494, 2, 0, -256);  view_1494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1495: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_288, [96, 4, 256, 770]);  slice_scatter_288 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_51: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1495, [0, -257]);  view_1495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1496: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_4, [-1]);  squeeze_4 = None
        index_add_3: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_1496);  view_1496 = None
        as_strided_131: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_3, [96, 1536, 64], [98304, 64, 1], 0);  index_add_3 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_52: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_131, [0, 0, -256, -256]);  as_strided_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1498: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_52, [8, 12, 1024, 64]);  constant_pad_nd_52 = None
        permute_1307: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_1498, [0, 2, 1, 3]);  view_1498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1499: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_51, [8, 12, 1024, 513]);  constant_pad_nd_51 = None
        permute_1308: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1499, [0, 2, 1, 3]);  view_1499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1309: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1307, [1, 0, 2, 3]);  permute_1307 = None
        clone_191: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1309, memory_format = torch.contiguous_format);  permute_1309 = None
        view_1500: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_191, [1024, 8, 768]);  clone_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_613: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_30, torch.bfloat16);  gt_30 = None
        mul_221: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_613, 1.1111111111111112);  convert_element_type_613 = None
        mul_222: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1308, mul_221);  permute_1308 = mul_221 = None
        clone_192: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_222, memory_format = torch.contiguous_format);  mul_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_614: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_192, torch.float32);  clone_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_99: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_614);  convert_element_type_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_452: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_1084, torch.float32);  permute_1084 = None
        clone_149: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_452, memory_format = torch.contiguous_format);  convert_element_type_452 = None
        sub_84: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_149, amax_10);  clone_149 = amax_10 = None
        exp_10: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
        div_107: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        mul_223: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_99, div_107);  where_99 = None
        sum_39: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_223, [-1], True)
        neg_1: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_107);  div_107 = None
        fma_1: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_1, sum_39, mul_223);  neg_1 = sum_39 = mul_223 = None
        convert_element_type_615: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1310: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_615, [0, 2, 1, 3]);  convert_element_type_615 = None
        clone_193: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1310, memory_format = torch.contiguous_format);  permute_1310 = None
        view_1501: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_193, [96, 4, 256, 513]);  clone_193 = None
        view_1504: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1501, [8, 12, 1024, 513]);  view_1501 = None
        permute_1312: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1504, [0, 2, 1, 3]);  view_1504 = None
        clone_194: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1312, memory_format = torch.contiguous_format)
        copy_158: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_1312, clone_194);  permute_1312 = clone_194 = None
        permute_1313: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_158, [0, 2, 1, 3]);  copy_158 = None
        view_1506: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1313, [96, 4, 256, 513]);  permute_1313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1512: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1506, [8, 12, 1024, 513]);  view_1506 = None
        permute_1318: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1512, [0, 2, 1, 3]);  view_1512 = None
        slice_1652: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1318, 1, -256, 9223372036854775807)
        slice_1653: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1652, 3, -257, 9223372036854775807)
        clone_195: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1653, memory_format = torch.contiguous_format)
        copy_160: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1653, full_default_52);  slice_1653 = None
        slice_scatter_289: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1652, copy_160, 3, -257, 9223372036854775807);  slice_1652 = copy_160 = None
        slice_scatter_290: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1318, slice_scatter_289, 1, -256, 9223372036854775807);  permute_1318 = slice_scatter_289 = None
        permute_1320: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_290, [0, 2, 1, 3]);  slice_scatter_290 = None
        view_1514: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1320, [96, 4, 256, 513]);  permute_1320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_100: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_195);  clone_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_291: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_100, 3, -257, 9223372036854775807);  where_100 = None
        slice_scatter_292: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_291, 1, -256, 9223372036854775807);  slice_scatter_291 = None
        permute_1322: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_292, [0, 2, 1, 3]);  slice_scatter_292 = None
        clone_196: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1322, memory_format = torch.contiguous_format);  permute_1322 = None
        view_1516: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_196, [96, 4, 256, 513]);  clone_196 = None
        add_194: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1514, view_1516);  view_1514 = view_1516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1521: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_194, [8, 12, 1024, 513]);  add_194 = None
        permute_1326: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1521, [0, 2, 1, 3]);  view_1521 = None
        slice_1660: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1326, 1, 0, 256)
        slice_1661: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1660, 3, 0, 257)
        clone_197: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1661, memory_format = torch.contiguous_format)
        copy_162: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1661, full_default_52);  slice_1661 = None
        slice_scatter_293: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1660, copy_162, 3, 0, 257);  slice_1660 = copy_162 = None
        slice_scatter_294: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1326, slice_scatter_293, 1, 0, 256);  permute_1326 = slice_scatter_293 = None
        permute_1328: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_294, [0, 2, 1, 3]);  slice_scatter_294 = None
        view_1523: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1328, [96, 4, 256, 513]);  permute_1328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_101: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_197);  clone_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_295: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_101, 3, 0, 257);  where_101 = None
        slice_scatter_296: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_295, 1, 0, 256);  slice_scatter_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1330: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_296, [0, 2, 1, 3]);  slice_scatter_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_198: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1330, memory_format = torch.contiguous_format);  permute_1330 = None
        view_1525: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_198, [96, 4, 256, 513]);  clone_198 = None
        add_195: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1523, view_1525);  view_1523 = view_1525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_386: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_195, 1, 0)
        slice_1668: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_386, 1, 1, 256)
        slice_1669: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1668, 2, 1, 256)
        clone_199: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_1669, memory_format = torch.contiguous_format)
        copy_164: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1669, full_default_60);  slice_1669 = None
        slice_scatter_297: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1668, copy_164, 2, 1, 256);  slice_1668 = copy_164 = None
        slice_scatter_298: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_386, slice_scatter_297, 1, 1, 256);  select_386 = slice_scatter_297 = None
        select_scatter_52: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_195, slice_scatter_298, 1, 0);  add_195 = slice_scatter_298 = None
        slice_scatter_299: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_199, 2, -255, 9223372036854775807);  clone_199 = None
        slice_scatter_300: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_299, 1, 0, 255);  slice_scatter_299 = None
        select_scatter_53: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_300, 1, 0);  slice_scatter_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1676: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_52, 1, 1, 9223372036854775807)
        slice_1677: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1676, 3, 0, 256)
        clone_200: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_1677, memory_format = torch.contiguous_format)
        copy_166: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1677, full_default_64);  slice_1677 = None
        slice_scatter_301: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1676, copy_166, 3, 0, 256);  slice_1676 = copy_166 = None
        slice_scatter_302: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_52, slice_scatter_301, 1, 1, 9223372036854775807);  select_scatter_52 = slice_scatter_301 = None
        slice_scatter_303: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_200, 3, 257, 9223372036854775807);  clone_200 = None
        slice_scatter_304: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_303, 2, -257, -1);  slice_scatter_303 = None
        add_196: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_53, slice_scatter_304);  select_scatter_53 = slice_scatter_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_391: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_302, 1, -1)
        slice_1682: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_391, 2, 256, 9223372036854775807)
        clone_201: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1682, memory_format = torch.contiguous_format)
        copy_168: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1682, full_default_67);  slice_1682 = None
        slice_scatter_305: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_391, copy_168, 2, 256, 9223372036854775807);  select_391 = copy_168 = None
        select_scatter_54: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_302, slice_scatter_305, 1, -1);  slice_scatter_302 = slice_scatter_305 = None
        slice_scatter_306: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_201, 2, 0, 257);  clone_201 = None
        slice_scatter_307: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_306, 1, 256, 9223372036854775807);  slice_scatter_306 = None
        select_scatter_55: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_307, 1, -1);  slice_scatter_307 = None
        add_197: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_196, select_scatter_55);  add_196 = select_scatter_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1687: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_54, 1, 0, -1);  select_scatter_54 = None
        slice_1688: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1687, 3, 256, 9223372036854775807);  slice_1687 = None
        clone_202: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1688, memory_format = torch.contiguous_format);  slice_1688 = None
        slice_scatter_308: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_202, 3, 0, 257);  clone_202 = None
        slice_scatter_309: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_308, 2, 0, 256);  slice_scatter_308 = None
        add_198: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_197, slice_scatter_309);  add_197 = slice_scatter_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1526: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_198, [96, 3, 513, 512]);  add_198 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_53: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1526, [0, 0, 0, -1]);  view_1526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1527: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_53, [96, 3, 512, 512, 1]);  constant_pad_nd_53 = None
        permute_1331: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1527, [0, 1, 2, 4, 3]);  view_1527 = None
        view_1528: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1331, [288, 512, 512]);  permute_1331 = None
        bmm_30: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1332, view_1528);  permute_1332 = None
        bmm_31: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1528, permute_1333);  view_1528 = permute_1333 = None
        view_1529: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [96, 3, 64, 512, 1]);  bmm_30 = None
        permute_1334: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_1529, [0, 1, 4, 3, 2]);  view_1529 = None
        view_1530: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_31, [96, 3, 512, 64, 1]);  bmm_31 = None
        permute_1336: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_1334, [0, 1, 3, 4, 2]);  permute_1334 = None
        squeeze_6: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_1336, 4);  permute_1336 = None
        squeeze_7: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1530, 4);  view_1530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_203: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_6, memory_format = torch.contiguous_format);  squeeze_6 = None
        view_1531: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_203, [9437184]);  clone_203 = None
        index_add_4: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1531);  view_1531 = None
        view_1534: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_7, [-1]);  squeeze_7 = None
        index_add_5: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1534);  view_1534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_146: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_5, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_5 = None
        view_1555: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_146, [96, 1024, 64]);  as_strided_146 = None
        view_1556: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1555, [8, 12, 1024, 64]);  view_1555 = None
        permute_1348: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1556, [0, 2, 1, 3]);  view_1556 = None
        permute_1349: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1348, [1, 0, 2, 3]);  permute_1348 = None
        view_1557: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1349, [1024, 8, 768]);  permute_1349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_125: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_1557, 8.0);  view_1557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_40: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1500, [0, 1], True, dtype = torch.float32)
        view_1558: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_40, [768]);  sum_40 = None
        convert_element_type_620: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1558, torch.bfloat16);  view_1558 = None
        view_1559: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1500, [8192, 768]);  view_1500 = None
        permute_1350: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1559, [1, 0])
        mm_66: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1350, view_1150);  permute_1350 = None
        mm_67: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1559, permute_1352);  view_1559 = permute_1352 = None
        view_1560: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_67, [1024, 8, 768]);  mm_67 = None
        convert_element_type_625: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1560, torch.float32);  view_1560 = None
        convert_element_type_626: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        convert_element_type_627: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_620, torch.float32);  convert_element_type_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_147: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_4, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_4 = None
        view_1561: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_147, [96, 1024, 64]);  as_strided_147 = None
        view_1562: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1561, [8, 12, 1024, 64]);  view_1561 = None
        permute_1354: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1562, [0, 2, 1, 3]);  view_1562 = None
        permute_1355: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1354, [1, 0, 2, 3]);  permute_1354 = None
        view_1563: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1355, [1024, 8, 768]);  permute_1355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_41: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1563, [0, 1], True, dtype = torch.float32)
        view_1564: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_41, [768]);  sum_41 = None
        convert_element_type_628: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1564, torch.bfloat16);  view_1564 = None
        view_1569: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1563, [8192, 768]);  view_1563 = None
        permute_1361: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1569, [1, 0])
        mm_68: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1361, view_1150);  permute_1361 = None
        mm_69: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1569, permute_1363);  view_1569 = permute_1363 = None
        view_1574: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_69, [1024, 8, 768]);  mm_69 = None
        convert_element_type_633: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1574, torch.float32);  view_1574 = None
        add_199: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_625, convert_element_type_633);  convert_element_type_625 = convert_element_type_633 = None
        convert_element_type_634: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        convert_element_type_635: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_628, torch.float32);  convert_element_type_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_42: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_125, [0, 1], True, dtype = torch.float32)
        view_1575: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [768]);  sum_42 = None
        convert_element_type_636: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1575, torch.bfloat16);  view_1575 = None
        view_1576: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_125, [8192, 768]);  div_125 = None
        permute_1365: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1576, [1, 0])
        mm_70: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1365, view_1150);  permute_1365 = view_1150 = None
        mm_71: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1576, permute_1367);  view_1576 = permute_1367 = None
        view_1577: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_71, [1024, 8, 768]);  mm_71 = None
        convert_element_type_641: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1577, torch.float32);  view_1577 = None
        add_200: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_199, convert_element_type_641);  add_199 = convert_element_type_641 = None
        convert_element_type_642: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        convert_element_type_643: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_636, torch.float32);  convert_element_type_636 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1369: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_200, [1, 0, 2]);  add_200 = None
        add_201: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_217, permute_1369);  mul_217 = permute_1369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_225: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_201, primals_162);  primals_162 = None
        mul_226: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_225, 768)
        sum_43: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_225, [2], True)
        mul_227: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_225, mul_138);  mul_225 = None
        sum_44: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True);  mul_227 = None
        mul_228: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_138, sum_44);  sum_44 = None
        sub_109: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_226, sum_43);  mul_226 = sum_43 = None
        sub_110: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_109, mul_228);  sub_109 = mul_228 = None
        mul_229: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_126, sub_110);  div_126 = sub_110 = None
        mul_230: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_201, mul_138);  mul_138 = None
        sum_45: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1]);  mul_230 = None
        sum_46: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_201, [0, 1]);  add_201 = None
        convert_element_type_644: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_229, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_645: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_29, torch.bfloat16);  gt_29 = None
        mul_231: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_645, 1.1111111111111112);  convert_element_type_645 = None
        mul_232: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_644, mul_231);  convert_element_type_644 = mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1578: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_232, [8192, 768]);  mul_232 = None
        mm_72: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_1578, permute_1370);  permute_1370 = None
        permute_1371: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1578, [1, 0])
        mm_73: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_1371, view_1148);  permute_1371 = view_1148 = None
        sum_47: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1578, [0], True, dtype = torch.float32);  view_1578 = None
        view_1579: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_47, [768]);  sum_47 = None
        convert_element_type_650: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1579, torch.bfloat16);  view_1579 = None
        view_1580: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [8, 1024, 3072]);  mm_72 = None
        convert_element_type_651: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_652: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_650, torch.float32);  convert_element_type_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_653: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1580, torch.float32);  view_1580 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1147: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [8, 1024, 3072]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_423: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1147, torch.float32);  view_1147 = None
        mul_134: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_423, 0.7071067811865476)
        erf_9: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_134);  mul_134 = None
        add_146: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_234: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_146, 0.5);  add_146 = None
        mul_235: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_423, convert_element_type_423)
        mul_236: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_235, -0.5);  mul_235 = None
        exp_14: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_236);  mul_236 = None
        mul_237: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
        mul_238: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_423, mul_237);  convert_element_type_423 = mul_237 = None
        add_203: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_234, mul_238);  mul_234 = mul_238 = None
        mul_239: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_653, add_203);  convert_element_type_653 = add_203 = None
        convert_element_type_655: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_239, torch.bfloat16);  mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1581: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_655, [8192, 3072]);  convert_element_type_655 = None
        mm_74: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1581, permute_1374);  permute_1374 = None
        permute_1375: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_1581, [1, 0])
        mm_75: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1375, view_1146);  permute_1375 = view_1146 = None
        sum_48: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1581, [0], True, dtype = torch.float32);  view_1581 = None
        view_1582: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [3072]);  sum_48 = None
        convert_element_type_660: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1582, torch.bfloat16);  view_1582 = None
        view_1583: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [8, 1024, 768]);  mm_74 = None
        convert_element_type_661: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1583, torch.float32);  view_1583 = None
        add_204: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_229, convert_element_type_661);  mul_229 = convert_element_type_661 = None
        convert_element_type_662: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_663: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_660, torch.float32);  convert_element_type_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_241: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_204, primals_156);  primals_156 = None
        mul_242: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, 768)
        sum_49: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
        mul_243: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, mul_131);  mul_241 = None
        sum_50: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
        mul_244: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_131, sum_50);  sum_50 = None
        sub_112: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_242, sum_49);  mul_242 = sum_49 = None
        sub_113: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_112, mul_244);  sub_112 = mul_244 = None
        mul_245: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_127, sub_113);  div_127 = sub_113 = None
        mul_246: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_204, mul_131);  mul_131 = None
        sum_51: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
        sum_52: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_204, [0, 1]);  add_204 = None
        convert_element_type_664: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_245, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_665: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_28, torch.bfloat16);  gt_28 = None
        mul_247: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_665, 1.1111111111111112);  convert_element_type_665 = None
        mul_248: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_664, mul_247);  convert_element_type_664 = mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_53: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1], True, dtype = torch.float32)
        view_1584: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_53, [768]);  sum_53 = None
        convert_element_type_666: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1584, torch.bfloat16);  view_1584 = None
        view_1585: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_248, [8192, 768]);  mul_248 = None
        permute_1378: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1585, [1, 0])
        mm_76: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1378, view_1144);  permute_1378 = view_1144 = None
        mm_77: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1585, permute_1380);  view_1585 = permute_1380 = None
        view_1586: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_77, [8, 1024, 768]);  mm_77 = None
        convert_element_type_671: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        convert_element_type_672: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_666, torch.float32);  convert_element_type_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1382: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_1586, [1, 0, 2]);  view_1586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1587: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1382, [1024, 8, 12, 64]);  permute_1382 = None
        permute_1383: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1587, [1, 0, 2, 3]);  view_1587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1384: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1383, [0, 2, 1, 3]);  permute_1383 = None
        clone_208: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1384, memory_format = torch.contiguous_format);  permute_1384 = None
        view_1588: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_208, [96, 4, 256, 64]);  clone_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1589: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1588, [96, 4, 256, 64, 1]);  view_1588 = None
        permute_1385: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1589, [0, 1, 2, 4, 3]);  view_1589 = None
        view_1590: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1385, [384, 256, 64]);  permute_1385 = None
        bmm_32: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1386, view_1590);  permute_1386 = None
        bmm_33: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_1590, permute_1387);  view_1590 = permute_1387 = None
        view_1591: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_32, [96, 4, 768, 64, 1]);  bmm_32 = None
        view_1592: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_33, [96, 4, 256, 768, 1]);  bmm_33 = None
        squeeze_8: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1591, 4);  view_1591 = None
        squeeze_9: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1592, 4);  view_1592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_310: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_9, 3, 0, -1);  squeeze_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1593: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_310, [96, 4, 196864]);  slice_scatter_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_311: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_1593, 2, 0, -256);  view_1593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1594: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_311, [96, 4, 256, 770]);  slice_scatter_311 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_54: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1594, [0, -257]);  view_1594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1595: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_8, [-1]);  squeeze_8 = None
        index_add_6: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_1595);  view_1595 = None
        as_strided_152: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_6, [96, 1536, 64], [98304, 64, 1], 0);  index_add_6 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_55: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_152, [0, 0, -256, -256]);  as_strided_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1597: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_55, [8, 12, 1024, 64]);  constant_pad_nd_55 = None
        permute_1392: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_1597, [0, 2, 1, 3]);  view_1597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1598: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_54, [8, 12, 1024, 513]);  constant_pad_nd_54 = None
        permute_1393: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1598, [0, 2, 1, 3]);  view_1598 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1394: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1392, [1, 0, 2, 3]);  permute_1392 = None
        clone_210: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1394, memory_format = torch.contiguous_format);  permute_1394 = None
        view_1599: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_210, [1024, 8, 768]);  clone_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_677: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_27, torch.bfloat16);  gt_27 = None
        mul_249: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_677, 1.1111111111111112);  convert_element_type_677 = None
        mul_250: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1393, mul_249);  permute_1393 = mul_249 = None
        clone_211: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_250, memory_format = torch.contiguous_format);  mul_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_678: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_211, torch.float32);  clone_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_102: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_678);  convert_element_type_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_409: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_984, torch.float32);  permute_984 = None
        clone_135: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_409, memory_format = torch.contiguous_format);  convert_element_type_409 = None
        sub_76: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_135, amax_9);  clone_135 = amax_9 = None
        exp_9: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        div_97: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        mul_251: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_102, div_97);  where_102 = None
        sum_54: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
        neg_2: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_97);  div_97 = None
        fma_2: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_2, sum_54, mul_251);  neg_2 = sum_54 = mul_251 = None
        convert_element_type_679: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1395: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_679, [0, 2, 1, 3]);  convert_element_type_679 = None
        clone_212: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1395, memory_format = torch.contiguous_format);  permute_1395 = None
        view_1600: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_212, [96, 4, 256, 513]);  clone_212 = None
        view_1603: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1600, [8, 12, 1024, 513]);  view_1600 = None
        permute_1397: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1603, [0, 2, 1, 3]);  view_1603 = None
        clone_213: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1397, memory_format = torch.contiguous_format)
        copy_171: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_1397, clone_213);  permute_1397 = clone_213 = None
        permute_1398: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_171, [0, 2, 1, 3]);  copy_171 = None
        view_1605: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1398, [96, 4, 256, 513]);  permute_1398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1611: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1605, [8, 12, 1024, 513]);  view_1605 = None
        permute_1403: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1611, [0, 2, 1, 3]);  view_1611 = None
        slice_1692: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1403, 1, -256, 9223372036854775807)
        slice_1693: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1692, 3, -257, 9223372036854775807)
        clone_214: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1693, memory_format = torch.contiguous_format)
        copy_173: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1693, full_default_52);  slice_1693 = None
        slice_scatter_312: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1692, copy_173, 3, -257, 9223372036854775807);  slice_1692 = copy_173 = None
        slice_scatter_313: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1403, slice_scatter_312, 1, -256, 9223372036854775807);  permute_1403 = slice_scatter_312 = None
        permute_1405: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_313, [0, 2, 1, 3]);  slice_scatter_313 = None
        view_1613: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1405, [96, 4, 256, 513]);  permute_1405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_103: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_214);  clone_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_314: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_103, 3, -257, 9223372036854775807);  where_103 = None
        slice_scatter_315: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_314, 1, -256, 9223372036854775807);  slice_scatter_314 = None
        permute_1407: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_315, [0, 2, 1, 3]);  slice_scatter_315 = None
        clone_215: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1407, memory_format = torch.contiguous_format);  permute_1407 = None
        view_1615: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_215, [96, 4, 256, 513]);  clone_215 = None
        add_205: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1613, view_1615);  view_1613 = view_1615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1620: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_205, [8, 12, 1024, 513]);  add_205 = None
        permute_1411: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1620, [0, 2, 1, 3]);  view_1620 = None
        slice_1700: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1411, 1, 0, 256)
        slice_1701: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1700, 3, 0, 257)
        clone_216: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1701, memory_format = torch.contiguous_format)
        copy_175: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1701, full_default_52);  slice_1701 = None
        slice_scatter_316: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1700, copy_175, 3, 0, 257);  slice_1700 = copy_175 = None
        slice_scatter_317: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1411, slice_scatter_316, 1, 0, 256);  permute_1411 = slice_scatter_316 = None
        permute_1413: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_317, [0, 2, 1, 3]);  slice_scatter_317 = None
        view_1622: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1413, [96, 4, 256, 513]);  permute_1413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_104: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_216);  clone_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_318: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_104, 3, 0, 257);  where_104 = None
        slice_scatter_319: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_318, 1, 0, 256);  slice_scatter_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1415: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_319, [0, 2, 1, 3]);  slice_scatter_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_217: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1415, memory_format = torch.contiguous_format);  permute_1415 = None
        view_1624: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_217, [96, 4, 256, 513]);  clone_217 = None
        add_206: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1622, view_1624);  view_1622 = view_1624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_397: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_206, 1, 0)
        slice_1708: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_397, 1, 1, 256)
        slice_1709: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1708, 2, 1, 256)
        clone_218: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_1709, memory_format = torch.contiguous_format)
        copy_177: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1709, full_default_60);  slice_1709 = None
        slice_scatter_320: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1708, copy_177, 2, 1, 256);  slice_1708 = copy_177 = None
        slice_scatter_321: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_397, slice_scatter_320, 1, 1, 256);  select_397 = slice_scatter_320 = None
        select_scatter_56: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_206, slice_scatter_321, 1, 0);  add_206 = slice_scatter_321 = None
        slice_scatter_322: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_218, 2, -255, 9223372036854775807);  clone_218 = None
        slice_scatter_323: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_322, 1, 0, 255);  slice_scatter_322 = None
        select_scatter_57: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_323, 1, 0);  slice_scatter_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1716: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_56, 1, 1, 9223372036854775807)
        slice_1717: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1716, 3, 0, 256)
        clone_219: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_1717, memory_format = torch.contiguous_format)
        copy_179: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1717, full_default_64);  slice_1717 = None
        slice_scatter_324: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1716, copy_179, 3, 0, 256);  slice_1716 = copy_179 = None
        slice_scatter_325: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_56, slice_scatter_324, 1, 1, 9223372036854775807);  select_scatter_56 = slice_scatter_324 = None
        slice_scatter_326: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_219, 3, 257, 9223372036854775807);  clone_219 = None
        slice_scatter_327: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_326, 2, -257, -1);  slice_scatter_326 = None
        add_207: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_57, slice_scatter_327);  select_scatter_57 = slice_scatter_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_402: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_325, 1, -1)
        slice_1722: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_402, 2, 256, 9223372036854775807)
        clone_220: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1722, memory_format = torch.contiguous_format)
        copy_181: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1722, full_default_67);  slice_1722 = None
        slice_scatter_328: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_402, copy_181, 2, 256, 9223372036854775807);  select_402 = copy_181 = None
        select_scatter_58: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_325, slice_scatter_328, 1, -1);  slice_scatter_325 = slice_scatter_328 = None
        slice_scatter_329: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_220, 2, 0, 257);  clone_220 = None
        slice_scatter_330: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_329, 1, 256, 9223372036854775807);  slice_scatter_329 = None
        select_scatter_59: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_330, 1, -1);  slice_scatter_330 = None
        add_208: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_207, select_scatter_59);  add_207 = select_scatter_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1727: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_58, 1, 0, -1);  select_scatter_58 = None
        slice_1728: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1727, 3, 256, 9223372036854775807);  slice_1727 = None
        clone_221: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1728, memory_format = torch.contiguous_format);  slice_1728 = None
        slice_scatter_331: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_221, 3, 0, 257);  clone_221 = None
        slice_scatter_332: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_331, 2, 0, 256);  slice_scatter_331 = None
        add_209: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_208, slice_scatter_332);  add_208 = slice_scatter_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1625: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_209, [96, 3, 513, 512]);  add_209 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_56: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1625, [0, 0, 0, -1]);  view_1625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1626: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_56, [96, 3, 512, 512, 1]);  constant_pad_nd_56 = None
        permute_1416: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1626, [0, 1, 2, 4, 3]);  view_1626 = None
        view_1627: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1416, [288, 512, 512]);  permute_1416 = None
        bmm_34: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1417, view_1627);  permute_1417 = None
        bmm_35: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1627, permute_1418);  view_1627 = permute_1418 = None
        view_1628: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_34, [96, 3, 64, 512, 1]);  bmm_34 = None
        permute_1419: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_1628, [0, 1, 4, 3, 2]);  view_1628 = None
        view_1629: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_35, [96, 3, 512, 64, 1]);  bmm_35 = None
        permute_1421: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_1419, [0, 1, 3, 4, 2]);  permute_1419 = None
        squeeze_10: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_1421, 4);  permute_1421 = None
        squeeze_11: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1629, 4);  view_1629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_222: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_10, memory_format = torch.contiguous_format);  squeeze_10 = None
        view_1630: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_222, [9437184]);  clone_222 = None
        index_add_7: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1630);  view_1630 = None
        view_1633: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_11, [-1]);  squeeze_11 = None
        index_add_8: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1633);  view_1633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_167: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_8, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_8 = None
        view_1654: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_167, [96, 1024, 64]);  as_strided_167 = None
        view_1655: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1654, [8, 12, 1024, 64]);  view_1654 = None
        permute_1433: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1655, [0, 2, 1, 3]);  view_1655 = None
        permute_1434: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1433, [1, 0, 2, 3]);  permute_1433 = None
        view_1656: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1434, [1024, 8, 768]);  permute_1434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_128: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_1656, 8.0);  view_1656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_55: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1599, [0, 1], True, dtype = torch.float32)
        view_1657: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_55, [768]);  sum_55 = None
        convert_element_type_684: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1657, torch.bfloat16);  view_1657 = None
        view_1658: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1599, [8192, 768]);  view_1599 = None
        permute_1435: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1658, [1, 0])
        mm_78: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1435, view_1035);  permute_1435 = None
        mm_79: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1658, permute_1437);  view_1658 = permute_1437 = None
        view_1659: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_79, [1024, 8, 768]);  mm_79 = None
        convert_element_type_689: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1659, torch.float32);  view_1659 = None
        convert_element_type_690: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        convert_element_type_691: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_684, torch.float32);  convert_element_type_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_168: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_7, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_7 = None
        view_1660: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_168, [96, 1024, 64]);  as_strided_168 = None
        view_1661: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1660, [8, 12, 1024, 64]);  view_1660 = None
        permute_1439: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1661, [0, 2, 1, 3]);  view_1661 = None
        permute_1440: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1439, [1, 0, 2, 3]);  permute_1439 = None
        view_1662: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1440, [1024, 8, 768]);  permute_1440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_56: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1662, [0, 1], True, dtype = torch.float32)
        view_1663: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_56, [768]);  sum_56 = None
        convert_element_type_692: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1663, torch.bfloat16);  view_1663 = None
        view_1668: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1662, [8192, 768]);  view_1662 = None
        permute_1446: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1668, [1, 0])
        mm_80: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1446, view_1035);  permute_1446 = None
        mm_81: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1668, permute_1448);  view_1668 = permute_1448 = None
        view_1673: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_81, [1024, 8, 768]);  mm_81 = None
        convert_element_type_697: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1673, torch.float32);  view_1673 = None
        add_210: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_689, convert_element_type_697);  convert_element_type_689 = convert_element_type_697 = None
        convert_element_type_698: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        convert_element_type_699: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_692, torch.float32);  convert_element_type_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_57: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_128, [0, 1], True, dtype = torch.float32)
        view_1674: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [768]);  sum_57 = None
        convert_element_type_700: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1674, torch.bfloat16);  view_1674 = None
        view_1675: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_128, [8192, 768]);  div_128 = None
        permute_1450: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1675, [1, 0])
        mm_82: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1450, view_1035);  permute_1450 = view_1035 = None
        mm_83: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1675, permute_1452);  view_1675 = permute_1452 = None
        view_1676: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_83, [1024, 8, 768]);  mm_83 = None
        convert_element_type_705: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1676, torch.float32);  view_1676 = None
        add_211: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_210, convert_element_type_705);  add_210 = convert_element_type_705 = None
        convert_element_type_706: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        convert_element_type_707: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_700, torch.float32);  convert_element_type_700 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1454: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_211, [1, 0, 2]);  add_211 = None
        add_212: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_245, permute_1454);  mul_245 = permute_1454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_253: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_212, primals_146);  primals_146 = None
        mul_254: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, 768)
        sum_58: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
        mul_255: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, mul_124);  mul_253 = None
        sum_59: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
        mul_256: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_124, sum_59);  sum_59 = None
        sub_115: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_254, sum_58);  mul_254 = sum_58 = None
        sub_116: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_115, mul_256);  sub_115 = mul_256 = None
        mul_257: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_129, sub_116);  div_129 = sub_116 = None
        mul_258: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_212, mul_124);  mul_124 = None
        sum_60: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
        sum_61: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None
        convert_element_type_708: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_257, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_709: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_26, torch.bfloat16);  gt_26 = None
        mul_259: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_709, 1.1111111111111112);  convert_element_type_709 = None
        mul_260: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_708, mul_259);  convert_element_type_708 = mul_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1677: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_260, [8192, 768]);  mul_260 = None
        mm_84: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_1677, permute_1455);  permute_1455 = None
        permute_1456: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1677, [1, 0])
        mm_85: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_1456, view_1033);  permute_1456 = view_1033 = None
        sum_62: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1677, [0], True, dtype = torch.float32);  view_1677 = None
        view_1678: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_62, [768]);  sum_62 = None
        convert_element_type_714: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1678, torch.bfloat16);  view_1678 = None
        view_1679: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [8, 1024, 3072]);  mm_84 = None
        convert_element_type_715: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_716: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_714, torch.float32);  convert_element_type_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_717: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1679, torch.float32);  view_1679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1032: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [8, 1024, 3072]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_380: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1032, torch.float32);  view_1032 = None
        mul_120: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, 0.7071067811865476)
        erf_8: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_120);  mul_120 = None
        add_131: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_262: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_131, 0.5);  add_131 = None
        mul_263: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, convert_element_type_380)
        mul_264: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
        exp_15: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
        mul_265: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
        mul_266: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_380, mul_265);  convert_element_type_380 = mul_265 = None
        add_214: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
        mul_267: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_717, add_214);  convert_element_type_717 = add_214 = None
        convert_element_type_719: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_267, torch.bfloat16);  mul_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1680: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_719, [8192, 3072]);  convert_element_type_719 = None
        mm_86: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1680, permute_1459);  permute_1459 = None
        permute_1460: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_1680, [1, 0])
        mm_87: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1460, view_1031);  permute_1460 = view_1031 = None
        sum_63: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1680, [0], True, dtype = torch.float32);  view_1680 = None
        view_1681: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [3072]);  sum_63 = None
        convert_element_type_724: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1681, torch.bfloat16);  view_1681 = None
        view_1682: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [8, 1024, 768]);  mm_86 = None
        convert_element_type_725: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1682, torch.float32);  view_1682 = None
        add_215: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_257, convert_element_type_725);  mul_257 = convert_element_type_725 = None
        convert_element_type_726: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_727: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_724, torch.float32);  convert_element_type_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_269: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_215, primals_140);  primals_140 = None
        mul_270: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, 768)
        sum_64: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
        mul_271: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, mul_117);  mul_269 = None
        sum_65: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
        mul_272: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_117, sum_65);  sum_65 = None
        sub_118: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_270, sum_64);  mul_270 = sum_64 = None
        sub_119: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, mul_272);  sub_118 = mul_272 = None
        mul_273: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_130, sub_119);  div_130 = sub_119 = None
        mul_274: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_215, mul_117);  mul_117 = None
        sum_66: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
        sum_67: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_215, [0, 1]);  add_215 = None
        convert_element_type_728: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_273, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_729: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_25, torch.bfloat16);  gt_25 = None
        mul_275: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_729, 1.1111111111111112);  convert_element_type_729 = None
        mul_276: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_728, mul_275);  convert_element_type_728 = mul_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_68: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1], True, dtype = torch.float32)
        view_1683: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_68, [768]);  sum_68 = None
        convert_element_type_730: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1683, torch.bfloat16);  view_1683 = None
        view_1684: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_276, [8192, 768]);  mul_276 = None
        permute_1463: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1684, [1, 0])
        mm_88: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1463, view_1029);  permute_1463 = view_1029 = None
        mm_89: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1684, permute_1465);  view_1684 = permute_1465 = None
        view_1685: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_89, [8, 1024, 768]);  mm_89 = None
        convert_element_type_735: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        convert_element_type_736: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_730, torch.float32);  convert_element_type_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1467: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_1685, [1, 0, 2]);  view_1685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1686: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1467, [1024, 8, 12, 64]);  permute_1467 = None
        permute_1468: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1686, [1, 0, 2, 3]);  view_1686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1469: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1468, [0, 2, 1, 3]);  permute_1468 = None
        clone_227: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1469, memory_format = torch.contiguous_format);  permute_1469 = None
        view_1687: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_227, [96, 4, 256, 64]);  clone_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1688: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1687, [96, 4, 256, 64, 1]);  view_1687 = None
        permute_1470: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1688, [0, 1, 2, 4, 3]);  view_1688 = None
        view_1689: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1470, [384, 256, 64]);  permute_1470 = None
        bmm_36: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1471, view_1689);  permute_1471 = None
        bmm_37: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_1689, permute_1472);  view_1689 = permute_1472 = None
        view_1690: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [96, 4, 768, 64, 1]);  bmm_36 = None
        view_1691: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [96, 4, 256, 768, 1]);  bmm_37 = None
        squeeze_12: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1690, 4);  view_1690 = None
        squeeze_13: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1691, 4);  view_1691 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_333: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_13, 3, 0, -1);  squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1692: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_333, [96, 4, 196864]);  slice_scatter_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_334: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_1692, 2, 0, -256);  view_1692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1693: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_334, [96, 4, 256, 770]);  slice_scatter_334 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_57: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1693, [0, -257]);  view_1693 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1694: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_12, [-1]);  squeeze_12 = None
        index_add_9: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_1694);  view_1694 = None
        as_strided_173: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_9, [96, 1536, 64], [98304, 64, 1], 0);  index_add_9 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_58: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_173, [0, 0, -256, -256]);  as_strided_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1696: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_58, [8, 12, 1024, 64]);  constant_pad_nd_58 = None
        permute_1477: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_1696, [0, 2, 1, 3]);  view_1696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1697: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_57, [8, 12, 1024, 513]);  constant_pad_nd_57 = None
        permute_1478: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1697, [0, 2, 1, 3]);  view_1697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1479: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1477, [1, 0, 2, 3]);  permute_1477 = None
        clone_229: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1479, memory_format = torch.contiguous_format);  permute_1479 = None
        view_1698: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_229, [1024, 8, 768]);  clone_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_741: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_24, torch.bfloat16);  gt_24 = None
        mul_277: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_741, 1.1111111111111112);  convert_element_type_741 = None
        mul_278: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1478, mul_277);  permute_1478 = mul_277 = None
        clone_230: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_278, memory_format = torch.contiguous_format);  mul_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_742: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_230, torch.float32);  clone_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_105: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_742);  convert_element_type_742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_366: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_884, torch.float32);  permute_884 = None
        clone_121: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_366, memory_format = torch.contiguous_format);  convert_element_type_366 = None
        sub_68: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_121, amax_8);  clone_121 = amax_8 = None
        exp_8: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_68);  sub_68 = None
        div_87: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        mul_279: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_105, div_87);  where_105 = None
        sum_69: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_279, [-1], True)
        neg_3: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_87);  div_87 = None
        fma_3: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_3, sum_69, mul_279);  neg_3 = sum_69 = mul_279 = None
        convert_element_type_743: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1480: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_743, [0, 2, 1, 3]);  convert_element_type_743 = None
        clone_231: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1480, memory_format = torch.contiguous_format);  permute_1480 = None
        view_1699: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_231, [96, 4, 256, 513]);  clone_231 = None
        view_1702: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1699, [8, 12, 1024, 513]);  view_1699 = None
        permute_1482: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1702, [0, 2, 1, 3]);  view_1702 = None
        clone_232: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1482, memory_format = torch.contiguous_format)
        copy_184: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_1482, clone_232);  permute_1482 = clone_232 = None
        permute_1483: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_184, [0, 2, 1, 3]);  copy_184 = None
        view_1704: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1483, [96, 4, 256, 513]);  permute_1483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1710: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1704, [8, 12, 1024, 513]);  view_1704 = None
        permute_1488: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1710, [0, 2, 1, 3]);  view_1710 = None
        slice_1732: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1488, 1, -256, 9223372036854775807)
        slice_1733: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1732, 3, -257, 9223372036854775807)
        clone_233: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1733, memory_format = torch.contiguous_format)
        copy_186: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1733, full_default_52);  slice_1733 = None
        slice_scatter_335: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1732, copy_186, 3, -257, 9223372036854775807);  slice_1732 = copy_186 = None
        slice_scatter_336: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1488, slice_scatter_335, 1, -256, 9223372036854775807);  permute_1488 = slice_scatter_335 = None
        permute_1490: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_336, [0, 2, 1, 3]);  slice_scatter_336 = None
        view_1712: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1490, [96, 4, 256, 513]);  permute_1490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_106: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_233);  clone_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_337: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_106, 3, -257, 9223372036854775807);  where_106 = None
        slice_scatter_338: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_337, 1, -256, 9223372036854775807);  slice_scatter_337 = None
        permute_1492: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_338, [0, 2, 1, 3]);  slice_scatter_338 = None
        clone_234: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1492, memory_format = torch.contiguous_format);  permute_1492 = None
        view_1714: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_234, [96, 4, 256, 513]);  clone_234 = None
        add_216: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1712, view_1714);  view_1712 = view_1714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1719: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_216, [8, 12, 1024, 513]);  add_216 = None
        permute_1496: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1719, [0, 2, 1, 3]);  view_1719 = None
        slice_1740: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1496, 1, 0, 256)
        slice_1741: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1740, 3, 0, 257)
        clone_235: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1741, memory_format = torch.contiguous_format)
        copy_188: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1741, full_default_52);  slice_1741 = None
        slice_scatter_339: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1740, copy_188, 3, 0, 257);  slice_1740 = copy_188 = None
        slice_scatter_340: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1496, slice_scatter_339, 1, 0, 256);  permute_1496 = slice_scatter_339 = None
        permute_1498: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_340, [0, 2, 1, 3]);  slice_scatter_340 = None
        view_1721: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1498, [96, 4, 256, 513]);  permute_1498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_107: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_235);  clone_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_341: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_107, 3, 0, 257);  where_107 = None
        slice_scatter_342: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_341, 1, 0, 256);  slice_scatter_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1500: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_342, [0, 2, 1, 3]);  slice_scatter_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_236: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1500, memory_format = torch.contiguous_format);  permute_1500 = None
        view_1723: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_236, [96, 4, 256, 513]);  clone_236 = None
        add_217: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1721, view_1723);  view_1721 = view_1723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_408: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_217, 1, 0)
        slice_1748: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_408, 1, 1, 256)
        slice_1749: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1748, 2, 1, 256)
        clone_237: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_1749, memory_format = torch.contiguous_format)
        copy_190: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1749, full_default_60);  slice_1749 = None
        slice_scatter_343: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1748, copy_190, 2, 1, 256);  slice_1748 = copy_190 = None
        slice_scatter_344: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_408, slice_scatter_343, 1, 1, 256);  select_408 = slice_scatter_343 = None
        select_scatter_60: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_217, slice_scatter_344, 1, 0);  add_217 = slice_scatter_344 = None
        slice_scatter_345: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_237, 2, -255, 9223372036854775807);  clone_237 = None
        slice_scatter_346: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_345, 1, 0, 255);  slice_scatter_345 = None
        select_scatter_61: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_346, 1, 0);  slice_scatter_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1756: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_60, 1, 1, 9223372036854775807)
        slice_1757: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1756, 3, 0, 256)
        clone_238: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_1757, memory_format = torch.contiguous_format)
        copy_192: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1757, full_default_64);  slice_1757 = None
        slice_scatter_347: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1756, copy_192, 3, 0, 256);  slice_1756 = copy_192 = None
        slice_scatter_348: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_60, slice_scatter_347, 1, 1, 9223372036854775807);  select_scatter_60 = slice_scatter_347 = None
        slice_scatter_349: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_238, 3, 257, 9223372036854775807);  clone_238 = None
        slice_scatter_350: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_349, 2, -257, -1);  slice_scatter_349 = None
        add_218: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_61, slice_scatter_350);  select_scatter_61 = slice_scatter_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_413: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_348, 1, -1)
        slice_1762: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_413, 2, 256, 9223372036854775807)
        clone_239: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1762, memory_format = torch.contiguous_format)
        copy_194: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1762, full_default_67);  slice_1762 = None
        slice_scatter_351: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_413, copy_194, 2, 256, 9223372036854775807);  select_413 = copy_194 = None
        select_scatter_62: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_348, slice_scatter_351, 1, -1);  slice_scatter_348 = slice_scatter_351 = None
        slice_scatter_352: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_239, 2, 0, 257);  clone_239 = None
        slice_scatter_353: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_352, 1, 256, 9223372036854775807);  slice_scatter_352 = None
        select_scatter_63: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_353, 1, -1);  slice_scatter_353 = None
        add_219: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_218, select_scatter_63);  add_218 = select_scatter_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1767: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_62, 1, 0, -1);  select_scatter_62 = None
        slice_1768: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1767, 3, 256, 9223372036854775807);  slice_1767 = None
        clone_240: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1768, memory_format = torch.contiguous_format);  slice_1768 = None
        slice_scatter_354: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_240, 3, 0, 257);  clone_240 = None
        slice_scatter_355: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_354, 2, 0, 256);  slice_scatter_354 = None
        add_220: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_219, slice_scatter_355);  add_219 = slice_scatter_355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1724: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_220, [96, 3, 513, 512]);  add_220 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_59: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1724, [0, 0, 0, -1]);  view_1724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1725: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_59, [96, 3, 512, 512, 1]);  constant_pad_nd_59 = None
        permute_1501: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1725, [0, 1, 2, 4, 3]);  view_1725 = None
        view_1726: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1501, [288, 512, 512]);  permute_1501 = None
        bmm_38: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1502, view_1726);  permute_1502 = None
        bmm_39: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1726, permute_1503);  view_1726 = permute_1503 = None
        view_1727: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [96, 3, 64, 512, 1]);  bmm_38 = None
        permute_1504: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_1727, [0, 1, 4, 3, 2]);  view_1727 = None
        view_1728: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_39, [96, 3, 512, 64, 1]);  bmm_39 = None
        permute_1506: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_1504, [0, 1, 3, 4, 2]);  permute_1504 = None
        squeeze_14: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_1506, 4);  permute_1506 = None
        squeeze_15: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1728, 4);  view_1728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_241: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_14, memory_format = torch.contiguous_format);  squeeze_14 = None
        view_1729: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_241, [9437184]);  clone_241 = None
        index_add_10: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1729);  view_1729 = None
        view_1732: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_15, [-1]);  squeeze_15 = None
        index_add_11: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1732);  view_1732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_188: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_11, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_11 = None
        view_1753: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_188, [96, 1024, 64]);  as_strided_188 = None
        view_1754: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1753, [8, 12, 1024, 64]);  view_1753 = None
        permute_1518: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1754, [0, 2, 1, 3]);  view_1754 = None
        permute_1519: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1518, [1, 0, 2, 3]);  permute_1518 = None
        view_1755: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1519, [1024, 8, 768]);  permute_1519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_131: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_1755, 8.0);  view_1755 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_70: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1698, [0, 1], True, dtype = torch.float32)
        view_1756: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [768]);  sum_70 = None
        convert_element_type_748: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1756, torch.bfloat16);  view_1756 = None
        view_1757: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1698, [8192, 768]);  view_1698 = None
        permute_1520: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1757, [1, 0])
        mm_90: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1520, view_920);  permute_1520 = None
        mm_91: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1757, permute_1522);  view_1757 = permute_1522 = None
        view_1758: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_91, [1024, 8, 768]);  mm_91 = None
        convert_element_type_753: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1758, torch.float32);  view_1758 = None
        convert_element_type_754: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        convert_element_type_755: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_748, torch.float32);  convert_element_type_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_189: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_10, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_10 = None
        view_1759: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_189, [96, 1024, 64]);  as_strided_189 = None
        view_1760: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1759, [8, 12, 1024, 64]);  view_1759 = None
        permute_1524: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1760, [0, 2, 1, 3]);  view_1760 = None
        permute_1525: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1524, [1, 0, 2, 3]);  permute_1524 = None
        view_1761: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1525, [1024, 8, 768]);  permute_1525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_71: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1761, [0, 1], True, dtype = torch.float32)
        view_1762: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_71, [768]);  sum_71 = None
        convert_element_type_756: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1762, torch.bfloat16);  view_1762 = None
        view_1767: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1761, [8192, 768]);  view_1761 = None
        permute_1531: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1767, [1, 0])
        mm_92: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1531, view_920);  permute_1531 = None
        mm_93: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1767, permute_1533);  view_1767 = permute_1533 = None
        view_1772: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_93, [1024, 8, 768]);  mm_93 = None
        convert_element_type_761: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1772, torch.float32);  view_1772 = None
        add_221: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_753, convert_element_type_761);  convert_element_type_753 = convert_element_type_761 = None
        convert_element_type_762: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        convert_element_type_763: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_756, torch.float32);  convert_element_type_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_72: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_131, [0, 1], True, dtype = torch.float32)
        view_1773: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_72, [768]);  sum_72 = None
        convert_element_type_764: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1773, torch.bfloat16);  view_1773 = None
        view_1774: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_131, [8192, 768]);  div_131 = None
        permute_1535: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1774, [1, 0])
        mm_94: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1535, view_920);  permute_1535 = view_920 = None
        mm_95: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1774, permute_1537);  view_1774 = permute_1537 = None
        view_1775: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_95, [1024, 8, 768]);  mm_95 = None
        convert_element_type_769: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1775, torch.float32);  view_1775 = None
        add_222: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_221, convert_element_type_769);  add_221 = convert_element_type_769 = None
        convert_element_type_770: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        convert_element_type_771: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_764, torch.float32);  convert_element_type_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1539: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_222, [1, 0, 2]);  add_222 = None
        add_223: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_273, permute_1539);  mul_273 = permute_1539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_281: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_223, primals_130);  primals_130 = None
        mul_282: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, 768)
        sum_73: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_281, [2], True)
        mul_283: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, mul_110);  mul_281 = None
        sum_74: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True);  mul_283 = None
        mul_284: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_110, sum_74);  sum_74 = None
        sub_121: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_282, sum_73);  mul_282 = sum_73 = None
        sub_122: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_121, mul_284);  sub_121 = mul_284 = None
        mul_285: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_132, sub_122);  div_132 = sub_122 = None
        mul_286: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_223, mul_110);  mul_110 = None
        sum_75: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 1]);  mul_286 = None
        sum_76: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_223, [0, 1]);  add_223 = None
        convert_element_type_772: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_285, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_773: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_23, torch.bfloat16);  gt_23 = None
        mul_287: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_773, 1.1111111111111112);  convert_element_type_773 = None
        mul_288: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_772, mul_287);  convert_element_type_772 = mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1776: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_288, [8192, 768]);  mul_288 = None
        mm_96: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_1776, permute_1540);  permute_1540 = None
        permute_1541: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1776, [1, 0])
        mm_97: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_1541, view_918);  permute_1541 = view_918 = None
        sum_77: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1776, [0], True, dtype = torch.float32);  view_1776 = None
        view_1777: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [768]);  sum_77 = None
        convert_element_type_778: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1777, torch.bfloat16);  view_1777 = None
        view_1778: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [8, 1024, 3072]);  mm_96 = None
        convert_element_type_779: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_780: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_778, torch.float32);  convert_element_type_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_781: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1778, torch.float32);  view_1778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_917: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [8, 1024, 3072]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_337: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_917, torch.float32);  view_917 = None
        mul_106: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_337, 0.7071067811865476)
        erf_7: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_106);  mul_106 = None
        add_116: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_290: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_116, 0.5);  add_116 = None
        mul_291: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_337, convert_element_type_337)
        mul_292: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_291, -0.5);  mul_291 = None
        exp_16: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_292);  mul_292 = None
        mul_293: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
        mul_294: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_337, mul_293);  convert_element_type_337 = mul_293 = None
        add_225: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_290, mul_294);  mul_290 = mul_294 = None
        mul_295: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_781, add_225);  convert_element_type_781 = add_225 = None
        convert_element_type_783: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_295, torch.bfloat16);  mul_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1779: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_783, [8192, 3072]);  convert_element_type_783 = None
        mm_98: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1779, permute_1544);  permute_1544 = None
        permute_1545: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_1779, [1, 0])
        mm_99: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1545, view_916);  permute_1545 = view_916 = None
        sum_78: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1779, [0], True, dtype = torch.float32);  view_1779 = None
        view_1780: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_78, [3072]);  sum_78 = None
        convert_element_type_788: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1780, torch.bfloat16);  view_1780 = None
        view_1781: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [8, 1024, 768]);  mm_98 = None
        convert_element_type_789: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1781, torch.float32);  view_1781 = None
        add_226: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_285, convert_element_type_789);  mul_285 = convert_element_type_789 = None
        convert_element_type_790: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_791: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_788, torch.float32);  convert_element_type_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_297: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_226, primals_124);  primals_124 = None
        mul_298: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, 768)
        sum_79: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
        mul_299: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, mul_103);  mul_297 = None
        sum_80: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
        mul_300: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, sum_80);  sum_80 = None
        sub_124: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_298, sum_79);  mul_298 = sum_79 = None
        sub_125: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_124, mul_300);  sub_124 = mul_300 = None
        mul_301: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_133, sub_125);  div_133 = sub_125 = None
        mul_302: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_226, mul_103);  mul_103 = None
        sum_81: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
        sum_82: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_226, [0, 1]);  add_226 = None
        convert_element_type_792: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_301, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_793: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_22, torch.bfloat16);  gt_22 = None
        mul_303: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_793, 1.1111111111111112);  convert_element_type_793 = None
        mul_304: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_792, mul_303);  convert_element_type_792 = mul_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_83: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_304, [0, 1], True, dtype = torch.float32)
        view_1782: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_83, [768]);  sum_83 = None
        convert_element_type_794: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1782, torch.bfloat16);  view_1782 = None
        view_1783: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_304, [8192, 768]);  mul_304 = None
        permute_1548: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1783, [1, 0])
        mm_100: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1548, view_914);  permute_1548 = view_914 = None
        mm_101: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1783, permute_1550);  view_1783 = permute_1550 = None
        view_1784: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_101, [8, 1024, 768]);  mm_101 = None
        convert_element_type_799: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        convert_element_type_800: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_794, torch.float32);  convert_element_type_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1552: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_1784, [1, 0, 2]);  view_1784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1785: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1552, [1024, 8, 12, 64]);  permute_1552 = None
        permute_1553: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1785, [1, 0, 2, 3]);  view_1785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1554: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1553, [0, 2, 1, 3]);  permute_1553 = None
        clone_246: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1554, memory_format = torch.contiguous_format);  permute_1554 = None
        view_1786: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_246, [96, 4, 256, 64]);  clone_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1787: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1786, [96, 4, 256, 64, 1]);  view_1786 = None
        permute_1555: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1787, [0, 1, 2, 4, 3]);  view_1787 = None
        view_1788: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1555, [384, 256, 64]);  permute_1555 = None
        bmm_40: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1556, view_1788);  permute_1556 = None
        bmm_41: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_1788, permute_1557);  view_1788 = permute_1557 = None
        view_1789: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_40, [96, 4, 768, 64, 1]);  bmm_40 = None
        view_1790: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_41, [96, 4, 256, 768, 1]);  bmm_41 = None
        squeeze_16: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1789, 4);  view_1789 = None
        squeeze_17: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1790, 4);  view_1790 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_356: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_17, 3, 0, -1);  squeeze_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1791: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_356, [96, 4, 196864]);  slice_scatter_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_357: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_1791, 2, 0, -256);  view_1791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1792: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_357, [96, 4, 256, 770]);  slice_scatter_357 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_60: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1792, [0, -257]);  view_1792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1793: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_16, [-1]);  squeeze_16 = None
        index_add_12: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_1793);  view_1793 = None
        as_strided_194: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_12, [96, 1536, 64], [98304, 64, 1], 0);  index_add_12 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_61: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_194, [0, 0, -256, -256]);  as_strided_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1795: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_61, [8, 12, 1024, 64]);  constant_pad_nd_61 = None
        permute_1562: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_1795, [0, 2, 1, 3]);  view_1795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1796: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_60, [8, 12, 1024, 513]);  constant_pad_nd_60 = None
        permute_1563: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1796, [0, 2, 1, 3]);  view_1796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1564: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1562, [1, 0, 2, 3]);  permute_1562 = None
        clone_248: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1564, memory_format = torch.contiguous_format);  permute_1564 = None
        view_1797: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_248, [1024, 8, 768]);  clone_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_805: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_21, torch.bfloat16);  gt_21 = None
        mul_305: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_805, 1.1111111111111112);  convert_element_type_805 = None
        mul_306: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1563, mul_305);  permute_1563 = mul_305 = None
        clone_249: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_306, memory_format = torch.contiguous_format);  mul_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_806: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_249, torch.float32);  clone_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_108: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_806);  convert_element_type_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_323: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_784, torch.float32);  permute_784 = None
        clone_107: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_323, memory_format = torch.contiguous_format);  convert_element_type_323 = None
        sub_60: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_107, amax_7);  clone_107 = amax_7 = None
        exp_7: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
        div_77: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        mul_307: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_108, div_77);  where_108 = None
        sum_84: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_307, [-1], True)
        neg_4: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_77);  div_77 = None
        fma_4: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_4, sum_84, mul_307);  neg_4 = sum_84 = mul_307 = None
        convert_element_type_807: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1565: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_807, [0, 2, 1, 3]);  convert_element_type_807 = None
        clone_250: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1565, memory_format = torch.contiguous_format);  permute_1565 = None
        view_1798: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_250, [96, 4, 256, 513]);  clone_250 = None
        view_1801: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1798, [8, 12, 1024, 513]);  view_1798 = None
        permute_1567: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1801, [0, 2, 1, 3]);  view_1801 = None
        clone_251: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1567, memory_format = torch.contiguous_format)
        copy_197: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_1567, clone_251);  permute_1567 = clone_251 = None
        permute_1568: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_197, [0, 2, 1, 3]);  copy_197 = None
        view_1803: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1568, [96, 4, 256, 513]);  permute_1568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1809: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1803, [8, 12, 1024, 513]);  view_1803 = None
        permute_1573: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1809, [0, 2, 1, 3]);  view_1809 = None
        slice_1772: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1573, 1, -256, 9223372036854775807)
        slice_1773: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1772, 3, -257, 9223372036854775807)
        clone_252: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1773, memory_format = torch.contiguous_format)
        copy_199: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1773, full_default_52);  slice_1773 = None
        slice_scatter_358: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1772, copy_199, 3, -257, 9223372036854775807);  slice_1772 = copy_199 = None
        slice_scatter_359: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1573, slice_scatter_358, 1, -256, 9223372036854775807);  permute_1573 = slice_scatter_358 = None
        permute_1575: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_359, [0, 2, 1, 3]);  slice_scatter_359 = None
        view_1811: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1575, [96, 4, 256, 513]);  permute_1575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_109: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_252);  clone_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_360: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_109, 3, -257, 9223372036854775807);  where_109 = None
        slice_scatter_361: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_360, 1, -256, 9223372036854775807);  slice_scatter_360 = None
        permute_1577: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_361, [0, 2, 1, 3]);  slice_scatter_361 = None
        clone_253: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1577, memory_format = torch.contiguous_format);  permute_1577 = None
        view_1813: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_253, [96, 4, 256, 513]);  clone_253 = None
        add_227: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1811, view_1813);  view_1811 = view_1813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1818: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_227, [8, 12, 1024, 513]);  add_227 = None
        permute_1581: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1818, [0, 2, 1, 3]);  view_1818 = None
        slice_1780: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1581, 1, 0, 256)
        slice_1781: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1780, 3, 0, 257)
        clone_254: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1781, memory_format = torch.contiguous_format)
        copy_201: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1781, full_default_52);  slice_1781 = None
        slice_scatter_362: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1780, copy_201, 3, 0, 257);  slice_1780 = copy_201 = None
        slice_scatter_363: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1581, slice_scatter_362, 1, 0, 256);  permute_1581 = slice_scatter_362 = None
        permute_1583: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_363, [0, 2, 1, 3]);  slice_scatter_363 = None
        view_1820: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1583, [96, 4, 256, 513]);  permute_1583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_110: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_254);  clone_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_364: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_110, 3, 0, 257);  where_110 = None
        slice_scatter_365: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_364, 1, 0, 256);  slice_scatter_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1585: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_365, [0, 2, 1, 3]);  slice_scatter_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_255: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1585, memory_format = torch.contiguous_format);  permute_1585 = None
        view_1822: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_255, [96, 4, 256, 513]);  clone_255 = None
        add_228: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1820, view_1822);  view_1820 = view_1822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_419: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_228, 1, 0)
        slice_1788: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_419, 1, 1, 256)
        slice_1789: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1788, 2, 1, 256)
        clone_256: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_1789, memory_format = torch.contiguous_format)
        copy_203: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1789, full_default_60);  slice_1789 = None
        slice_scatter_366: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1788, copy_203, 2, 1, 256);  slice_1788 = copy_203 = None
        slice_scatter_367: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_419, slice_scatter_366, 1, 1, 256);  select_419 = slice_scatter_366 = None
        select_scatter_64: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_228, slice_scatter_367, 1, 0);  add_228 = slice_scatter_367 = None
        slice_scatter_368: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_256, 2, -255, 9223372036854775807);  clone_256 = None
        slice_scatter_369: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_368, 1, 0, 255);  slice_scatter_368 = None
        select_scatter_65: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_369, 1, 0);  slice_scatter_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1796: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_64, 1, 1, 9223372036854775807)
        slice_1797: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1796, 3, 0, 256)
        clone_257: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_1797, memory_format = torch.contiguous_format)
        copy_205: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1797, full_default_64);  slice_1797 = None
        slice_scatter_370: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1796, copy_205, 3, 0, 256);  slice_1796 = copy_205 = None
        slice_scatter_371: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_64, slice_scatter_370, 1, 1, 9223372036854775807);  select_scatter_64 = slice_scatter_370 = None
        slice_scatter_372: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_257, 3, 257, 9223372036854775807);  clone_257 = None
        slice_scatter_373: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_372, 2, -257, -1);  slice_scatter_372 = None
        add_229: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_65, slice_scatter_373);  select_scatter_65 = slice_scatter_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_424: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_371, 1, -1)
        slice_1802: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_424, 2, 256, 9223372036854775807)
        clone_258: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1802, memory_format = torch.contiguous_format)
        copy_207: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1802, full_default_67);  slice_1802 = None
        slice_scatter_374: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_424, copy_207, 2, 256, 9223372036854775807);  select_424 = copy_207 = None
        select_scatter_66: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_371, slice_scatter_374, 1, -1);  slice_scatter_371 = slice_scatter_374 = None
        slice_scatter_375: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_258, 2, 0, 257);  clone_258 = None
        slice_scatter_376: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_375, 1, 256, 9223372036854775807);  slice_scatter_375 = None
        select_scatter_67: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_376, 1, -1);  slice_scatter_376 = None
        add_230: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_229, select_scatter_67);  add_229 = select_scatter_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1807: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_66, 1, 0, -1);  select_scatter_66 = None
        slice_1808: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1807, 3, 256, 9223372036854775807);  slice_1807 = None
        clone_259: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1808, memory_format = torch.contiguous_format);  slice_1808 = None
        slice_scatter_377: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_259, 3, 0, 257);  clone_259 = None
        slice_scatter_378: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_377, 2, 0, 256);  slice_scatter_377 = None
        add_231: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_230, slice_scatter_378);  add_230 = slice_scatter_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1823: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_231, [96, 3, 513, 512]);  add_231 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_62: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1823, [0, 0, 0, -1]);  view_1823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1824: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_62, [96, 3, 512, 512, 1]);  constant_pad_nd_62 = None
        permute_1586: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1824, [0, 1, 2, 4, 3]);  view_1824 = None
        view_1825: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1586, [288, 512, 512]);  permute_1586 = None
        bmm_42: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1587, view_1825);  permute_1587 = None
        bmm_43: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1825, permute_1588);  view_1825 = permute_1588 = None
        view_1826: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_42, [96, 3, 64, 512, 1]);  bmm_42 = None
        permute_1589: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_1826, [0, 1, 4, 3, 2]);  view_1826 = None
        view_1827: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_43, [96, 3, 512, 64, 1]);  bmm_43 = None
        permute_1591: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_1589, [0, 1, 3, 4, 2]);  permute_1589 = None
        squeeze_18: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_1591, 4);  permute_1591 = None
        squeeze_19: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1827, 4);  view_1827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_260: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_18, memory_format = torch.contiguous_format);  squeeze_18 = None
        view_1828: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_260, [9437184]);  clone_260 = None
        index_add_13: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1828);  view_1828 = None
        view_1831: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_19, [-1]);  squeeze_19 = None
        index_add_14: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1831);  view_1831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_209: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_14, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_14 = None
        view_1852: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_209, [96, 1024, 64]);  as_strided_209 = None
        view_1853: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1852, [8, 12, 1024, 64]);  view_1852 = None
        permute_1603: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1853, [0, 2, 1, 3]);  view_1853 = None
        permute_1604: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1603, [1, 0, 2, 3]);  permute_1603 = None
        view_1854: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1604, [1024, 8, 768]);  permute_1604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_134: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_1854, 8.0);  view_1854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_85: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1797, [0, 1], True, dtype = torch.float32)
        view_1855: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_85, [768]);  sum_85 = None
        convert_element_type_812: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1855, torch.bfloat16);  view_1855 = None
        view_1856: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1797, [8192, 768]);  view_1797 = None
        permute_1605: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1856, [1, 0])
        mm_102: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1605, view_805);  permute_1605 = None
        mm_103: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1856, permute_1607);  view_1856 = permute_1607 = None
        view_1857: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_103, [1024, 8, 768]);  mm_103 = None
        convert_element_type_817: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1857, torch.float32);  view_1857 = None
        convert_element_type_818: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        convert_element_type_819: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_812, torch.float32);  convert_element_type_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_210: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_13, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_13 = None
        view_1858: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_210, [96, 1024, 64]);  as_strided_210 = None
        view_1859: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1858, [8, 12, 1024, 64]);  view_1858 = None
        permute_1609: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1859, [0, 2, 1, 3]);  view_1859 = None
        permute_1610: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1609, [1, 0, 2, 3]);  permute_1609 = None
        view_1860: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1610, [1024, 8, 768]);  permute_1610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_86: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1860, [0, 1], True, dtype = torch.float32)
        view_1861: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_86, [768]);  sum_86 = None
        convert_element_type_820: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1861, torch.bfloat16);  view_1861 = None
        view_1866: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1860, [8192, 768]);  view_1860 = None
        permute_1616: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1866, [1, 0])
        mm_104: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1616, view_805);  permute_1616 = None
        mm_105: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1866, permute_1618);  view_1866 = permute_1618 = None
        view_1871: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_105, [1024, 8, 768]);  mm_105 = None
        convert_element_type_825: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1871, torch.float32);  view_1871 = None
        add_232: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_817, convert_element_type_825);  convert_element_type_817 = convert_element_type_825 = None
        convert_element_type_826: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        convert_element_type_827: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_820, torch.float32);  convert_element_type_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_87: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_134, [0, 1], True, dtype = torch.float32)
        view_1872: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [768]);  sum_87 = None
        convert_element_type_828: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1872, torch.bfloat16);  view_1872 = None
        view_1873: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_134, [8192, 768]);  div_134 = None
        permute_1620: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1873, [1, 0])
        mm_106: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1620, view_805);  permute_1620 = view_805 = None
        mm_107: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1873, permute_1622);  view_1873 = permute_1622 = None
        view_1874: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_107, [1024, 8, 768]);  mm_107 = None
        convert_element_type_833: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1874, torch.float32);  view_1874 = None
        add_233: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_232, convert_element_type_833);  add_232 = convert_element_type_833 = None
        convert_element_type_834: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        convert_element_type_835: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_828, torch.float32);  convert_element_type_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1624: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_233, [1, 0, 2]);  add_233 = None
        add_234: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_301, permute_1624);  mul_301 = permute_1624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_309: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_234, primals_114);  primals_114 = None
        mul_310: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, 768)
        sum_88: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
        mul_311: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_309, mul_96);  mul_309 = None
        sum_89: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
        mul_312: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, sum_89);  sum_89 = None
        sub_127: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_310, sum_88);  mul_310 = sum_88 = None
        sub_128: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_127, mul_312);  sub_127 = mul_312 = None
        mul_313: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_135, sub_128);  div_135 = sub_128 = None
        mul_314: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_234, mul_96);  mul_96 = None
        sum_90: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
        sum_91: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_234, [0, 1]);  add_234 = None
        convert_element_type_836: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_313, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_837: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_20, torch.bfloat16);  gt_20 = None
        mul_315: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_837, 1.1111111111111112);  convert_element_type_837 = None
        mul_316: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_836, mul_315);  convert_element_type_836 = mul_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1875: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_316, [8192, 768]);  mul_316 = None
        mm_108: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_1875, permute_1625);  permute_1625 = None
        permute_1626: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1875, [1, 0])
        mm_109: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_1626, view_803);  permute_1626 = view_803 = None
        sum_92: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1875, [0], True, dtype = torch.float32);  view_1875 = None
        view_1876: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_92, [768]);  sum_92 = None
        convert_element_type_842: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1876, torch.bfloat16);  view_1876 = None
        view_1877: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [8, 1024, 3072]);  mm_108 = None
        convert_element_type_843: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_844: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_842, torch.float32);  convert_element_type_842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_845: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1877, torch.float32);  view_1877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_802: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [8, 1024, 3072]);  addmm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_294: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_802, torch.float32);  view_802 = None
        mul_92: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_294, 0.7071067811865476)
        erf_6: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
        add_101: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_318: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_101, 0.5);  add_101 = None
        mul_319: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_294, convert_element_type_294)
        mul_320: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_319, -0.5);  mul_319 = None
        exp_17: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_320);  mul_320 = None
        mul_321: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
        mul_322: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_294, mul_321);  convert_element_type_294 = mul_321 = None
        add_236: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_318, mul_322);  mul_318 = mul_322 = None
        mul_323: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_845, add_236);  convert_element_type_845 = add_236 = None
        convert_element_type_847: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_323, torch.bfloat16);  mul_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1878: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_847, [8192, 3072]);  convert_element_type_847 = None
        mm_110: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1878, permute_1629);  permute_1629 = None
        permute_1630: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_1878, [1, 0])
        mm_111: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1630, view_801);  permute_1630 = view_801 = None
        sum_93: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1878, [0], True, dtype = torch.float32);  view_1878 = None
        view_1879: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [3072]);  sum_93 = None
        convert_element_type_852: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1879, torch.bfloat16);  view_1879 = None
        view_1880: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [8, 1024, 768]);  mm_110 = None
        convert_element_type_853: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1880, torch.float32);  view_1880 = None
        add_237: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_313, convert_element_type_853);  mul_313 = convert_element_type_853 = None
        convert_element_type_854: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_855: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_852, torch.float32);  convert_element_type_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_325: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_237, primals_108);  primals_108 = None
        mul_326: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, 768)
        sum_94: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
        mul_327: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, mul_89);  mul_325 = None
        sum_95: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
        mul_328: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, sum_95);  sum_95 = None
        sub_130: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_326, sum_94);  mul_326 = sum_94 = None
        sub_131: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_130, mul_328);  sub_130 = mul_328 = None
        mul_329: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_136, sub_131);  div_136 = sub_131 = None
        mul_330: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_237, mul_89);  mul_89 = None
        sum_96: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
        sum_97: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_237, [0, 1]);  add_237 = None
        convert_element_type_856: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_329, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_857: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_19, torch.bfloat16);  gt_19 = None
        mul_331: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_857, 1.1111111111111112);  convert_element_type_857 = None
        mul_332: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_856, mul_331);  convert_element_type_856 = mul_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_98: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1], True, dtype = torch.float32)
        view_1881: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_98, [768]);  sum_98 = None
        convert_element_type_858: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1881, torch.bfloat16);  view_1881 = None
        view_1882: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_332, [8192, 768]);  mul_332 = None
        permute_1633: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1882, [1, 0])
        mm_112: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1633, view_799);  permute_1633 = view_799 = None
        mm_113: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1882, permute_1635);  view_1882 = permute_1635 = None
        view_1883: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_113, [8, 1024, 768]);  mm_113 = None
        convert_element_type_863: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None
        convert_element_type_864: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_858, torch.float32);  convert_element_type_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1637: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_1883, [1, 0, 2]);  view_1883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1884: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1637, [1024, 8, 12, 64]);  permute_1637 = None
        permute_1638: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1884, [1, 0, 2, 3]);  view_1884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1639: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1638, [0, 2, 1, 3]);  permute_1638 = None
        clone_265: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1639, memory_format = torch.contiguous_format);  permute_1639 = None
        view_1885: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_265, [96, 4, 256, 64]);  clone_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1886: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1885, [96, 4, 256, 64, 1]);  view_1885 = None
        permute_1640: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1886, [0, 1, 2, 4, 3]);  view_1886 = None
        view_1887: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1640, [384, 256, 64]);  permute_1640 = None
        bmm_44: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1641, view_1887);  permute_1641 = None
        bmm_45: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_1887, permute_1642);  view_1887 = permute_1642 = None
        view_1888: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [96, 4, 768, 64, 1]);  bmm_44 = None
        view_1889: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [96, 4, 256, 768, 1]);  bmm_45 = None
        squeeze_20: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1888, 4);  view_1888 = None
        squeeze_21: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1889, 4);  view_1889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_379: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_21, 3, 0, -1);  squeeze_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1890: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_379, [96, 4, 196864]);  slice_scatter_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_380: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_1890, 2, 0, -256);  view_1890 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1891: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_380, [96, 4, 256, 770]);  slice_scatter_380 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_63: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1891, [0, -257]);  view_1891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1892: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_20, [-1]);  squeeze_20 = None
        index_add_15: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_1892);  view_1892 = None
        as_strided_215: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_15, [96, 1536, 64], [98304, 64, 1], 0);  index_add_15 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_64: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_215, [0, 0, -256, -256]);  as_strided_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1894: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_64, [8, 12, 1024, 64]);  constant_pad_nd_64 = None
        permute_1647: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_1894, [0, 2, 1, 3]);  view_1894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1895: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_63, [8, 12, 1024, 513]);  constant_pad_nd_63 = None
        permute_1648: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1895, [0, 2, 1, 3]);  view_1895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1649: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1647, [1, 0, 2, 3]);  permute_1647 = None
        clone_267: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1649, memory_format = torch.contiguous_format);  permute_1649 = None
        view_1896: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_267, [1024, 8, 768]);  clone_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_869: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_18, torch.bfloat16);  gt_18 = None
        mul_333: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_869, 1.1111111111111112);  convert_element_type_869 = None
        mul_334: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1648, mul_333);  permute_1648 = mul_333 = None
        clone_268: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_334, memory_format = torch.contiguous_format);  mul_334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_870: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_268, torch.float32);  clone_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_111: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_870);  convert_element_type_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_280: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_684, torch.float32);  permute_684 = None
        clone_93: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_280, memory_format = torch.contiguous_format);  convert_element_type_280 = None
        sub_52: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_93, amax_6);  clone_93 = amax_6 = None
        exp_6: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
        div_67: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        mul_335: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_111, div_67);  where_111 = None
        sum_99: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_335, [-1], True)
        neg_5: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_67);  div_67 = None
        fma_5: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_5, sum_99, mul_335);  neg_5 = sum_99 = mul_335 = None
        convert_element_type_871: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1650: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_871, [0, 2, 1, 3]);  convert_element_type_871 = None
        clone_269: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1650, memory_format = torch.contiguous_format);  permute_1650 = None
        view_1897: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_269, [96, 4, 256, 513]);  clone_269 = None
        view_1900: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1897, [8, 12, 1024, 513]);  view_1897 = None
        permute_1652: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1900, [0, 2, 1, 3]);  view_1900 = None
        clone_270: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1652, memory_format = torch.contiguous_format)
        copy_210: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_1652, clone_270);  permute_1652 = clone_270 = None
        permute_1653: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_210, [0, 2, 1, 3]);  copy_210 = None
        view_1902: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1653, [96, 4, 256, 513]);  permute_1653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_1908: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1902, [8, 12, 1024, 513]);  view_1902 = None
        permute_1658: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1908, [0, 2, 1, 3]);  view_1908 = None
        slice_1812: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1658, 1, -256, 9223372036854775807)
        slice_1813: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1812, 3, -257, 9223372036854775807)
        clone_271: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1813, memory_format = torch.contiguous_format)
        copy_212: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1813, full_default_52);  slice_1813 = None
        slice_scatter_381: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1812, copy_212, 3, -257, 9223372036854775807);  slice_1812 = copy_212 = None
        slice_scatter_382: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1658, slice_scatter_381, 1, -256, 9223372036854775807);  permute_1658 = slice_scatter_381 = None
        permute_1660: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_382, [0, 2, 1, 3]);  slice_scatter_382 = None
        view_1910: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1660, [96, 4, 256, 513]);  permute_1660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_112: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_271);  clone_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_383: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_112, 3, -257, 9223372036854775807);  where_112 = None
        slice_scatter_384: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_383, 1, -256, 9223372036854775807);  slice_scatter_383 = None
        permute_1662: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_384, [0, 2, 1, 3]);  slice_scatter_384 = None
        clone_272: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1662, memory_format = torch.contiguous_format);  permute_1662 = None
        view_1912: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_272, [96, 4, 256, 513]);  clone_272 = None
        add_238: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1910, view_1912);  view_1910 = view_1912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_1917: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_238, [8, 12, 1024, 513]);  add_238 = None
        permute_1666: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1917, [0, 2, 1, 3]);  view_1917 = None
        slice_1820: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1666, 1, 0, 256)
        slice_1821: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1820, 3, 0, 257)
        clone_273: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1821, memory_format = torch.contiguous_format)
        copy_214: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1821, full_default_52);  slice_1821 = None
        slice_scatter_385: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1820, copy_214, 3, 0, 257);  slice_1820 = copy_214 = None
        slice_scatter_386: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1666, slice_scatter_385, 1, 0, 256);  permute_1666 = slice_scatter_385 = None
        permute_1668: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_386, [0, 2, 1, 3]);  slice_scatter_386 = None
        view_1919: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1668, [96, 4, 256, 513]);  permute_1668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_113: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_273);  clone_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_387: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_113, 3, 0, 257);  where_113 = None
        slice_scatter_388: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_387, 1, 0, 256);  slice_scatter_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1670: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_388, [0, 2, 1, 3]);  slice_scatter_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_274: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1670, memory_format = torch.contiguous_format);  permute_1670 = None
        view_1921: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_274, [96, 4, 256, 513]);  clone_274 = None
        add_239: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_1919, view_1921);  view_1919 = view_1921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_430: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_239, 1, 0)
        slice_1828: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_430, 1, 1, 256)
        slice_1829: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1828, 2, 1, 256)
        clone_275: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_1829, memory_format = torch.contiguous_format)
        copy_216: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1829, full_default_60);  slice_1829 = None
        slice_scatter_389: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1828, copy_216, 2, 1, 256);  slice_1828 = copy_216 = None
        slice_scatter_390: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_430, slice_scatter_389, 1, 1, 256);  select_430 = slice_scatter_389 = None
        select_scatter_68: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_239, slice_scatter_390, 1, 0);  add_239 = slice_scatter_390 = None
        slice_scatter_391: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_275, 2, -255, 9223372036854775807);  clone_275 = None
        slice_scatter_392: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_391, 1, 0, 255);  slice_scatter_391 = None
        select_scatter_69: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_392, 1, 0);  slice_scatter_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1836: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_68, 1, 1, 9223372036854775807)
        slice_1837: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1836, 3, 0, 256)
        clone_276: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_1837, memory_format = torch.contiguous_format)
        copy_218: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1837, full_default_64);  slice_1837 = None
        slice_scatter_393: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1836, copy_218, 3, 0, 256);  slice_1836 = copy_218 = None
        slice_scatter_394: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_68, slice_scatter_393, 1, 1, 9223372036854775807);  select_scatter_68 = slice_scatter_393 = None
        slice_scatter_395: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_276, 3, 257, 9223372036854775807);  clone_276 = None
        slice_scatter_396: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_395, 2, -257, -1);  slice_scatter_395 = None
        add_240: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_69, slice_scatter_396);  select_scatter_69 = slice_scatter_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_435: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_394, 1, -1)
        slice_1842: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_435, 2, 256, 9223372036854775807)
        clone_277: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1842, memory_format = torch.contiguous_format)
        copy_220: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1842, full_default_67);  slice_1842 = None
        slice_scatter_397: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_435, copy_220, 2, 256, 9223372036854775807);  select_435 = copy_220 = None
        select_scatter_70: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_394, slice_scatter_397, 1, -1);  slice_scatter_394 = slice_scatter_397 = None
        slice_scatter_398: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_277, 2, 0, 257);  clone_277 = None
        slice_scatter_399: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_398, 1, 256, 9223372036854775807);  slice_scatter_398 = None
        select_scatter_71: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_399, 1, -1);  slice_scatter_399 = None
        add_241: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_240, select_scatter_71);  add_240 = select_scatter_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1847: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_70, 1, 0, -1);  select_scatter_70 = None
        slice_1848: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1847, 3, 256, 9223372036854775807);  slice_1847 = None
        clone_278: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1848, memory_format = torch.contiguous_format);  slice_1848 = None
        slice_scatter_400: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_278, 3, 0, 257);  clone_278 = None
        slice_scatter_401: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_400, 2, 0, 256);  slice_scatter_400 = None
        add_242: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_241, slice_scatter_401);  add_241 = slice_scatter_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_1922: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_242, [96, 3, 513, 512]);  add_242 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_65: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1922, [0, 0, 0, -1]);  view_1922 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_1923: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_65, [96, 3, 512, 512, 1]);  constant_pad_nd_65 = None
        permute_1671: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1923, [0, 1, 2, 4, 3]);  view_1923 = None
        view_1924: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1671, [288, 512, 512]);  permute_1671 = None
        bmm_46: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1672, view_1924);  permute_1672 = None
        bmm_47: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_1924, permute_1673);  view_1924 = permute_1673 = None
        view_1925: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [96, 3, 64, 512, 1]);  bmm_46 = None
        permute_1674: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_1925, [0, 1, 4, 3, 2]);  view_1925 = None
        view_1926: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_47, [96, 3, 512, 64, 1]);  bmm_47 = None
        permute_1676: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_1674, [0, 1, 3, 4, 2]);  permute_1674 = None
        squeeze_22: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_1676, 4);  permute_1676 = None
        squeeze_23: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1926, 4);  view_1926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_279: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_22, memory_format = torch.contiguous_format);  squeeze_22 = None
        view_1927: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_279, [9437184]);  clone_279 = None
        index_add_16: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1927);  view_1927 = None
        view_1930: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_23, [-1]);  squeeze_23 = None
        index_add_17: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_1930);  view_1930 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_230: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_17, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_17 = None
        view_1951: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_230, [96, 1024, 64]);  as_strided_230 = None
        view_1952: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1951, [8, 12, 1024, 64]);  view_1951 = None
        permute_1688: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1952, [0, 2, 1, 3]);  view_1952 = None
        permute_1689: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1688, [1, 0, 2, 3]);  permute_1688 = None
        view_1953: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1689, [1024, 8, 768]);  permute_1689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_137: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_1953, 8.0);  view_1953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_100: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1896, [0, 1], True, dtype = torch.float32)
        view_1954: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_100, [768]);  sum_100 = None
        convert_element_type_876: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1954, torch.bfloat16);  view_1954 = None
        view_1955: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1896, [8192, 768]);  view_1896 = None
        permute_1690: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1955, [1, 0])
        mm_114: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1690, view_690);  permute_1690 = None
        mm_115: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1955, permute_1692);  view_1955 = permute_1692 = None
        view_1956: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_115, [1024, 8, 768]);  mm_115 = None
        convert_element_type_881: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1956, torch.float32);  view_1956 = None
        convert_element_type_882: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        convert_element_type_883: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_876, torch.float32);  convert_element_type_876 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_231: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_16, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_16 = None
        view_1957: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_231, [96, 1024, 64]);  as_strided_231 = None
        view_1958: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_1957, [8, 12, 1024, 64]);  view_1957 = None
        permute_1694: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1958, [0, 2, 1, 3]);  view_1958 = None
        permute_1695: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1694, [1, 0, 2, 3]);  permute_1694 = None
        view_1959: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1695, [1024, 8, 768]);  permute_1695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_101: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1959, [0, 1], True, dtype = torch.float32)
        view_1960: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_101, [768]);  sum_101 = None
        convert_element_type_884: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1960, torch.bfloat16);  view_1960 = None
        view_1965: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1959, [8192, 768]);  view_1959 = None
        permute_1701: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1965, [1, 0])
        mm_116: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1701, view_690);  permute_1701 = None
        mm_117: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1965, permute_1703);  view_1965 = permute_1703 = None
        view_1970: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_117, [1024, 8, 768]);  mm_117 = None
        convert_element_type_889: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1970, torch.float32);  view_1970 = None
        add_243: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_881, convert_element_type_889);  convert_element_type_881 = convert_element_type_889 = None
        convert_element_type_890: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        convert_element_type_891: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_884, torch.float32);  convert_element_type_884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_102: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_137, [0, 1], True, dtype = torch.float32)
        view_1971: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_102, [768]);  sum_102 = None
        convert_element_type_892: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1971, torch.bfloat16);  view_1971 = None
        view_1972: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_137, [8192, 768]);  div_137 = None
        permute_1705: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1972, [1, 0])
        mm_118: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1705, view_690);  permute_1705 = view_690 = None
        mm_119: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1972, permute_1707);  view_1972 = permute_1707 = None
        view_1973: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_119, [1024, 8, 768]);  mm_119 = None
        convert_element_type_897: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1973, torch.float32);  view_1973 = None
        add_244: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_243, convert_element_type_897);  add_243 = convert_element_type_897 = None
        convert_element_type_898: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        convert_element_type_899: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_892, torch.float32);  convert_element_type_892 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1709: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_244, [1, 0, 2]);  add_244 = None
        add_245: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_329, permute_1709);  mul_329 = permute_1709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_337: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, primals_98);  primals_98 = None
        mul_338: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, 768)
        sum_103: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True)
        mul_339: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, mul_82);  mul_337 = None
        sum_104: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True);  mul_339 = None
        mul_340: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, sum_104);  sum_104 = None
        sub_133: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_338, sum_103);  mul_338 = sum_103 = None
        sub_134: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_133, mul_340);  sub_133 = mul_340 = None
        mul_341: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_138, sub_134);  div_138 = sub_134 = None
        mul_342: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, mul_82);  mul_82 = None
        sum_105: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1]);  mul_342 = None
        sum_106: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None
        convert_element_type_900: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_341, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_901: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_17, torch.bfloat16);  gt_17 = None
        mul_343: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_901, 1.1111111111111112);  convert_element_type_901 = None
        mul_344: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_900, mul_343);  convert_element_type_900 = mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_1974: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_344, [8192, 768]);  mul_344 = None
        mm_120: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_1974, permute_1710);  permute_1710 = None
        permute_1711: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1974, [1, 0])
        mm_121: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_1711, view_688);  permute_1711 = view_688 = None
        sum_107: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1974, [0], True, dtype = torch.float32);  view_1974 = None
        view_1975: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_107, [768]);  sum_107 = None
        convert_element_type_906: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1975, torch.bfloat16);  view_1975 = None
        view_1976: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [8, 1024, 3072]);  mm_120 = None
        convert_element_type_907: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_908: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_906, torch.float32);  convert_element_type_906 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_909: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1976, torch.float32);  view_1976 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_687: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [8, 1024, 3072]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_251: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_687, torch.float32);  view_687 = None
        mul_78: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_251, 0.7071067811865476)
        erf_5: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_86: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_346: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
        mul_347: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_251, convert_element_type_251)
        mul_348: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_347, -0.5);  mul_347 = None
        exp_18: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_348);  mul_348 = None
        mul_349: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
        mul_350: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_251, mul_349);  convert_element_type_251 = mul_349 = None
        add_247: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_346, mul_350);  mul_346 = mul_350 = None
        mul_351: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_909, add_247);  convert_element_type_909 = add_247 = None
        convert_element_type_911: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_351, torch.bfloat16);  mul_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_1977: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_911, [8192, 3072]);  convert_element_type_911 = None
        mm_122: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1977, permute_1714);  permute_1714 = None
        permute_1715: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_1977, [1, 0])
        mm_123: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1715, view_686);  permute_1715 = view_686 = None
        sum_108: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1977, [0], True, dtype = torch.float32);  view_1977 = None
        view_1978: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_108, [3072]);  sum_108 = None
        convert_element_type_916: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1978, torch.bfloat16);  view_1978 = None
        view_1979: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [8, 1024, 768]);  mm_122 = None
        convert_element_type_917: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1979, torch.float32);  view_1979 = None
        add_248: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_341, convert_element_type_917);  mul_341 = convert_element_type_917 = None
        convert_element_type_918: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_919: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_916, torch.float32);  convert_element_type_916 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_353: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_248, primals_92);  primals_92 = None
        mul_354: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, 768)
        sum_109: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
        mul_355: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, mul_75);  mul_353 = None
        sum_110: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
        mul_356: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, sum_110);  sum_110 = None
        sub_136: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_354, sum_109);  mul_354 = sum_109 = None
        sub_137: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_136, mul_356);  sub_136 = mul_356 = None
        mul_357: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_139, sub_137);  div_139 = sub_137 = None
        mul_358: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_248, mul_75);  mul_75 = None
        sum_111: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
        sum_112: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None
        convert_element_type_920: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_357, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_921: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_16, torch.bfloat16);  gt_16 = None
        mul_359: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_921, 1.1111111111111112);  convert_element_type_921 = None
        mul_360: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_920, mul_359);  convert_element_type_920 = mul_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_113: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1], True, dtype = torch.float32)
        view_1980: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_113, [768]);  sum_113 = None
        convert_element_type_922: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1980, torch.bfloat16);  view_1980 = None
        view_1981: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_360, [8192, 768]);  mul_360 = None
        permute_1718: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_1981, [1, 0])
        mm_124: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1718, view_684);  permute_1718 = view_684 = None
        mm_125: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_1981, permute_1720);  view_1981 = permute_1720 = None
        view_1982: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_125, [8, 1024, 768]);  mm_125 = None
        convert_element_type_927: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None
        convert_element_type_928: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_922, torch.float32);  convert_element_type_922 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1722: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_1982, [1, 0, 2]);  view_1982 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_1983: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1722, [1024, 8, 12, 64]);  permute_1722 = None
        permute_1723: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_1983, [1, 0, 2, 3]);  view_1983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1724: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1723, [0, 2, 1, 3]);  permute_1723 = None
        clone_284: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1724, memory_format = torch.contiguous_format);  permute_1724 = None
        view_1984: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_284, [96, 4, 256, 64]);  clone_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_1985: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_1984, [96, 4, 256, 64, 1]);  view_1984 = None
        permute_1725: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_1985, [0, 1, 2, 4, 3]);  view_1985 = None
        view_1986: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1725, [384, 256, 64]);  permute_1725 = None
        bmm_48: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1726, view_1986);  permute_1726 = None
        bmm_49: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_1986, permute_1727);  view_1986 = permute_1727 = None
        view_1987: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_48, [96, 4, 768, 64, 1]);  bmm_48 = None
        view_1988: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_49, [96, 4, 256, 768, 1]);  bmm_49 = None
        squeeze_24: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1987, 4);  view_1987 = None
        squeeze_25: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1988, 4);  view_1988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_402: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_25, 3, 0, -1);  squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1989: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_402, [96, 4, 196864]);  slice_scatter_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_403: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_1989, 2, 0, -256);  view_1989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_1990: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_403, [96, 4, 256, 770]);  slice_scatter_403 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_66: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_1990, [0, -257]);  view_1990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_1991: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_24, [-1]);  squeeze_24 = None
        index_add_18: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_1991);  view_1991 = None
        as_strided_236: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_18, [96, 1536, 64], [98304, 64, 1], 0);  index_add_18 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_67: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_236, [0, 0, -256, -256]);  as_strided_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_1993: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_67, [8, 12, 1024, 64]);  constant_pad_nd_67 = None
        permute_1732: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_1993, [0, 2, 1, 3]);  view_1993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_1994: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_66, [8, 12, 1024, 513]);  constant_pad_nd_66 = None
        permute_1733: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1994, [0, 2, 1, 3]);  view_1994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1734: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1732, [1, 0, 2, 3]);  permute_1732 = None
        clone_286: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1734, memory_format = torch.contiguous_format);  permute_1734 = None
        view_1995: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_286, [1024, 8, 768]);  clone_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_933: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_15, torch.bfloat16);  gt_15 = None
        mul_361: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_933, 1.1111111111111112);  convert_element_type_933 = None
        mul_362: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1733, mul_361);  permute_1733 = mul_361 = None
        clone_287: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_362, memory_format = torch.contiguous_format);  mul_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_934: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_287, torch.float32);  clone_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_114: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_934);  convert_element_type_934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_237: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_584, torch.float32);  permute_584 = None
        clone_79: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_237, memory_format = torch.contiguous_format);  convert_element_type_237 = None
        sub_44: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_79, amax_5);  clone_79 = amax_5 = None
        exp_5: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_44);  sub_44 = None
        div_57: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        mul_363: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_114, div_57);  where_114 = None
        sum_114: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_363, [-1], True)
        neg_6: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_57);  div_57 = None
        fma_6: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_6, sum_114, mul_363);  neg_6 = sum_114 = mul_363 = None
        convert_element_type_935: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1735: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_935, [0, 2, 1, 3]);  convert_element_type_935 = None
        clone_288: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1735, memory_format = torch.contiguous_format);  permute_1735 = None
        view_1996: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_288, [96, 4, 256, 513]);  clone_288 = None
        view_1999: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_1996, [8, 12, 1024, 513]);  view_1996 = None
        permute_1737: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_1999, [0, 2, 1, 3]);  view_1999 = None
        clone_289: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1737, memory_format = torch.contiguous_format)
        copy_223: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_1737, clone_289);  permute_1737 = clone_289 = None
        permute_1738: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_223, [0, 2, 1, 3]);  copy_223 = None
        view_2001: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1738, [96, 4, 256, 513]);  permute_1738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2007: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2001, [8, 12, 1024, 513]);  view_2001 = None
        permute_1743: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2007, [0, 2, 1, 3]);  view_2007 = None
        slice_1852: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1743, 1, -256, 9223372036854775807)
        slice_1853: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1852, 3, -257, 9223372036854775807)
        clone_290: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1853, memory_format = torch.contiguous_format)
        copy_225: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1853, full_default_52);  slice_1853 = None
        slice_scatter_404: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1852, copy_225, 3, -257, 9223372036854775807);  slice_1852 = copy_225 = None
        slice_scatter_405: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1743, slice_scatter_404, 1, -256, 9223372036854775807);  permute_1743 = slice_scatter_404 = None
        permute_1745: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_405, [0, 2, 1, 3]);  slice_scatter_405 = None
        view_2009: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1745, [96, 4, 256, 513]);  permute_1745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_115: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_290);  clone_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_406: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_115, 3, -257, 9223372036854775807);  where_115 = None
        slice_scatter_407: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_406, 1, -256, 9223372036854775807);  slice_scatter_406 = None
        permute_1747: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_407, [0, 2, 1, 3]);  slice_scatter_407 = None
        clone_291: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1747, memory_format = torch.contiguous_format);  permute_1747 = None
        view_2011: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_291, [96, 4, 256, 513]);  clone_291 = None
        add_249: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2009, view_2011);  view_2009 = view_2011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2016: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_249, [8, 12, 1024, 513]);  add_249 = None
        permute_1751: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2016, [0, 2, 1, 3]);  view_2016 = None
        slice_1860: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1751, 1, 0, 256)
        slice_1861: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1860, 3, 0, 257)
        clone_292: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1861, memory_format = torch.contiguous_format)
        copy_227: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1861, full_default_52);  slice_1861 = None
        slice_scatter_408: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1860, copy_227, 3, 0, 257);  slice_1860 = copy_227 = None
        slice_scatter_409: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1751, slice_scatter_408, 1, 0, 256);  permute_1751 = slice_scatter_408 = None
        permute_1753: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_409, [0, 2, 1, 3]);  slice_scatter_409 = None
        view_2018: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1753, [96, 4, 256, 513]);  permute_1753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_116: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_292);  clone_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_410: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_116, 3, 0, 257);  where_116 = None
        slice_scatter_411: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_410, 1, 0, 256);  slice_scatter_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1755: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_411, [0, 2, 1, 3]);  slice_scatter_411 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_293: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1755, memory_format = torch.contiguous_format);  permute_1755 = None
        view_2020: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_293, [96, 4, 256, 513]);  clone_293 = None
        add_250: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2018, view_2020);  view_2018 = view_2020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_441: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_250, 1, 0)
        slice_1868: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_441, 1, 1, 256)
        slice_1869: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1868, 2, 1, 256)
        clone_294: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_1869, memory_format = torch.contiguous_format)
        copy_229: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1869, full_default_60);  slice_1869 = None
        slice_scatter_412: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1868, copy_229, 2, 1, 256);  slice_1868 = copy_229 = None
        slice_scatter_413: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_441, slice_scatter_412, 1, 1, 256);  select_441 = slice_scatter_412 = None
        select_scatter_72: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_250, slice_scatter_413, 1, 0);  add_250 = slice_scatter_413 = None
        slice_scatter_414: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_294, 2, -255, 9223372036854775807);  clone_294 = None
        slice_scatter_415: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_414, 1, 0, 255);  slice_scatter_414 = None
        select_scatter_73: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_415, 1, 0);  slice_scatter_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1876: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_72, 1, 1, 9223372036854775807)
        slice_1877: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1876, 3, 0, 256)
        clone_295: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_1877, memory_format = torch.contiguous_format)
        copy_231: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1877, full_default_64);  slice_1877 = None
        slice_scatter_416: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1876, copy_231, 3, 0, 256);  slice_1876 = copy_231 = None
        slice_scatter_417: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_72, slice_scatter_416, 1, 1, 9223372036854775807);  select_scatter_72 = slice_scatter_416 = None
        slice_scatter_418: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_295, 3, 257, 9223372036854775807);  clone_295 = None
        slice_scatter_419: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_418, 2, -257, -1);  slice_scatter_418 = None
        add_251: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_73, slice_scatter_419);  select_scatter_73 = slice_scatter_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_446: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_417, 1, -1)
        slice_1882: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_446, 2, 256, 9223372036854775807)
        clone_296: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1882, memory_format = torch.contiguous_format)
        copy_233: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1882, full_default_67);  slice_1882 = None
        slice_scatter_420: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_446, copy_233, 2, 256, 9223372036854775807);  select_446 = copy_233 = None
        select_scatter_74: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_417, slice_scatter_420, 1, -1);  slice_scatter_417 = slice_scatter_420 = None
        slice_scatter_421: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_296, 2, 0, 257);  clone_296 = None
        slice_scatter_422: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_421, 1, 256, 9223372036854775807);  slice_scatter_421 = None
        select_scatter_75: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_422, 1, -1);  slice_scatter_422 = None
        add_252: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_251, select_scatter_75);  add_251 = select_scatter_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1887: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_74, 1, 0, -1);  select_scatter_74 = None
        slice_1888: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1887, 3, 256, 9223372036854775807);  slice_1887 = None
        clone_297: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1888, memory_format = torch.contiguous_format);  slice_1888 = None
        slice_scatter_423: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_297, 3, 0, 257);  clone_297 = None
        slice_scatter_424: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_423, 2, 0, 256);  slice_scatter_423 = None
        add_253: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_252, slice_scatter_424);  add_252 = slice_scatter_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2021: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_253, [96, 3, 513, 512]);  add_253 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_68: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2021, [0, 0, 0, -1]);  view_2021 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2022: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_68, [96, 3, 512, 512, 1]);  constant_pad_nd_68 = None
        permute_1756: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2022, [0, 1, 2, 4, 3]);  view_2022 = None
        view_2023: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1756, [288, 512, 512]);  permute_1756 = None
        bmm_50: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1757, view_2023);  permute_1757 = None
        bmm_51: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_2023, permute_1758);  view_2023 = permute_1758 = None
        view_2024: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_50, [96, 3, 64, 512, 1]);  bmm_50 = None
        permute_1759: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_2024, [0, 1, 4, 3, 2]);  view_2024 = None
        view_2025: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_51, [96, 3, 512, 64, 1]);  bmm_51 = None
        permute_1761: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_1759, [0, 1, 3, 4, 2]);  permute_1759 = None
        squeeze_26: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_1761, 4);  permute_1761 = None
        squeeze_27: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2025, 4);  view_2025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_298: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_26, memory_format = torch.contiguous_format);  squeeze_26 = None
        view_2026: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_298, [9437184]);  clone_298 = None
        index_add_19: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2026);  view_2026 = None
        view_2029: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_27, [-1]);  squeeze_27 = None
        index_add_20: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2029);  view_2029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_251: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_20, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_20 = None
        view_2050: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_251, [96, 1024, 64]);  as_strided_251 = None
        view_2051: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2050, [8, 12, 1024, 64]);  view_2050 = None
        permute_1773: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2051, [0, 2, 1, 3]);  view_2051 = None
        permute_1774: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1773, [1, 0, 2, 3]);  permute_1773 = None
        view_2052: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1774, [1024, 8, 768]);  permute_1774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_140: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_2052, 8.0);  view_2052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_115: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1995, [0, 1], True, dtype = torch.float32)
        view_2053: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_115, [768]);  sum_115 = None
        convert_element_type_940: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2053, torch.bfloat16);  view_2053 = None
        view_2054: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_1995, [8192, 768]);  view_1995 = None
        permute_1775: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2054, [1, 0])
        mm_126: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1775, view_575);  permute_1775 = None
        mm_127: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2054, permute_1777);  view_2054 = permute_1777 = None
        view_2055: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_127, [1024, 8, 768]);  mm_127 = None
        convert_element_type_945: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2055, torch.float32);  view_2055 = None
        convert_element_type_946: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        convert_element_type_947: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_940, torch.float32);  convert_element_type_940 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_252: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_19, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_19 = None
        view_2056: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_252, [96, 1024, 64]);  as_strided_252 = None
        view_2057: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2056, [8, 12, 1024, 64]);  view_2056 = None
        permute_1779: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2057, [0, 2, 1, 3]);  view_2057 = None
        permute_1780: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1779, [1, 0, 2, 3]);  permute_1779 = None
        view_2058: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1780, [1024, 8, 768]);  permute_1780 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_116: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2058, [0, 1], True, dtype = torch.float32)
        view_2059: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_116, [768]);  sum_116 = None
        convert_element_type_948: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2059, torch.bfloat16);  view_2059 = None
        view_2064: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2058, [8192, 768]);  view_2058 = None
        permute_1786: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2064, [1, 0])
        mm_128: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1786, view_575);  permute_1786 = None
        mm_129: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2064, permute_1788);  view_2064 = permute_1788 = None
        view_2069: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_129, [1024, 8, 768]);  mm_129 = None
        convert_element_type_953: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2069, torch.float32);  view_2069 = None
        add_254: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_945, convert_element_type_953);  convert_element_type_945 = convert_element_type_953 = None
        convert_element_type_954: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        convert_element_type_955: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_948, torch.float32);  convert_element_type_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_117: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_140, [0, 1], True, dtype = torch.float32)
        view_2070: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [768]);  sum_117 = None
        convert_element_type_956: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2070, torch.bfloat16);  view_2070 = None
        view_2071: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_140, [8192, 768]);  div_140 = None
        permute_1790: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2071, [1, 0])
        mm_130: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1790, view_575);  permute_1790 = view_575 = None
        mm_131: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2071, permute_1792);  view_2071 = permute_1792 = None
        view_2072: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_131, [1024, 8, 768]);  mm_131 = None
        convert_element_type_961: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2072, torch.float32);  view_2072 = None
        add_255: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_254, convert_element_type_961);  add_254 = convert_element_type_961 = None
        convert_element_type_962: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        convert_element_type_963: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_956, torch.float32);  convert_element_type_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1794: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_255, [1, 0, 2]);  add_255 = None
        add_256: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_357, permute_1794);  mul_357 = permute_1794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_365: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_256, primals_82);  primals_82 = None
        mul_366: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, 768)
        sum_118: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_365, [2], True)
        mul_367: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, mul_68);  mul_365 = None
        sum_119: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True);  mul_367 = None
        mul_368: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, sum_119);  sum_119 = None
        sub_139: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_366, sum_118);  mul_366 = sum_118 = None
        sub_140: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_139, mul_368);  sub_139 = mul_368 = None
        mul_369: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_141, sub_140);  div_141 = sub_140 = None
        mul_370: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_256, mul_68);  mul_68 = None
        sum_120: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 1]);  mul_370 = None
        sum_121: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_256, [0, 1]);  add_256 = None
        convert_element_type_964: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_369, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_965: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_14, torch.bfloat16);  gt_14 = None
        mul_371: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_965, 1.1111111111111112);  convert_element_type_965 = None
        mul_372: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_964, mul_371);  convert_element_type_964 = mul_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_2073: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_372, [8192, 768]);  mul_372 = None
        mm_132: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_2073, permute_1795);  permute_1795 = None
        permute_1796: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2073, [1, 0])
        mm_133: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_1796, view_573);  permute_1796 = view_573 = None
        sum_122: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2073, [0], True, dtype = torch.float32);  view_2073 = None
        view_2074: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_122, [768]);  sum_122 = None
        convert_element_type_970: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2074, torch.bfloat16);  view_2074 = None
        view_2075: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [8, 1024, 3072]);  mm_132 = None
        convert_element_type_971: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_972: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_970, torch.float32);  convert_element_type_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_973: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2075, torch.float32);  view_2075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_572: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [8, 1024, 3072]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_208: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_572, torch.float32);  view_572 = None
        mul_64: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_208, 0.7071067811865476)
        erf_4: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_71: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_374: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_71, 0.5);  add_71 = None
        mul_375: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_208, convert_element_type_208)
        mul_376: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_375, -0.5);  mul_375 = None
        exp_19: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_376);  mul_376 = None
        mul_377: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
        mul_378: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_208, mul_377);  convert_element_type_208 = mul_377 = None
        add_258: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_374, mul_378);  mul_374 = mul_378 = None
        mul_379: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_973, add_258);  convert_element_type_973 = add_258 = None
        convert_element_type_975: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_379, torch.bfloat16);  mul_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_2076: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_975, [8192, 3072]);  convert_element_type_975 = None
        mm_134: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2076, permute_1799);  permute_1799 = None
        permute_1800: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_2076, [1, 0])
        mm_135: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1800, view_571);  permute_1800 = view_571 = None
        sum_123: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2076, [0], True, dtype = torch.float32);  view_2076 = None
        view_2077: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [3072]);  sum_123 = None
        convert_element_type_980: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2077, torch.bfloat16);  view_2077 = None
        view_2078: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [8, 1024, 768]);  mm_134 = None
        convert_element_type_981: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2078, torch.float32);  view_2078 = None
        add_259: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_369, convert_element_type_981);  mul_369 = convert_element_type_981 = None
        convert_element_type_982: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_983: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_980, torch.float32);  convert_element_type_980 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_381: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_259, primals_76);  primals_76 = None
        mul_382: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, 768)
        sum_124: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_381, [2], True)
        mul_383: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, mul_61);  mul_381 = None
        sum_125: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True);  mul_383 = None
        mul_384: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, sum_125);  sum_125 = None
        sub_142: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_382, sum_124);  mul_382 = sum_124 = None
        sub_143: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, mul_384);  sub_142 = mul_384 = None
        mul_385: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_142, sub_143);  div_142 = sub_143 = None
        mul_386: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_259, mul_61);  mul_61 = None
        sum_126: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 1]);  mul_386 = None
        sum_127: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_259, [0, 1]);  add_259 = None
        convert_element_type_984: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_385, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_985: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_13, torch.bfloat16);  gt_13 = None
        mul_387: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_985, 1.1111111111111112);  convert_element_type_985 = None
        mul_388: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_984, mul_387);  convert_element_type_984 = mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_128: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1], True, dtype = torch.float32)
        view_2079: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_128, [768]);  sum_128 = None
        convert_element_type_986: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2079, torch.bfloat16);  view_2079 = None
        view_2080: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_388, [8192, 768]);  mul_388 = None
        permute_1803: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2080, [1, 0])
        mm_136: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1803, view_569);  permute_1803 = view_569 = None
        mm_137: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2080, permute_1805);  view_2080 = permute_1805 = None
        view_2081: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [8, 1024, 768]);  mm_137 = None
        convert_element_type_991: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        convert_element_type_992: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_986, torch.float32);  convert_element_type_986 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1807: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_2081, [1, 0, 2]);  view_2081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_2082: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1807, [1024, 8, 12, 64]);  permute_1807 = None
        permute_1808: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2082, [1, 0, 2, 3]);  view_2082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1809: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1808, [0, 2, 1, 3]);  permute_1808 = None
        clone_303: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1809, memory_format = torch.contiguous_format);  permute_1809 = None
        view_2083: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_303, [96, 4, 256, 64]);  clone_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_2084: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_2083, [96, 4, 256, 64, 1]);  view_2083 = None
        permute_1810: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2084, [0, 1, 2, 4, 3]);  view_2084 = None
        view_2085: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1810, [384, 256, 64]);  permute_1810 = None
        bmm_52: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1811, view_2085);  permute_1811 = None
        bmm_53: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_2085, permute_1812);  view_2085 = permute_1812 = None
        view_2086: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [96, 4, 768, 64, 1]);  bmm_52 = None
        view_2087: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [96, 4, 256, 768, 1]);  bmm_53 = None
        squeeze_28: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2086, 4);  view_2086 = None
        squeeze_29: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2087, 4);  view_2087 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_425: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_29, 3, 0, -1);  squeeze_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2088: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_425, [96, 4, 196864]);  slice_scatter_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_426: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_2088, 2, 0, -256);  view_2088 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2089: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_426, [96, 4, 256, 770]);  slice_scatter_426 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_69: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2089, [0, -257]);  view_2089 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_2090: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_28, [-1]);  squeeze_28 = None
        index_add_21: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_2090);  view_2090 = None
        as_strided_257: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_21, [96, 1536, 64], [98304, 64, 1], 0);  index_add_21 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_70: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_257, [0, 0, -256, -256]);  as_strided_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_2092: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_70, [8, 12, 1024, 64]);  constant_pad_nd_70 = None
        permute_1817: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_2092, [0, 2, 1, 3]);  view_2092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_2093: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_69, [8, 12, 1024, 513]);  constant_pad_nd_69 = None
        permute_1818: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2093, [0, 2, 1, 3]);  view_2093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1819: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1817, [1, 0, 2, 3]);  permute_1817 = None
        clone_305: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1819, memory_format = torch.contiguous_format);  permute_1819 = None
        view_2094: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_305, [1024, 8, 768]);  clone_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_997: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_12, torch.bfloat16);  gt_12 = None
        mul_389: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_997, 1.1111111111111112);  convert_element_type_997 = None
        mul_390: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1818, mul_389);  permute_1818 = mul_389 = None
        clone_306: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_390, memory_format = torch.contiguous_format);  mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_998: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_306, torch.float32);  clone_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_117: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_998);  convert_element_type_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_194: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_484, torch.float32);  permute_484 = None
        clone_65: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_194, memory_format = torch.contiguous_format);  convert_element_type_194 = None
        sub_36: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_65, amax_4);  clone_65 = amax_4 = None
        exp_4: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        div_47: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        mul_391: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_117, div_47);  where_117 = None
        sum_129: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_391, [-1], True)
        neg_7: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_47);  div_47 = None
        fma_7: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_7, sum_129, mul_391);  neg_7 = sum_129 = mul_391 = None
        convert_element_type_999: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1820: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_999, [0, 2, 1, 3]);  convert_element_type_999 = None
        clone_307: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1820, memory_format = torch.contiguous_format);  permute_1820 = None
        view_2095: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_307, [96, 4, 256, 513]);  clone_307 = None
        view_2098: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2095, [8, 12, 1024, 513]);  view_2095 = None
        permute_1822: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2098, [0, 2, 1, 3]);  view_2098 = None
        clone_308: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1822, memory_format = torch.contiguous_format)
        copy_236: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_1822, clone_308);  permute_1822 = clone_308 = None
        permute_1823: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_236, [0, 2, 1, 3]);  copy_236 = None
        view_2100: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1823, [96, 4, 256, 513]);  permute_1823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2106: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2100, [8, 12, 1024, 513]);  view_2100 = None
        permute_1828: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2106, [0, 2, 1, 3]);  view_2106 = None
        slice_1892: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1828, 1, -256, 9223372036854775807)
        slice_1893: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1892, 3, -257, 9223372036854775807)
        clone_309: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1893, memory_format = torch.contiguous_format)
        copy_238: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1893, full_default_52);  slice_1893 = None
        slice_scatter_427: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1892, copy_238, 3, -257, 9223372036854775807);  slice_1892 = copy_238 = None
        slice_scatter_428: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1828, slice_scatter_427, 1, -256, 9223372036854775807);  permute_1828 = slice_scatter_427 = None
        permute_1830: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_428, [0, 2, 1, 3]);  slice_scatter_428 = None
        view_2108: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1830, [96, 4, 256, 513]);  permute_1830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_118: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_309);  clone_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_429: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_118, 3, -257, 9223372036854775807);  where_118 = None
        slice_scatter_430: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_429, 1, -256, 9223372036854775807);  slice_scatter_429 = None
        permute_1832: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_430, [0, 2, 1, 3]);  slice_scatter_430 = None
        clone_310: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1832, memory_format = torch.contiguous_format);  permute_1832 = None
        view_2110: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_310, [96, 4, 256, 513]);  clone_310 = None
        add_260: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2108, view_2110);  view_2108 = view_2110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2115: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_260, [8, 12, 1024, 513]);  add_260 = None
        permute_1836: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2115, [0, 2, 1, 3]);  view_2115 = None
        slice_1900: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1836, 1, 0, 256)
        slice_1901: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1900, 3, 0, 257)
        clone_311: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1901, memory_format = torch.contiguous_format)
        copy_240: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1901, full_default_52);  slice_1901 = None
        slice_scatter_431: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1900, copy_240, 3, 0, 257);  slice_1900 = copy_240 = None
        slice_scatter_432: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1836, slice_scatter_431, 1, 0, 256);  permute_1836 = slice_scatter_431 = None
        permute_1838: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_432, [0, 2, 1, 3]);  slice_scatter_432 = None
        view_2117: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1838, [96, 4, 256, 513]);  permute_1838 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_119: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_311);  clone_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_433: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_119, 3, 0, 257);  where_119 = None
        slice_scatter_434: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_433, 1, 0, 256);  slice_scatter_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1840: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_434, [0, 2, 1, 3]);  slice_scatter_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_312: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1840, memory_format = torch.contiguous_format);  permute_1840 = None
        view_2119: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_312, [96, 4, 256, 513]);  clone_312 = None
        add_261: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2117, view_2119);  view_2117 = view_2119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_452: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_261, 1, 0)
        slice_1908: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_452, 1, 1, 256)
        slice_1909: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1908, 2, 1, 256)
        clone_313: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_1909, memory_format = torch.contiguous_format)
        copy_242: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1909, full_default_60);  slice_1909 = None
        slice_scatter_435: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1908, copy_242, 2, 1, 256);  slice_1908 = copy_242 = None
        slice_scatter_436: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_452, slice_scatter_435, 1, 1, 256);  select_452 = slice_scatter_435 = None
        select_scatter_76: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_261, slice_scatter_436, 1, 0);  add_261 = slice_scatter_436 = None
        slice_scatter_437: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_313, 2, -255, 9223372036854775807);  clone_313 = None
        slice_scatter_438: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_437, 1, 0, 255);  slice_scatter_437 = None
        select_scatter_77: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_438, 1, 0);  slice_scatter_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1916: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_76, 1, 1, 9223372036854775807)
        slice_1917: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1916, 3, 0, 256)
        clone_314: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_1917, memory_format = torch.contiguous_format)
        copy_244: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1917, full_default_64);  slice_1917 = None
        slice_scatter_439: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1916, copy_244, 3, 0, 256);  slice_1916 = copy_244 = None
        slice_scatter_440: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_76, slice_scatter_439, 1, 1, 9223372036854775807);  select_scatter_76 = slice_scatter_439 = None
        slice_scatter_441: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_314, 3, 257, 9223372036854775807);  clone_314 = None
        slice_scatter_442: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_441, 2, -257, -1);  slice_scatter_441 = None
        add_262: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_77, slice_scatter_442);  select_scatter_77 = slice_scatter_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_457: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_440, 1, -1)
        slice_1922: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_457, 2, 256, 9223372036854775807)
        clone_315: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1922, memory_format = torch.contiguous_format)
        copy_246: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1922, full_default_67);  slice_1922 = None
        slice_scatter_443: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_457, copy_246, 2, 256, 9223372036854775807);  select_457 = copy_246 = None
        select_scatter_78: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_440, slice_scatter_443, 1, -1);  slice_scatter_440 = slice_scatter_443 = None
        slice_scatter_444: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_315, 2, 0, 257);  clone_315 = None
        slice_scatter_445: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_444, 1, 256, 9223372036854775807);  slice_scatter_444 = None
        select_scatter_79: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_445, 1, -1);  slice_scatter_445 = None
        add_263: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_262, select_scatter_79);  add_262 = select_scatter_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1927: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_78, 1, 0, -1);  select_scatter_78 = None
        slice_1928: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1927, 3, 256, 9223372036854775807);  slice_1927 = None
        clone_316: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1928, memory_format = torch.contiguous_format);  slice_1928 = None
        slice_scatter_446: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_316, 3, 0, 257);  clone_316 = None
        slice_scatter_447: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_446, 2, 0, 256);  slice_scatter_446 = None
        add_264: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_263, slice_scatter_447);  add_263 = slice_scatter_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2120: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_264, [96, 3, 513, 512]);  add_264 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_71: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2120, [0, 0, 0, -1]);  view_2120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2121: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_71, [96, 3, 512, 512, 1]);  constant_pad_nd_71 = None
        permute_1841: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2121, [0, 1, 2, 4, 3]);  view_2121 = None
        view_2122: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1841, [288, 512, 512]);  permute_1841 = None
        bmm_54: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1842, view_2122);  permute_1842 = None
        bmm_55: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_2122, permute_1843);  view_2122 = permute_1843 = None
        view_2123: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [96, 3, 64, 512, 1]);  bmm_54 = None
        permute_1844: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_2123, [0, 1, 4, 3, 2]);  view_2123 = None
        view_2124: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_55, [96, 3, 512, 64, 1]);  bmm_55 = None
        permute_1846: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_1844, [0, 1, 3, 4, 2]);  permute_1844 = None
        squeeze_30: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_1846, 4);  permute_1846 = None
        squeeze_31: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2124, 4);  view_2124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_317: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_30, memory_format = torch.contiguous_format);  squeeze_30 = None
        view_2125: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_317, [9437184]);  clone_317 = None
        index_add_22: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2125);  view_2125 = None
        view_2128: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_31, [-1]);  squeeze_31 = None
        index_add_23: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2128);  view_2128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_272: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_23, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_23 = None
        view_2149: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_272, [96, 1024, 64]);  as_strided_272 = None
        view_2150: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2149, [8, 12, 1024, 64]);  view_2149 = None
        permute_1858: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2150, [0, 2, 1, 3]);  view_2150 = None
        permute_1859: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1858, [1, 0, 2, 3]);  permute_1858 = None
        view_2151: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1859, [1024, 8, 768]);  permute_1859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_143: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_2151, 8.0);  view_2151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_130: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2094, [0, 1], True, dtype = torch.float32)
        view_2152: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_130, [768]);  sum_130 = None
        convert_element_type_1004: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2152, torch.bfloat16);  view_2152 = None
        view_2153: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2094, [8192, 768]);  view_2094 = None
        permute_1860: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2153, [1, 0])
        mm_138: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1860, view_460);  permute_1860 = None
        mm_139: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2153, permute_1862);  view_2153 = permute_1862 = None
        view_2154: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_139, [1024, 8, 768]);  mm_139 = None
        convert_element_type_1009: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2154, torch.float32);  view_2154 = None
        convert_element_type_1010: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        convert_element_type_1011: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1004, torch.float32);  convert_element_type_1004 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_273: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_22, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_22 = None
        view_2155: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_273, [96, 1024, 64]);  as_strided_273 = None
        view_2156: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2155, [8, 12, 1024, 64]);  view_2155 = None
        permute_1864: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2156, [0, 2, 1, 3]);  view_2156 = None
        permute_1865: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1864, [1, 0, 2, 3]);  permute_1864 = None
        view_2157: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1865, [1024, 8, 768]);  permute_1865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_131: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2157, [0, 1], True, dtype = torch.float32)
        view_2158: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_131, [768]);  sum_131 = None
        convert_element_type_1012: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2158, torch.bfloat16);  view_2158 = None
        view_2163: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2157, [8192, 768]);  view_2157 = None
        permute_1871: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2163, [1, 0])
        mm_140: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1871, view_460);  permute_1871 = None
        mm_141: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2163, permute_1873);  view_2163 = permute_1873 = None
        view_2168: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_141, [1024, 8, 768]);  mm_141 = None
        convert_element_type_1017: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2168, torch.float32);  view_2168 = None
        add_265: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1009, convert_element_type_1017);  convert_element_type_1009 = convert_element_type_1017 = None
        convert_element_type_1018: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        convert_element_type_1019: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1012, torch.float32);  convert_element_type_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_132: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_143, [0, 1], True, dtype = torch.float32)
        view_2169: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_132, [768]);  sum_132 = None
        convert_element_type_1020: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2169, torch.bfloat16);  view_2169 = None
        view_2170: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_143, [8192, 768]);  div_143 = None
        permute_1875: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2170, [1, 0])
        mm_142: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1875, view_460);  permute_1875 = view_460 = None
        mm_143: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2170, permute_1877);  view_2170 = permute_1877 = None
        view_2171: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_143, [1024, 8, 768]);  mm_143 = None
        convert_element_type_1025: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2171, torch.float32);  view_2171 = None
        add_266: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_265, convert_element_type_1025);  add_265 = convert_element_type_1025 = None
        convert_element_type_1026: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        convert_element_type_1027: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1020, torch.float32);  convert_element_type_1020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1879: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_266, [1, 0, 2]);  add_266 = None
        add_267: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_385, permute_1879);  mul_385 = permute_1879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_393: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_267, primals_66);  primals_66 = None
        mul_394: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_393, 768)
        sum_133: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True)
        mul_395: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_393, mul_54);  mul_393 = None
        sum_134: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True);  mul_395 = None
        mul_396: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_54, sum_134);  sum_134 = None
        sub_145: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_394, sum_133);  mul_394 = sum_133 = None
        sub_146: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_145, mul_396);  sub_145 = mul_396 = None
        mul_397: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_144, sub_146);  div_144 = sub_146 = None
        mul_398: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_267, mul_54);  mul_54 = None
        sum_135: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_398, [0, 1]);  mul_398 = None
        sum_136: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_267, [0, 1]);  add_267 = None
        convert_element_type_1028: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_397, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1029: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_11, torch.bfloat16);  gt_11 = None
        mul_399: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1029, 1.1111111111111112);  convert_element_type_1029 = None
        mul_400: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1028, mul_399);  convert_element_type_1028 = mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_2172: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_400, [8192, 768]);  mul_400 = None
        mm_144: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_2172, permute_1880);  permute_1880 = None
        permute_1881: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2172, [1, 0])
        mm_145: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_1881, view_458);  permute_1881 = view_458 = None
        sum_137: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2172, [0], True, dtype = torch.float32);  view_2172 = None
        view_2173: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_137, [768]);  sum_137 = None
        convert_element_type_1034: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2173, torch.bfloat16);  view_2173 = None
        view_2174: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [8, 1024, 3072]);  mm_144 = None
        convert_element_type_1035: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_1036: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1034, torch.float32);  convert_element_type_1034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1037: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2174, torch.float32);  view_2174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_457: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [8, 1024, 3072]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_165: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_457, torch.float32);  view_457 = None
        mul_50: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, 0.7071067811865476)
        erf_3: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_50);  mul_50 = None
        add_56: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_402: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_403: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, convert_element_type_165)
        mul_404: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_403, -0.5);  mul_403 = None
        exp_20: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_404);  mul_404 = None
        mul_405: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
        mul_406: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_165, mul_405);  convert_element_type_165 = mul_405 = None
        add_269: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_402, mul_406);  mul_402 = mul_406 = None
        mul_407: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1037, add_269);  convert_element_type_1037 = add_269 = None
        convert_element_type_1039: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_407, torch.bfloat16);  mul_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_2175: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1039, [8192, 3072]);  convert_element_type_1039 = None
        mm_146: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2175, permute_1884);  permute_1884 = None
        permute_1885: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_2175, [1, 0])
        mm_147: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1885, view_456);  permute_1885 = view_456 = None
        sum_138: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2175, [0], True, dtype = torch.float32);  view_2175 = None
        view_2176: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_138, [3072]);  sum_138 = None
        convert_element_type_1044: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2176, torch.bfloat16);  view_2176 = None
        view_2177: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [8, 1024, 768]);  mm_146 = None
        convert_element_type_1045: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2177, torch.float32);  view_2177 = None
        add_270: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_397, convert_element_type_1045);  mul_397 = convert_element_type_1045 = None
        convert_element_type_1046: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_1047: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1044, torch.float32);  convert_element_type_1044 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_409: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_270, primals_60);  primals_60 = None
        mul_410: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_409, 768)
        sum_139: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True)
        mul_411: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_409, mul_47);  mul_409 = None
        sum_140: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True);  mul_411 = None
        mul_412: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_47, sum_140);  sum_140 = None
        sub_148: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_410, sum_139);  mul_410 = sum_139 = None
        sub_149: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_148, mul_412);  sub_148 = mul_412 = None
        mul_413: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_145, sub_149);  div_145 = sub_149 = None
        mul_414: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_270, mul_47);  mul_47 = None
        sum_141: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 1]);  mul_414 = None
        sum_142: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_270, [0, 1]);  add_270 = None
        convert_element_type_1048: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_413, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1049: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_10, torch.bfloat16);  gt_10 = None
        mul_415: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1049, 1.1111111111111112);  convert_element_type_1049 = None
        mul_416: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1048, mul_415);  convert_element_type_1048 = mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_143: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1], True, dtype = torch.float32)
        view_2178: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_143, [768]);  sum_143 = None
        convert_element_type_1050: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2178, torch.bfloat16);  view_2178 = None
        view_2179: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_416, [8192, 768]);  mul_416 = None
        permute_1888: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2179, [1, 0])
        mm_148: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1888, view_454);  permute_1888 = view_454 = None
        mm_149: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2179, permute_1890);  view_2179 = permute_1890 = None
        view_2180: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_149, [8, 1024, 768]);  mm_149 = None
        convert_element_type_1055: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None
        convert_element_type_1056: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1050, torch.float32);  convert_element_type_1050 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1892: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_2180, [1, 0, 2]);  view_2180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_2181: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1892, [1024, 8, 12, 64]);  permute_1892 = None
        permute_1893: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2181, [1, 0, 2, 3]);  view_2181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1894: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1893, [0, 2, 1, 3]);  permute_1893 = None
        clone_322: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1894, memory_format = torch.contiguous_format);  permute_1894 = None
        view_2182: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_322, [96, 4, 256, 64]);  clone_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_2183: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_2182, [96, 4, 256, 64, 1]);  view_2182 = None
        permute_1895: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2183, [0, 1, 2, 4, 3]);  view_2183 = None
        view_2184: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1895, [384, 256, 64]);  permute_1895 = None
        bmm_56: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1896, view_2184);  permute_1896 = None
        bmm_57: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_2184, permute_1897);  view_2184 = permute_1897 = None
        view_2185: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_56, [96, 4, 768, 64, 1]);  bmm_56 = None
        view_2186: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_57, [96, 4, 256, 768, 1]);  bmm_57 = None
        squeeze_32: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2185, 4);  view_2185 = None
        squeeze_33: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2186, 4);  view_2186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_448: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_33, 3, 0, -1);  squeeze_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2187: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_448, [96, 4, 196864]);  slice_scatter_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_449: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_2187, 2, 0, -256);  view_2187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2188: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_449, [96, 4, 256, 770]);  slice_scatter_449 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_72: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2188, [0, -257]);  view_2188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_2189: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_32, [-1]);  squeeze_32 = None
        index_add_24: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_2189);  view_2189 = None
        as_strided_278: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_24, [96, 1536, 64], [98304, 64, 1], 0);  index_add_24 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_73: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_278, [0, 0, -256, -256]);  as_strided_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_2191: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_73, [8, 12, 1024, 64]);  constant_pad_nd_73 = None
        permute_1902: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_2191, [0, 2, 1, 3]);  view_2191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_2192: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_72, [8, 12, 1024, 513]);  constant_pad_nd_72 = None
        permute_1903: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2192, [0, 2, 1, 3]);  view_2192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1904: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1902, [1, 0, 2, 3]);  permute_1902 = None
        clone_324: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1904, memory_format = torch.contiguous_format);  permute_1904 = None
        view_2193: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_324, [1024, 8, 768]);  clone_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_1061: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_9, torch.bfloat16);  gt_9 = None
        mul_417: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1061, 1.1111111111111112);  convert_element_type_1061 = None
        mul_418: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1903, mul_417);  permute_1903 = mul_417 = None
        clone_325: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_418, memory_format = torch.contiguous_format);  mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_1062: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_325, torch.float32);  clone_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_120: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_1062);  convert_element_type_1062 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_151: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_384, torch.float32);  permute_384 = None
        clone_51: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_151, memory_format = torch.contiguous_format);  convert_element_type_151 = None
        sub_28: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_51, amax_3);  clone_51 = amax_3 = None
        exp_3: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        div_37: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        mul_419: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_120, div_37);  where_120 = None
        sum_144: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_419, [-1], True)
        neg_8: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_37);  div_37 = None
        fma_8: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_8, sum_144, mul_419);  neg_8 = sum_144 = mul_419 = None
        convert_element_type_1063: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1905: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1063, [0, 2, 1, 3]);  convert_element_type_1063 = None
        clone_326: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1905, memory_format = torch.contiguous_format);  permute_1905 = None
        view_2194: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_326, [96, 4, 256, 513]);  clone_326 = None
        view_2197: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2194, [8, 12, 1024, 513]);  view_2194 = None
        permute_1907: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2197, [0, 2, 1, 3]);  view_2197 = None
        clone_327: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1907, memory_format = torch.contiguous_format)
        copy_249: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_1907, clone_327);  permute_1907 = clone_327 = None
        permute_1908: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_249, [0, 2, 1, 3]);  copy_249 = None
        view_2199: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1908, [96, 4, 256, 513]);  permute_1908 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2205: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2199, [8, 12, 1024, 513]);  view_2199 = None
        permute_1913: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2205, [0, 2, 1, 3]);  view_2205 = None
        slice_1932: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1913, 1, -256, 9223372036854775807)
        slice_1933: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1932, 3, -257, 9223372036854775807)
        clone_328: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1933, memory_format = torch.contiguous_format)
        copy_251: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1933, full_default_52);  slice_1933 = None
        slice_scatter_450: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1932, copy_251, 3, -257, 9223372036854775807);  slice_1932 = copy_251 = None
        slice_scatter_451: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1913, slice_scatter_450, 1, -256, 9223372036854775807);  permute_1913 = slice_scatter_450 = None
        permute_1915: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_451, [0, 2, 1, 3]);  slice_scatter_451 = None
        view_2207: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1915, [96, 4, 256, 513]);  permute_1915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_121: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_328);  clone_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_452: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_121, 3, -257, 9223372036854775807);  where_121 = None
        slice_scatter_453: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_452, 1, -256, 9223372036854775807);  slice_scatter_452 = None
        permute_1917: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_453, [0, 2, 1, 3]);  slice_scatter_453 = None
        clone_329: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1917, memory_format = torch.contiguous_format);  permute_1917 = None
        view_2209: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_329, [96, 4, 256, 513]);  clone_329 = None
        add_271: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2207, view_2209);  view_2207 = view_2209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2214: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_271, [8, 12, 1024, 513]);  add_271 = None
        permute_1921: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2214, [0, 2, 1, 3]);  view_2214 = None
        slice_1940: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1921, 1, 0, 256)
        slice_1941: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1940, 3, 0, 257)
        clone_330: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1941, memory_format = torch.contiguous_format)
        copy_253: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1941, full_default_52);  slice_1941 = None
        slice_scatter_454: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1940, copy_253, 3, 0, 257);  slice_1940 = copy_253 = None
        slice_scatter_455: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1921, slice_scatter_454, 1, 0, 256);  permute_1921 = slice_scatter_454 = None
        permute_1923: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_455, [0, 2, 1, 3]);  slice_scatter_455 = None
        view_2216: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1923, [96, 4, 256, 513]);  permute_1923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_122: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_330);  clone_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_456: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_122, 3, 0, 257);  where_122 = None
        slice_scatter_457: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_456, 1, 0, 256);  slice_scatter_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_1925: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_457, [0, 2, 1, 3]);  slice_scatter_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_331: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1925, memory_format = torch.contiguous_format);  permute_1925 = None
        view_2218: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_331, [96, 4, 256, 513]);  clone_331 = None
        add_272: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2216, view_2218);  view_2216 = view_2218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_463: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_272, 1, 0)
        slice_1948: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_463, 1, 1, 256)
        slice_1949: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1948, 2, 1, 256)
        clone_332: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_1949, memory_format = torch.contiguous_format)
        copy_255: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1949, full_default_60);  slice_1949 = None
        slice_scatter_458: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1948, copy_255, 2, 1, 256);  slice_1948 = copy_255 = None
        slice_scatter_459: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_463, slice_scatter_458, 1, 1, 256);  select_463 = slice_scatter_458 = None
        select_scatter_80: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_272, slice_scatter_459, 1, 0);  add_272 = slice_scatter_459 = None
        slice_scatter_460: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_332, 2, -255, 9223372036854775807);  clone_332 = None
        slice_scatter_461: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_460, 1, 0, 255);  slice_scatter_460 = None
        select_scatter_81: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_461, 1, 0);  slice_scatter_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1956: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_80, 1, 1, 9223372036854775807)
        slice_1957: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1956, 3, 0, 256)
        clone_333: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_1957, memory_format = torch.contiguous_format)
        copy_257: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1957, full_default_64);  slice_1957 = None
        slice_scatter_462: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1956, copy_257, 3, 0, 256);  slice_1956 = copy_257 = None
        slice_scatter_463: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_80, slice_scatter_462, 1, 1, 9223372036854775807);  select_scatter_80 = slice_scatter_462 = None
        slice_scatter_464: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_333, 3, 257, 9223372036854775807);  clone_333 = None
        slice_scatter_465: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_464, 2, -257, -1);  slice_scatter_464 = None
        add_273: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_81, slice_scatter_465);  select_scatter_81 = slice_scatter_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_468: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_463, 1, -1)
        slice_1962: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_468, 2, 256, 9223372036854775807)
        clone_334: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1962, memory_format = torch.contiguous_format)
        copy_259: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1962, full_default_67);  slice_1962 = None
        slice_scatter_466: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_468, copy_259, 2, 256, 9223372036854775807);  select_468 = copy_259 = None
        select_scatter_82: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_463, slice_scatter_466, 1, -1);  slice_scatter_463 = slice_scatter_466 = None
        slice_scatter_467: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_334, 2, 0, 257);  clone_334 = None
        slice_scatter_468: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_467, 1, 256, 9223372036854775807);  slice_scatter_467 = None
        select_scatter_83: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_468, 1, -1);  slice_scatter_468 = None
        add_274: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_273, select_scatter_83);  add_273 = select_scatter_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_1967: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_82, 1, 0, -1);  select_scatter_82 = None
        slice_1968: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1967, 3, 256, 9223372036854775807);  slice_1967 = None
        clone_335: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1968, memory_format = torch.contiguous_format);  slice_1968 = None
        slice_scatter_469: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_335, 3, 0, 257);  clone_335 = None
        slice_scatter_470: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_469, 2, 0, 256);  slice_scatter_469 = None
        add_275: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_274, slice_scatter_470);  add_274 = slice_scatter_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2219: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_275, [96, 3, 513, 512]);  add_275 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_74: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2219, [0, 0, 0, -1]);  view_2219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2220: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_74, [96, 3, 512, 512, 1]);  constant_pad_nd_74 = None
        permute_1926: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2220, [0, 1, 2, 4, 3]);  view_2220 = None
        view_2221: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1926, [288, 512, 512]);  permute_1926 = None
        bmm_58: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1927, view_2221);  permute_1927 = None
        bmm_59: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_2221, permute_1928);  view_2221 = permute_1928 = None
        view_2222: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_58, [96, 3, 64, 512, 1]);  bmm_58 = None
        permute_1929: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_2222, [0, 1, 4, 3, 2]);  view_2222 = None
        view_2223: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_59, [96, 3, 512, 64, 1]);  bmm_59 = None
        permute_1931: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_1929, [0, 1, 3, 4, 2]);  permute_1929 = None
        squeeze_34: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_1931, 4);  permute_1931 = None
        squeeze_35: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2223, 4);  view_2223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_336: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_34, memory_format = torch.contiguous_format);  squeeze_34 = None
        view_2224: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_336, [9437184]);  clone_336 = None
        index_add_25: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2224);  view_2224 = None
        view_2227: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_35, [-1]);  squeeze_35 = None
        index_add_26: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2227);  view_2227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_293: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_26, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_26 = None
        view_2248: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_293, [96, 1024, 64]);  as_strided_293 = None
        view_2249: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2248, [8, 12, 1024, 64]);  view_2248 = None
        permute_1943: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2249, [0, 2, 1, 3]);  view_2249 = None
        permute_1944: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1943, [1, 0, 2, 3]);  permute_1943 = None
        view_2250: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1944, [1024, 8, 768]);  permute_1944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_146: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_2250, 8.0);  view_2250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_145: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2193, [0, 1], True, dtype = torch.float32)
        view_2251: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_145, [768]);  sum_145 = None
        convert_element_type_1068: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2251, torch.bfloat16);  view_2251 = None
        view_2252: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2193, [8192, 768]);  view_2193 = None
        permute_1945: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2252, [1, 0])
        mm_150: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1945, view_345);  permute_1945 = None
        mm_151: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2252, permute_1947);  view_2252 = permute_1947 = None
        view_2253: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_151, [1024, 8, 768]);  mm_151 = None
        convert_element_type_1073: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2253, torch.float32);  view_2253 = None
        convert_element_type_1074: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        convert_element_type_1075: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1068, torch.float32);  convert_element_type_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_294: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_25, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_25 = None
        view_2254: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_294, [96, 1024, 64]);  as_strided_294 = None
        view_2255: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2254, [8, 12, 1024, 64]);  view_2254 = None
        permute_1949: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2255, [0, 2, 1, 3]);  view_2255 = None
        permute_1950: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_1949, [1, 0, 2, 3]);  permute_1949 = None
        view_2256: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1950, [1024, 8, 768]);  permute_1950 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_146: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2256, [0, 1], True, dtype = torch.float32)
        view_2257: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_146, [768]);  sum_146 = None
        convert_element_type_1076: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2257, torch.bfloat16);  view_2257 = None
        view_2262: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2256, [8192, 768]);  view_2256 = None
        permute_1956: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2262, [1, 0])
        mm_152: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1956, view_345);  permute_1956 = None
        mm_153: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2262, permute_1958);  view_2262 = permute_1958 = None
        view_2267: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_153, [1024, 8, 768]);  mm_153 = None
        convert_element_type_1081: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2267, torch.float32);  view_2267 = None
        add_276: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1073, convert_element_type_1081);  convert_element_type_1073 = convert_element_type_1081 = None
        convert_element_type_1082: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        convert_element_type_1083: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1076, torch.float32);  convert_element_type_1076 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_147: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_146, [0, 1], True, dtype = torch.float32)
        view_2268: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [768]);  sum_147 = None
        convert_element_type_1084: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2268, torch.bfloat16);  view_2268 = None
        view_2269: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_146, [8192, 768]);  div_146 = None
        permute_1960: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2269, [1, 0])
        mm_154: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1960, view_345);  permute_1960 = view_345 = None
        mm_155: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2269, permute_1962);  view_2269 = permute_1962 = None
        view_2270: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_155, [1024, 8, 768]);  mm_155 = None
        convert_element_type_1089: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2270, torch.float32);  view_2270 = None
        add_277: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_276, convert_element_type_1089);  add_276 = convert_element_type_1089 = None
        convert_element_type_1090: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        convert_element_type_1091: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1084, torch.float32);  convert_element_type_1084 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_1964: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_277, [1, 0, 2]);  add_277 = None
        add_278: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_413, permute_1964);  mul_413 = permute_1964 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_421: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_278, primals_50);  primals_50 = None
        mul_422: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_421, 768)
        sum_148: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True)
        mul_423: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_421, mul_40);  mul_421 = None
        sum_149: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_423, [2], True);  mul_423 = None
        mul_424: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, sum_149);  sum_149 = None
        sub_151: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_422, sum_148);  mul_422 = sum_148 = None
        sub_152: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_151, mul_424);  sub_151 = mul_424 = None
        mul_425: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_147, sub_152);  div_147 = sub_152 = None
        mul_426: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_278, mul_40);  mul_40 = None
        sum_150: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 1]);  mul_426 = None
        sum_151: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None
        convert_element_type_1092: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_425, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1093: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_8, torch.bfloat16);  gt_8 = None
        mul_427: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1093, 1.1111111111111112);  convert_element_type_1093 = None
        mul_428: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1092, mul_427);  convert_element_type_1092 = mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_2271: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_428, [8192, 768]);  mul_428 = None
        mm_156: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_2271, permute_1965);  permute_1965 = None
        permute_1966: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2271, [1, 0])
        mm_157: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_1966, view_343);  permute_1966 = view_343 = None
        sum_152: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2271, [0], True, dtype = torch.float32);  view_2271 = None
        view_2272: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_152, [768]);  sum_152 = None
        convert_element_type_1098: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2272, torch.bfloat16);  view_2272 = None
        view_2273: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_156, [8, 1024, 3072]);  mm_156 = None
        convert_element_type_1099: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None
        convert_element_type_1100: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1098, torch.float32);  convert_element_type_1098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1101: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2273, torch.float32);  view_2273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_342: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [8, 1024, 3072]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_122: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_342, torch.float32);  view_342 = None
        mul_36: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, 0.7071067811865476)
        erf_2: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
        add_41: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_430: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_41, 0.5);  add_41 = None
        mul_431: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, convert_element_type_122)
        mul_432: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_431, -0.5);  mul_431 = None
        exp_21: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_432);  mul_432 = None
        mul_433: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
        mul_434: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_122, mul_433);  convert_element_type_122 = mul_433 = None
        add_280: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_430, mul_434);  mul_430 = mul_434 = None
        mul_435: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1101, add_280);  convert_element_type_1101 = add_280 = None
        convert_element_type_1103: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_435, torch.bfloat16);  mul_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_2274: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1103, [8192, 3072]);  convert_element_type_1103 = None
        mm_158: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2274, permute_1969);  permute_1969 = None
        permute_1970: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_2274, [1, 0])
        mm_159: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1970, view_341);  permute_1970 = view_341 = None
        sum_153: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2274, [0], True, dtype = torch.float32);  view_2274 = None
        view_2275: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [3072]);  sum_153 = None
        convert_element_type_1108: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2275, torch.bfloat16);  view_2275 = None
        view_2276: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_158, [8, 1024, 768]);  mm_158 = None
        convert_element_type_1109: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2276, torch.float32);  view_2276 = None
        add_281: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_425, convert_element_type_1109);  mul_425 = convert_element_type_1109 = None
        convert_element_type_1110: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_1111: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1108, torch.float32);  convert_element_type_1108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_437: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_281, primals_44);  primals_44 = None
        mul_438: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_437, 768)
        sum_154: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True)
        mul_439: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_437, mul_33);  mul_437 = None
        sum_155: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True);  mul_439 = None
        mul_440: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_33, sum_155);  sum_155 = None
        sub_154: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_438, sum_154);  mul_438 = sum_154 = None
        sub_155: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_154, mul_440);  sub_154 = mul_440 = None
        mul_441: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_148, sub_155);  div_148 = sub_155 = None
        mul_442: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_281, mul_33);  mul_33 = None
        sum_156: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 1]);  mul_442 = None
        sum_157: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None
        convert_element_type_1112: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_441, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1113: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_7, torch.bfloat16);  gt_7 = None
        mul_443: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1113, 1.1111111111111112);  convert_element_type_1113 = None
        mul_444: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1112, mul_443);  convert_element_type_1112 = mul_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_158: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 1], True, dtype = torch.float32)
        view_2277: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_158, [768]);  sum_158 = None
        convert_element_type_1114: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2277, torch.bfloat16);  view_2277 = None
        view_2278: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_444, [8192, 768]);  mul_444 = None
        permute_1973: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2278, [1, 0])
        mm_160: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_1973, view_339);  permute_1973 = view_339 = None
        mm_161: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2278, permute_1975);  view_2278 = permute_1975 = None
        view_2279: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_161, [8, 1024, 768]);  mm_161 = None
        convert_element_type_1119: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        convert_element_type_1120: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1114, torch.float32);  convert_element_type_1114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_1977: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_2279, [1, 0, 2]);  view_2279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_2280: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1977, [1024, 8, 12, 64]);  permute_1977 = None
        permute_1978: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2280, [1, 0, 2, 3]);  view_2280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_1979: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1978, [0, 2, 1, 3]);  permute_1978 = None
        clone_341: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1979, memory_format = torch.contiguous_format);  permute_1979 = None
        view_2281: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_341, [96, 4, 256, 64]);  clone_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_2282: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_2281, [96, 4, 256, 64, 1]);  view_2281 = None
        permute_1980: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2282, [0, 1, 2, 4, 3]);  view_2282 = None
        view_2283: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1980, [384, 256, 64]);  permute_1980 = None
        bmm_60: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_1981, view_2283);  permute_1981 = None
        bmm_61: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_2283, permute_1982);  view_2283 = permute_1982 = None
        view_2284: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [96, 4, 768, 64, 1]);  bmm_60 = None
        view_2285: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [96, 4, 256, 768, 1]);  bmm_61 = None
        squeeze_36: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2284, 4);  view_2284 = None
        squeeze_37: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2285, 4);  view_2285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_471: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_37, 3, 0, -1);  squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2286: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_471, [96, 4, 196864]);  slice_scatter_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_472: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_2286, 2, 0, -256);  view_2286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2287: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_472, [96, 4, 256, 770]);  slice_scatter_472 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_75: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2287, [0, -257]);  view_2287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_2288: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_36, [-1]);  squeeze_36 = None
        index_add_27: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_2288);  view_2288 = None
        as_strided_299: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_27, [96, 1536, 64], [98304, 64, 1], 0);  index_add_27 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_76: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_299, [0, 0, -256, -256]);  as_strided_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_2290: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_76, [8, 12, 1024, 64]);  constant_pad_nd_76 = None
        permute_1987: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_2290, [0, 2, 1, 3]);  view_2290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_2291: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_75, [8, 12, 1024, 513]);  constant_pad_nd_75 = None
        permute_1988: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2291, [0, 2, 1, 3]);  view_2291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_1989: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_1987, [1, 0, 2, 3]);  permute_1987 = None
        clone_343: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1989, memory_format = torch.contiguous_format);  permute_1989 = None
        view_2292: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_343, [1024, 8, 768]);  clone_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_1125: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_6, torch.bfloat16);  gt_6 = None
        mul_445: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1125, 1.1111111111111112);  convert_element_type_1125 = None
        mul_446: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1988, mul_445);  permute_1988 = mul_445 = None
        clone_344: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_446, memory_format = torch.contiguous_format);  mul_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_1126: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_344, torch.float32);  clone_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_123: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_1126);  convert_element_type_1126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_108: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_284, torch.float32);  permute_284 = None
        clone_37: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_108, memory_format = torch.contiguous_format);  convert_element_type_108 = None
        sub_20: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_37, amax_2);  clone_37 = amax_2 = None
        exp_2: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        div_27: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        mul_447: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_123, div_27);  where_123 = None
        sum_159: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_447, [-1], True)
        neg_9: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_27);  div_27 = None
        fma_9: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_9, sum_159, mul_447);  neg_9 = sum_159 = mul_447 = None
        convert_element_type_1127: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_1990: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1127, [0, 2, 1, 3]);  convert_element_type_1127 = None
        clone_345: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1990, memory_format = torch.contiguous_format);  permute_1990 = None
        view_2293: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_345, [96, 4, 256, 513]);  clone_345 = None
        view_2296: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2293, [8, 12, 1024, 513]);  view_2293 = None
        permute_1992: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2296, [0, 2, 1, 3]);  view_2296 = None
        clone_346: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_1992, memory_format = torch.contiguous_format)
        copy_262: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_1992, clone_346);  permute_1992 = clone_346 = None
        permute_1993: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_262, [0, 2, 1, 3]);  copy_262 = None
        view_2298: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1993, [96, 4, 256, 513]);  permute_1993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2304: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2298, [8, 12, 1024, 513]);  view_2298 = None
        permute_1998: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2304, [0, 2, 1, 3]);  view_2304 = None
        slice_1972: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_1998, 1, -256, 9223372036854775807)
        slice_1973: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1972, 3, -257, 9223372036854775807)
        clone_347: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1973, memory_format = torch.contiguous_format)
        copy_264: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1973, full_default_52);  slice_1973 = None
        slice_scatter_473: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1972, copy_264, 3, -257, 9223372036854775807);  slice_1972 = copy_264 = None
        slice_scatter_474: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_1998, slice_scatter_473, 1, -256, 9223372036854775807);  permute_1998 = slice_scatter_473 = None
        permute_2000: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_474, [0, 2, 1, 3]);  slice_scatter_474 = None
        view_2306: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2000, [96, 4, 256, 513]);  permute_2000 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_124: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_347);  clone_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_475: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_124, 3, -257, 9223372036854775807);  where_124 = None
        slice_scatter_476: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_475, 1, -256, 9223372036854775807);  slice_scatter_475 = None
        permute_2002: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_476, [0, 2, 1, 3]);  slice_scatter_476 = None
        clone_348: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_2002, memory_format = torch.contiguous_format);  permute_2002 = None
        view_2308: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_348, [96, 4, 256, 513]);  clone_348 = None
        add_282: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2306, view_2308);  view_2306 = view_2308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2313: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_282, [8, 12, 1024, 513]);  add_282 = None
        permute_2006: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2313, [0, 2, 1, 3]);  view_2313 = None
        slice_1980: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_2006, 1, 0, 256)
        slice_1981: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1980, 3, 0, 257)
        clone_349: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_1981, memory_format = torch.contiguous_format)
        copy_266: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_1981, full_default_52);  slice_1981 = None
        slice_scatter_477: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1980, copy_266, 3, 0, 257);  slice_1980 = copy_266 = None
        slice_scatter_478: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_2006, slice_scatter_477, 1, 0, 256);  permute_2006 = slice_scatter_477 = None
        permute_2008: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_478, [0, 2, 1, 3]);  slice_scatter_478 = None
        view_2315: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2008, [96, 4, 256, 513]);  permute_2008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_125: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_349);  clone_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_479: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_125, 3, 0, 257);  where_125 = None
        slice_scatter_480: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_479, 1, 0, 256);  slice_scatter_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_2010: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_480, [0, 2, 1, 3]);  slice_scatter_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_350: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_2010, memory_format = torch.contiguous_format);  permute_2010 = None
        view_2317: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_350, [96, 4, 256, 513]);  clone_350 = None
        add_283: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2315, view_2317);  view_2315 = view_2317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_474: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_283, 1, 0)
        slice_1988: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_474, 1, 1, 256)
        slice_1989: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1988, 2, 1, 256)
        clone_351: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_1989, memory_format = torch.contiguous_format)
        copy_268: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1989, full_default_60);  slice_1989 = None
        slice_scatter_481: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1988, copy_268, 2, 1, 256);  slice_1988 = copy_268 = None
        slice_scatter_482: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_474, slice_scatter_481, 1, 1, 256);  select_474 = slice_scatter_481 = None
        select_scatter_84: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_283, slice_scatter_482, 1, 0);  add_283 = slice_scatter_482 = None
        slice_scatter_483: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_351, 2, -255, 9223372036854775807);  clone_351 = None
        slice_scatter_484: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_483, 1, 0, 255);  slice_scatter_483 = None
        select_scatter_85: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_484, 1, 0);  slice_scatter_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_1996: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_84, 1, 1, 9223372036854775807)
        slice_1997: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_1996, 3, 0, 256)
        clone_352: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_1997, memory_format = torch.contiguous_format)
        copy_270: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_1997, full_default_64);  slice_1997 = None
        slice_scatter_485: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_1996, copy_270, 3, 0, 256);  slice_1996 = copy_270 = None
        slice_scatter_486: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_84, slice_scatter_485, 1, 1, 9223372036854775807);  select_scatter_84 = slice_scatter_485 = None
        slice_scatter_487: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_352, 3, 257, 9223372036854775807);  clone_352 = None
        slice_scatter_488: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_487, 2, -257, -1);  slice_scatter_487 = None
        add_284: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_85, slice_scatter_488);  select_scatter_85 = slice_scatter_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_479: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_486, 1, -1)
        slice_2002: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_479, 2, 256, 9223372036854775807)
        clone_353: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_2002, memory_format = torch.contiguous_format)
        copy_272: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_2002, full_default_67);  slice_2002 = None
        slice_scatter_489: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_479, copy_272, 2, 256, 9223372036854775807);  select_479 = copy_272 = None
        select_scatter_86: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_486, slice_scatter_489, 1, -1);  slice_scatter_486 = slice_scatter_489 = None
        slice_scatter_490: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_353, 2, 0, 257);  clone_353 = None
        slice_scatter_491: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_490, 1, 256, 9223372036854775807);  slice_scatter_490 = None
        select_scatter_87: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_491, 1, -1);  slice_scatter_491 = None
        add_285: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_284, select_scatter_87);  add_284 = select_scatter_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2007: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_86, 1, 0, -1);  select_scatter_86 = None
        slice_2008: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2007, 3, 256, 9223372036854775807);  slice_2007 = None
        clone_354: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_2008, memory_format = torch.contiguous_format);  slice_2008 = None
        slice_scatter_492: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_354, 3, 0, 257);  clone_354 = None
        slice_scatter_493: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_492, 2, 0, 256);  slice_scatter_492 = None
        add_286: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_285, slice_scatter_493);  add_285 = slice_scatter_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2318: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_286, [96, 3, 513, 512]);  add_286 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_77: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2318, [0, 0, 0, -1]);  view_2318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2319: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_77, [96, 3, 512, 512, 1]);  constant_pad_nd_77 = None
        permute_2011: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2319, [0, 1, 2, 4, 3]);  view_2319 = None
        view_2320: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2011, [288, 512, 512]);  permute_2011 = None
        bmm_62: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_2012, view_2320);  permute_2012 = None
        bmm_63: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_2320, permute_2013);  view_2320 = permute_2013 = None
        view_2321: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [96, 3, 64, 512, 1]);  bmm_62 = None
        permute_2014: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_2321, [0, 1, 4, 3, 2]);  view_2321 = None
        view_2322: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_63, [96, 3, 512, 64, 1]);  bmm_63 = None
        permute_2016: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_2014, [0, 1, 3, 4, 2]);  permute_2014 = None
        squeeze_38: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_2016, 4);  permute_2016 = None
        squeeze_39: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2322, 4);  view_2322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_355: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_38, memory_format = torch.contiguous_format);  squeeze_38 = None
        view_2323: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_355, [9437184]);  clone_355 = None
        index_add_28: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2323);  view_2323 = None
        view_2326: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_39, [-1]);  squeeze_39 = None
        index_add_29: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2326);  view_2326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_314: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_29, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_29 = None
        view_2347: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_314, [96, 1024, 64]);  as_strided_314 = None
        view_2348: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2347, [8, 12, 1024, 64]);  view_2347 = None
        permute_2028: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2348, [0, 2, 1, 3]);  view_2348 = None
        permute_2029: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_2028, [1, 0, 2, 3]);  permute_2028 = None
        view_2349: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2029, [1024, 8, 768]);  permute_2029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_149: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_2349, 8.0);  view_2349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_160: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2292, [0, 1], True, dtype = torch.float32)
        view_2350: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_160, [768]);  sum_160 = None
        convert_element_type_1132: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2350, torch.bfloat16);  view_2350 = None
        view_2351: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2292, [8192, 768]);  view_2292 = None
        permute_2030: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2351, [1, 0])
        mm_162: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2030, view_230);  permute_2030 = None
        mm_163: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2351, permute_2032);  view_2351 = permute_2032 = None
        view_2352: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_163, [1024, 8, 768]);  mm_163 = None
        convert_element_type_1137: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2352, torch.float32);  view_2352 = None
        convert_element_type_1138: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        convert_element_type_1139: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1132, torch.float32);  convert_element_type_1132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_315: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_28, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_28 = None
        view_2353: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_315, [96, 1024, 64]);  as_strided_315 = None
        view_2354: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2353, [8, 12, 1024, 64]);  view_2353 = None
        permute_2034: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2354, [0, 2, 1, 3]);  view_2354 = None
        permute_2035: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_2034, [1, 0, 2, 3]);  permute_2034 = None
        view_2355: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2035, [1024, 8, 768]);  permute_2035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_161: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2355, [0, 1], True, dtype = torch.float32)
        view_2356: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_161, [768]);  sum_161 = None
        convert_element_type_1140: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2356, torch.bfloat16);  view_2356 = None
        view_2361: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2355, [8192, 768]);  view_2355 = None
        permute_2041: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2361, [1, 0])
        mm_164: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2041, view_230);  permute_2041 = None
        mm_165: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2361, permute_2043);  view_2361 = permute_2043 = None
        view_2366: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_165, [1024, 8, 768]);  mm_165 = None
        convert_element_type_1145: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2366, torch.float32);  view_2366 = None
        add_287: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1137, convert_element_type_1145);  convert_element_type_1137 = convert_element_type_1145 = None
        convert_element_type_1146: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        convert_element_type_1147: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1140, torch.float32);  convert_element_type_1140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_162: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_149, [0, 1], True, dtype = torch.float32)
        view_2367: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_162, [768]);  sum_162 = None
        convert_element_type_1148: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2367, torch.bfloat16);  view_2367 = None
        view_2368: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_149, [8192, 768]);  div_149 = None
        permute_2045: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2368, [1, 0])
        mm_166: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2045, view_230);  permute_2045 = view_230 = None
        mm_167: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2368, permute_2047);  view_2368 = permute_2047 = None
        view_2369: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_167, [1024, 8, 768]);  mm_167 = None
        convert_element_type_1153: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2369, torch.float32);  view_2369 = None
        add_288: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_287, convert_element_type_1153);  add_287 = convert_element_type_1153 = None
        convert_element_type_1154: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        convert_element_type_1155: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1148, torch.float32);  convert_element_type_1148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_2049: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_288, [1, 0, 2]);  add_288 = None
        add_289: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_441, permute_2049);  mul_441 = permute_2049 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_449: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_289, primals_34);  primals_34 = None
        mul_450: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_449, 768)
        sum_163: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True)
        mul_451: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_449, mul_26);  mul_449 = None
        sum_164: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True);  mul_451 = None
        mul_452: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, sum_164);  sum_164 = None
        sub_157: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_450, sum_163);  mul_450 = sum_163 = None
        sub_158: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_157, mul_452);  sub_157 = mul_452 = None
        mul_453: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_150, sub_158);  div_150 = sub_158 = None
        mul_454: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_289, mul_26);  mul_26 = None
        sum_165: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 1]);  mul_454 = None
        sum_166: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_289, [0, 1]);  add_289 = None
        convert_element_type_1156: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_453, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1157: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_5, torch.bfloat16);  gt_5 = None
        mul_455: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1157, 1.1111111111111112);  convert_element_type_1157 = None
        mul_456: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1156, mul_455);  convert_element_type_1156 = mul_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_2370: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_456, [8192, 768]);  mul_456 = None
        mm_168: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_2370, permute_2050);  permute_2050 = None
        permute_2051: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2370, [1, 0])
        mm_169: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_2051, view_228);  permute_2051 = view_228 = None
        sum_167: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2370, [0], True, dtype = torch.float32);  view_2370 = None
        view_2371: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_167, [768]);  sum_167 = None
        convert_element_type_1162: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2371, torch.bfloat16);  view_2371 = None
        view_2372: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_168, [8, 1024, 3072]);  mm_168 = None
        convert_element_type_1163: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_169, torch.float32);  mm_169 = None
        convert_element_type_1164: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1162, torch.float32);  convert_element_type_1162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1165: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2372, torch.float32);  view_2372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_227: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [8, 1024, 3072]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_79: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        mul_22: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, 0.7071067811865476)
        erf_1: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_22);  mul_22 = None
        add_26: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_458: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_26, 0.5);  add_26 = None
        mul_459: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, convert_element_type_79)
        mul_460: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_459, -0.5);  mul_459 = None
        exp_22: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_460);  mul_460 = None
        mul_461: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
        mul_462: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_79, mul_461);  convert_element_type_79 = mul_461 = None
        add_291: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_458, mul_462);  mul_458 = mul_462 = None
        mul_463: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1165, add_291);  convert_element_type_1165 = add_291 = None
        convert_element_type_1167: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_463, torch.bfloat16);  mul_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_2373: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1167, [8192, 3072]);  convert_element_type_1167 = None
        mm_170: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2373, permute_2054);  permute_2054 = None
        permute_2055: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_2373, [1, 0])
        mm_171: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2055, view_226);  permute_2055 = view_226 = None
        sum_168: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2373, [0], True, dtype = torch.float32);  view_2373 = None
        view_2374: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [3072]);  sum_168 = None
        convert_element_type_1172: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2374, torch.bfloat16);  view_2374 = None
        view_2375: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_170, [8, 1024, 768]);  mm_170 = None
        convert_element_type_1173: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2375, torch.float32);  view_2375 = None
        add_292: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_453, convert_element_type_1173);  mul_453 = convert_element_type_1173 = None
        convert_element_type_1174: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        convert_element_type_1175: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1172, torch.float32);  convert_element_type_1172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_465: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_292, primals_28);  primals_28 = None
        mul_466: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_465, 768)
        sum_169: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_465, [2], True)
        mul_467: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_465, mul_19);  mul_465 = None
        sum_170: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True);  mul_467 = None
        mul_468: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_19, sum_170);  sum_170 = None
        sub_160: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_466, sum_169);  mul_466 = sum_169 = None
        sub_161: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_160, mul_468);  sub_160 = mul_468 = None
        mul_469: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_151, sub_161);  div_151 = sub_161 = None
        mul_470: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_292, mul_19);  mul_19 = None
        sum_171: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_470, [0, 1]);  mul_470 = None
        sum_172: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_292, [0, 1]);  add_292 = None
        convert_element_type_1176: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_469, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1177: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_4, torch.bfloat16);  gt_4 = None
        mul_471: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1177, 1.1111111111111112);  convert_element_type_1177 = None
        mul_472: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1176, mul_471);  convert_element_type_1176 = mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_173: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1], True, dtype = torch.float32)
        view_2376: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_173, [768]);  sum_173 = None
        convert_element_type_1178: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2376, torch.bfloat16);  view_2376 = None
        view_2377: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_472, [8192, 768]);  mul_472 = None
        permute_2058: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2377, [1, 0])
        mm_172: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2058, view_224);  permute_2058 = view_224 = None
        mm_173: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2377, permute_2060);  view_2377 = permute_2060 = None
        view_2378: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_173, [8, 1024, 768]);  mm_173 = None
        convert_element_type_1183: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None
        convert_element_type_1184: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1178, torch.float32);  convert_element_type_1178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_2062: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_2378, [1, 0, 2]);  view_2378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_2379: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2062, [1024, 8, 12, 64]);  permute_2062 = None
        permute_2063: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2379, [1, 0, 2, 3]);  view_2379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_2064: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_2063, [0, 2, 1, 3]);  permute_2063 = None
        clone_360: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_2064, memory_format = torch.contiguous_format);  permute_2064 = None
        view_2380: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_360, [96, 4, 256, 64]);  clone_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_2381: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_2380, [96, 4, 256, 64, 1]);  view_2380 = None
        permute_2065: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2381, [0, 1, 2, 4, 3]);  view_2381 = None
        view_2382: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2065, [384, 256, 64]);  permute_2065 = None
        bmm_64: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_2066, view_2382);  permute_2066 = None
        bmm_65: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_2382, permute_2067);  view_2382 = permute_2067 = None
        view_2383: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_64, [96, 4, 768, 64, 1]);  bmm_64 = None
        view_2384: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_65, [96, 4, 256, 768, 1]);  bmm_65 = None
        squeeze_40: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2383, 4);  view_2383 = None
        squeeze_41: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2384, 4);  view_2384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_494: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_41, 3, 0, -1);  squeeze_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2385: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_494, [96, 4, 196864]);  slice_scatter_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_495: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_2385, 2, 0, -256);  view_2385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2386: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_495, [96, 4, 256, 770]);  slice_scatter_495 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_78: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2386, [0, -257]);  view_2386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_2387: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_40, [-1]);  squeeze_40 = None
        index_add_30: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_2387);  view_2387 = None
        as_strided_320: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_30, [96, 1536, 64], [98304, 64, 1], 0);  index_add_30 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_79: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_320, [0, 0, -256, -256]);  as_strided_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_2389: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_79, [8, 12, 1024, 64]);  constant_pad_nd_79 = None
        permute_2072: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_2389, [0, 2, 1, 3]);  view_2389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_2390: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_78, [8, 12, 1024, 513]);  constant_pad_nd_78 = None
        permute_2073: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2390, [0, 2, 1, 3]);  view_2390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_2074: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_2072, [1, 0, 2, 3]);  permute_2072 = None
        clone_362: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_2074, memory_format = torch.contiguous_format);  permute_2074 = None
        view_2391: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_362, [1024, 8, 768]);  clone_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_1189: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_473: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1189, 1.1111111111111112);  convert_element_type_1189 = None
        mul_474: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_2073, mul_473);  permute_2073 = mul_473 = None
        clone_363: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_474, memory_format = torch.contiguous_format);  mul_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_1190: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_363, torch.float32);  clone_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_126: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_1190);  convert_element_type_1190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_65: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_184, torch.float32);  permute_184 = None
        clone_23: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_65, memory_format = torch.contiguous_format);  convert_element_type_65 = None
        sub_12: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_23, amax_1);  clone_23 = amax_1 = None
        exp_1: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        div_17: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        mul_475: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_126, div_17);  where_126 = None
        sum_174: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_475, [-1], True)
        neg_10: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_10: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_10, sum_174, mul_475);  neg_10 = sum_174 = mul_475 = None
        convert_element_type_1191: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_2075: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1191, [0, 2, 1, 3]);  convert_element_type_1191 = None
        clone_364: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_2075, memory_format = torch.contiguous_format);  permute_2075 = None
        view_2392: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_364, [96, 4, 256, 513]);  clone_364 = None
        view_2395: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2392, [8, 12, 1024, 513]);  view_2392 = None
        permute_2077: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2395, [0, 2, 1, 3]);  view_2395 = None
        clone_365: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_2077, memory_format = torch.contiguous_format)
        copy_275: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_2077, clone_365);  permute_2077 = clone_365 = None
        permute_2078: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_275, [0, 2, 1, 3]);  copy_275 = None
        view_2397: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2078, [96, 4, 256, 513]);  permute_2078 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2403: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2397, [8, 12, 1024, 513]);  view_2397 = None
        permute_2083: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2403, [0, 2, 1, 3]);  view_2403 = None
        slice_2012: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_2083, 1, -256, 9223372036854775807)
        slice_2013: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2012, 3, -257, 9223372036854775807)
        clone_366: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_2013, memory_format = torch.contiguous_format)
        copy_277: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_2013, full_default_52);  slice_2013 = None
        slice_scatter_496: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_2012, copy_277, 3, -257, 9223372036854775807);  slice_2012 = copy_277 = None
        slice_scatter_497: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_2083, slice_scatter_496, 1, -256, 9223372036854775807);  permute_2083 = slice_scatter_496 = None
        permute_2085: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_497, [0, 2, 1, 3]);  slice_scatter_497 = None
        view_2405: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2085, [96, 4, 256, 513]);  permute_2085 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_127: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_366);  clone_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_498: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_127, 3, -257, 9223372036854775807);  where_127 = None
        slice_scatter_499: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_498, 1, -256, 9223372036854775807);  slice_scatter_498 = None
        permute_2087: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_499, [0, 2, 1, 3]);  slice_scatter_499 = None
        clone_367: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_2087, memory_format = torch.contiguous_format);  permute_2087 = None
        view_2407: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_367, [96, 4, 256, 513]);  clone_367 = None
        add_293: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2405, view_2407);  view_2405 = view_2407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2412: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_293, [8, 12, 1024, 513]);  add_293 = None
        permute_2091: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2412, [0, 2, 1, 3]);  view_2412 = None
        slice_2020: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_2091, 1, 0, 256)
        slice_2021: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2020, 3, 0, 257)
        clone_368: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_2021, memory_format = torch.contiguous_format)
        copy_279: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_2021, full_default_52);  slice_2021 = None
        slice_scatter_500: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_2020, copy_279, 3, 0, 257);  slice_2020 = copy_279 = None
        slice_scatter_501: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_2091, slice_scatter_500, 1, 0, 256);  permute_2091 = slice_scatter_500 = None
        permute_2093: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_501, [0, 2, 1, 3]);  slice_scatter_501 = None
        view_2414: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2093, [96, 4, 256, 513]);  permute_2093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_128: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_368);  clone_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_502: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_128, 3, 0, 257);  where_128 = None
        slice_scatter_503: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_502, 1, 0, 256);  slice_scatter_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_2095: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_503, [0, 2, 1, 3]);  slice_scatter_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_369: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_2095, memory_format = torch.contiguous_format);  permute_2095 = None
        view_2416: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_369, [96, 4, 256, 513]);  clone_369 = None
        add_294: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2414, view_2416);  view_2414 = view_2416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_485: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_294, 1, 0)
        slice_2028: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_485, 1, 1, 256)
        slice_2029: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2028, 2, 1, 256)
        clone_370: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_2029, memory_format = torch.contiguous_format)
        copy_281: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_2029, full_default_60);  slice_2029 = None
        slice_scatter_504: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_2028, copy_281, 2, 1, 256);  slice_2028 = copy_281 = None
        slice_scatter_505: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_485, slice_scatter_504, 1, 1, 256);  select_485 = slice_scatter_504 = None
        select_scatter_88: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_294, slice_scatter_505, 1, 0);  add_294 = slice_scatter_505 = None
        slice_scatter_506: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_370, 2, -255, 9223372036854775807);  clone_370 = None
        slice_scatter_507: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_506, 1, 0, 255);  slice_scatter_506 = None
        select_scatter_89: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_507, 1, 0);  slice_scatter_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2036: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_88, 1, 1, 9223372036854775807)
        slice_2037: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2036, 3, 0, 256)
        clone_371: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_2037, memory_format = torch.contiguous_format)
        copy_283: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_2037, full_default_64);  slice_2037 = None
        slice_scatter_508: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_2036, copy_283, 3, 0, 256);  slice_2036 = copy_283 = None
        slice_scatter_509: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_88, slice_scatter_508, 1, 1, 9223372036854775807);  select_scatter_88 = slice_scatter_508 = None
        slice_scatter_510: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_371, 3, 257, 9223372036854775807);  clone_371 = None
        slice_scatter_511: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_510, 2, -257, -1);  slice_scatter_510 = None
        add_295: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_89, slice_scatter_511);  select_scatter_89 = slice_scatter_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_490: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_509, 1, -1)
        slice_2042: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_490, 2, 256, 9223372036854775807)
        clone_372: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_2042, memory_format = torch.contiguous_format)
        copy_285: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_2042, full_default_67);  slice_2042 = None
        slice_scatter_512: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_490, copy_285, 2, 256, 9223372036854775807);  select_490 = copy_285 = None
        select_scatter_90: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_509, slice_scatter_512, 1, -1);  slice_scatter_509 = slice_scatter_512 = None
        slice_scatter_513: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_372, 2, 0, 257);  clone_372 = None
        slice_scatter_514: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_513, 1, 256, 9223372036854775807);  slice_scatter_513 = None
        select_scatter_91: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_514, 1, -1);  slice_scatter_514 = None
        add_296: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_295, select_scatter_91);  add_295 = select_scatter_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2047: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_90, 1, 0, -1);  select_scatter_90 = None
        slice_2048: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2047, 3, 256, 9223372036854775807);  slice_2047 = None
        clone_373: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_2048, memory_format = torch.contiguous_format);  slice_2048 = None
        slice_scatter_515: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_373, 3, 0, 257);  clone_373 = None
        slice_scatter_516: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_515, 2, 0, 256);  slice_scatter_515 = None
        add_297: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_296, slice_scatter_516);  add_296 = slice_scatter_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2417: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_297, [96, 3, 513, 512]);  add_297 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_80: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2417, [0, 0, 0, -1]);  view_2417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2418: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_80, [96, 3, 512, 512, 1]);  constant_pad_nd_80 = None
        permute_2096: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2418, [0, 1, 2, 4, 3]);  view_2418 = None
        view_2419: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2096, [288, 512, 512]);  permute_2096 = None
        bmm_66: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_2097, view_2419);  permute_2097 = None
        bmm_67: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_2419, permute_2098);  view_2419 = permute_2098 = None
        view_2420: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_66, [96, 3, 64, 512, 1]);  bmm_66 = None
        permute_2099: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_2420, [0, 1, 4, 3, 2]);  view_2420 = None
        view_2421: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_67, [96, 3, 512, 64, 1]);  bmm_67 = None
        permute_2101: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_2099, [0, 1, 3, 4, 2]);  permute_2099 = None
        squeeze_42: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_2101, 4);  permute_2101 = None
        squeeze_43: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2421, 4);  view_2421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_374: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_42, memory_format = torch.contiguous_format);  squeeze_42 = None
        view_2422: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_374, [9437184]);  clone_374 = None
        index_add_31: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2422);  view_2422 = None
        view_2425: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_43, [-1]);  squeeze_43 = None
        index_add_32: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2425);  view_2425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_335: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_32, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_32 = None
        view_2446: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_335, [96, 1024, 64]);  as_strided_335 = None
        view_2447: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2446, [8, 12, 1024, 64]);  view_2446 = None
        permute_2113: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2447, [0, 2, 1, 3]);  view_2447 = None
        permute_2114: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_2113, [1, 0, 2, 3]);  permute_2113 = None
        view_2448: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2114, [1024, 8, 768]);  permute_2114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_152: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_2448, 8.0);  view_2448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_175: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2391, [0, 1], True, dtype = torch.float32)
        view_2449: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_175, [768]);  sum_175 = None
        convert_element_type_1196: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2449, torch.bfloat16);  view_2449 = None
        view_2450: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2391, [8192, 768]);  view_2391 = None
        permute_2115: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2450, [1, 0])
        mm_174: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2115, view_115);  permute_2115 = None
        mm_175: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2450, permute_2117);  view_2450 = permute_2117 = None
        view_2451: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_175, [1024, 8, 768]);  mm_175 = None
        convert_element_type_1201: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2451, torch.float32);  view_2451 = None
        convert_element_type_1202: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        convert_element_type_1203: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1196, torch.float32);  convert_element_type_1196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_336: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_31, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_31 = None
        view_2452: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_336, [96, 1024, 64]);  as_strided_336 = None
        view_2453: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2452, [8, 12, 1024, 64]);  view_2452 = None
        permute_2119: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2453, [0, 2, 1, 3]);  view_2453 = None
        permute_2120: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_2119, [1, 0, 2, 3]);  permute_2119 = None
        view_2454: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2120, [1024, 8, 768]);  permute_2120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_176: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2454, [0, 1], True, dtype = torch.float32)
        view_2455: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_176, [768]);  sum_176 = None
        convert_element_type_1204: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2455, torch.bfloat16);  view_2455 = None
        view_2460: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2454, [8192, 768]);  view_2454 = None
        permute_2126: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2460, [1, 0])
        mm_176: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2126, view_115);  permute_2126 = None
        mm_177: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2460, permute_2128);  view_2460 = permute_2128 = None
        view_2465: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_177, [1024, 8, 768]);  mm_177 = None
        convert_element_type_1209: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2465, torch.float32);  view_2465 = None
        add_298: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1201, convert_element_type_1209);  convert_element_type_1201 = convert_element_type_1209 = None
        convert_element_type_1210: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        convert_element_type_1211: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1204, torch.float32);  convert_element_type_1204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_177: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_152, [0, 1], True, dtype = torch.float32)
        view_2466: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_177, [768]);  sum_177 = None
        convert_element_type_1212: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2466, torch.bfloat16);  view_2466 = None
        view_2467: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_152, [8192, 768]);  div_152 = None
        permute_2130: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2467, [1, 0])
        mm_178: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2130, view_115);  permute_2130 = view_115 = None
        mm_179: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2467, permute_2132);  view_2467 = permute_2132 = None
        view_2468: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_179, [1024, 8, 768]);  mm_179 = None
        convert_element_type_1217: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2468, torch.float32);  view_2468 = None
        add_299: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_298, convert_element_type_1217);  add_298 = convert_element_type_1217 = None
        convert_element_type_1218: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        convert_element_type_1219: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1212, torch.float32);  convert_element_type_1212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_2134: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_299, [1, 0, 2]);  add_299 = None
        add_300: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_469, permute_2134);  mul_469 = permute_2134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1129 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_477: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_300, primals_18);  primals_18 = None
        mul_478: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_477, 768)
        sum_178: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True)
        mul_479: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_477, mul_12);  mul_477 = None
        sum_179: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
        mul_480: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_12, sum_179);  sum_179 = None
        sub_163: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_478, sum_178);  mul_478 = sum_178 = None
        sub_164: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_163, mul_480);  sub_163 = mul_480 = None
        mul_481: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_153, sub_164);  div_153 = sub_164 = None
        mul_482: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_300, mul_12);  mul_12 = None
        sum_180: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1]);  mul_482 = None
        sum_181: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_300, [0, 1]);  add_300 = None
        convert_element_type_1220: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_481, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1128 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1221: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_483: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1221, 1.1111111111111112);  convert_element_type_1221 = None
        mul_484: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1220, mul_483);  convert_element_type_1220 = mul_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1127 in forward, code: hidden_states = self.dense(hidden_states)
        view_2469: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_484, [8192, 768]);  mul_484 = None
        mm_180: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_2469, permute_2135);  permute_2135 = None
        permute_2136: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2469, [1, 0])
        mm_181: "bf16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_2136, view_113);  permute_2136 = view_113 = None
        sum_182: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2469, [0], True, dtype = torch.float32);  view_2469 = None
        view_2470: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_182, [768]);  sum_182 = None
        convert_element_type_1226: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2470, torch.bfloat16);  view_2470 = None
        view_2471: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_180, [8, 1024, 3072]);  mm_180 = None
        convert_element_type_1227: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None
        convert_element_type_1228: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1226, torch.float32);  convert_element_type_1226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1229: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2471, torch.float32);  view_2471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_112: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [8, 1024, 3072]);  addmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_36: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_112, torch.float32);  view_112 = None
        mul_8: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_36, 0.7071067811865476)
        erf: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_11: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_486: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_11, 0.5);  add_11 = None
        mul_487: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_36, convert_element_type_36)
        mul_488: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_487, -0.5);  mul_487 = None
        exp_23: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.exp.default(mul_488);  mul_488 = None
        mul_489: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
        mul_490: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_36, mul_489);  convert_element_type_36 = mul_489 = None
        add_302: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_486, mul_490);  mul_486 = mul_490 = None
        mul_491: "f32[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1229, add_302);  convert_element_type_1229 = add_302 = None
        convert_element_type_1231: "bf16[8, 1024, 3072][3145728, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_491, torch.bfloat16);  mul_491 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1113 in forward, code: hidden_states = self.dense(hidden_states)
        view_2472: "bf16[8192, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1231, [8192, 3072]);  convert_element_type_1231 = None
        mm_182: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2472, permute_2139);  permute_2139 = None
        permute_2140: "bf16[3072, 8192][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_2472, [1, 0])
        mm_183: "bf16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2140, view_111);  permute_2140 = view_111 = None
        sum_183: "f32[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2472, [0], True, dtype = torch.float32);  view_2472 = None
        view_2473: "f32[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [3072]);  sum_183 = None
        convert_element_type_1236: "bf16[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2473, torch.bfloat16);  view_2473 = None
        view_2474: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_182, [8, 1024, 768]);  mm_182 = None
        convert_element_type_1237: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2474, torch.float32);  view_2474 = None
        add_303: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_481, convert_element_type_1237);  mul_481 = convert_element_type_1237 = None
        convert_element_type_1238: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_1239: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1236, torch.float32);  convert_element_type_1236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1070 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_493: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_303, primals_12);  primals_12 = None
        mul_494: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_493, 768)
        sum_184: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_493, [2], True)
        mul_495: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_493, mul_5);  mul_493 = None
        sum_185: "f32[8, 1024, 1][1024, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True);  mul_495 = None
        mul_496: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_5, sum_185);  sum_185 = None
        sub_166: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_494, sum_184);  mul_494 = sum_184 = None
        sub_167: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_166, mul_496);  sub_166 = mul_496 = None
        mul_497: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_154, sub_167);  div_154 = sub_167 = None
        mul_498: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_303, mul_5);  mul_5 = None
        sum_186: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_498, [0, 1]);  mul_498 = None
        sum_187: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_303, [0, 1]);  add_303 = None
        convert_element_type_1240: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_497, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1069 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1241: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_1, torch.bfloat16);  gt_1 = None
        mul_499: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1241, 1.1111111111111112);  convert_element_type_1241 = None
        mul_500: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1240, mul_499);  convert_element_type_1240 = mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:1068 in forward, code: hidden_states = self.dense(hidden_states)
        sum_188: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1], True, dtype = torch.float32)
        view_2475: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_188, [768]);  sum_188 = None
        convert_element_type_1242: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2475, torch.bfloat16);  view_2475 = None
        view_2476: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(mul_500, [8192, 768]);  mul_500 = None
        permute_2143: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2476, [1, 0])
        mm_184: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2143, view_109);  permute_2143 = view_109 = None
        mm_185: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2476, permute_2145);  view_2476 = permute_2145 = None
        view_2477: "bf16[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_185, [8, 1024, 768]);  mm_185 = None
        convert_element_type_1247: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        convert_element_type_1248: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1242, torch.float32);  convert_element_type_1242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:634 in forward, code: outputs = (attn_output.transpose(0, 1),)
        permute_2147: "bf16[1024, 8, 768][768, 786432, 1]cuda:0" = torch.ops.aten.permute.default(view_2477, [1, 0, 2]);  view_2477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:606 in forward, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        view_2478: "bf16[1024, 8, 12, 64][768, 786432, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2147, [1024, 8, 12, 64]);  permute_2147 = None
        permute_2148: "bf16[8, 1024, 12, 64][786432, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2478, [1, 0, 2, 3]);  view_2478 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:866 in _sliding_chunks_matmul_attn_probs_value, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
        permute_2149: "bf16[8, 12, 1024, 64][786432, 64, 768, 1]cuda:0" = torch.ops.aten.permute.default(permute_2148, [0, 2, 1, 3]);  permute_2148 = None
        clone_379: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_2149, memory_format = torch.contiguous_format);  permute_2149 = None
        view_2479: "bf16[96, 4, 256, 64][65536, 16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(clone_379, [96, 4, 256, 64]);  clone_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:865 in _sliding_chunks_matmul_attn_probs_value, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
        view_2480: "bf16[96, 4, 256, 64, 1][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(view_2479, [96, 4, 256, 64, 1]);  view_2479 = None
        permute_2150: "bf16[96, 4, 256, 1, 64][65536, 16384, 64, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2480, [0, 1, 2, 4, 3]);  view_2480 = None
        view_2481: "bf16[384, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2150, [384, 256, 64]);  permute_2150 = None
        bmm_68: "bf16[384, 768, 64][49152, 64, 1]cuda:0" = torch.ops.aten.bmm.default(permute_2151, view_2481);  permute_2151 = None
        bmm_69: "bf16[384, 256, 768][196608, 768, 1]cuda:0" = torch.ops.aten.bmm.default(view_2481, permute_2152);  view_2481 = permute_2152 = None
        view_2482: "bf16[96, 4, 768, 64, 1][196608, 49152, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [96, 4, 768, 64, 1]);  bmm_68 = None
        view_2483: "bf16[96, 4, 256, 768, 1][786432, 196608, 768, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [96, 4, 256, 768, 1]);  bmm_69 = None
        squeeze_44: "bf16[96, 4, 768, 64][196608, 49152, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2482, 4);  view_2482 = None
        squeeze_45: "bf16[96, 4, 256, 768][786432, 196608, 768, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2483, 4);  view_2483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:698 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        slice_scatter_517: "bf16[96, 4, 256, 769][787456, 196864, 769, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_48, squeeze_45, 3, 0, -1);  full_default_48 = squeeze_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:695 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2484: "bf16[96, 4, 196864][787456, 196864, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_517, [96, 4, 196864]);  slice_scatter_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:692 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states[
        slice_scatter_518: "bf16[96, 4, 197120][788480, 197120, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_49, view_2484, 2, 0, -256);  full_default_49 = view_2484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:689 in _pad_and_diagonalize, code: chunked_hidden_states = chunked_hidden_states.view(
        view_2485: "bf16[96, 4, 256, 770][788480, 197120, 770, 1]cuda:0" = torch.ops.aten.reshape.default(slice_scatter_518, [96, 4, 256, 770]);  slice_scatter_518 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_81: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2485, [0, -257]);  view_2485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:861 in _sliding_chunks_matmul_attn_probs_value, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        view_2486: "bf16[18874368][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_44, [-1]);  squeeze_44 = None
        index_add_33: "bf16[9437184][1]cuda:0" = torch.ops.aten.index_add.default(full_default_50, 0, view_1398, view_2486);  full_default_50 = view_1398 = view_2486 = None
        as_strided_341: "bf16[96, 1536, 64][98304, 64, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_33, [96, 1536, 64], [98304, 64, 1], 0);  index_add_33 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_82: "bf16[96, 1024, 64][65536, 64, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(as_strided_341, [0, 0, -256, -256]);  as_strided_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:847 in _sliding_chunks_matmul_attn_probs_value, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        view_2488: "bf16[8, 12, 1024, 64][786432, 65536, 64, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_82, [8, 12, 1024, 64]);  constant_pad_nd_82 = None
        permute_2157: "bf16[8, 1024, 12, 64][786432, 64, 65536, 1]cuda:0" = torch.ops.aten.permute.default(view_2488, [0, 2, 1, 3]);  view_2488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:839 in _sliding_chunks_matmul_attn_probs_value, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
        view_2489: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_81, [8, 12, 1024, 513]);  constant_pad_nd_81 = None
        permute_2158: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2489, [0, 2, 1, 3]);  view_2489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:587 in forward, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        permute_2159: "bf16[1024, 8, 12, 64][64, 786432, 65536, 1]cuda:0" = torch.ops.aten.permute.default(permute_2157, [1, 0, 2, 3]);  permute_2157 = None
        clone_381: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_2159, memory_format = torch.contiguous_format);  permute_2159 = None
        view_2490: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_381, [1024, 8, 768]);  clone_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:585 in forward, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
        convert_element_type_1253: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.bfloat16);  gt = None
        mul_501: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1253, 1.1111111111111112);  convert_element_type_1253 = None
        mul_502: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_2158, mul_501);  permute_2158 = mul_501 = None
        clone_382: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(mul_502, memory_format = torch.contiguous_format);  mul_502 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:579 in forward, code: attn_probs = attn_probs.type_as(attn_scores)
        convert_element_type_1254: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_382, torch.float32);  clone_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:578 in forward, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        where_129: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_18, full_default_3, convert_element_type_1254);  unsqueeze_18 = full_default_3 = convert_element_type_1254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        permute_52: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_10, [0, 2, 1, 3]);  slice_scatter_10 = None
        permute_78: "bf16[8, 1, 1024, 513][525312, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_21, [0, 2, 1, 3]);  slice_scatter_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_81: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [0, 2, 1, 3]);  permute_52 = None
        permute_82: "bf16[8, 1024, 1, 513][525312, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_78, [0, 2, 1, 3]);  permute_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        add_5: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_81, permute_82);  permute_81 = permute_82 = None
        permute_83: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(add_5, [0, 2, 1, 3]);  add_5 = None
        permute_84: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(permute_83, [0, 2, 1, 3]);  permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:573 in forward, code: attn_probs = nn.functional.softmax(
        convert_element_type_22: "f32[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_84, torch.float32);  permute_84 = None
        clone_9: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_22, memory_format = torch.contiguous_format);  convert_element_type_22 = None
        sub_4: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.sub.Tensor(clone_9, amax);  clone_9 = amax = None
        exp: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div_7: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_503: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_129, div_7);  where_129 = None
        sum_189: "f32[8, 1024, 12, 1][12288, 12, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_503, [-1], True)
        neg_11: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_11: "f32[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.fma.default(neg_11, sum_189, mul_503);  neg_11 = sum_189 = mul_503 = None
        convert_element_type_1255: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:535 in forward, code: attn_scores += diagonal_mask
        permute_2160: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1255, [0, 2, 1, 3]);  convert_element_type_1255 = None
        clone_383: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_2160, memory_format = torch.contiguous_format);  permute_2160 = None
        view_2491: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_383, [96, 4, 256, 513]);  clone_383 = None
        view_2494: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2491, [8, 12, 1024, 513]);  view_2491 = None
        permute_2162: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2494, [0, 2, 1, 3]);  view_2494 = None
        clone_384: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_2162, memory_format = torch.contiguous_format)
        copy_288: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(permute_2162, clone_384);  permute_2162 = clone_384 = None
        permute_2163: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(copy_288, [0, 2, 1, 3]);  copy_288 = None
        view_2496: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2163, [96, 4, 256, 513]);  permute_2163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:754 in _mask_invalid_locations, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
        view_2502: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(view_2496, [8, 12, 1024, 513]);  view_2496 = None
        permute_2168: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2502, [0, 2, 1, 3]);  view_2502 = None
        slice_2052: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_2168, 1, -256, 9223372036854775807)
        slice_2053: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2052, 3, -257, 9223372036854775807)
        clone_385: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_2053, memory_format = torch.contiguous_format)
        copy_290: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_2053, full_default_52);  slice_2053 = None
        slice_scatter_519: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_2052, copy_290, 3, -257, 9223372036854775807);  slice_2052 = copy_290 = None
        slice_scatter_520: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_2168, slice_scatter_519, 1, -256, 9223372036854775807);  permute_2168 = slice_scatter_519 = None
        permute_2170: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_520, [0, 2, 1, 3]);  slice_scatter_520 = None
        view_2504: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2170, [96, 4, 256, 513]);  permute_2170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:756 in _mask_invalid_locations, code: ).where(ending_mask.bool(), ending_input)
        where_130: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_18, full_default_53, clone_385);  convert_element_type_18 = clone_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:752 in _mask_invalid_locations, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
        slice_scatter_521: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_130, 3, -257, 9223372036854775807);  where_130 = None
        slice_scatter_522: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_521, 1, -256, 9223372036854775807);  slice_scatter_521 = None
        permute_2172: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_522, [0, 2, 1, 3]);  slice_scatter_522 = None
        clone_386: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_2172, memory_format = torch.contiguous_format);  permute_2172 = None
        view_2506: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_386, [96, 4, 256, 513]);  clone_386 = None
        add_304: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2504, view_2506);  view_2504 = view_2506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:749 in _mask_invalid_locations, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
        view_2511: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.reshape.default(add_304, [8, 12, 1024, 513]);  add_304 = None
        permute_2176: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.permute.default(view_2511, [0, 2, 1, 3]);  view_2511 = None
        slice_2060: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(permute_2176, 1, 0, 256)
        slice_2061: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2060, 3, 0, 257)
        clone_387: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_2061, memory_format = torch.contiguous_format)
        copy_292: "bf16[8, 256, 12, 257][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.copy.default(slice_2061, full_default_52);  slice_2061 = full_default_52 = None
        slice_scatter_523: "bf16[8, 256, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_2060, copy_292, 3, 0, 257);  slice_2060 = copy_292 = None
        slice_scatter_524: "bf16[8, 1024, 12, 513][6303744, 513, 525312, 1]cuda:0" = torch.ops.aten.slice_scatter.default(permute_2176, slice_scatter_523, 1, 0, 256);  permute_2176 = slice_scatter_523 = None
        permute_2178: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_524, [0, 2, 1, 3]);  slice_scatter_524 = None
        view_2513: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2178, [96, 4, 256, 513]);  permute_2178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:751 in _mask_invalid_locations, code: ).where(beginning_mask.bool(), beginning_input)
        where_131: "bf16[8, 256, 12, 257][789504, 3084, 257, 1]cuda:0" = torch.ops.aten.where.self(convert_element_type_17, full_default_53, clone_387);  convert_element_type_17 = full_default_53 = clone_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:747 in _mask_invalid_locations, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
        slice_scatter_525: "bf16[8, 256, 12, 513][1575936, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_54, where_131, 3, 0, 257);  full_default_54 = where_131 = None
        slice_scatter_526: "bf16[8, 1024, 12, 513][6303744, 6156, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_55, slice_scatter_525, 1, 0, 256);  full_default_55 = slice_scatter_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:819 in _sliding_chunks_query_key_matmul, code: ).transpose(2, 1)
        permute_2180: "bf16[8, 12, 1024, 513][6303744, 513, 6156, 1]cuda:0" = torch.ops.aten.permute.default(slice_scatter_526, [0, 2, 1, 3]);  slice_scatter_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:817 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores = diagonal_attention_scores.view(
        clone_388: "bf16[8, 12, 1024, 513][6303744, 525312, 513, 1]cuda:0" = torch.ops.aten.clone.default(permute_2180, memory_format = torch.contiguous_format);  permute_2180 = None
        view_2515: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.reshape.default(clone_388, [96, 4, 256, 513]);  clone_388 = None
        add_305: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(view_2513, view_2515);  view_2513 = view_2515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:812 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
        select_496: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(add_305, 1, 0)
        slice_2068: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_496, 1, 1, 256)
        slice_2069: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2068, 2, 1, 256)
        clone_389: "bf16[96, 255, 255][65025, 255, 1]cuda:0" = torch.ops.aten.clone.default(slice_2069, memory_format = torch.contiguous_format)
        copy_294: "bf16[96, 255, 255][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_2069, full_default_60);  slice_2069 = full_default_60 = None
        slice_scatter_527: "bf16[96, 255, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_2068, copy_294, 2, 1, 256);  slice_2068 = copy_294 = None
        slice_scatter_528: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_496, slice_scatter_527, 1, 1, 256);  select_496 = slice_scatter_527 = None
        select_scatter_92: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(add_305, slice_scatter_528, 1, 0);  add_305 = slice_scatter_528 = None
        slice_scatter_529: "bf16[96, 255, 513][130815, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_61, clone_389, 2, -255, 9223372036854775807);  full_default_61 = clone_389 = None
        slice_scatter_530: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_529, 1, 0, 255);  slice_scatter_529 = None
        select_scatter_93: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_530, 1, 0);  slice_scatter_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:808 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
        slice_2076: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_92, 1, 1, 9223372036854775807)
        slice_2077: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2076, 3, 0, 256)
        clone_390: "bf16[96, 3, 256, 256][196608, 65536, 256, 1]cuda:0" = torch.ops.aten.clone.default(slice_2077, memory_format = torch.contiguous_format)
        copy_296: "bf16[96, 3, 256, 256][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_2077, full_default_64);  slice_2077 = full_default_64 = None
        slice_scatter_531: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(slice_2076, copy_296, 3, 0, 256);  slice_2076 = copy_296 = None
        slice_scatter_532: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_scatter_92, slice_scatter_531, 1, 1, 9223372036854775807);  select_scatter_92 = slice_scatter_531 = None
        slice_scatter_533: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_390, 3, 257, 9223372036854775807);  clone_390 = None
        slice_scatter_534: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_533, 2, -257, -1);  slice_scatter_533 = None
        add_306: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_93, slice_scatter_534);  select_scatter_93 = slice_scatter_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:804 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
        select_501: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.select.int(slice_scatter_532, 1, -1)
        slice_2082: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_501, 2, 256, 9223372036854775807)
        clone_391: "bf16[96, 256, 257][65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_2082, memory_format = torch.contiguous_format)
        copy_298: "bf16[96, 256, 257][525312, 513, 1]cuda:0" = torch.ops.aten.copy.default(slice_2082, full_default_67);  slice_2082 = full_default_67 = None
        slice_scatter_535: "bf16[96, 256, 513][525312, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(select_501, copy_298, 2, 256, 9223372036854775807);  select_501 = copy_298 = None
        select_scatter_94: "bf16[96, 4, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(slice_scatter_532, slice_scatter_535, 1, -1);  slice_scatter_532 = slice_scatter_535 = None
        slice_scatter_536: "bf16[96, 256, 513][131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_68, clone_391, 2, 0, 257);  full_default_68 = clone_391 = None
        slice_scatter_537: "bf16[96, 512, 513][262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_62, slice_scatter_536, 1, 256, 9223372036854775807);  full_default_62 = slice_scatter_536 = None
        select_scatter_95: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_63, slice_scatter_537, 1, -1);  slice_scatter_537 = None
        add_307: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_306, select_scatter_95);  add_306 = select_scatter_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:801 in _sliding_chunks_query_key_matmul, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
        slice_2087: "bf16[96, 3, 256, 513][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(select_scatter_94, 1, 0, -1);  select_scatter_94 = None
        slice_2088: "bf16[96, 3, 256, 257][525312, 131328, 513, 1]cuda:0" = torch.ops.aten.slice.Tensor(slice_2087, 3, 256, 9223372036854775807);  slice_2087 = None
        clone_392: "bf16[96, 3, 256, 257][197376, 65792, 257, 1]cuda:0" = torch.ops.aten.clone.default(slice_2088, memory_format = torch.contiguous_format);  slice_2088 = None
        slice_scatter_538: "bf16[96, 3, 256, 513][393984, 131328, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_65, clone_392, 3, 0, 257);  full_default_65 = clone_392 = None
        slice_scatter_539: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.slice_scatter.default(full_default_63, slice_scatter_538, 2, 0, 256);  full_default_63 = slice_scatter_538 = None
        add_308: "bf16[96, 3, 512, 513][787968, 262656, 513, 1]cuda:0" = torch.ops.aten.add.Tensor(add_307, slice_scatter_539);  add_307 = slice_scatter_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:647 in _pad_and_transpose_last_two_dims, code: hidden_states_padded = hidden_states_padded.view(
        view_2516: "bf16[96, 3, 513, 512][787968, 262656, 512, 1]cuda:0" = torch.ops.aten.reshape.default(add_308, [96, 3, 513, 512]);  add_308 = None

        # File: /tmp/pytorch-work/torch/nn/functional.py:5737 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd_83: "bf16[96, 3, 512, 512][786432, 262144, 512, 1]cuda:0" = torch.ops.aten.constant_pad_nd.default(view_2516, [0, 0, 0, -1]);  view_2516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:783 in _sliding_chunks_query_key_matmul, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
        view_2517: "bf16[96, 3, 512, 512, 1][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(constant_pad_nd_83, [96, 3, 512, 512, 1]);  constant_pad_nd_83 = None
        permute_2181: "bf16[96, 3, 512, 1, 512][786432, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.permute.default(view_2517, [0, 1, 2, 4, 3]);  view_2517 = None
        view_2518: "bf16[288, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2181, [288, 512, 512]);  permute_2181 = None
        bmm_70: "bf16[288, 64, 512][32768, 512, 1]cuda:0" = torch.ops.aten.bmm.default(permute_2182, view_2518);  permute_2182 = None
        bmm_71: "bf16[288, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_2518, permute_2183);  view_2518 = permute_2183 = None
        view_2519: "bf16[96, 3, 64, 512, 1][98304, 32768, 512, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [96, 3, 64, 512, 1]);  bmm_70 = None
        permute_2184: "bf16[96, 3, 1, 512, 64][98304, 32768, 1, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_2519, [0, 1, 4, 3, 2]);  view_2519 = None
        view_2520: "bf16[96, 3, 512, 64, 1][98304, 32768, 64, 1, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_71, [96, 3, 512, 64, 1]);  bmm_71 = None
        permute_2186: "bf16[96, 3, 512, 64, 1][98304, 32768, 1, 512, 1]cuda:0" = torch.ops.aten.permute.default(permute_2184, [0, 1, 3, 4, 2]);  permute_2184 = None
        squeeze_46: "bf16[96, 3, 512, 64][98304, 32768, 1, 512]cuda:0" = torch.ops.aten.squeeze.dim(permute_2186, 4);  permute_2186 = None
        squeeze_47: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_2520, 4);  view_2520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:718 in _chunk, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
        clone_393: "bf16[96, 3, 512, 64][98304, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_46, memory_format = torch.contiguous_format);  squeeze_46 = None
        view_2521: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(clone_393, [9437184]);  clone_393 = None
        index_add_34: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2521);  view_2521 = None
        view_2524: "bf16[9437184][1]cuda:0" = torch.ops.aten.reshape.default(squeeze_47, [-1]);  squeeze_47 = None
        index_add_35: "bf16[6291456][1]cuda:0" = torch.ops.aten.index_add.default(full_default_73, 0, view_1433, view_2524);  full_default_73 = view_1433 = view_2524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:515 in forward, code: query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_356: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_35, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_35 = None
        view_2545: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_356, [96, 1024, 64]);  as_strided_356 = None
        view_2546: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2545, [8, 12, 1024, 64]);  view_2545 = None
        permute_2198: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2546, [0, 2, 1, 3]);  view_2546 = None
        permute_2199: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_2198, [1, 0, 2, 3]);  permute_2198 = None
        view_2547: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2199, [1024, 8, 768]);  permute_2199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:513 in forward, code: query_vectors /= math.sqrt(self.head_dim)
        div_155: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.div.Tensor(view_2547, 8.0);  view_2547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:505 in forward, code: value_vectors = self.value(hidden_states)
        sum_190: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2490, [0, 1], True, dtype = torch.float32)
        view_2548: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_190, [768]);  sum_190 = None
        convert_element_type_1260: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2548, torch.bfloat16);  view_2548 = None
        view_2549: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2490, [8192, 768]);  view_2490 = None
        permute_2200: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2549, [1, 0])
        mm_186: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2200, view);  permute_2200 = None
        mm_187: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2549, permute_2202);  view_2549 = permute_2202 = None
        view_2550: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_187, [1024, 8, 768]);  mm_187 = None
        convert_element_type_1265: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2550, torch.float32);  view_2550 = None
        convert_element_type_1266: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        convert_element_type_1267: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1260, torch.float32);  convert_element_type_1260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:516 in forward, code: key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        as_strided_357: "bf16[96, 2, 512, 64][64, 3145728, 6144, 1]cuda:0" = torch.ops.aten.as_strided.default(index_add_34, [96, 2, 512, 64], [64, 3145728, 6144, 1], 0);  index_add_34 = None
        view_2551: "bf16[96, 1024, 64][64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(as_strided_357, [96, 1024, 64]);  as_strided_357 = None
        view_2552: "bf16[8, 12, 1024, 64][768, 64, 6144, 1]cuda:0" = torch.ops.aten.reshape.default(view_2551, [8, 12, 1024, 64]);  view_2551 = None
        permute_2204: "bf16[8, 1024, 12, 64][768, 6144, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_2552, [0, 2, 1, 3]);  view_2552 = None
        permute_2205: "bf16[1024, 8, 12, 64][6144, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(permute_2204, [1, 0, 2, 3]);  permute_2204 = None
        view_2553: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_2205, [1024, 8, 768]);  permute_2205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:504 in forward, code: key_vectors = self.key(hidden_states)
        sum_191: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_2553, [0, 1], True, dtype = torch.float32)
        view_2554: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_191, [768]);  sum_191 = None
        convert_element_type_1268: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2554, torch.bfloat16);  view_2554 = None
        view_2559: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(view_2553, [8192, 768]);  view_2553 = None
        permute_2211: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2559, [1, 0])
        mm_188: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2211, view);  permute_2211 = None
        mm_189: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2559, permute_2213);  view_2559 = permute_2213 = None
        view_2564: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_189, [1024, 8, 768]);  mm_189 = None
        convert_element_type_1273: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2564, torch.float32);  view_2564 = None
        add_309: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1265, convert_element_type_1273);  convert_element_type_1265 = convert_element_type_1273 = None
        convert_element_type_1274: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        convert_element_type_1275: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1268, torch.float32);  convert_element_type_1268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:503 in forward, code: query_vectors = self.query(hidden_states)
        sum_192: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(div_155, [0, 1], True, dtype = torch.float32)
        view_2565: "f32[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_192, [768]);  sum_192 = None
        convert_element_type_1276: "bf16[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2565, torch.bfloat16);  view_2565 = None
        view_2566: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(div_155, [8192, 768]);  div_155 = None
        permute_2215: "bf16[768, 8192][1, 768]cuda:0" = torch.ops.aten.permute.default(view_2566, [1, 0])
        mm_190: "bf16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_2215, view);  permute_2215 = view = None
        mm_191: "bf16[8192, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_2566, permute_2217);  view_2566 = permute_2217 = None
        view_2567: "bf16[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_191, [1024, 8, 768]);  mm_191 = None
        convert_element_type_1281: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_2567, torch.float32);  view_2567 = None
        add_310: "f32[1024, 8, 768][6144, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_309, convert_element_type_1281);  add_309 = convert_element_type_1281 = None
        convert_element_type_1282: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        convert_element_type_1283: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1276, torch.float32);  convert_element_type_1276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/longformer/modeling_longformer.py:500 in forward, code: hidden_states = hidden_states.transpose(0, 1)
        permute_2219: "f32[8, 1024, 768][768, 6144, 1]cuda:0" = torch.ops.aten.permute.default(add_310, [1, 0, 2]);  add_310 = None
        add_311: "f32[8, 1024, 768][786432, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_497, permute_2219);  mul_497 = permute_2219 = None
        return (add_311, convert_element_type_1282, convert_element_type_1283, convert_element_type_1274, convert_element_type_1275, convert_element_type_1266, convert_element_type_1267, None, None, convert_element_type_1247, convert_element_type_1248, sum_186, sum_187, convert_element_type_1238, convert_element_type_1239, convert_element_type_1227, convert_element_type_1228, sum_180, sum_181, convert_element_type_1218, convert_element_type_1219, convert_element_type_1210, convert_element_type_1211, convert_element_type_1202, convert_element_type_1203, convert_element_type_1183, convert_element_type_1184, sum_171, sum_172, convert_element_type_1174, convert_element_type_1175, convert_element_type_1163, convert_element_type_1164, sum_165, sum_166, convert_element_type_1154, convert_element_type_1155, convert_element_type_1146, convert_element_type_1147, convert_element_type_1138, convert_element_type_1139, convert_element_type_1119, convert_element_type_1120, sum_156, sum_157, convert_element_type_1110, convert_element_type_1111, convert_element_type_1099, convert_element_type_1100, sum_150, sum_151, convert_element_type_1090, convert_element_type_1091, convert_element_type_1082, convert_element_type_1083, convert_element_type_1074, convert_element_type_1075, convert_element_type_1055, convert_element_type_1056, sum_141, sum_142, convert_element_type_1046, convert_element_type_1047, convert_element_type_1035, convert_element_type_1036, sum_135, sum_136, convert_element_type_1026, convert_element_type_1027, convert_element_type_1018, convert_element_type_1019, convert_element_type_1010, convert_element_type_1011, convert_element_type_991, convert_element_type_992, sum_126, sum_127, convert_element_type_982, convert_element_type_983, convert_element_type_971, convert_element_type_972, sum_120, sum_121, convert_element_type_962, convert_element_type_963, convert_element_type_954, convert_element_type_955, convert_element_type_946, convert_element_type_947, convert_element_type_927, convert_element_type_928, sum_111, sum_112, convert_element_type_918, convert_element_type_919, convert_element_type_907, convert_element_type_908, sum_105, sum_106, convert_element_type_898, convert_element_type_899, convert_element_type_890, convert_element_type_891, convert_element_type_882, convert_element_type_883, convert_element_type_863, convert_element_type_864, sum_96, sum_97, convert_element_type_854, convert_element_type_855, convert_element_type_843, convert_element_type_844, sum_90, sum_91, convert_element_type_834, convert_element_type_835, convert_element_type_826, convert_element_type_827, convert_element_type_818, convert_element_type_819, convert_element_type_799, convert_element_type_800, sum_81, sum_82, convert_element_type_790, convert_element_type_791, convert_element_type_779, convert_element_type_780, sum_75, sum_76, convert_element_type_770, convert_element_type_771, convert_element_type_762, convert_element_type_763, convert_element_type_754, convert_element_type_755, convert_element_type_735, convert_element_type_736, sum_66, sum_67, convert_element_type_726, convert_element_type_727, convert_element_type_715, convert_element_type_716, sum_60, sum_61, convert_element_type_706, convert_element_type_707, convert_element_type_698, convert_element_type_699, convert_element_type_690, convert_element_type_691, convert_element_type_671, convert_element_type_672, sum_51, sum_52, convert_element_type_662, convert_element_type_663, convert_element_type_651, convert_element_type_652, sum_45, sum_46, convert_element_type_642, convert_element_type_643, convert_element_type_634, convert_element_type_635, convert_element_type_626, convert_element_type_627, convert_element_type_607, convert_element_type_608, sum_36, sum_37, convert_element_type_598, convert_element_type_599, convert_element_type_587, convert_element_type_588, sum_30, sum_31, convert_element_type_578, convert_element_type_579, convert_element_type_570, convert_element_type_571, convert_element_type_562, convert_element_type_563, convert_element_type_543, convert_element_type_544, sum_21, sum_22, convert_element_type_534, convert_element_type_535, convert_element_type_523, convert_element_type_524, sum_15, sum_16)
