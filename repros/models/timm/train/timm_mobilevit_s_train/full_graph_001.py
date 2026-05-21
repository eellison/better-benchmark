class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0", primals_2: "f32[128, 3, 256, 256][196608, 1, 768, 3]cuda:0", primals_6: "f32[16][1]cuda:0", primals_7: "f32[16][1]cuda:0", primals_8: "f32[64, 16, 1, 1][16, 1, 16, 16]cuda:0", primals_12: "f32[64][1]cuda:0", primals_13: "f32[64][1]cuda:0", primals_14: "f32[64, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_18: "f32[64][1]cuda:0", primals_19: "f32[64][1]cuda:0", primals_20: "f32[32, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_24: "f32[32][1]cuda:0", primals_26: "f32[128, 32, 1, 1][32, 1, 32, 32]cuda:0", primals_30: "f32[128][1]cuda:0", primals_31: "f32[128][1]cuda:0", primals_32: "f32[128, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_36: "f32[128][1]cuda:0", primals_37: "f32[128][1]cuda:0", primals_38: "f32[64, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_42: "f32[64][1]cuda:0", primals_44: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_48: "f32[256][1]cuda:0", primals_49: "f32[256][1]cuda:0", primals_50: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_54: "f32[256][1]cuda:0", primals_55: "f32[256][1]cuda:0", primals_56: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_60: "f32[64][1]cuda:0", primals_62: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_66: "f32[256][1]cuda:0", primals_67: "f32[256][1]cuda:0", primals_68: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_72: "f32[256][1]cuda:0", primals_73: "f32[256][1]cuda:0", primals_74: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_78: "f32[64][1]cuda:0", primals_80: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0", primals_84: "f32[256][1]cuda:0", primals_85: "f32[256][1]cuda:0", primals_86: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_90: "f32[256][1]cuda:0", primals_91: "f32[256][1]cuda:0", primals_92: "f32[96, 256, 1, 1][256, 1, 256, 256]cuda:0", primals_96: "f32[96][1]cuda:0", primals_98: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0", primals_102: "f32[96][1]cuda:0", primals_103: "f32[96][1]cuda:0", primals_104: "f32[144, 96, 1, 1][96, 1, 96, 96]cuda:0", primals_105: "f32[144][1]cuda:0", primals_107: "f32[432, 144][144, 1]cuda:0", primals_109: "f32[144, 144][144, 1]cuda:0", primals_111: "f32[144][1]cuda:0", primals_113: "f32[288, 144][144, 1]cuda:0", primals_115: "f32[144, 288][288, 1]cuda:0", primals_117: "f32[144][1]cuda:0", primals_119: "f32[432, 144][144, 1]cuda:0", primals_121: "f32[144, 144][144, 1]cuda:0", primals_123: "f32[144][1]cuda:0", primals_125: "f32[288, 144][144, 1]cuda:0", primals_127: "f32[144, 288][288, 1]cuda:0", primals_129: "f32[144][1]cuda:0", primals_131: "f32[96, 144, 1, 1][144, 1, 144, 144]cuda:0", primals_135: "f32[96][1]cuda:0", primals_136: "f32[96][1]cuda:0", primals_137: "f32[96, 192, 3, 3][1728, 1, 576, 192]cuda:0", primals_141: "f32[96][1]cuda:0", primals_142: "f32[96][1]cuda:0", primals_143: "f32[384, 96, 1, 1][96, 1, 96, 96]cuda:0", primals_147: "f32[384][1]cuda:0", primals_148: "f32[384][1]cuda:0", primals_149: "f32[384, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_153: "f32[384][1]cuda:0", primals_154: "f32[384][1]cuda:0", primals_155: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0", primals_159: "f32[128][1]cuda:0", primals_161: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0", primals_165: "f32[128][1]cuda:0", primals_166: "f32[128][1]cuda:0", primals_167: "f32[192, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_168: "f32[192][1]cuda:0", primals_170: "f32[576, 192][192, 1]cuda:0", primals_172: "f32[192, 192][192, 1]cuda:0", primals_174: "f32[192][1]cuda:0", primals_176: "f32[384, 192][192, 1]cuda:0", primals_178: "f32[192, 384][384, 1]cuda:0", primals_180: "f32[192][1]cuda:0", primals_182: "f32[576, 192][192, 1]cuda:0", primals_184: "f32[192, 192][192, 1]cuda:0", primals_186: "f32[192][1]cuda:0", primals_188: "f32[384, 192][192, 1]cuda:0", primals_190: "f32[192, 384][384, 1]cuda:0", primals_192: "f32[192][1]cuda:0", primals_194: "f32[576, 192][192, 1]cuda:0", primals_196: "f32[192, 192][192, 1]cuda:0", primals_198: "f32[192][1]cuda:0", primals_200: "f32[384, 192][192, 1]cuda:0", primals_202: "f32[192, 384][384, 1]cuda:0", primals_204: "f32[192][1]cuda:0", primals_206: "f32[576, 192][192, 1]cuda:0", primals_208: "f32[192, 192][192, 1]cuda:0", primals_210: "f32[192][1]cuda:0", primals_212: "f32[384, 192][192, 1]cuda:0", primals_214: "f32[192, 384][384, 1]cuda:0", primals_216: "f32[192][1]cuda:0", primals_218: "f32[128, 192, 1, 1][192, 1, 192, 192]cuda:0", primals_222: "f32[128][1]cuda:0", primals_223: "f32[128][1]cuda:0", primals_224: "f32[128, 256, 3, 3][2304, 1, 768, 256]cuda:0", primals_228: "f32[128][1]cuda:0", primals_229: "f32[128][1]cuda:0", primals_230: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0", primals_234: "f32[512][1]cuda:0", primals_235: "f32[512][1]cuda:0", primals_236: "f32[512, 1, 3, 3][9, 1, 3, 1]cuda:0", primals_240: "f32[512][1]cuda:0", primals_241: "f32[512][1]cuda:0", primals_242: "f32[160, 512, 1, 1][512, 1, 512, 512]cuda:0", primals_246: "f32[160][1]cuda:0", primals_248: "f32[160, 160, 3, 3][1440, 1, 480, 160]cuda:0", primals_252: "f32[160][1]cuda:0", primals_253: "f32[160][1]cuda:0", primals_254: "f32[240, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_255: "f32[240][1]cuda:0", primals_257: "f32[720, 240][240, 1]cuda:0", primals_259: "f32[240, 240][240, 1]cuda:0", primals_261: "f32[240][1]cuda:0", primals_263: "f32[480, 240][240, 1]cuda:0", primals_265: "f32[240, 480][480, 1]cuda:0", primals_267: "f32[240][1]cuda:0", primals_269: "f32[720, 240][240, 1]cuda:0", primals_271: "f32[240, 240][240, 1]cuda:0", primals_273: "f32[240][1]cuda:0", primals_275: "f32[480, 240][240, 1]cuda:0", primals_277: "f32[240, 480][480, 1]cuda:0", primals_279: "f32[240][1]cuda:0", primals_281: "f32[720, 240][240, 1]cuda:0", primals_283: "f32[240, 240][240, 1]cuda:0", primals_285: "f32[240][1]cuda:0", primals_287: "f32[480, 240][240, 1]cuda:0", primals_289: "f32[240, 480][480, 1]cuda:0", primals_291: "f32[240][1]cuda:0", primals_293: "f32[160, 240, 1, 1][240, 1, 240, 240]cuda:0", primals_297: "f32[160][1]cuda:0", primals_298: "f32[160][1]cuda:0", primals_299: "f32[160, 320, 3, 3][2880, 1, 960, 320]cuda:0", primals_303: "f32[160][1]cuda:0", primals_304: "f32[160][1]cuda:0", primals_305: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0", primals_309: "f32[640][1]cuda:0", primals_310: "f32[640][1]cuda:0", primals_311: "f32[1000, 640][640, 1]cuda:0", convolution: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0", getitem_1: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0", rsqrt: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0", div: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0", convolution_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0", getitem_3: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", rsqrt_1: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", div_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0", convolution_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0", getitem_5: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", rsqrt_2: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", div_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0", convolution_3: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0", squeeze_10: "f32[32][1]cuda:0", add_22: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0", convolution_4: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0", getitem_9: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", rsqrt_4: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", div_3: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0", convolution_5: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0", getitem_11: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", rsqrt_5: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", div_4: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0", convolution_6: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", squeeze_19: "f32[64][1]cuda:0", add_39: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", convolution_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", getitem_15: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_7: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", div_5: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", convolution_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", getitem_17: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_8: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", div_6: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", convolution_9: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", squeeze_28: "f32[64][1]cuda:0", add_57: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", convolution_10: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", getitem_21: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_10: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", div_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", convolution_11: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", getitem_23: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_11: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", div_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", convolution_12: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", squeeze_37: "f32[64][1]cuda:0", add_75: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0", convolution_13: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", getitem_27: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_13: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", div_9: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0", convolution_14: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0", getitem_29: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", rsqrt_14: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0", div_10: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0", convolution_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", squeeze_46: "f32[96][1]cuda:0", add_92: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", convolution_16: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", getitem_33: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", rsqrt_16: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", div_11: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", mul_119: "f32[512, 256, 144][36864, 144, 1]cuda:0", view_3: "f32[131072, 144][144, 1]cuda:0", getitem_36: "f32[512, 4, 256, 36][110592, 36, 432, 1]cuda:0", getitem_37: "f32[512, 4, 256, 36][110592, 36, 432, 1]cuda:0", getitem_38: "f32[512, 4, 256, 36][110592, 36, 432, 1]cuda:0", getitem_39: "f32[512, 4, 256, 36][36864, 36, 144, 1]cuda:0", getitem_40: "f32[512, 4, 256][1024, 256, 1]cuda:0", getitem_41: "i64[][]cuda:0", getitem_42: "i64[][]cuda:0", mul_121: "f32[512, 256, 144][36864, 144, 1]cuda:0", view_9: "f32[131072, 144][144, 1]cuda:0", addmm_2: "f32[131072, 288][288, 1]cuda:0", view_11: "f32[131072, 288][288, 1]cuda:0", mul_123: "f32[512, 256, 144][36864, 144, 1]cuda:0", view_13: "f32[131072, 144][144, 1]cuda:0", getitem_47: "f32[512, 4, 256, 36][110592, 36, 432, 1]cuda:0", getitem_48: "f32[512, 4, 256, 36][110592, 36, 432, 1]cuda:0", getitem_49: "f32[512, 4, 256, 36][110592, 36, 432, 1]cuda:0", getitem_50: "f32[512, 4, 256, 36][36864, 36, 144, 1]cuda:0", getitem_51: "f32[512, 4, 256][1024, 256, 1]cuda:0", getitem_52: "i64[][]cuda:0", getitem_53: "i64[][]cuda:0", mul_125: "f32[512, 256, 144][36864, 144, 1]cuda:0", view_19: "f32[131072, 144][144, 1]cuda:0", addmm_6: "f32[131072, 288][288, 1]cuda:0", view_21: "f32[131072, 288][288, 1]cuda:0", mul_127: "f32[512, 256, 144][36864, 144, 1]cuda:0", view_25: "f32[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0", convolution_18: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", getitem_59: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", rsqrt_22: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", cat: "f32[128, 192, 32, 32][196608, 1, 6144, 192]cuda:0", convolution_19: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", getitem_61: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", rsqrt_23: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", div_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0", convolution_20: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0", getitem_63: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", rsqrt_24: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", div_16: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0", convolution_21: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0", getitem_65: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", rsqrt_25: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0", div_17: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0", convolution_22: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", squeeze_64: "f32[128][1]cuda:0", add_143: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", convolution_23: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", getitem_69: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", rsqrt_27: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", div_18: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", mul_171: "f32[512, 64, 192][12288, 192, 1]cuda:0", view_29: "f32[32768, 192][192, 1]cuda:0", getitem_72: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_73: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_74: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_75: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0", getitem_76: "f32[512, 4, 64][256, 64, 1]cuda:0", getitem_77: "i64[][]cuda:0", getitem_78: "i64[][]cuda:0", mul_173: "f32[512, 64, 192][12288, 192, 1]cuda:0", view_35: "f32[32768, 192][192, 1]cuda:0", addmm_10: "f32[32768, 384][384, 1]cuda:0", view_37: "f32[32768, 384][384, 1]cuda:0", mul_175: "f32[512, 64, 192][12288, 192, 1]cuda:0", view_39: "f32[32768, 192][192, 1]cuda:0", getitem_83: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_84: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_85: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_86: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0", getitem_87: "f32[512, 4, 64][256, 64, 1]cuda:0", getitem_88: "i64[][]cuda:0", getitem_89: "i64[][]cuda:0", mul_177: "f32[512, 64, 192][12288, 192, 1]cuda:0", view_45: "f32[32768, 192][192, 1]cuda:0", addmm_14: "f32[32768, 384][384, 1]cuda:0", view_47: "f32[32768, 384][384, 1]cuda:0", mul_179: "f32[512, 64, 192][12288, 192, 1]cuda:0", view_49: "f32[32768, 192][192, 1]cuda:0", getitem_94: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_95: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_96: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_97: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0", getitem_98: "f32[512, 4, 64][256, 64, 1]cuda:0", getitem_99: "i64[][]cuda:0", getitem_100: "i64[][]cuda:0", mul_181: "f32[512, 64, 192][12288, 192, 1]cuda:0", view_55: "f32[32768, 192][192, 1]cuda:0", addmm_18: "f32[32768, 384][384, 1]cuda:0", view_57: "f32[32768, 384][384, 1]cuda:0", mul_183: "f32[512, 64, 192][12288, 192, 1]cuda:0", view_59: "f32[32768, 192][192, 1]cuda:0", getitem_105: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_106: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_107: "f32[512, 4, 64, 48][36864, 48, 576, 1]cuda:0", getitem_108: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0", getitem_109: "f32[512, 4, 64][256, 64, 1]cuda:0", getitem_110: "i64[][]cuda:0", getitem_111: "i64[][]cuda:0", mul_185: "f32[512, 64, 192][12288, 192, 1]cuda:0", view_65: "f32[32768, 192][192, 1]cuda:0", addmm_22: "f32[32768, 384][384, 1]cuda:0", view_67: "f32[32768, 384][384, 1]cuda:0", mul_187: "f32[512, 64, 192][12288, 192, 1]cuda:0", view_71: "f32[128, 192, 16, 16][49152, 256, 16, 1]cuda:0", convolution_25: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", getitem_117: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", rsqrt_37: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", cat_1: "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0", convolution_26: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", getitem_119: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", rsqrt_38: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", div_24: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0", convolution_27: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0", getitem_121: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", rsqrt_39: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", div_25: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0", convolution_28: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0", getitem_123: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", rsqrt_40: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0", div_26: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0", convolution_29: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", squeeze_82: "f32[160][1]cuda:0", add_208: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", convolution_30: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", getitem_127: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", rsqrt_42: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", div_27: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", mul_231: "f32[512, 16, 240][3840, 240, 1]cuda:0", view_75: "f32[8192, 240][240, 1]cuda:0", getitem_130: "f32[512, 4, 16, 60][11520, 60, 720, 1]cuda:0", getitem_131: "f32[512, 4, 16, 60][11520, 60, 720, 1]cuda:0", getitem_132: "f32[512, 4, 16, 60][11520, 60, 720, 1]cuda:0", getitem_133: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0", getitem_134: "f32[512, 4, 32][128, 32, 1]cuda:0", getitem_135: "i64[][]cuda:0", getitem_136: "i64[][]cuda:0", mul_233: "f32[512, 16, 240][3840, 240, 1]cuda:0", view_81: "f32[8192, 240][240, 1]cuda:0", addmm_26: "f32[8192, 480][480, 1]cuda:0", view_83: "f32[8192, 480][480, 1]cuda:0", mul_235: "f32[512, 16, 240][3840, 240, 1]cuda:0", view_85: "f32[8192, 240][240, 1]cuda:0", getitem_141: "f32[512, 4, 16, 60][11520, 60, 720, 1]cuda:0", getitem_142: "f32[512, 4, 16, 60][11520, 60, 720, 1]cuda:0", getitem_143: "f32[512, 4, 16, 60][11520, 60, 720, 1]cuda:0", getitem_144: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0", getitem_145: "f32[512, 4, 32][128, 32, 1]cuda:0", getitem_146: "i64[][]cuda:0", getitem_147: "i64[][]cuda:0", mul_237: "f32[512, 16, 240][3840, 240, 1]cuda:0", view_91: "f32[8192, 240][240, 1]cuda:0", addmm_30: "f32[8192, 480][480, 1]cuda:0", view_93: "f32[8192, 480][480, 1]cuda:0", mul_239: "f32[512, 16, 240][3840, 240, 1]cuda:0", view_95: "f32[8192, 240][240, 1]cuda:0", getitem_152: "f32[512, 4, 16, 60][11520, 60, 720, 1]cuda:0", getitem_153: "f32[512, 4, 16, 60][11520, 60, 720, 1]cuda:0", getitem_154: "f32[512, 4, 16, 60][11520, 60, 720, 1]cuda:0", getitem_155: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0", getitem_156: "f32[512, 4, 32][128, 32, 1]cuda:0", getitem_157: "i64[][]cuda:0", getitem_158: "i64[][]cuda:0", mul_241: "f32[512, 16, 240][3840, 240, 1]cuda:0", view_101: "f32[8192, 240][240, 1]cuda:0", addmm_34: "f32[8192, 480][480, 1]cuda:0", view_103: "f32[8192, 480][480, 1]cuda:0", mul_243: "f32[512, 16, 240][3840, 240, 1]cuda:0", view_107: "f32[128, 240, 8, 8][15360, 64, 8, 1]cuda:0", convolution_32: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", getitem_164: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", rsqrt_50: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", cat_2: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0", convolution_33: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", getitem_166: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", rsqrt_51: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", div_32: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0", convolution_34: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0", getitem_168: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0", rsqrt_52: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0", view_108: "f32[128, 640][640, 1]cuda:0", div_35: "f32[512, 16, 1][16, 1, 1]cuda:0", div_36: "f32[512, 16, 1][16, 1, 1]cuda:0", div_37: "f32[512, 16, 1][16, 1, 1]cuda:0", div_38: "f32[512, 16, 1][16, 1, 1]cuda:0", div_39: "f32[512, 16, 1][16, 1, 1]cuda:0", div_40: "f32[512, 16, 1][16, 1, 1]cuda:0", div_41: "f32[512, 16, 1][16, 1, 1]cuda:0", unsqueeze_178: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0", div_42: "f32[512, 64, 1][64, 1, 1]cuda:0", div_43: "f32[512, 64, 1][64, 1, 1]cuda:0", div_44: "f32[512, 64, 1][64, 1, 1]cuda:0", div_45: "f32[512, 64, 1][64, 1, 1]cuda:0", div_46: "f32[512, 64, 1][64, 1, 1]cuda:0", div_47: "f32[512, 64, 1][64, 1, 1]cuda:0", div_48: "f32[512, 64, 1][64, 1, 1]cuda:0", div_49: "f32[512, 64, 1][64, 1, 1]cuda:0", div_50: "f32[512, 64, 1][64, 1, 1]cuda:0", unsqueeze_250: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0", div_51: "f32[512, 256, 1][256, 1, 1]cuda:0", div_52: "f32[512, 256, 1][256, 1, 1]cuda:0", div_53: "f32[512, 256, 1][256, 1, 1]cuda:0", div_54: "f32[512, 256, 1][256, 1, 1]cuda:0", div_55: "f32[512, 256, 1][256, 1, 1]cuda:0", unsqueeze_322: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0", unsqueeze_358: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_394: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_430: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0", unsqueeze_466: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0", tangents_1: "f32[128, 1000][1000, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_66: "f32[640, 1000][1, 640]cuda:0" = torch.ops.aten.permute.default(primals_311, [1, 0]);  primals_311 = None
        permute_67: "f32[1000, 640][640, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm: "f32[128, 640][640, 1]cuda:0" = torch.ops.aten.mm.default(tangents_1, permute_67);  permute_67 = None
        permute_68: "f32[1000, 128][1, 1000]cuda:0" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        mm_1: "f32[1000, 640][640, 1]cuda:0" = torch.ops.aten.mm.default(permute_68, view_108);  permute_68 = view_108 = None
        sum_1: "f32[1, 1000][1000, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        view_109: "f32[1000][1]cuda:0" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:173 in forward, code: x = self.flatten(x)
        view_110: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [128, 640, 1, 1]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/adaptive_avgmax_pool.py:172 in forward, code: x = self.pool(x)
        squeeze_96: "f32[128, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_110, 3);  view_110 = None
        squeeze_97: "f32[128, 640][640, 1]cuda:0" = torch.ops.aten.squeeze.dim(squeeze_96, 2);  squeeze_96 = None
        full_32: "f32[81920][1]cuda:0" = torch.ops.aten.full.default([81920], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        as_strided_scatter: "f32[81920][1]cuda:0" = torch.ops.aten.as_strided_scatter.default(full_32, squeeze_97, [128, 640], [640, 1], 0);  full_32 = squeeze_97 = None
        as_strided_5: "f32[128, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.as_strided.default(as_strided_scatter, [128, 640, 1, 1], [640, 1, 1, 1], 0);  as_strided_scatter = None
        expand_1: "f32[128, 640, 8, 8][640, 1, 0, 0]cuda:0" = torch.ops.aten.expand.default(as_strided_5, [128, 640, 8, 8]);  as_strided_5 = None
        div_34: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.div.Scalar(expand_1, 64);  expand_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_52: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, getitem_168)
        mul_259: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = None
        unsqueeze_124: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_309, -1)
        unsqueeze_125: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_265: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_125);  mul_259 = unsqueeze_125 = None
        unsqueeze_126: "f32[640, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_310, -1);  primals_310 = None
        unsqueeze_127: "f32[640, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_254: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_265, unsqueeze_127);  mul_265 = unsqueeze_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_33: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.neg.default(add_254)
        exp_33: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.exp.default(neg_33);  neg_33 = None
        add_255: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.add.Tensor(exp_33, 1);  exp_33 = None
        reciprocal: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.reciprocal.default(add_255);  add_255 = None
        mul_266: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        mul_267: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_34, mul_266);  div_34 = None
        sub_53: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_266);  mul_266 = None
        mul_268: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(add_254, sub_53);  add_254 = sub_53 = None
        add_257: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.add.Tensor(mul_268, 1);  mul_268 = None
        mul_269: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_267, add_257);  mul_267 = add_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_93: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_168, [0, 2, 3]);  getitem_168 = None
        unsqueeze_128: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_93, 0);  squeeze_93 = None
        unsqueeze_129: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 2);  unsqueeze_128 = None
        unsqueeze_130: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_129, 3);  unsqueeze_129 = None
        sum_2: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_269, [0, 2, 3])
        sub_54: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_130);  convolution_34 = unsqueeze_130 = None
        mul_270: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, sub_54)
        sum_3: "f32[640][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_270, [0, 2, 3]);  mul_270 = None
        mul_271: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_2, 0.0001220703125)
        unsqueeze_131: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_271, 0);  mul_271 = None
        unsqueeze_132: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 2);  unsqueeze_131 = None
        unsqueeze_133: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 3);  unsqueeze_132 = None
        mul_272: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, 0.0001220703125)
        squeeze_94: "f32[640][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_52, [0, 2, 3]);  rsqrt_52 = None
        mul_273: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
        mul_274: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
        unsqueeze_134: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_274, 0);  mul_274 = None
        unsqueeze_135: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 2);  unsqueeze_134 = None
        unsqueeze_136: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_135, 3);  unsqueeze_135 = None
        mul_275: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_94, primals_309);  primals_309 = None
        unsqueeze_137: "f32[1, 640][640, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_275, 0);  mul_275 = None
        unsqueeze_138: "f32[1, 640, 1][640, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_137, 2);  unsqueeze_137 = None
        unsqueeze_139: "f32[1, 640, 1, 1][640, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 3);  unsqueeze_138 = None
        mul_276: "f32[128, 640, 8, 8][40960, 1, 5120, 640]cuda:0" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_136);  sub_54 = unsqueeze_136 = None
        sub_56: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_269, mul_276);  mul_269 = mul_276 = None
        sub_57: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, unsqueeze_133);  sub_56 = unsqueeze_133 = None
        mul_277: "f32[128, 640, 8, 8][40960, 64, 8, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_139);  sub_57 = unsqueeze_139 = None
        mul_278: "f32[640][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_3, squeeze_94);  sum_3 = squeeze_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(mul_277, div_32, primals_305, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_277 = div_32 = primals_305 = None
        getitem_169: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = convolution_backward[0]
        getitem_170: "f32[640, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward[1];  convolution_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_51: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, getitem_166)
        mul_252: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = None
        unsqueeze_120: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_303, -1)
        unsqueeze_121: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        mul_258: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_252, unsqueeze_121);  mul_252 = unsqueeze_121 = None
        unsqueeze_122: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_304, -1);  primals_304 = None
        unsqueeze_123: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        add_248: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_258, unsqueeze_123);  mul_258 = unsqueeze_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_32: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.neg.default(add_248)
        exp_32: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.exp.default(neg_32);  neg_32 = None
        add_249: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(exp_32, 1);  exp_32 = None
        reciprocal_1: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.reciprocal.default(add_249);  add_249 = None
        mul_279: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        mul_280: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(getitem_169, mul_279);  getitem_169 = None
        sub_58: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_279);  mul_279 = None
        mul_281: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(add_248, sub_58);  add_248 = sub_58 = None
        add_259: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_281, 1);  mul_281 = None
        mul_282: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, add_259);  mul_280 = add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_90: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_166, [0, 2, 3]);  getitem_166 = None
        unsqueeze_140: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_90, 0);  squeeze_90 = None
        unsqueeze_141: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 2);  unsqueeze_140 = None
        unsqueeze_142: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_141, 3);  unsqueeze_141 = None
        sum_4: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 2, 3])
        sub_59: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_142);  convolution_33 = unsqueeze_142 = None
        mul_283: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, sub_59)
        sum_5: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 2, 3]);  mul_283 = None
        mul_284: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_4, 0.0001220703125)
        unsqueeze_143: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_284, 0);  mul_284 = None
        unsqueeze_144: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 2);  unsqueeze_143 = None
        unsqueeze_145: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_144, 3);  unsqueeze_144 = None
        mul_285: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, 0.0001220703125)
        squeeze_91: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_51, [0, 2, 3]);  rsqrt_51 = None
        mul_286: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
        mul_287: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_285, mul_286);  mul_285 = mul_286 = None
        unsqueeze_146: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_287, 0);  mul_287 = None
        unsqueeze_147: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 2);  unsqueeze_146 = None
        unsqueeze_148: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 3);  unsqueeze_147 = None
        mul_288: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_91, primals_303);  primals_303 = None
        unsqueeze_149: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_288, 0);  mul_288 = None
        unsqueeze_150: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_149, 2);  unsqueeze_149 = None
        unsqueeze_151: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_150, 3);  unsqueeze_150 = None
        mul_289: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_148);  sub_59 = unsqueeze_148 = None
        sub_61: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(mul_282, mul_289);  mul_282 = mul_289 = None
        sub_62: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_61, unsqueeze_145);  sub_61 = unsqueeze_145 = None
        mul_290: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_151);  sub_62 = unsqueeze_151 = None
        mul_291: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_5, squeeze_91);  sum_5 = squeeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_290, cat_2, primals_299, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_290 = cat_2 = primals_299 = None
        getitem_172: "f32[128, 320, 8, 8][20480, 1, 2560, 320]cuda:0" = convolution_backward_1[0]
        getitem_173: "f32[160, 320, 3, 3][2880, 1, 960, 320]cuda:0" = convolution_backward_1[1];  convolution_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        slice_1: "f32[128, 160, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.slice.Tensor(getitem_172, 1, 0, 160)
        slice_2: "f32[128, 160, 8, 8][20480, 1, 2560, 320]cuda:0" = torch.ops.aten.slice.Tensor(getitem_172, 1, 160, 320);  getitem_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_50: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, getitem_164)
        mul_245: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = None
        unsqueeze_116: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_297, -1)
        unsqueeze_117: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_251: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_245, unsqueeze_117);  mul_245 = unsqueeze_117 = None
        unsqueeze_118: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_298, -1);  primals_298 = None
        unsqueeze_119: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_242: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_119);  mul_251 = unsqueeze_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_31: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.neg.default(add_242)
        exp_31: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.exp.default(neg_31);  neg_31 = None
        add_243: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(exp_31, 1);  exp_31 = None
        reciprocal_2: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.reciprocal.default(add_243);  add_243 = None
        mul_292: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        mul_293: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(slice_2, mul_292);  slice_2 = None
        sub_63: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_292);  mul_292 = None
        mul_294: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(add_242, sub_63);  add_242 = sub_63 = None
        add_261: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_294, 1);  mul_294 = None
        mul_295: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_293, add_261);  mul_293 = add_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_87: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_164, [0, 2, 3]);  getitem_164 = None
        unsqueeze_152: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_87, 0);  squeeze_87 = None
        unsqueeze_153: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 2);  unsqueeze_152 = None
        unsqueeze_154: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 3);  unsqueeze_153 = None
        sum_6: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 2, 3])
        sub_64: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_154);  convolution_32 = unsqueeze_154 = None
        mul_296: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, sub_64)
        sum_7: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 2, 3]);  mul_296 = None
        mul_297: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_6, 0.0001220703125)
        unsqueeze_155: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_297, 0);  mul_297 = None
        unsqueeze_156: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 2);  unsqueeze_155 = None
        unsqueeze_157: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_156, 3);  unsqueeze_156 = None
        mul_298: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, 0.0001220703125)
        squeeze_88: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_50, [0, 2, 3]);  rsqrt_50 = None
        mul_299: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
        mul_300: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
        unsqueeze_158: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_300, 0);  mul_300 = None
        unsqueeze_159: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 2);  unsqueeze_158 = None
        unsqueeze_160: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 3);  unsqueeze_159 = None
        mul_301: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_88, primals_297);  primals_297 = None
        unsqueeze_161: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
        unsqueeze_162: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_161, 2);  unsqueeze_161 = None
        unsqueeze_163: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_162, 3);  unsqueeze_162 = None
        mul_302: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_160);  sub_64 = unsqueeze_160 = None
        sub_66: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(mul_295, mul_302);  mul_295 = mul_302 = None
        sub_67: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_157);  sub_66 = unsqueeze_157 = None
        mul_303: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_163);  sub_67 = unsqueeze_163 = None
        mul_304: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_7, squeeze_88);  sum_7 = squeeze_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_303, view_107, primals_293, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_303 = view_107 = primals_293 = None
        getitem_175: "f32[128, 240, 8, 8][15360, 1, 1920, 240]cuda:0" = convolution_backward_2[0]
        getitem_176: "f32[160, 240, 1, 1][240, 1, 240, 240]cuda:0" = convolution_backward_2[1];  convolution_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        clone_68: "f32[128, 240, 8, 8][15360, 64, 8, 1]cuda:0" = torch.ops.aten.clone.default(getitem_175, memory_format = torch.contiguous_format);  getitem_175 = None
        view_111: "f32[122880, 2, 4, 2][16, 8, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_68, [122880, 2, 4, 2]);  clone_68 = None
        permute_71: "f32[122880, 4, 2, 2][16, 2, 8, 1]cuda:0" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        clone_69: "f32[122880, 4, 2, 2][16, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
        view_112: "f32[128, 240, 16, 4][15360, 64, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_69, [128, 240, 16, 4]);  clone_69 = None
        permute_72: "f32[128, 4, 16, 240][15360, 1, 4, 64]cuda:0" = torch.ops.aten.permute.default(view_112, [0, 3, 2, 1]);  view_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        clone_70: "f32[128, 4, 16, 240][15360, 3840, 240, 1]cuda:0" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_113: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(clone_70, [512, 16, 240]);  clone_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_306: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_113, primals_291);  primals_291 = None
        mul_307: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, 240)
        sum_8: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_306, [2], True)
        mul_308: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, mul_243);  mul_306 = None
        sum_9: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_308, [2], True);  mul_308 = None
        mul_309: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, sum_9);  sum_9 = None
        sub_69: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_307, sum_8);  mul_307 = sum_8 = None
        sub_70: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_69, mul_309);  sub_69 = mul_309 = None
        mul_310: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_35, sub_70);  div_35 = sub_70 = None
        mul_311: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_113, mul_243);  mul_243 = None
        sum_10: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_311, [0, 1]);  mul_311 = None
        sum_11: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_113, [0, 1]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_114: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(mul_310, [8192, 240])
        permute_63: "f32[480, 240][1, 480]cuda:0" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
        permute_73: "f32[240, 480][480, 1]cuda:0" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_2: "f32[8192, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(view_114, permute_73);  permute_73 = None
        permute_74: "f32[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_114, [1, 0])
        mm_3: "f32[240, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(permute_74, view_103);  permute_74 = view_103 = None
        sum_12: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_114, [0], True);  view_114 = None
        view_115: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_12, [240]);  sum_12 = None
        view_116: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [512, 16, 480]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_102: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [512, 16, 480]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_30: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.neg.default(view_102)
        exp_30: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.exp.default(neg_30);  neg_30 = None
        add_234: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_30, 1);  exp_30 = None
        reciprocal_3: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reciprocal.default(add_234);  add_234 = None
        mul_312: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        mul_313: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_116, mul_312);  view_116 = None
        sub_71: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_312);  mul_312 = None
        mul_314: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_102, sub_71);  view_102 = sub_71 = None
        add_263: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_314, 1);  mul_314 = None
        mul_315: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_313, add_263);  mul_313 = add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_117: "f32[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(mul_315, [8192, 480]);  mul_315 = None
        permute_62: "f32[240, 480][1, 240]cuda:0" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
        permute_77: "f32[480, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
        mm_4: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_117, permute_77);  permute_77 = None
        permute_78: "f32[480, 8192][1, 480]cuda:0" = torch.ops.aten.permute.default(view_117, [1, 0])
        mm_5: "f32[480, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_78, view_101);  permute_78 = view_101 = None
        sum_13: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_117, [0], True);  view_117 = None
        view_118: "f32[480][1]cuda:0" = torch.ops.aten.reshape.default(sum_13, [480]);  sum_13 = None
        view_119: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [512, 16, 240]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_317: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_119, primals_285);  primals_285 = None
        mul_318: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_317, 240)
        sum_14: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_317, [2], True)
        mul_319: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_317, mul_241);  mul_317 = None
        sum_15: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True);  mul_319 = None
        mul_320: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_241, sum_15);  sum_15 = None
        sub_73: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_318, sum_14);  mul_318 = sum_14 = None
        sub_74: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_73, mul_320);  sub_73 = mul_320 = None
        mul_321: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_36, sub_74);  div_36 = sub_74 = None
        mul_322: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_119, mul_241);  mul_241 = None
        sum_16: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_322, [0, 1]);  mul_322 = None
        sum_17: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_119, [0, 1]);  view_119 = None
        add_264: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_310, mul_321);  mul_310 = mul_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_120: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(add_264, [8192, 240])
        permute_61: "f32[240, 240][1, 240]cuda:0" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
        permute_81: "f32[240, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        mm_6: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_120, permute_81);  permute_81 = None
        permute_82: "f32[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_120, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_60: "f32[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3])
        view_98: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(permute_60, [512, 16, 240]);  permute_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_99: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(view_98, [8192, 240]);  view_98 = None
        mm_7: "f32[240, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_82, view_99);  permute_82 = view_99 = None
        sum_18: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_120, [0], True);  view_120 = None
        view_121: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_18, [240]);  sum_18 = None
        view_122: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [512, 16, 240]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_123: "f32[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_122, [512, 16, 4, 60]);  view_122 = None
        permute_85: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_85, getitem_152, getitem_153, getitem_154, None, getitem_155, getitem_156, getitem_157, getitem_158, 0.0, [True, True, True, False]);  permute_85 = getitem_152 = getitem_153 = getitem_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = None
        getitem_178: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_179: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_180: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_3: "f32[1536, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
        view_124: "f32[3, 512, 4, 16, 60][1966080, 3840, 960, 60, 1]cuda:0" = torch.ops.aten.reshape.default(cat_3, [3, 512, 4, 16, 60]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_86: "f32[512, 16, 3, 4, 60][3840, 60, 1966080, 960, 1]cuda:0" = torch.ops.aten.permute.default(view_124, [1, 3, 0, 2, 4]);  view_124 = None
        clone_71: "f32[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
        view_125: "f32[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(clone_71, [512, 16, 720]);  clone_71 = None
        view_126: "f32[8192, 720][720, 1]cuda:0" = torch.ops.aten.reshape.default(view_125, [8192, 720]);  view_125 = None
        permute_58: "f32[240, 720][1, 240]cuda:0" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
        permute_87: "f32[720, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        mm_8: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_126, permute_87);  permute_87 = None
        permute_88: "f32[720, 8192][1, 720]cuda:0" = torch.ops.aten.permute.default(view_126, [1, 0])
        mm_9: "f32[720, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_88, view_95);  permute_88 = view_95 = None
        sum_19: "f32[1, 720][720, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
        view_127: "f32[720][1]cuda:0" = torch.ops.aten.reshape.default(sum_19, [720]);  sum_19 = None
        view_128: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [512, 16, 240]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_324: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_128, primals_279);  primals_279 = None
        mul_325: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_324, 240)
        sum_20: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True)
        mul_326: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_324, mul_239);  mul_324 = None
        sum_21: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True);  mul_326 = None
        mul_327: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_239, sum_21);  sum_21 = None
        sub_76: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_325, sum_20);  mul_325 = sum_20 = None
        sub_77: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_76, mul_327);  sub_76 = mul_327 = None
        mul_328: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_37, sub_77);  div_37 = sub_77 = None
        mul_329: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_128, mul_239);  mul_239 = None
        sum_22: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 1]);  mul_329 = None
        sum_23: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_128, [0, 1]);  view_128 = None
        add_265: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_264, mul_328);  add_264 = mul_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_129: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(add_265, [8192, 240])
        permute_57: "f32[480, 240][1, 480]cuda:0" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
        permute_91: "f32[240, 480][480, 1]cuda:0" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_10: "f32[8192, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(view_129, permute_91);  permute_91 = None
        permute_92: "f32[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_129, [1, 0])
        mm_11: "f32[240, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(permute_92, view_93);  permute_92 = view_93 = None
        sum_24: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
        view_130: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_24, [240]);  sum_24 = None
        view_131: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [512, 16, 480]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_92: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [512, 16, 480]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_29: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.neg.default(view_92)
        exp_29: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.exp.default(neg_29);  neg_29 = None
        add_227: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_29, 1);  exp_29 = None
        reciprocal_4: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reciprocal.default(add_227);  add_227 = None
        mul_330: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        mul_331: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_131, mul_330);  view_131 = None
        sub_78: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_330);  mul_330 = None
        mul_332: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_92, sub_78);  view_92 = sub_78 = None
        add_267: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_332, 1);  mul_332 = None
        mul_333: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_331, add_267);  mul_331 = add_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_132: "f32[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(mul_333, [8192, 480]);  mul_333 = None
        permute_56: "f32[240, 480][1, 240]cuda:0" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
        permute_95: "f32[480, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        mm_12: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_132, permute_95);  permute_95 = None
        permute_96: "f32[480, 8192][1, 480]cuda:0" = torch.ops.aten.permute.default(view_132, [1, 0])
        mm_13: "f32[480, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_96, view_91);  permute_96 = view_91 = None
        sum_25: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_132, [0], True);  view_132 = None
        view_133: "f32[480][1]cuda:0" = torch.ops.aten.reshape.default(sum_25, [480]);  sum_25 = None
        view_134: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [512, 16, 240]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_335: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_134, primals_273);  primals_273 = None
        mul_336: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_335, 240)
        sum_26: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_335, [2], True)
        mul_337: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_335, mul_237);  mul_335 = None
        sum_27: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_337, [2], True);  mul_337 = None
        mul_338: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, sum_27);  sum_27 = None
        sub_80: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_336, sum_26);  mul_336 = sum_26 = None
        sub_81: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_80, mul_338);  sub_80 = mul_338 = None
        mul_339: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, sub_81);  div_38 = sub_81 = None
        mul_340: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_134, mul_237);  mul_237 = None
        sum_28: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1]);  mul_340 = None
        sum_29: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_134, [0, 1]);  view_134 = None
        add_268: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_265, mul_339);  add_265 = mul_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_135: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(add_268, [8192, 240])
        permute_55: "f32[240, 240][1, 240]cuda:0" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
        permute_99: "f32[240, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_14: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_135, permute_99);  permute_99 = None
        permute_100: "f32[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_135, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_54: "f32[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.permute.default(getitem_144, [0, 2, 1, 3])
        view_88: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(permute_54, [512, 16, 240]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_89: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(view_88, [8192, 240]);  view_88 = None
        mm_15: "f32[240, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_100, view_89);  permute_100 = view_89 = None
        sum_30: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
        view_136: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_30, [240]);  sum_30 = None
        view_137: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [512, 16, 240]);  mm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_138: "f32[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [512, 16, 4, 60]);  view_137 = None
        permute_103: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_103, getitem_141, getitem_142, getitem_143, None, getitem_144, getitem_145, getitem_146, getitem_147, 0.0, [True, True, True, False]);  permute_103 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = getitem_146 = getitem_147 = None
        getitem_182: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_183: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_184: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_4: "f32[1536, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.cat.default([getitem_182, getitem_183, getitem_184]);  getitem_182 = getitem_183 = getitem_184 = None
        view_139: "f32[3, 512, 4, 16, 60][1966080, 3840, 960, 60, 1]cuda:0" = torch.ops.aten.reshape.default(cat_4, [3, 512, 4, 16, 60]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_104: "f32[512, 16, 3, 4, 60][3840, 60, 1966080, 960, 1]cuda:0" = torch.ops.aten.permute.default(view_139, [1, 3, 0, 2, 4]);  view_139 = None
        clone_72: "f32[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_140: "f32[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(clone_72, [512, 16, 720]);  clone_72 = None
        view_141: "f32[8192, 720][720, 1]cuda:0" = torch.ops.aten.reshape.default(view_140, [8192, 720]);  view_140 = None
        permute_52: "f32[240, 720][1, 240]cuda:0" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
        permute_105: "f32[720, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_16: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_141, permute_105);  permute_105 = None
        permute_106: "f32[720, 8192][1, 720]cuda:0" = torch.ops.aten.permute.default(view_141, [1, 0])
        mm_17: "f32[720, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_106, view_85);  permute_106 = view_85 = None
        sum_31: "f32[1, 720][720, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
        view_142: "f32[720][1]cuda:0" = torch.ops.aten.reshape.default(sum_31, [720]);  sum_31 = None
        view_143: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [512, 16, 240]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_342: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_143, primals_267);  primals_267 = None
        mul_343: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, 240)
        sum_32: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_342, [2], True)
        mul_344: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_342, mul_235);  mul_342 = None
        sum_33: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_344, [2], True);  mul_344 = None
        mul_345: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_235, sum_33);  sum_33 = None
        sub_83: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_343, sum_32);  mul_343 = sum_32 = None
        sub_84: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_83, mul_345);  sub_83 = mul_345 = None
        mul_346: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_39, sub_84);  div_39 = sub_84 = None
        mul_347: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_143, mul_235);  mul_235 = None
        sum_34: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_347, [0, 1]);  mul_347 = None
        sum_35: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_143, [0, 1]);  view_143 = None
        add_269: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_268, mul_346);  add_268 = mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_144: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(add_269, [8192, 240])
        permute_51: "f32[480, 240][1, 480]cuda:0" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
        permute_109: "f32[240, 480][480, 1]cuda:0" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        mm_18: "f32[8192, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(view_144, permute_109);  permute_109 = None
        permute_110: "f32[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_144, [1, 0])
        mm_19: "f32[240, 480][480, 1]cuda:0" = torch.ops.aten.mm.default(permute_110, view_83);  permute_110 = view_83 = None
        sum_36: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
        view_145: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_36, [240]);  sum_36 = None
        view_146: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [512, 16, 480]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_82: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [512, 16, 480]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_28: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.neg.default(view_82)
        exp_28: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.exp.default(neg_28);  neg_28 = None
        add_220: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_28, 1);  exp_28 = None
        reciprocal_5: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.reciprocal.default(add_220);  add_220 = None
        mul_348: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        mul_349: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_146, mul_348);  view_146 = None
        sub_85: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_348);  mul_348 = None
        mul_350: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_82, sub_85);  view_82 = sub_85 = None
        add_271: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_350, 1);  mul_350 = None
        mul_351: "f32[512, 16, 480][7680, 480, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_349, add_271);  mul_349 = add_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_147: "f32[8192, 480][480, 1]cuda:0" = torch.ops.aten.reshape.default(mul_351, [8192, 480]);  mul_351 = None
        permute_50: "f32[240, 480][1, 240]cuda:0" = torch.ops.aten.permute.default(primals_263, [1, 0]);  primals_263 = None
        permute_113: "f32[480, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
        mm_20: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_147, permute_113);  permute_113 = None
        permute_114: "f32[480, 8192][1, 480]cuda:0" = torch.ops.aten.permute.default(view_147, [1, 0])
        mm_21: "f32[480, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_114, view_81);  permute_114 = view_81 = None
        sum_37: "f32[1, 480][480, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_147, [0], True);  view_147 = None
        view_148: "f32[480][1]cuda:0" = torch.ops.aten.reshape.default(sum_37, [480]);  sum_37 = None
        view_149: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [512, 16, 240]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_353: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_149, primals_261);  primals_261 = None
        mul_354: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, 240)
        sum_38: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
        mul_355: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_353, mul_233);  mul_353 = None
        sum_39: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
        mul_356: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, sum_39);  sum_39 = None
        sub_87: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_354, sum_38);  mul_354 = sum_38 = None
        sub_88: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_356);  sub_87 = mul_356 = None
        mul_357: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_40, sub_88);  div_40 = sub_88 = None
        mul_358: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_149, mul_233);  mul_233 = None
        sum_40: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
        sum_41: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_149, [0, 1]);  view_149 = None
        add_272: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_269, mul_357);  add_269 = mul_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_150: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(add_272, [8192, 240])
        permute_49: "f32[240, 240][1, 240]cuda:0" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
        permute_117: "f32[240, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        mm_22: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_150, permute_117);  permute_117 = None
        permute_118: "f32[240, 8192][1, 240]cuda:0" = torch.ops.aten.permute.default(view_150, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_48: "f32[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3])
        view_78: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(permute_48, [512, 16, 240]);  permute_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_79: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.reshape.default(view_78, [8192, 240]);  view_78 = None
        mm_23: "f32[240, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_118, view_79);  permute_118 = view_79 = None
        sum_42: "f32[1, 240][240, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
        view_151: "f32[240][1]cuda:0" = torch.ops.aten.reshape.default(sum_42, [240]);  sum_42 = None
        view_152: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [512, 16, 240]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_153: "f32[512, 16, 4, 60][3840, 240, 60, 1]cuda:0" = torch.ops.aten.reshape.default(view_152, [512, 16, 4, 60]);  view_152 = None
        permute_121: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_121, getitem_130, getitem_131, getitem_132, None, getitem_133, getitem_134, getitem_135, getitem_136, 0.0, [True, True, True, False]);  permute_121 = getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = None
        getitem_186: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[0]
        getitem_187: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[1]
        getitem_188: "f32[512, 4, 16, 60][3840, 60, 240, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_5: "f32[1536, 4, 16, 60][3840, 960, 60, 1]cuda:0" = torch.ops.aten.cat.default([getitem_186, getitem_187, getitem_188]);  getitem_186 = getitem_187 = getitem_188 = None
        view_154: "f32[3, 512, 4, 16, 60][1966080, 3840, 960, 60, 1]cuda:0" = torch.ops.aten.reshape.default(cat_5, [3, 512, 4, 16, 60]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_122: "f32[512, 16, 3, 4, 60][3840, 60, 1966080, 960, 1]cuda:0" = torch.ops.aten.permute.default(view_154, [1, 3, 0, 2, 4]);  view_154 = None
        clone_73: "f32[512, 16, 3, 4, 60][11520, 720, 240, 60, 1]cuda:0" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
        view_155: "f32[512, 16, 720][11520, 720, 1]cuda:0" = torch.ops.aten.reshape.default(clone_73, [512, 16, 720]);  clone_73 = None
        view_156: "f32[8192, 720][720, 1]cuda:0" = torch.ops.aten.reshape.default(view_155, [8192, 720]);  view_155 = None
        permute_46: "f32[240, 720][1, 240]cuda:0" = torch.ops.aten.permute.default(primals_257, [1, 0]);  primals_257 = None
        permute_123: "f32[720, 240][240, 1]cuda:0" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_24: "f32[8192, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(view_156, permute_123);  permute_123 = None
        permute_124: "f32[720, 8192][1, 720]cuda:0" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_25: "f32[720, 240][240, 1]cuda:0" = torch.ops.aten.mm.default(permute_124, view_75);  permute_124 = view_75 = None
        sum_43: "f32[1, 720][720, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
        view_157: "f32[720][1]cuda:0" = torch.ops.aten.reshape.default(sum_43, [720]);  sum_43 = None
        view_158: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [512, 16, 240]);  mm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_360: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_158, primals_255);  primals_255 = None
        mul_361: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, 240)
        sum_44: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_360, [2], True)
        mul_362: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_360, mul_231);  mul_360 = None
        sum_45: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_362, [2], True);  mul_362 = None
        mul_363: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, sum_45);  sum_45 = None
        sub_90: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_361, sum_44);  mul_361 = sum_44 = None
        sub_91: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_90, mul_363);  sub_90 = mul_363 = None
        mul_364: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_41, sub_91);  div_41 = sub_91 = None
        mul_365: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_158, mul_231);  mul_231 = None
        sum_46: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_365, [0, 1]);  mul_365 = None
        sum_47: "f32[240][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_158, [0, 1]);  view_158 = None
        add_273: "f32[512, 16, 240][3840, 240, 1]cuda:0" = torch.ops.aten.add.Tensor(add_272, mul_364);  add_272 = mul_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        view_159: "f32[128, 4, 16, 240][15360, 3840, 240, 1]cuda:0" = torch.ops.aten.reshape.default(add_273, [128, 4, 16, 240]);  add_273 = None
        permute_127: "f32[128, 240, 16, 4][15360, 1, 240, 3840]cuda:0" = torch.ops.aten.permute.default(view_159, [0, 3, 2, 1]);  view_159 = None
        clone_74: "f32[128, 240, 16, 4][15360, 64, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_160: "f32[122880, 4, 2, 2][16, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_74, [122880, 4, 2, 2]);  clone_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        permute_128: "f32[122880, 2, 4, 2][16, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None
        clone_75: "f32[122880, 2, 4, 2][16, 8, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_161: "f32[128, 240, 8, 8][15360, 64, 8, 1]cuda:0" = torch.ops.aten.reshape.default(clone_75, [128, 240, 8, 8]);  clone_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_161, div_27, primals_254, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_161 = div_27 = primals_254 = None
        getitem_190: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = convolution_backward_3[0]
        getitem_191: "f32[240, 160, 1, 1][160, 1, 160, 160]cuda:0" = convolution_backward_3[1];  convolution_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_42: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, getitem_127)
        mul_224: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = None
        unsqueeze_112: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_252, -1)
        unsqueeze_113: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        mul_230: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, unsqueeze_113);  mul_224 = unsqueeze_113 = None
        unsqueeze_114: "f32[160, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_253, -1);  primals_253 = None
        unsqueeze_115: "f32[160, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        add_213: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_115);  mul_230 = unsqueeze_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_27: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.neg.default(add_213)
        exp_27: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.exp.default(neg_27);  neg_27 = None
        add_214: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(exp_27, 1);  exp_27 = None
        reciprocal_6: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.reciprocal.default(add_214);  add_214 = None
        mul_366: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        mul_367: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(getitem_190, mul_366);  getitem_190 = None
        sub_92: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_366);  mul_366 = None
        mul_368: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(add_213, sub_92);  add_213 = sub_92 = None
        add_275: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(mul_368, 1);  mul_368 = None
        mul_369: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, add_275);  mul_367 = add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_84: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_127, [0, 2, 3]);  getitem_127 = None
        unsqueeze_164: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_84, 0);  squeeze_84 = None
        unsqueeze_165: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 2);  unsqueeze_164 = None
        unsqueeze_166: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_165, 3);  unsqueeze_165 = None
        sum_48: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 2, 3])
        sub_93: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_166);  convolution_30 = unsqueeze_166 = None
        mul_370: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(mul_369, sub_93)
        sum_49: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 2, 3]);  mul_370 = None
        mul_371: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_48, 0.0001220703125)
        unsqueeze_167: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
        unsqueeze_168: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
        unsqueeze_169: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
        mul_372: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, 0.0001220703125)
        squeeze_85: "f32[160][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_42, [0, 2, 3]);  rsqrt_42 = None
        mul_373: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
        mul_374: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_372, mul_373);  mul_372 = mul_373 = None
        unsqueeze_170: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
        unsqueeze_171: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
        unsqueeze_172: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
        mul_375: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_85, primals_252);  primals_252 = None
        unsqueeze_173: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_375, 0);  mul_375 = None
        unsqueeze_174: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
        unsqueeze_175: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
        mul_376: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_172);  sub_93 = unsqueeze_172 = None
        sub_95: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(mul_369, mul_376);  mul_369 = mul_376 = None
        sub_96: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_169);  sub_95 = unsqueeze_169 = None
        mul_377: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_175);  sub_96 = unsqueeze_175 = None
        mul_378: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_49, squeeze_85);  sum_49 = squeeze_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_377, add_208, primals_248, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_377 = add_208 = primals_248 = None
        getitem_193: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = convolution_backward_4[0]
        getitem_194: "f32[160, 160, 3, 3][1440, 1, 480, 160]cuda:0" = convolution_backward_4[1];  convolution_backward_4 = None
        add_276: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.add.Tensor(slice_1, getitem_193);  slice_1 = getitem_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_50: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_276, [0, 2, 3])
        sub_97: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_178);  convolution_29 = unsqueeze_178 = None
        mul_379: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(add_276, sub_97)
        sum_51: "f32[160][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 2, 3]);  mul_379 = None
        mul_380: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_50, 0.0001220703125)
        unsqueeze_179: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_380, 0);  mul_380 = None
        unsqueeze_180: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
        unsqueeze_181: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
        mul_381: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
        mul_382: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
        mul_383: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, mul_382);  mul_381 = mul_382 = None
        unsqueeze_182: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
        unsqueeze_183: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
        unsqueeze_184: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
        mul_384: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_82, primals_246);  primals_246 = None
        unsqueeze_185: "f32[1, 160][160, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_384, 0);  mul_384 = None
        unsqueeze_186: "f32[1, 160, 1][160, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
        unsqueeze_187: "f32[1, 160, 1, 1][160, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
        mul_385: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_184);  sub_97 = unsqueeze_184 = None
        sub_99: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(add_276, mul_385);  add_276 = mul_385 = None
        sub_100: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_181);  sub_99 = unsqueeze_181 = None
        mul_386: "f32[128, 160, 8, 8][10240, 1, 1280, 160]cuda:0" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_187);  sub_100 = unsqueeze_187 = None
        mul_387: "f32[160][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_51, squeeze_82);  sum_51 = squeeze_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_386, div_26, primals_242, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_386 = div_26 = primals_242 = None
        getitem_196: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = convolution_backward_5[0]
        getitem_197: "f32[160, 512, 1, 1][512, 1, 512, 512]cuda:0" = convolution_backward_5[1];  convolution_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_40: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, getitem_123)
        mul_210: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = None
        unsqueeze_104: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_240, -1)
        unsqueeze_105: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        mul_216: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, unsqueeze_105);  mul_210 = unsqueeze_105 = None
        unsqueeze_106: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_241, -1);  primals_241 = None
        unsqueeze_107: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        add_202: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_216, unsqueeze_107);  mul_216 = unsqueeze_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_26: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.neg.default(add_202)
        exp_26: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.exp.default(neg_26);  neg_26 = None
        add_203: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.add.Tensor(exp_26, 1);  exp_26 = None
        reciprocal_7: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.reciprocal.default(add_203);  add_203 = None
        mul_388: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        mul_389: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(getitem_196, mul_388);  getitem_196 = None
        sub_101: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_388);  mul_388 = None
        mul_390: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(add_202, sub_101);  add_202 = sub_101 = None
        add_278: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_390, 1);  mul_390 = None
        mul_391: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, add_278);  mul_389 = add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_78: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_123, [0, 2, 3]);  getitem_123 = None
        unsqueeze_188: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_78, 0);  squeeze_78 = None
        unsqueeze_189: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 2);  unsqueeze_188 = None
        unsqueeze_190: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_189, 3);  unsqueeze_189 = None
        sum_52: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 2, 3])
        sub_102: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_190);  convolution_28 = unsqueeze_190 = None
        mul_392: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_391, sub_102)
        sum_53: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 2, 3]);  mul_392 = None
        mul_393: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_52, 0.0001220703125)
        unsqueeze_191: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_393, 0);  mul_393 = None
        unsqueeze_192: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
        unsqueeze_193: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
        mul_394: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
        squeeze_79: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_40, [0, 2, 3]);  rsqrt_40 = None
        mul_395: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
        mul_396: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_394, mul_395);  mul_394 = mul_395 = None
        unsqueeze_194: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
        unsqueeze_195: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
        unsqueeze_196: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
        mul_397: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_79, primals_240);  primals_240 = None
        unsqueeze_197: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
        unsqueeze_198: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
        unsqueeze_199: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
        mul_398: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_196);  sub_102 = unsqueeze_196 = None
        sub_104: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(mul_391, mul_398);  mul_391 = mul_398 = None
        sub_105: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_104, unsqueeze_193);  sub_104 = unsqueeze_193 = None
        mul_399: "f32[128, 512, 8, 8][32768, 1, 4096, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_199);  sub_105 = unsqueeze_199 = None
        mul_400: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_53, squeeze_79);  sum_53 = squeeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_399, div_25, primals_236, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 512, [True, True, False]);  mul_399 = div_25 = primals_236 = None
        getitem_199: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = convolution_backward_6[0]
        getitem_200: "f32[512, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_6[1];  convolution_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_39: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, getitem_121)
        mul_203: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = None
        unsqueeze_100: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_234, -1)
        unsqueeze_101: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_209: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, unsqueeze_101);  mul_203 = unsqueeze_101 = None
        unsqueeze_102: "f32[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_235, -1);  primals_235 = None
        unsqueeze_103: "f32[512, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_196: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_103);  mul_209 = unsqueeze_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_25: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.neg.default(add_196)
        exp_25: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.exp.default(neg_25);  neg_25 = None
        add_197: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.add.Tensor(exp_25, 1);  exp_25 = None
        reciprocal_8: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.reciprocal.default(add_197);  add_197 = None
        mul_401: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        mul_402: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(getitem_199, mul_401);  getitem_199 = None
        sub_106: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_401);  mul_401 = None
        mul_403: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(add_196, sub_106);  add_196 = sub_106 = None
        add_280: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.add.Tensor(mul_403, 1);  mul_403 = None
        mul_404: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, add_280);  mul_402 = add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_75: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_121, [0, 2, 3]);  getitem_121 = None
        unsqueeze_200: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_75, 0);  squeeze_75 = None
        unsqueeze_201: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 2);  unsqueeze_200 = None
        unsqueeze_202: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 3);  unsqueeze_201 = None
        sum_54: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 2, 3])
        sub_107: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_202);  convolution_27 = unsqueeze_202 = None
        mul_405: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(mul_404, sub_107)
        sum_55: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 2, 3]);  mul_405 = None
        mul_406: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_54, 3.0517578125e-05)
        unsqueeze_203: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_406, 0);  mul_406 = None
        unsqueeze_204: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
        unsqueeze_205: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
        mul_407: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, 3.0517578125e-05)
        squeeze_76: "f32[512][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_39, [0, 2, 3]);  rsqrt_39 = None
        mul_408: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
        mul_409: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
        unsqueeze_206: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
        unsqueeze_207: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
        unsqueeze_208: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
        mul_410: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_76, primals_234);  primals_234 = None
        unsqueeze_209: "f32[1, 512][512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
        unsqueeze_210: "f32[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
        unsqueeze_211: "f32[1, 512, 1, 1][512, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
        mul_411: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_208);  sub_107 = unsqueeze_208 = None
        sub_109: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(mul_404, mul_411);  mul_404 = mul_411 = None
        sub_110: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.sub.Tensor(sub_109, unsqueeze_205);  sub_109 = unsqueeze_205 = None
        mul_412: "f32[128, 512, 16, 16][131072, 1, 8192, 512]cuda:0" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_211);  sub_110 = unsqueeze_211 = None
        mul_413: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_55, squeeze_76);  sum_55 = squeeze_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_412, div_24, primals_230, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_412 = div_24 = primals_230 = None
        getitem_202: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = convolution_backward_7[0]
        getitem_203: "f32[512, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_7[1];  convolution_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_38: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, getitem_119)
        mul_196: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = None
        unsqueeze_96: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_228, -1)
        unsqueeze_97: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        mul_202: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_97);  mul_196 = unsqueeze_97 = None
        unsqueeze_98: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_229, -1);  primals_229 = None
        unsqueeze_99: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        add_190: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_202, unsqueeze_99);  mul_202 = unsqueeze_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_24: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.neg.default(add_190)
        exp_24: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.exp.default(neg_24);  neg_24 = None
        add_191: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_24, 1);  exp_24 = None
        reciprocal_9: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_191);  add_191 = None
        mul_414: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        mul_415: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(getitem_202, mul_414);  getitem_202 = None
        sub_111: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_414);  mul_414 = None
        mul_416: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(add_190, sub_111);  add_190 = sub_111 = None
        add_282: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_416, 1);  mul_416 = None
        mul_417: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_415, add_282);  mul_415 = add_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_72: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_119, [0, 2, 3]);  getitem_119 = None
        unsqueeze_212: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_72, 0);  squeeze_72 = None
        unsqueeze_213: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 2);  unsqueeze_212 = None
        unsqueeze_214: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 3);  unsqueeze_213 = None
        sum_56: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3])
        sub_112: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_214);  convolution_26 = unsqueeze_214 = None
        mul_418: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_417, sub_112)
        sum_57: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
        mul_419: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_56, 3.0517578125e-05)
        unsqueeze_215: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
        unsqueeze_216: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
        unsqueeze_217: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
        mul_420: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, 3.0517578125e-05)
        squeeze_73: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_38, [0, 2, 3]);  rsqrt_38 = None
        mul_421: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
        mul_422: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
        unsqueeze_218: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
        unsqueeze_219: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
        unsqueeze_220: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
        mul_423: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_73, primals_228);  primals_228 = None
        unsqueeze_221: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
        unsqueeze_222: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
        unsqueeze_223: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
        mul_424: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_220);  sub_112 = unsqueeze_220 = None
        sub_114: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(mul_417, mul_424);  mul_417 = mul_424 = None
        sub_115: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_217);  sub_114 = unsqueeze_217 = None
        mul_425: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_223);  sub_115 = unsqueeze_223 = None
        mul_426: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_57, squeeze_73);  sum_57 = squeeze_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_425, cat_1, primals_224, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_425 = cat_1 = primals_224 = None
        getitem_205: "f32[128, 256, 16, 16][65536, 1, 4096, 256]cuda:0" = convolution_backward_8[0]
        getitem_206: "f32[128, 256, 3, 3][2304, 1, 768, 256]cuda:0" = convolution_backward_8[1];  convolution_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        slice_3: "f32[128, 128, 16, 16][65536, 1, 4096, 256]cuda:0" = torch.ops.aten.slice.Tensor(getitem_205, 1, 0, 128)
        slice_4: "f32[128, 128, 16, 16][65536, 1, 4096, 256]cuda:0" = torch.ops.aten.slice.Tensor(getitem_205, 1, 128, 256);  getitem_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_37: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, getitem_117)
        mul_189: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = None
        unsqueeze_92: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_222, -1)
        unsqueeze_93: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_195: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, unsqueeze_93);  mul_189 = unsqueeze_93 = None
        unsqueeze_94: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_223, -1);  primals_223 = None
        unsqueeze_95: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_184: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_195, unsqueeze_95);  mul_195 = unsqueeze_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_23: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.neg.default(add_184)
        exp_23: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.exp.default(neg_23);  neg_23 = None
        add_185: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_23, 1);  exp_23 = None
        reciprocal_10: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_185);  add_185 = None
        mul_427: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        mul_428: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(slice_4, mul_427);  slice_4 = None
        sub_116: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_427);  mul_427 = None
        mul_429: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(add_184, sub_116);  add_184 = sub_116 = None
        add_284: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_429, 1);  mul_429 = None
        mul_430: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_428, add_284);  mul_428 = add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_69: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_117, [0, 2, 3]);  getitem_117 = None
        unsqueeze_224: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_69, 0);  squeeze_69 = None
        unsqueeze_225: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 2);  unsqueeze_224 = None
        unsqueeze_226: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 3);  unsqueeze_225 = None
        sum_58: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 2, 3])
        sub_117: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_226);  convolution_25 = unsqueeze_226 = None
        mul_431: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_430, sub_117)
        sum_59: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 2, 3]);  mul_431 = None
        mul_432: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_58, 3.0517578125e-05)
        unsqueeze_227: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
        unsqueeze_228: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
        unsqueeze_229: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
        mul_433: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, 3.0517578125e-05)
        squeeze_70: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_37, [0, 2, 3]);  rsqrt_37 = None
        mul_434: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
        mul_435: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
        unsqueeze_230: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
        unsqueeze_231: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
        unsqueeze_232: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
        mul_436: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_70, primals_222);  primals_222 = None
        unsqueeze_233: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
        unsqueeze_234: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
        unsqueeze_235: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
        mul_437: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_232);  sub_117 = unsqueeze_232 = None
        sub_119: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(mul_430, mul_437);  mul_430 = mul_437 = None
        sub_120: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_229);  sub_119 = unsqueeze_229 = None
        mul_438: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_235);  sub_120 = unsqueeze_235 = None
        mul_439: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_59, squeeze_70);  sum_59 = squeeze_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_438, view_71, primals_218, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_438 = view_71 = primals_218 = None
        getitem_208: "f32[128, 192, 16, 16][49152, 1, 3072, 192]cuda:0" = convolution_backward_9[0]
        getitem_209: "f32[128, 192, 1, 1][192, 1, 192, 192]cuda:0" = convolution_backward_9[1];  convolution_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        clone_76: "f32[128, 192, 16, 16][49152, 256, 16, 1]cuda:0" = torch.ops.aten.clone.default(getitem_208, memory_format = torch.contiguous_format);  getitem_208 = None
        view_162: "f32[196608, 2, 8, 2][32, 16, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_76, [196608, 2, 8, 2]);  clone_76 = None
        permute_129: "f32[196608, 8, 2, 2][32, 2, 16, 1]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        clone_77: "f32[196608, 8, 2, 2][32, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_163: "f32[128, 192, 64, 4][49152, 256, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_77, [128, 192, 64, 4]);  clone_77 = None
        permute_130: "f32[128, 4, 64, 192][49152, 1, 4, 256]cuda:0" = torch.ops.aten.permute.default(view_163, [0, 3, 2, 1]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        clone_78: "f32[128, 4, 64, 192][49152, 12288, 192, 1]cuda:0" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_164: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(clone_78, [512, 64, 192]);  clone_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_441: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_164, primals_216);  primals_216 = None
        mul_442: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, 192)
        sum_60: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True)
        mul_443: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_441, mul_187);  mul_441 = None
        sum_61: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True);  mul_443 = None
        mul_444: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, sum_61);  sum_61 = None
        sub_122: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_442, sum_60);  mul_442 = sum_60 = None
        sub_123: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, mul_444);  sub_122 = mul_444 = None
        mul_445: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_42, sub_123);  div_42 = sub_123 = None
        mul_446: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_164, mul_187);  mul_187 = None
        sum_62: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1]);  mul_446 = None
        sum_63: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_164, [0, 1]);  view_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_165: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(mul_445, [32768, 192])
        permute_41: "f32[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(primals_214, [1, 0]);  primals_214 = None
        permute_131: "f32[192, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_26: "f32[32768, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_165, permute_131);  permute_131 = None
        permute_132: "f32[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_27: "f32[192, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_132, view_67);  permute_132 = view_67 = None
        sum_64: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
        view_166: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [192]);  sum_64 = None
        view_167: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [512, 64, 384]);  mm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_66: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [512, 64, 384]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_22: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(view_66)
        exp_22: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_22);  neg_22 = None
        add_176: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_22, 1);  exp_22 = None
        reciprocal_11: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reciprocal.default(add_176);  add_176 = None
        mul_447: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        mul_448: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_167, mul_447);  view_167 = None
        sub_124: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_447);  mul_447 = None
        mul_449: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_66, sub_124);  view_66 = sub_124 = None
        add_286: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_449, 1);  mul_449 = None
        mul_450: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_448, add_286);  mul_448 = add_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_168: "f32[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(mul_450, [32768, 384]);  mul_450 = None
        permute_40: "f32[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_212, [1, 0]);  primals_212 = None
        permute_135: "f32[384, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        mm_28: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_168, permute_135);  permute_135 = None
        permute_136: "f32[384, 32768][1, 384]cuda:0" = torch.ops.aten.permute.default(view_168, [1, 0])
        mm_29: "f32[384, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_136, view_65);  permute_136 = view_65 = None
        sum_65: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
        view_169: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_65, [384]);  sum_65 = None
        view_170: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [512, 64, 192]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_452: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_170, primals_210);  primals_210 = None
        mul_453: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_452, 192)
        sum_66: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_452, [2], True)
        mul_454: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_452, mul_185);  mul_452 = None
        sum_67: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True);  mul_454 = None
        mul_455: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, sum_67);  sum_67 = None
        sub_126: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_453, sum_66);  mul_453 = sum_66 = None
        sub_127: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, mul_455);  sub_126 = mul_455 = None
        mul_456: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_43, sub_127);  div_43 = sub_127 = None
        mul_457: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_170, mul_185);  mul_185 = None
        sum_68: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 1]);  mul_457 = None
        sum_69: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_170, [0, 1]);  view_170 = None
        add_287: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_445, mul_456);  mul_445 = mul_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_171: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_287, [32768, 192])
        permute_39: "f32[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_208, [1, 0]);  primals_208 = None
        permute_139: "f32[192, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        mm_30: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_171, permute_139);  permute_139 = None
        permute_140: "f32[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_171, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_38: "f32[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_108, [0, 2, 1, 3])
        view_62: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_38, [512, 64, 192]);  permute_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_63: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_62, [32768, 192]);  view_62 = None
        mm_31: "f32[192, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_140, view_63);  permute_140 = view_63 = None
        sum_70: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_171, [0], True);  view_171 = None
        view_172: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [192]);  sum_70 = None
        view_173: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [512, 64, 192]);  mm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_174: "f32[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_173, [512, 64, 4, 48]);  view_173 = None
        permute_143: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_143, getitem_105, getitem_106, getitem_107, None, getitem_108, getitem_109, getitem_110, getitem_111, 0.0, [True, True, True, False]);  permute_143 = getitem_105 = getitem_106 = getitem_107 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = None
        getitem_211: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[0]
        getitem_212: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[1]
        getitem_213: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_6: "f32[1536, 4, 64, 48][12288, 3072, 48, 1]cuda:0" = torch.ops.aten.cat.default([getitem_211, getitem_212, getitem_213]);  getitem_211 = getitem_212 = getitem_213 = None
        view_175: "f32[3, 512, 4, 64, 48][6291456, 12288, 3072, 48, 1]cuda:0" = torch.ops.aten.reshape.default(cat_6, [3, 512, 4, 64, 48]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_144: "f32[512, 64, 3, 4, 48][12288, 48, 6291456, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_175, [1, 3, 0, 2, 4]);  view_175 = None
        clone_79: "f32[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
        view_176: "f32[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_79, [512, 64, 576]);  clone_79 = None
        view_177: "f32[32768, 576][576, 1]cuda:0" = torch.ops.aten.reshape.default(view_176, [32768, 576]);  view_176 = None
        permute_36: "f32[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_206, [1, 0]);  primals_206 = None
        permute_145: "f32[576, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_32: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_177, permute_145);  permute_145 = None
        permute_146: "f32[576, 32768][1, 576]cuda:0" = torch.ops.aten.permute.default(view_177, [1, 0])
        mm_33: "f32[576, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_146, view_59);  permute_146 = view_59 = None
        sum_71: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_177, [0], True);  view_177 = None
        view_178: "f32[576][1]cuda:0" = torch.ops.aten.reshape.default(sum_71, [576]);  sum_71 = None
        view_179: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [512, 64, 192]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_459: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_179, primals_204);  primals_204 = None
        mul_460: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_459, 192)
        sum_72: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
        mul_461: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_459, mul_183);  mul_459 = None
        sum_73: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
        mul_462: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, sum_73);  sum_73 = None
        sub_129: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_460, sum_72);  mul_460 = sum_72 = None
        sub_130: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_129, mul_462);  sub_129 = mul_462 = None
        mul_463: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_44, sub_130);  div_44 = sub_130 = None
        mul_464: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_179, mul_183);  mul_183 = None
        sum_74: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
        sum_75: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_179, [0, 1]);  view_179 = None
        add_288: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_287, mul_463);  add_287 = mul_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_180: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_288, [32768, 192])
        permute_35: "f32[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(primals_202, [1, 0]);  primals_202 = None
        permute_149: "f32[192, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_34: "f32[32768, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_180, permute_149);  permute_149 = None
        permute_150: "f32[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_180, [1, 0])
        mm_35: "f32[192, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_150, view_57);  permute_150 = view_57 = None
        sum_76: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
        view_181: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_76, [192]);  sum_76 = None
        view_182: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [512, 64, 384]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_56: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [512, 64, 384]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_21: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(view_56)
        exp_21: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_21);  neg_21 = None
        add_169: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_21, 1);  exp_21 = None
        reciprocal_12: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reciprocal.default(add_169);  add_169 = None
        mul_465: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        mul_466: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_182, mul_465);  view_182 = None
        sub_131: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_465);  mul_465 = None
        mul_467: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_56, sub_131);  view_56 = sub_131 = None
        add_290: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_467, 1);  mul_467 = None
        mul_468: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_466, add_290);  mul_466 = add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_183: "f32[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(mul_468, [32768, 384]);  mul_468 = None
        permute_34: "f32[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_200, [1, 0]);  primals_200 = None
        permute_153: "f32[384, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_36: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_183, permute_153);  permute_153 = None
        permute_154: "f32[384, 32768][1, 384]cuda:0" = torch.ops.aten.permute.default(view_183, [1, 0])
        mm_37: "f32[384, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_154, view_55);  permute_154 = view_55 = None
        sum_77: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
        view_184: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [384]);  sum_77 = None
        view_185: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [512, 64, 192]);  mm_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_470: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_185, primals_198);  primals_198 = None
        mul_471: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_470, 192)
        sum_78: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_470, [2], True)
        mul_472: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_470, mul_181);  mul_470 = None
        sum_79: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
        mul_473: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_181, sum_79);  sum_79 = None
        sub_133: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_471, sum_78);  mul_471 = sum_78 = None
        sub_134: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_133, mul_473);  sub_133 = mul_473 = None
        mul_474: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_45, sub_134);  div_45 = sub_134 = None
        mul_475: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_185, mul_181);  mul_181 = None
        sum_80: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
        sum_81: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_185, [0, 1]);  view_185 = None
        add_291: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_288, mul_474);  add_288 = mul_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_186: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_291, [32768, 192])
        permute_33: "f32[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
        permute_157: "f32[192, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_38: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_186, permute_157);  permute_157 = None
        permute_158: "f32[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_186, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_32: "f32[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_97, [0, 2, 1, 3])
        view_52: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_32, [512, 64, 192]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_53: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_52, [32768, 192]);  view_52 = None
        mm_39: "f32[192, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_158, view_53);  permute_158 = view_53 = None
        sum_82: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
        view_187: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [192]);  sum_82 = None
        view_188: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [512, 64, 192]);  mm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_189: "f32[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_188, [512, 64, 4, 48]);  view_188 = None
        permute_161: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_161, getitem_94, getitem_95, getitem_96, None, getitem_97, getitem_98, getitem_99, getitem_100, 0.0, [True, True, True, False]);  permute_161 = getitem_94 = getitem_95 = getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = None
        getitem_215: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[0]
        getitem_216: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[1]
        getitem_217: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_7: "f32[1536, 4, 64, 48][12288, 3072, 48, 1]cuda:0" = torch.ops.aten.cat.default([getitem_215, getitem_216, getitem_217]);  getitem_215 = getitem_216 = getitem_217 = None
        view_190: "f32[3, 512, 4, 64, 48][6291456, 12288, 3072, 48, 1]cuda:0" = torch.ops.aten.reshape.default(cat_7, [3, 512, 4, 64, 48]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_162: "f32[512, 64, 3, 4, 48][12288, 48, 6291456, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_190, [1, 3, 0, 2, 4]);  view_190 = None
        clone_80: "f32[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
        view_191: "f32[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_80, [512, 64, 576]);  clone_80 = None
        view_192: "f32[32768, 576][576, 1]cuda:0" = torch.ops.aten.reshape.default(view_191, [32768, 576]);  view_191 = None
        permute_30: "f32[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
        permute_163: "f32[576, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_40: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_192, permute_163);  permute_163 = None
        permute_164: "f32[576, 32768][1, 576]cuda:0" = torch.ops.aten.permute.default(view_192, [1, 0])
        mm_41: "f32[576, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_164, view_49);  permute_164 = view_49 = None
        sum_83: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_192, [0], True);  view_192 = None
        view_193: "f32[576][1]cuda:0" = torch.ops.aten.reshape.default(sum_83, [576]);  sum_83 = None
        view_194: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [512, 64, 192]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_477: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_194, primals_192);  primals_192 = None
        mul_478: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_477, 192)
        sum_84: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True)
        mul_479: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_477, mul_179);  mul_477 = None
        sum_85: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
        mul_480: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_179, sum_85);  sum_85 = None
        sub_136: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_478, sum_84);  mul_478 = sum_84 = None
        sub_137: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_136, mul_480);  sub_136 = mul_480 = None
        mul_481: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_46, sub_137);  div_46 = sub_137 = None
        mul_482: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_194, mul_179);  mul_179 = None
        sum_86: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1]);  mul_482 = None
        sum_87: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_194, [0, 1]);  view_194 = None
        add_292: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_291, mul_481);  add_291 = mul_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_195: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_292, [32768, 192])
        permute_29: "f32[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        permute_167: "f32[192, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        mm_42: "f32[32768, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_195, permute_167);  permute_167 = None
        permute_168: "f32[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_195, [1, 0])
        mm_43: "f32[192, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_168, view_47);  permute_168 = view_47 = None
        sum_88: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_195, [0], True);  view_195 = None
        view_196: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [192]);  sum_88 = None
        view_197: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [512, 64, 384]);  mm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_46: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [512, 64, 384]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_20: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(view_46)
        exp_20: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_20);  neg_20 = None
        add_162: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_20, 1);  exp_20 = None
        reciprocal_13: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reciprocal.default(add_162);  add_162 = None
        mul_483: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        mul_484: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_197, mul_483);  view_197 = None
        sub_138: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_483);  mul_483 = None
        mul_485: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_46, sub_138);  view_46 = sub_138 = None
        add_294: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_485, 1);  mul_485 = None
        mul_486: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_484, add_294);  mul_484 = add_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_198: "f32[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(mul_486, [32768, 384]);  mul_486 = None
        permute_28: "f32[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        permute_171: "f32[384, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
        mm_44: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_198, permute_171);  permute_171 = None
        permute_172: "f32[384, 32768][1, 384]cuda:0" = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_45: "f32[384, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_172, view_45);  permute_172 = view_45 = None
        sum_89: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
        view_199: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_89, [384]);  sum_89 = None
        view_200: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [512, 64, 192]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_488: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_200, primals_186);  primals_186 = None
        mul_489: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_488, 192)
        sum_90: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_488, [2], True)
        mul_490: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_488, mul_177);  mul_488 = None
        sum_91: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_490, [2], True);  mul_490 = None
        mul_491: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_177, sum_91);  sum_91 = None
        sub_140: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_489, sum_90);  mul_489 = sum_90 = None
        sub_141: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_140, mul_491);  sub_140 = mul_491 = None
        mul_492: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_47, sub_141);  div_47 = sub_141 = None
        mul_493: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_200, mul_177);  mul_177 = None
        sum_92: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_493, [0, 1]);  mul_493 = None
        sum_93: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_200, [0, 1]);  view_200 = None
        add_295: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_292, mul_492);  add_292 = mul_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_201: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_295, [32768, 192])
        permute_27: "f32[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_175: "f32[192, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        mm_46: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_201, permute_175);  permute_175 = None
        permute_176: "f32[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_201, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_26: "f32[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_86, [0, 2, 1, 3])
        view_42: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_26, [512, 64, 192]);  permute_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_43: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_42, [32768, 192]);  view_42 = None
        mm_47: "f32[192, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_176, view_43);  permute_176 = view_43 = None
        sum_94: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
        view_202: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [192]);  sum_94 = None
        view_203: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [512, 64, 192]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_204: "f32[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_203, [512, 64, 4, 48]);  view_203 = None
        permute_179: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = torch.ops.aten.permute.default(view_204, [0, 2, 1, 3]);  view_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_179, getitem_83, getitem_84, getitem_85, None, getitem_86, getitem_87, getitem_88, getitem_89, 0.0, [True, True, True, False]);  permute_179 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = None
        getitem_219: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[0]
        getitem_220: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[1]
        getitem_221: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_8: "f32[1536, 4, 64, 48][12288, 3072, 48, 1]cuda:0" = torch.ops.aten.cat.default([getitem_219, getitem_220, getitem_221]);  getitem_219 = getitem_220 = getitem_221 = None
        view_205: "f32[3, 512, 4, 64, 48][6291456, 12288, 3072, 48, 1]cuda:0" = torch.ops.aten.reshape.default(cat_8, [3, 512, 4, 64, 48]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_180: "f32[512, 64, 3, 4, 48][12288, 48, 6291456, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_205, [1, 3, 0, 2, 4]);  view_205 = None
        clone_81: "f32[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_206: "f32[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_81, [512, 64, 576]);  clone_81 = None
        view_207: "f32[32768, 576][576, 1]cuda:0" = torch.ops.aten.reshape.default(view_206, [32768, 576]);  view_206 = None
        permute_24: "f32[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
        permute_181: "f32[576, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_48: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_207, permute_181);  permute_181 = None
        permute_182: "f32[576, 32768][1, 576]cuda:0" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_49: "f32[576, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_182, view_39);  permute_182 = view_39 = None
        sum_95: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_207, [0], True);  view_207 = None
        view_208: "f32[576][1]cuda:0" = torch.ops.aten.reshape.default(sum_95, [576]);  sum_95 = None
        view_209: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [512, 64, 192]);  mm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_495: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_209, primals_180);  primals_180 = None
        mul_496: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, 192)
        sum_96: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
        mul_497: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, mul_175);  mul_495 = None
        sum_97: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
        mul_498: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_175, sum_97);  sum_97 = None
        sub_143: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_496, sum_96);  mul_496 = sum_96 = None
        sub_144: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_143, mul_498);  sub_143 = mul_498 = None
        mul_499: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_48, sub_144);  div_48 = sub_144 = None
        mul_500: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_209, mul_175);  mul_175 = None
        sum_98: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
        sum_99: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_209, [0, 1]);  view_209 = None
        add_296: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_295, mul_499);  add_295 = mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_210: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_296, [32768, 192])
        permute_23: "f32[384, 192][1, 384]cuda:0" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        permute_185: "f32[192, 384][384, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_50: "f32[32768, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(view_210, permute_185);  permute_185 = None
        permute_186: "f32[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_51: "f32[192, 384][384, 1]cuda:0" = torch.ops.aten.mm.default(permute_186, view_37);  permute_186 = view_37 = None
        sum_100: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
        view_211: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_100, [192]);  sum_100 = None
        view_212: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [512, 64, 384]);  mm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_36: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [512, 64, 384]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_19: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.neg.default(view_36)
        exp_19: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.exp.default(neg_19);  neg_19 = None
        add_155: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_19, 1);  exp_19 = None
        reciprocal_14: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.reciprocal.default(add_155);  add_155 = None
        mul_501: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        mul_502: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_212, mul_501);  view_212 = None
        sub_145: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_501);  mul_501 = None
        mul_503: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_36, sub_145);  view_36 = sub_145 = None
        add_298: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_503, 1);  mul_503 = None
        mul_504: "f32[512, 64, 384][24576, 384, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_502, add_298);  mul_502 = add_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_213: "f32[32768, 384][384, 1]cuda:0" = torch.ops.aten.reshape.default(mul_504, [32768, 384]);  mul_504 = None
        permute_22: "f32[192, 384][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        permute_189: "f32[384, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_52: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_213, permute_189);  permute_189 = None
        permute_190: "f32[384, 32768][1, 384]cuda:0" = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_53: "f32[384, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_190, view_35);  permute_190 = view_35 = None
        sum_101: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_213, [0], True);  view_213 = None
        view_214: "f32[384][1]cuda:0" = torch.ops.aten.reshape.default(sum_101, [384]);  sum_101 = None
        view_215: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [512, 64, 192]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_506: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_215, primals_174);  primals_174 = None
        mul_507: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, 192)
        sum_102: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True)
        mul_508: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_506, mul_173);  mul_506 = None
        sum_103: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True);  mul_508 = None
        mul_509: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, sum_103);  sum_103 = None
        sub_147: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_507, sum_102);  mul_507 = sum_102 = None
        sub_148: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_147, mul_509);  sub_147 = mul_509 = None
        mul_510: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_49, sub_148);  div_49 = sub_148 = None
        mul_511: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_215, mul_173);  mul_173 = None
        sum_104: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1]);  mul_511 = None
        sum_105: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_215, [0, 1]);  view_215 = None
        add_299: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_296, mul_510);  add_296 = mul_510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_216: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(add_299, [32768, 192])
        permute_21: "f32[192, 192][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        permute_193: "f32[192, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_54: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_216, permute_193);  permute_193 = None
        permute_194: "f32[192, 32768][1, 192]cuda:0" = torch.ops.aten.permute.default(view_216, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_20: "f32[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.permute.default(getitem_75, [0, 2, 1, 3])
        view_32: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(permute_20, [512, 64, 192]);  permute_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_33: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.reshape.default(view_32, [32768, 192]);  view_32 = None
        mm_55: "f32[192, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_194, view_33);  permute_194 = view_33 = None
        sum_106: "f32[1, 192][192, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_216, [0], True);  view_216 = None
        view_217: "f32[192][1]cuda:0" = torch.ops.aten.reshape.default(sum_106, [192]);  sum_106 = None
        view_218: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [512, 64, 192]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_219: "f32[512, 64, 4, 48][12288, 192, 48, 1]cuda:0" = torch.ops.aten.reshape.default(view_218, [512, 64, 4, 48]);  view_218 = None
        permute_197: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = torch.ops.aten.permute.default(view_219, [0, 2, 1, 3]);  view_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_197, getitem_72, getitem_73, getitem_74, None, getitem_75, getitem_76, getitem_77, getitem_78, 0.0, [True, True, True, False]);  permute_197 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = None
        getitem_223: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[0]
        getitem_224: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[1]
        getitem_225: "f32[512, 4, 64, 48][12288, 48, 192, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_9: "f32[1536, 4, 64, 48][12288, 3072, 48, 1]cuda:0" = torch.ops.aten.cat.default([getitem_223, getitem_224, getitem_225]);  getitem_223 = getitem_224 = getitem_225 = None
        view_220: "f32[3, 512, 4, 64, 48][6291456, 12288, 3072, 48, 1]cuda:0" = torch.ops.aten.reshape.default(cat_9, [3, 512, 4, 64, 48]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_198: "f32[512, 64, 3, 4, 48][12288, 48, 6291456, 3072, 1]cuda:0" = torch.ops.aten.permute.default(view_220, [1, 3, 0, 2, 4]);  view_220 = None
        clone_82: "f32[512, 64, 3, 4, 48][36864, 576, 192, 48, 1]cuda:0" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
        view_221: "f32[512, 64, 576][36864, 576, 1]cuda:0" = torch.ops.aten.reshape.default(clone_82, [512, 64, 576]);  clone_82 = None
        view_222: "f32[32768, 576][576, 1]cuda:0" = torch.ops.aten.reshape.default(view_221, [32768, 576]);  view_221 = None
        permute_18: "f32[192, 576][1, 192]cuda:0" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        permute_199: "f32[576, 192][192, 1]cuda:0" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        mm_56: "f32[32768, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(view_222, permute_199);  permute_199 = None
        permute_200: "f32[576, 32768][1, 576]cuda:0" = torch.ops.aten.permute.default(view_222, [1, 0])
        mm_57: "f32[576, 192][192, 1]cuda:0" = torch.ops.aten.mm.default(permute_200, view_29);  permute_200 = view_29 = None
        sum_107: "f32[1, 576][576, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
        view_223: "f32[576][1]cuda:0" = torch.ops.aten.reshape.default(sum_107, [576]);  sum_107 = None
        view_224: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [512, 64, 192]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_513: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_224, primals_168);  primals_168 = None
        mul_514: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_513, 192)
        sum_108: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_513, [2], True)
        mul_515: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_513, mul_171);  mul_513 = None
        sum_109: "f32[512, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_515, [2], True);  mul_515 = None
        mul_516: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, sum_109);  sum_109 = None
        sub_150: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_514, sum_108);  mul_514 = sum_108 = None
        sub_151: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_150, mul_516);  sub_150 = mul_516 = None
        mul_517: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, sub_151);  div_50 = sub_151 = None
        mul_518: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_224, mul_171);  mul_171 = None
        sum_110: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 1]);  mul_518 = None
        sum_111: "f32[192][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1]);  view_224 = None
        add_300: "f32[512, 64, 192][12288, 192, 1]cuda:0" = torch.ops.aten.add.Tensor(add_299, mul_517);  add_299 = mul_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        view_225: "f32[128, 4, 64, 192][49152, 12288, 192, 1]cuda:0" = torch.ops.aten.reshape.default(add_300, [128, 4, 64, 192]);  add_300 = None
        permute_203: "f32[128, 192, 64, 4][49152, 1, 192, 12288]cuda:0" = torch.ops.aten.permute.default(view_225, [0, 3, 2, 1]);  view_225 = None
        clone_83: "f32[128, 192, 64, 4][49152, 256, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
        view_226: "f32[196608, 8, 2, 2][32, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_83, [196608, 8, 2, 2]);  clone_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        permute_204: "f32[196608, 2, 8, 2][32, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_84: "f32[196608, 2, 8, 2][32, 16, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_227: "f32[128, 192, 16, 16][49152, 256, 16, 1]cuda:0" = torch.ops.aten.reshape.default(clone_84, [128, 192, 16, 16]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_backward_10 = torch.ops.aten.convolution_backward.default(view_227, div_18, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_227 = div_18 = primals_167 = None
        getitem_227: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = convolution_backward_10[0]
        getitem_228: "f32[192, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_10[1];  convolution_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_27: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, getitem_69)
        mul_164: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = None
        unsqueeze_88: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_165, -1)
        unsqueeze_89: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        mul_170: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, unsqueeze_89);  mul_164 = unsqueeze_89 = None
        unsqueeze_90: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_166, -1);  primals_166 = None
        unsqueeze_91: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        add_148: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_91);  mul_170 = unsqueeze_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_18: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.neg.default(add_148)
        exp_18: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.exp.default(neg_18);  neg_18 = None
        add_149: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_18, 1);  exp_18 = None
        reciprocal_15: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_149);  add_149 = None
        mul_519: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        mul_520: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(getitem_227, mul_519);  getitem_227 = None
        sub_152: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_519);  mul_519 = None
        mul_521: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(add_148, sub_152);  add_148 = sub_152 = None
        add_302: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_521, 1);  mul_521 = None
        mul_522: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_520, add_302);  mul_520 = add_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_66: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_69, [0, 2, 3]);  getitem_69 = None
        unsqueeze_236: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_66, 0);  squeeze_66 = None
        unsqueeze_237: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 2);  unsqueeze_236 = None
        unsqueeze_238: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 3);  unsqueeze_237 = None
        sum_112: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_522, [0, 2, 3])
        sub_153: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_238);  convolution_23 = unsqueeze_238 = None
        mul_523: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_522, sub_153)
        sum_113: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_523, [0, 2, 3]);  mul_523 = None
        mul_524: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_112, 3.0517578125e-05)
        unsqueeze_239: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
        unsqueeze_240: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
        unsqueeze_241: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
        mul_525: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, 3.0517578125e-05)
        squeeze_67: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_27, [0, 2, 3]);  rsqrt_27 = None
        mul_526: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
        mul_527: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_525, mul_526);  mul_525 = mul_526 = None
        unsqueeze_242: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
        unsqueeze_243: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
        unsqueeze_244: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
        mul_528: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_67, primals_165);  primals_165 = None
        unsqueeze_245: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
        unsqueeze_246: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
        unsqueeze_247: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
        mul_529: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_244);  sub_153 = unsqueeze_244 = None
        sub_155: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(mul_522, mul_529);  mul_522 = mul_529 = None
        sub_156: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_241);  sub_155 = unsqueeze_241 = None
        mul_530: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_247);  sub_156 = unsqueeze_247 = None
        mul_531: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_113, squeeze_67);  sum_113 = squeeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_530, add_143, primals_161, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_530 = add_143 = primals_161 = None
        getitem_230: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = convolution_backward_11[0]
        getitem_231: "f32[128, 128, 3, 3][1152, 1, 384, 128]cuda:0" = convolution_backward_11[1];  convolution_backward_11 = None
        add_303: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.add.Tensor(slice_3, getitem_230);  slice_3 = getitem_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_114: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_303, [0, 2, 3])
        sub_157: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_250);  convolution_22 = unsqueeze_250 = None
        mul_532: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(add_303, sub_157)
        sum_115: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 2, 3]);  mul_532 = None
        mul_533: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_114, 3.0517578125e-05)
        unsqueeze_251: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
        unsqueeze_252: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
        unsqueeze_253: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
        mul_534: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, 3.0517578125e-05)
        mul_535: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
        mul_536: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_534, mul_535);  mul_534 = mul_535 = None
        unsqueeze_254: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
        unsqueeze_255: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
        unsqueeze_256: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
        mul_537: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_64, primals_159);  primals_159 = None
        unsqueeze_257: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
        unsqueeze_258: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
        unsqueeze_259: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
        mul_538: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_256);  sub_157 = unsqueeze_256 = None
        sub_159: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(add_303, mul_538);  add_303 = mul_538 = None
        sub_160: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_253);  sub_159 = unsqueeze_253 = None
        mul_539: "f32[128, 128, 16, 16][32768, 1, 2048, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_259);  sub_160 = unsqueeze_259 = None
        mul_540: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_115, squeeze_64);  sum_115 = squeeze_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_539, div_17, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_539 = div_17 = primals_155 = None
        getitem_233: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = convolution_backward_12[0]
        getitem_234: "f32[128, 384, 1, 1][384, 1, 384, 384]cuda:0" = convolution_backward_12[1];  convolution_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_25: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, getitem_65)
        mul_150: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
        unsqueeze_80: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_153, -1)
        unsqueeze_81: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        mul_156: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_150, unsqueeze_81);  mul_150 = unsqueeze_81 = None
        unsqueeze_82: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_154, -1);  primals_154 = None
        unsqueeze_83: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        add_137: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_156, unsqueeze_83);  mul_156 = unsqueeze_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_17: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.neg.default(add_137)
        exp_17: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.exp.default(neg_17);  neg_17 = None
        add_138: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_17, 1);  exp_17 = None
        reciprocal_16: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_138);  add_138 = None
        mul_541: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        mul_542: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_233, mul_541);  getitem_233 = None
        sub_161: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_541);  mul_541 = None
        mul_543: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_137, sub_161);  add_137 = sub_161 = None
        add_305: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_543, 1);  mul_543 = None
        mul_544: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_542, add_305);  mul_542 = add_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_60: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_65, [0, 2, 3]);  getitem_65 = None
        unsqueeze_260: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_60, 0);  squeeze_60 = None
        unsqueeze_261: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 2);  unsqueeze_260 = None
        unsqueeze_262: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 3);  unsqueeze_261 = None
        sum_116: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3])
        sub_162: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_262);  convolution_21 = unsqueeze_262 = None
        mul_545: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_544, sub_162)
        sum_117: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 2, 3]);  mul_545 = None
        mul_546: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_116, 3.0517578125e-05)
        unsqueeze_263: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
        unsqueeze_264: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
        unsqueeze_265: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
        mul_547: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, 3.0517578125e-05)
        squeeze_61: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_25, [0, 2, 3]);  rsqrt_25 = None
        mul_548: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
        mul_549: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
        unsqueeze_266: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
        unsqueeze_267: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
        unsqueeze_268: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
        mul_550: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_61, primals_153);  primals_153 = None
        unsqueeze_269: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
        unsqueeze_270: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
        unsqueeze_271: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
        mul_551: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_162, unsqueeze_268);  sub_162 = unsqueeze_268 = None
        sub_164: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(mul_544, mul_551);  mul_544 = mul_551 = None
        sub_165: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_164, unsqueeze_265);  sub_164 = unsqueeze_265 = None
        mul_552: "f32[128, 384, 16, 16][98304, 1, 6144, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_271);  sub_165 = unsqueeze_271 = None
        mul_553: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_117, squeeze_61);  sum_117 = squeeze_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_552, div_16, primals_149, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_552 = div_16 = primals_149 = None
        getitem_236: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = convolution_backward_13[0]
        getitem_237: "f32[384, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_13[1];  convolution_backward_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_24: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, getitem_63)
        mul_143: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
        unsqueeze_76: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_147, -1)
        unsqueeze_77: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_149: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_143, unsqueeze_77);  mul_143 = unsqueeze_77 = None
        unsqueeze_78: "f32[384, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_148, -1);  primals_148 = None
        unsqueeze_79: "f32[384, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_131: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_79);  mul_149 = unsqueeze_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_16: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.neg.default(add_131)
        exp_16: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.exp.default(neg_16);  neg_16 = None
        add_132: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.add.Tensor(exp_16, 1);  exp_16 = None
        reciprocal_17: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.reciprocal.default(add_132);  add_132 = None
        mul_554: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        mul_555: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(getitem_236, mul_554);  getitem_236 = None
        sub_166: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_554);  mul_554 = None
        mul_556: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(add_131, sub_166);  add_131 = sub_166 = None
        add_307: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.add.Tensor(mul_556, 1);  mul_556 = None
        mul_557: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_555, add_307);  mul_555 = add_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_57: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_63, [0, 2, 3]);  getitem_63 = None
        unsqueeze_272: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_57, 0);  squeeze_57 = None
        unsqueeze_273: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 2);  unsqueeze_272 = None
        unsqueeze_274: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 3);  unsqueeze_273 = None
        sum_118: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 2, 3])
        sub_167: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_274);  convolution_20 = unsqueeze_274 = None
        mul_558: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(mul_557, sub_167)
        sum_119: "f32[384][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_558, [0, 2, 3]);  mul_558 = None
        mul_559: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_118, 7.62939453125e-06)
        unsqueeze_275: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
        unsqueeze_276: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
        unsqueeze_277: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
        mul_560: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, 7.62939453125e-06)
        squeeze_58: "f32[384][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_24, [0, 2, 3]);  rsqrt_24 = None
        mul_561: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
        mul_562: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_560, mul_561);  mul_560 = mul_561 = None
        unsqueeze_278: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
        unsqueeze_279: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
        unsqueeze_280: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
        mul_563: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_58, primals_147);  primals_147 = None
        unsqueeze_281: "f32[1, 384][384, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
        unsqueeze_282: "f32[1, 384, 1][384, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
        unsqueeze_283: "f32[1, 384, 1, 1][384, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
        mul_564: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_280);  sub_167 = unsqueeze_280 = None
        sub_169: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(mul_557, mul_564);  mul_557 = mul_564 = None
        sub_170: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.sub.Tensor(sub_169, unsqueeze_277);  sub_169 = unsqueeze_277 = None
        mul_565: "f32[128, 384, 32, 32][393216, 1, 12288, 384]cuda:0" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_283);  sub_170 = unsqueeze_283 = None
        mul_566: "f32[384][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_119, squeeze_58);  sum_119 = squeeze_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_565, div_15, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_565 = div_15 = primals_143 = None
        getitem_239: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = convolution_backward_14[0]
        getitem_240: "f32[384, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_14[1];  convolution_backward_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_23: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, getitem_61)
        mul_136: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
        unsqueeze_72: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_141, -1)
        unsqueeze_73: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        mul_142: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_73);  mul_136 = unsqueeze_73 = None
        unsqueeze_74: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_142, -1);  primals_142 = None
        unsqueeze_75: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        add_125: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_142, unsqueeze_75);  mul_142 = unsqueeze_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.neg.default(add_125)
        exp_15: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.exp.default(neg_15);  neg_15 = None
        add_126: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_15, 1);  exp_15 = None
        reciprocal_18: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.reciprocal.default(add_126);  add_126 = None
        mul_567: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        mul_568: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(getitem_239, mul_567);  getitem_239 = None
        sub_171: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_567);  mul_567 = None
        mul_569: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(add_125, sub_171);  add_125 = sub_171 = None
        add_309: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_569, 1);  mul_569 = None
        mul_570: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_568, add_309);  mul_568 = add_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_54: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_61, [0, 2, 3]);  getitem_61 = None
        unsqueeze_284: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_54, 0);  squeeze_54 = None
        unsqueeze_285: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 2);  unsqueeze_284 = None
        unsqueeze_286: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 3);  unsqueeze_285 = None
        sum_120: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 2, 3])
        sub_172: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_286);  convolution_19 = unsqueeze_286 = None
        mul_571: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_570, sub_172)
        sum_121: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
        mul_572: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_120, 7.62939453125e-06)
        unsqueeze_287: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
        unsqueeze_288: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
        unsqueeze_289: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
        mul_573: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, 7.62939453125e-06)
        squeeze_55: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_23, [0, 2, 3]);  rsqrt_23 = None
        mul_574: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
        mul_575: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
        unsqueeze_290: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
        unsqueeze_291: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
        unsqueeze_292: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
        mul_576: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_55, primals_141);  primals_141 = None
        unsqueeze_293: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
        unsqueeze_294: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
        unsqueeze_295: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
        mul_577: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_292);  sub_172 = unsqueeze_292 = None
        sub_174: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(mul_570, mul_577);  mul_570 = mul_577 = None
        sub_175: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_289);  sub_174 = unsqueeze_289 = None
        mul_578: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_295);  sub_175 = unsqueeze_295 = None
        mul_579: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_121, squeeze_55);  sum_121 = squeeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_578, cat, primals_137, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_578 = cat = primals_137 = None
        getitem_242: "f32[128, 192, 32, 32][196608, 1, 6144, 192]cuda:0" = convolution_backward_15[0]
        getitem_243: "f32[96, 192, 3, 3][1728, 1, 576, 192]cuda:0" = convolution_backward_15[1];  convolution_backward_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:277 in forward, code: x = self.conv_fusion(torch.cat((shortcut, x), dim=1))
        slice_5: "f32[128, 96, 32, 32][196608, 1, 6144, 192]cuda:0" = torch.ops.aten.slice.Tensor(getitem_242, 1, 0, 96)
        slice_6: "f32[128, 96, 32, 32][196608, 1, 6144, 192]cuda:0" = torch.ops.aten.slice.Tensor(getitem_242, 1, 96, 192);  getitem_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_22: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, getitem_59)
        mul_129: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
        unsqueeze_68: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_135, -1)
        unsqueeze_69: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_135: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_129, unsqueeze_69);  mul_129 = unsqueeze_69 = None
        unsqueeze_70: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_136, -1);  primals_136 = None
        unsqueeze_71: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_119: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_135, unsqueeze_71);  mul_135 = unsqueeze_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_14: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.neg.default(add_119)
        exp_14: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.exp.default(neg_14);  neg_14 = None
        add_120: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_14, 1);  exp_14 = None
        reciprocal_19: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.reciprocal.default(add_120);  add_120 = None
        mul_580: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        mul_581: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(slice_6, mul_580);  slice_6 = None
        sub_176: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_580);  mul_580 = None
        mul_582: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(add_119, sub_176);  add_119 = sub_176 = None
        add_311: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_582, 1);  mul_582 = None
        mul_583: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_581, add_311);  mul_581 = add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_51: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_59, [0, 2, 3]);  getitem_59 = None
        unsqueeze_296: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_51, 0);  squeeze_51 = None
        unsqueeze_297: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 2);  unsqueeze_296 = None
        unsqueeze_298: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 3);  unsqueeze_297 = None
        sum_122: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_583, [0, 2, 3])
        sub_177: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_298);  convolution_18 = unsqueeze_298 = None
        mul_584: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_583, sub_177)
        sum_123: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3]);  mul_584 = None
        mul_585: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_122, 7.62939453125e-06)
        unsqueeze_299: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
        unsqueeze_300: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
        unsqueeze_301: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
        mul_586: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, 7.62939453125e-06)
        squeeze_52: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_22, [0, 2, 3]);  rsqrt_22 = None
        mul_587: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
        mul_588: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
        unsqueeze_302: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
        unsqueeze_303: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
        unsqueeze_304: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
        mul_589: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_52, primals_135);  primals_135 = None
        unsqueeze_305: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
        unsqueeze_306: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
        unsqueeze_307: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
        mul_590: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_304);  sub_177 = unsqueeze_304 = None
        sub_179: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(mul_583, mul_590);  mul_583 = mul_590 = None
        sub_180: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_301);  sub_179 = unsqueeze_301 = None
        mul_591: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_307);  sub_180 = unsqueeze_307 = None
        mul_592: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_123, squeeze_52);  sum_123 = squeeze_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_591, view_25, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_591 = view_25 = primals_131 = None
        getitem_245: "f32[128, 144, 32, 32][147456, 1, 4608, 144]cuda:0" = convolution_backward_16[0]
        getitem_246: "f32[96, 144, 1, 1][144, 1, 144, 144]cuda:0" = convolution_backward_16[1];  convolution_backward_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:271 in forward, code: x = x.transpose(1, 2).reshape(B, C, num_patch_h * patch_h, num_patch_w * patch_w)
        clone_85: "f32[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0" = torch.ops.aten.clone.default(getitem_245, memory_format = torch.contiguous_format);  getitem_245 = None
        view_228: "f32[294912, 2, 16, 2][64, 32, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_85, [294912, 2, 16, 2]);  clone_85 = None
        permute_205: "f32[294912, 16, 2, 2][64, 2, 32, 1]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:269 in forward, code: x = x.transpose(1, 3).reshape(B * C * num_patch_h, num_patch_w, patch_h, patch_w)
        clone_86: "f32[294912, 16, 2, 2][64, 4, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_229: "f32[128, 144, 256, 4][147456, 1024, 4, 1]cuda:0" = torch.ops.aten.reshape.default(clone_86, [128, 144, 256, 4]);  clone_86 = None
        permute_206: "f32[128, 4, 256, 144][147456, 1, 4, 1024]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 3, 2, 1]);  view_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:268 in forward, code: x = x.contiguous().view(B, self.patch_area, num_patches, -1)
        clone_87: "f32[128, 4, 256, 144][147456, 36864, 144, 1]cuda:0" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
        view_230: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(clone_87, [512, 256, 144]);  clone_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:264 in forward, code: x = self.norm(x)
        mul_594: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_230, primals_129);  primals_129 = None
        mul_595: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_594, 144)
        sum_124: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_594, [2], True)
        mul_596: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_594, mul_127);  mul_594 = None
        sum_125: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_596, [2], True);  mul_596 = None
        mul_597: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_127, sum_125);  sum_125 = None
        sub_182: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_595, sum_124);  mul_595 = sum_124 = None
        sub_183: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_182, mul_597);  sub_182 = mul_597 = None
        mul_598: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_51, sub_183);  div_51 = sub_183 = None
        mul_599: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_230, mul_127);  mul_127 = None
        sum_126: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 1]);  mul_599 = None
        sum_127: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_230, [0, 1]);  view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_231: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(mul_598, [131072, 144])
        permute_13: "f32[288, 144][1, 288]cuda:0" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_207: "f32[144, 288][288, 1]cuda:0" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_58: "f32[131072, 288][288, 1]cuda:0" = torch.ops.aten.mm.default(view_231, permute_207);  permute_207 = None
        permute_208: "f32[144, 131072][1, 144]cuda:0" = torch.ops.aten.permute.default(view_231, [1, 0])
        mm_59: "f32[144, 288][288, 1]cuda:0" = torch.ops.aten.mm.default(permute_208, view_21);  permute_208 = view_21 = None
        sum_128: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
        view_232: "f32[144][1]cuda:0" = torch.ops.aten.reshape.default(sum_128, [144]);  sum_128 = None
        view_233: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [512, 256, 288]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_20: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [512, 256, 288]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_13: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.neg.default(view_20)
        exp_13: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.exp.default(neg_13);  neg_13 = None
        add_111: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_13, 1);  exp_13 = None
        reciprocal_20: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reciprocal.default(add_111);  add_111 = None
        mul_600: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        mul_601: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_233, mul_600);  view_233 = None
        sub_184: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_600);  mul_600 = None
        mul_602: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_20, sub_184);  view_20 = sub_184 = None
        add_313: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_602, 1);  mul_602 = None
        mul_603: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_601, add_313);  mul_601 = add_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_234: "f32[131072, 288][288, 1]cuda:0" = torch.ops.aten.reshape.default(mul_603, [131072, 288]);  mul_603 = None
        permute_12: "f32[144, 288][1, 144]cuda:0" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        permute_211: "f32[288, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_60: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_234, permute_211);  permute_211 = None
        permute_212: "f32[288, 131072][1, 288]cuda:0" = torch.ops.aten.permute.default(view_234, [1, 0])
        mm_61: "f32[288, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_212, view_19);  permute_212 = view_19 = None
        sum_129: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
        view_235: "f32[288][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [288]);  sum_129 = None
        view_236: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [512, 256, 144]);  mm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_605: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_236, primals_123);  primals_123 = None
        mul_606: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_605, 144)
        sum_130: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_605, [2], True)
        mul_607: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_605, mul_125);  mul_605 = None
        sum_131: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_607, [2], True);  mul_607 = None
        mul_608: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, sum_131);  sum_131 = None
        sub_186: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_606, sum_130);  mul_606 = sum_130 = None
        sub_187: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_186, mul_608);  sub_186 = mul_608 = None
        mul_609: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, sub_187);  div_52 = sub_187 = None
        mul_610: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_236, mul_125);  mul_125 = None
        sum_132: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_610, [0, 1]);  mul_610 = None
        sum_133: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_236, [0, 1]);  view_236 = None
        add_314: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_598, mul_609);  mul_598 = mul_609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_237: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(add_314, [131072, 144])
        permute_11: "f32[144, 144][1, 144]cuda:0" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        permute_215: "f32[144, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_62: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_237, permute_215);  permute_215 = None
        permute_216: "f32[144, 131072][1, 144]cuda:0" = torch.ops.aten.permute.default(view_237, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_10: "f32[512, 256, 4, 36][36864, 144, 36, 1]cuda:0" = torch.ops.aten.permute.default(getitem_50, [0, 2, 1, 3])
        view_16: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_10, [512, 256, 144]);  permute_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_17: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(view_16, [131072, 144]);  view_16 = None
        mm_63: "f32[144, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_216, view_17);  permute_216 = view_17 = None
        sum_134: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_237, [0], True);  view_237 = None
        view_238: "f32[144][1]cuda:0" = torch.ops.aten.reshape.default(sum_134, [144]);  sum_134 = None
        view_239: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [512, 256, 144]);  mm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_240: "f32[512, 256, 4, 36][36864, 144, 36, 1]cuda:0" = torch.ops.aten.reshape.default(view_239, [512, 256, 4, 36]);  view_239 = None
        permute_219: "f32[512, 4, 256, 36][36864, 36, 144, 1]cuda:0" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_219, getitem_47, getitem_48, getitem_49, None, getitem_50, getitem_51, getitem_52, getitem_53, 0.0, [True, True, True, False]);  permute_219 = getitem_47 = getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = None
        getitem_248: "f32[512, 4, 256, 36][36864, 36, 144, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[0]
        getitem_249: "f32[512, 4, 256, 36][36864, 36, 144, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[1]
        getitem_250: "f32[512, 4, 256, 36][36864, 36, 144, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_10: "f32[1536, 4, 256, 36][36864, 9216, 36, 1]cuda:0" = torch.ops.aten.cat.default([getitem_248, getitem_249, getitem_250]);  getitem_248 = getitem_249 = getitem_250 = None
        view_241: "f32[3, 512, 4, 256, 36][18874368, 36864, 9216, 36, 1]cuda:0" = torch.ops.aten.reshape.default(cat_10, [3, 512, 4, 256, 36]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_220: "f32[512, 256, 3, 4, 36][36864, 36, 18874368, 9216, 1]cuda:0" = torch.ops.aten.permute.default(view_241, [1, 3, 0, 2, 4]);  view_241 = None
        clone_88: "f32[512, 256, 3, 4, 36][110592, 432, 144, 36, 1]cuda:0" = torch.ops.aten.clone.default(permute_220, memory_format = torch.contiguous_format);  permute_220 = None
        view_242: "f32[512, 256, 432][110592, 432, 1]cuda:0" = torch.ops.aten.reshape.default(clone_88, [512, 256, 432]);  clone_88 = None
        view_243: "f32[131072, 432][432, 1]cuda:0" = torch.ops.aten.reshape.default(view_242, [131072, 432]);  view_242 = None
        permute_8: "f32[144, 432][1, 144]cuda:0" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
        permute_221: "f32[432, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_64: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_243, permute_221);  permute_221 = None
        permute_222: "f32[432, 131072][1, 432]cuda:0" = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_65: "f32[432, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_222, view_13);  permute_222 = view_13 = None
        sum_135: "f32[1, 432][432, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
        view_244: "f32[432][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [432]);  sum_135 = None
        view_245: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [512, 256, 144]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_612: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_245, primals_117);  primals_117 = None
        mul_613: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_612, 144)
        sum_136: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_612, [2], True)
        mul_614: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_612, mul_123);  mul_612 = None
        sum_137: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_614, [2], True);  mul_614 = None
        mul_615: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, sum_137);  sum_137 = None
        sub_189: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_613, sum_136);  mul_613 = sum_136 = None
        sub_190: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_189, mul_615);  sub_189 = mul_615 = None
        mul_616: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_53, sub_190);  div_53 = sub_190 = None
        mul_617: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_245, mul_123);  mul_123 = None
        sum_138: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 1]);  mul_617 = None
        sum_139: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_245, [0, 1]);  view_245 = None
        add_315: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_314, mul_616);  add_314 = mul_616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_246: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(add_315, [131072, 144])
        permute_7: "f32[288, 144][1, 288]cuda:0" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_225: "f32[144, 288][288, 1]cuda:0" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        mm_66: "f32[131072, 288][288, 1]cuda:0" = torch.ops.aten.mm.default(view_246, permute_225);  permute_225 = None
        permute_226: "f32[144, 131072][1, 144]cuda:0" = torch.ops.aten.permute.default(view_246, [1, 0])
        mm_67: "f32[144, 288][288, 1]cuda:0" = torch.ops.aten.mm.default(permute_226, view_11);  permute_226 = view_11 = None
        sum_140: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
        view_247: "f32[144][1]cuda:0" = torch.ops.aten.reshape.default(sum_140, [144]);  sum_140 = None
        view_248: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [512, 256, 288]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_10: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [512, 256, 288]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        neg_12: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.neg.default(view_10)
        exp_12: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.exp.default(neg_12);  neg_12 = None
        add_104: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.add.Tensor(exp_12, 1);  exp_12 = None
        reciprocal_21: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.reciprocal.default(add_104);  add_104 = None
        mul_618: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        mul_619: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_248, mul_618);  view_248 = None
        sub_191: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_618);  mul_618 = None
        mul_620: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_10, sub_191);  view_10 = sub_191 = None
        add_317: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_620, 1);  mul_620 = None
        mul_621: "f32[512, 256, 288][73728, 288, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_619, add_317);  mul_619 = add_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_249: "f32[131072, 288][288, 1]cuda:0" = torch.ops.aten.reshape.default(mul_621, [131072, 288]);  mul_621 = None
        permute_6: "f32[144, 288][1, 144]cuda:0" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        permute_229: "f32[288, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
        mm_68: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_249, permute_229);  permute_229 = None
        permute_230: "f32[288, 131072][1, 288]cuda:0" = torch.ops.aten.permute.default(view_249, [1, 0])
        mm_69: "f32[288, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_230, view_9);  permute_230 = view_9 = None
        sum_141: "f32[1, 288][288, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
        view_250: "f32[288][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [288]);  sum_141 = None
        view_251: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [512, 256, 144]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        mul_623: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_251, primals_111);  primals_111 = None
        mul_624: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_623, 144)
        sum_142: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True)
        mul_625: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_623, mul_121);  mul_623 = None
        sum_143: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_625, [2], True);  mul_625 = None
        mul_626: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, sum_143);  sum_143 = None
        sub_193: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_624, sum_142);  mul_624 = sum_142 = None
        sub_194: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_193, mul_626);  sub_193 = mul_626 = None
        mul_627: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_54, sub_194);  div_54 = sub_194 = None
        mul_628: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_251, mul_121);  mul_121 = None
        sum_144: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_628, [0, 1]);  mul_628 = None
        sum_145: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_251, [0, 1]);  view_251 = None
        add_318: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_315, mul_627);  add_315 = mul_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_252: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(add_318, [131072, 144])
        permute_5: "f32[144, 144][1, 144]cuda:0" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_233: "f32[144, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        mm_70: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_252, permute_233);  permute_233 = None
        permute_234: "f32[144, 131072][1, 144]cuda:0" = torch.ops.aten.permute.default(view_252, [1, 0])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_4: "f32[512, 256, 4, 36][36864, 144, 36, 1]cuda:0" = torch.ops.aten.permute.default(getitem_39, [0, 2, 1, 3])
        view_6: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(permute_4, [512, 256, 144]);  permute_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_7: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.reshape.default(view_6, [131072, 144]);  view_6 = None
        mm_71: "f32[144, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_234, view_7);  permute_234 = view_7 = None
        sum_146: "f32[1, 144][144, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
        view_253: "f32[144][1]cuda:0" = torch.ops.aten.reshape.default(sum_146, [144]);  sum_146 = None
        view_254: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [512, 256, 144]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        view_255: "f32[512, 256, 4, 36][36864, 144, 36, 1]cuda:0" = torch.ops.aten.reshape.default(view_254, [512, 256, 4, 36]);  view_254 = None
        permute_237: "f32[512, 4, 256, 36][36864, 36, 144, 1]cuda:0" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_237, getitem_36, getitem_37, getitem_38, None, getitem_39, getitem_40, getitem_41, getitem_42, 0.0, [True, True, True, False]);  permute_237 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = getitem_42 = None
        getitem_252: "f32[512, 4, 256, 36][36864, 36, 144, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[0]
        getitem_253: "f32[512, 4, 256, 36][36864, 36, 144, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[1]
        getitem_254: "f32[512, 4, 256, 36][36864, 36, 144, 1]cuda:0" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        cat_11: "f32[1536, 4, 256, 36][36864, 9216, 36, 1]cuda:0" = torch.ops.aten.cat.default([getitem_252, getitem_253, getitem_254]);  getitem_252 = getitem_253 = getitem_254 = None
        view_256: "f32[3, 512, 4, 256, 36][18874368, 36864, 9216, 36, 1]cuda:0" = torch.ops.aten.reshape.default(cat_11, [3, 512, 4, 256, 36]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        permute_238: "f32[512, 256, 3, 4, 36][36864, 36, 18874368, 9216, 1]cuda:0" = torch.ops.aten.permute.default(view_256, [1, 3, 0, 2, 4]);  view_256 = None
        clone_89: "f32[512, 256, 3, 4, 36][110592, 432, 144, 36, 1]cuda:0" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
        view_257: "f32[512, 256, 432][110592, 432, 1]cuda:0" = torch.ops.aten.reshape.default(clone_89, [512, 256, 432]);  clone_89 = None
        view_258: "f32[131072, 432][432, 1]cuda:0" = torch.ops.aten.reshape.default(view_257, [131072, 432]);  view_257 = None
        permute_2: "f32[144, 432][1, 144]cuda:0" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        permute_239: "f32[432, 144][144, 1]cuda:0" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_72: "f32[131072, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(view_258, permute_239);  permute_239 = None
        permute_240: "f32[432, 131072][1, 432]cuda:0" = torch.ops.aten.permute.default(view_258, [1, 0])
        mm_73: "f32[432, 144][144, 1]cuda:0" = torch.ops.aten.mm.default(permute_240, view_3);  permute_240 = view_3 = None
        sum_147: "f32[1, 432][432, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
        view_259: "f32[432][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [432]);  sum_147 = None
        view_260: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [512, 256, 144]);  mm_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        mul_630: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_260, primals_105);  primals_105 = None
        mul_631: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_630, 144)
        sum_148: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_630, [2], True)
        mul_632: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_630, mul_119);  mul_630 = None
        sum_149: "f32[512, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_632, [2], True);  mul_632 = None
        mul_633: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, sum_149);  sum_149 = None
        sub_196: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_631, sum_148);  mul_631 = sum_148 = None
        sub_197: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_196, mul_633);  sub_196 = mul_633 = None
        mul_634: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_55, sub_197);  div_55 = sub_197 = None
        mul_635: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_260, mul_119);  mul_119 = None
        sum_150: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_635, [0, 1]);  mul_635 = None
        sum_151: "f32[144][1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_260, [0, 1]);  view_260 = None
        add_319: "f32[512, 256, 144][36864, 144, 1]cuda:0" = torch.ops.aten.add.Tensor(add_318, mul_634);  add_318 = mul_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:260 in forward, code: x = x.reshape(B, C, num_patches, self.patch_area).transpose(1, 3).reshape(B * self.patch_area, num_patches, -1)
        view_261: "f32[128, 4, 256, 144][147456, 36864, 144, 1]cuda:0" = torch.ops.aten.reshape.default(add_319, [128, 4, 256, 144]);  add_319 = None
        permute_243: "f32[128, 144, 256, 4][147456, 1, 144, 36864]cuda:0" = torch.ops.aten.permute.default(view_261, [0, 3, 2, 1]);  view_261 = None
        clone_90: "f32[128, 144, 256, 4][147456, 1024, 4, 1]cuda:0" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
        view_262: "f32[294912, 16, 2, 2][64, 4, 2, 1]cuda:0" = torch.ops.aten.reshape.default(clone_90, [294912, 16, 2, 2]);  clone_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:258 in forward, code: x = x.reshape(B * C * num_patch_h, patch_h, num_patch_w, patch_w).transpose(1, 2)
        permute_244: "f32[294912, 2, 16, 2][64, 2, 4, 1]cuda:0" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
        clone_91: "f32[294912, 2, 16, 2][64, 32, 2, 1]cuda:0" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
        view_263: "f32[128, 144, 32, 32][147456, 1024, 32, 1]cuda:0" = torch.ops.aten.reshape.default(clone_91, [128, 144, 32, 32]);  clone_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/mobilevit.py:243 in forward, code: x = self.conv_1x1(x)
        convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_263, div_11, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  view_263 = div_11 = primals_104 = None
        getitem_256: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = convolution_backward_17[0]
        getitem_257: "f32[144, 96, 1, 1][96, 1, 96, 96]cuda:0" = convolution_backward_17[1];  convolution_backward_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_16: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, getitem_33)
        mul_112: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
        unsqueeze_64: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_102, -1)
        unsqueeze_65: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        mul_118: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_65);  mul_112 = unsqueeze_65 = None
        unsqueeze_66: "f32[96, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_103, -1);  primals_103 = None
        unsqueeze_67: "f32[96, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        add_97: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_118, unsqueeze_67);  mul_118 = unsqueeze_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_11: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.neg.default(add_97)
        exp_11: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.exp.default(neg_11);  neg_11 = None
        add_98: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(exp_11, 1);  exp_11 = None
        reciprocal_22: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.reciprocal.default(add_98);  add_98 = None
        mul_636: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        mul_637: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(getitem_256, mul_636);  getitem_256 = None
        sub_198: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_636);  mul_636 = None
        mul_638: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(add_97, sub_198);  add_97 = sub_198 = None
        add_321: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(mul_638, 1);  mul_638 = None
        mul_639: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_637, add_321);  mul_637 = add_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_48: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_33, [0, 2, 3]);  getitem_33 = None
        unsqueeze_308: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_48, 0);  squeeze_48 = None
        unsqueeze_309: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 2);  unsqueeze_308 = None
        unsqueeze_310: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 3);  unsqueeze_309 = None
        sum_152: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_639, [0, 2, 3])
        sub_199: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_310);  convolution_16 = unsqueeze_310 = None
        mul_640: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(mul_639, sub_199)
        sum_153: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 2, 3]);  mul_640 = None
        mul_641: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_152, 7.62939453125e-06)
        unsqueeze_311: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_641, 0);  mul_641 = None
        unsqueeze_312: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
        unsqueeze_313: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
        mul_642: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, 7.62939453125e-06)
        squeeze_49: "f32[96][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_16, [0, 2, 3]);  rsqrt_16 = None
        mul_643: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
        mul_644: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
        unsqueeze_314: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
        unsqueeze_315: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
        unsqueeze_316: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
        mul_645: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_49, primals_102);  primals_102 = None
        unsqueeze_317: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
        unsqueeze_318: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
        unsqueeze_319: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
        mul_646: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_316);  sub_199 = unsqueeze_316 = None
        sub_201: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(mul_639, mul_646);  mul_639 = mul_646 = None
        sub_202: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_313);  sub_201 = unsqueeze_313 = None
        mul_647: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_319);  sub_202 = unsqueeze_319 = None
        mul_648: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_153, squeeze_49);  sum_153 = squeeze_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_647, add_92, primals_98, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_647 = add_92 = primals_98 = None
        getitem_259: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = convolution_backward_18[0]
        getitem_260: "f32[96, 96, 3, 3][864, 1, 288, 96]cuda:0" = convolution_backward_18[1];  convolution_backward_18 = None
        add_322: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.add.Tensor(slice_5, getitem_259);  slice_5 = getitem_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_154: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_322, [0, 2, 3])
        sub_203: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_322);  convolution_15 = unsqueeze_322 = None
        mul_649: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(add_322, sub_203)
        sum_155: "f32[96][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_649, [0, 2, 3]);  mul_649 = None
        mul_650: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_154, 7.62939453125e-06)
        unsqueeze_323: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_650, 0);  mul_650 = None
        unsqueeze_324: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
        unsqueeze_325: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
        mul_651: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, 7.62939453125e-06)
        mul_652: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
        mul_653: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_651, mul_652);  mul_651 = mul_652 = None
        unsqueeze_326: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
        unsqueeze_327: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
        unsqueeze_328: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
        mul_654: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_46, primals_96);  primals_96 = None
        unsqueeze_329: "f32[1, 96][96, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
        unsqueeze_330: "f32[1, 96, 1][96, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
        unsqueeze_331: "f32[1, 96, 1, 1][96, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
        mul_655: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_328);  sub_203 = unsqueeze_328 = None
        sub_205: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(add_322, mul_655);  add_322 = mul_655 = None
        sub_206: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.sub.Tensor(sub_205, unsqueeze_325);  sub_205 = unsqueeze_325 = None
        mul_656: "f32[128, 96, 32, 32][98304, 1, 3072, 96]cuda:0" = torch.ops.aten.mul.Tensor(sub_206, unsqueeze_331);  sub_206 = unsqueeze_331 = None
        mul_657: "f32[96][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_155, squeeze_46);  sum_155 = squeeze_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_656, div_10, primals_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_656 = div_10 = primals_92 = None
        getitem_262: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = convolution_backward_19[0]
        getitem_263: "f32[96, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_19[1];  convolution_backward_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_14: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, getitem_29)
        mul_98: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
        unsqueeze_56: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_90, -1)
        unsqueeze_57: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        mul_104: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, unsqueeze_57);  mul_98 = unsqueeze_57 = None
        unsqueeze_58: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_91, -1);  primals_91 = None
        unsqueeze_59: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        add_86: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_59);  mul_104 = unsqueeze_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_10: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.neg.default(add_86)
        exp_10: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.exp.default(neg_10);  neg_10 = None
        add_87: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_10, 1);  exp_10 = None
        reciprocal_23: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.reciprocal.default(add_87);  add_87 = None
        mul_658: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        mul_659: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(getitem_262, mul_658);  getitem_262 = None
        sub_207: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_658);  mul_658 = None
        mul_660: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(add_86, sub_207);  add_86 = sub_207 = None
        add_324: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_660, 1);  mul_660 = None
        mul_661: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_659, add_324);  mul_659 = add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_42: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_29, [0, 2, 3]);  getitem_29 = None
        unsqueeze_332: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_42, 0);  squeeze_42 = None
        unsqueeze_333: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 2);  unsqueeze_332 = None
        unsqueeze_334: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 3);  unsqueeze_333 = None
        sum_156: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3])
        sub_208: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_334);  convolution_14 = unsqueeze_334 = None
        mul_662: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_661, sub_208)
        sum_157: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3]);  mul_662 = None
        mul_663: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_156, 7.62939453125e-06)
        unsqueeze_335: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
        unsqueeze_336: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
        unsqueeze_337: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
        mul_664: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, 7.62939453125e-06)
        squeeze_43: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_14, [0, 2, 3]);  rsqrt_14 = None
        mul_665: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
        mul_666: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
        unsqueeze_338: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
        unsqueeze_339: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
        unsqueeze_340: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
        mul_667: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_43, primals_90);  primals_90 = None
        unsqueeze_341: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
        unsqueeze_342: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
        unsqueeze_343: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
        mul_668: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_340);  sub_208 = unsqueeze_340 = None
        sub_210: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(mul_661, mul_668);  mul_661 = mul_668 = None
        sub_211: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_337);  sub_210 = unsqueeze_337 = None
        mul_669: "f32[128, 256, 32, 32][262144, 1, 8192, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_343);  sub_211 = unsqueeze_343 = None
        mul_670: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_157, squeeze_43);  sum_157 = squeeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_669, div_9, primals_86, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_669 = div_9 = primals_86 = None
        getitem_265: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = convolution_backward_20[0]
        getitem_266: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_20[1];  convolution_backward_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_13: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, getitem_27)
        mul_91: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
        unsqueeze_52: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_84, -1)
        unsqueeze_53: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_97: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_53);  mul_91 = unsqueeze_53 = None
        unsqueeze_54: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_85, -1);  primals_85 = None
        unsqueeze_55: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_80: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_97, unsqueeze_55);  mul_97 = unsqueeze_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_9: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(add_80)
        exp_9: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_9);  neg_9 = None
        add_81: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_9, 1);  exp_9 = None
        reciprocal_24: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.reciprocal.default(add_81);  add_81 = None
        mul_671: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        mul_672: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(getitem_265, mul_671);  getitem_265 = None
        sub_212: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_671);  mul_671 = None
        mul_673: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(add_80, sub_212);  add_80 = sub_212 = None
        add_326: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_673, 1);  mul_673 = None
        mul_674: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_672, add_326);  mul_672 = add_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_39: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_27, [0, 2, 3]);  getitem_27 = None
        unsqueeze_344: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_39, 0);  squeeze_39 = None
        unsqueeze_345: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 2);  unsqueeze_344 = None
        unsqueeze_346: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 3);  unsqueeze_345 = None
        sum_158: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_674, [0, 2, 3])
        sub_213: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_346);  convolution_13 = unsqueeze_346 = None
        mul_675: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_674, sub_213)
        sum_159: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_675, [0, 2, 3]);  mul_675 = None
        mul_676: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_158, 1.9073486328125e-06)
        unsqueeze_347: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
        unsqueeze_348: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
        unsqueeze_349: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
        mul_677: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, 1.9073486328125e-06)
        squeeze_40: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_13, [0, 2, 3]);  rsqrt_13 = None
        mul_678: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
        mul_679: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_677, mul_678);  mul_677 = mul_678 = None
        unsqueeze_350: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_679, 0);  mul_679 = None
        unsqueeze_351: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
        unsqueeze_352: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
        mul_680: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_40, primals_84);  primals_84 = None
        unsqueeze_353: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
        unsqueeze_354: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
        unsqueeze_355: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
        mul_681: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_352);  sub_213 = unsqueeze_352 = None
        sub_215: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(mul_674, mul_681);  mul_674 = mul_681 = None
        sub_216: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_349);  sub_215 = unsqueeze_349 = None
        mul_682: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_355);  sub_216 = unsqueeze_355 = None
        mul_683: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_159, squeeze_40);  sum_159 = squeeze_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_682, add_75, primals_80, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_682 = add_75 = primals_80 = None
        getitem_268: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = convolution_backward_21[0]
        getitem_269: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_21[1];  convolution_backward_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_160: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_268, [0, 2, 3])
        sub_217: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_358);  convolution_12 = unsqueeze_358 = None
        mul_684: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(getitem_268, sub_217)
        sum_161: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_684, [0, 2, 3]);  mul_684 = None
        mul_685: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_160, 1.9073486328125e-06)
        unsqueeze_359: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
        unsqueeze_360: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
        unsqueeze_361: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
        mul_686: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, 1.9073486328125e-06)
        mul_687: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
        mul_688: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_686, mul_687);  mul_686 = mul_687 = None
        unsqueeze_362: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
        unsqueeze_363: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
        unsqueeze_364: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
        mul_689: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_37, primals_78);  primals_78 = None
        unsqueeze_365: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
        unsqueeze_366: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
        unsqueeze_367: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
        mul_690: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_364);  sub_217 = unsqueeze_364 = None
        sub_219: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(getitem_268, mul_690);  mul_690 = None
        sub_220: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_361);  sub_219 = unsqueeze_361 = None
        mul_691: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_367);  sub_220 = unsqueeze_367 = None
        mul_692: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_161, squeeze_37);  sum_161 = squeeze_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_691, div_8, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_691 = div_8 = primals_74 = None
        getitem_271: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = convolution_backward_22[0]
        getitem_272: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_22[1];  convolution_backward_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_11: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, getitem_23)
        mul_77: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
        unsqueeze_44: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_72, -1)
        unsqueeze_45: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_83: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, unsqueeze_45);  mul_77 = unsqueeze_45 = None
        unsqueeze_46: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_73, -1);  primals_73 = None
        unsqueeze_47: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_68: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_47);  mul_83 = unsqueeze_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(add_68)
        exp_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_8);  neg_8 = None
        add_69: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_8, 1);  exp_8 = None
        reciprocal_25: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.reciprocal.default(add_69);  add_69 = None
        mul_693: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        mul_694: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(getitem_271, mul_693);  getitem_271 = None
        sub_221: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_693);  mul_693 = None
        mul_695: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(add_68, sub_221);  add_68 = sub_221 = None
        add_328: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_695, 1);  mul_695 = None
        mul_696: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_694, add_328);  mul_694 = add_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_33: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_23, [0, 2, 3]);  getitem_23 = None
        unsqueeze_368: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_33, 0);  squeeze_33 = None
        unsqueeze_369: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
        unsqueeze_370: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
        sum_162: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_696, [0, 2, 3])
        sub_222: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_370);  convolution_11 = unsqueeze_370 = None
        mul_697: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_696, sub_222)
        sum_163: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
        mul_698: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_162, 1.9073486328125e-06)
        unsqueeze_371: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
        unsqueeze_372: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
        unsqueeze_373: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
        mul_699: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, 1.9073486328125e-06)
        squeeze_34: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_11, [0, 2, 3]);  rsqrt_11 = None
        mul_700: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
        mul_701: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
        unsqueeze_374: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
        unsqueeze_375: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
        unsqueeze_376: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
        mul_702: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_34, primals_72);  primals_72 = None
        unsqueeze_377: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
        unsqueeze_378: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
        unsqueeze_379: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
        mul_703: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_222, unsqueeze_376);  sub_222 = unsqueeze_376 = None
        sub_224: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(mul_696, mul_703);  mul_696 = mul_703 = None
        sub_225: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_224, unsqueeze_373);  sub_224 = unsqueeze_373 = None
        mul_704: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_379);  sub_225 = unsqueeze_379 = None
        mul_705: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_163, squeeze_34);  sum_163 = squeeze_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_704, div_7, primals_68, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_704 = div_7 = primals_68 = None
        getitem_274: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = convolution_backward_23[0]
        getitem_275: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_23[1];  convolution_backward_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_10: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, getitem_21)
        mul_70: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
        unsqueeze_40: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_66, -1)
        unsqueeze_41: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        mul_76: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_41);  mul_70 = unsqueeze_41 = None
        unsqueeze_42: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_67, -1);  primals_67 = None
        unsqueeze_43: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        add_62: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_76, unsqueeze_43);  mul_76 = unsqueeze_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(add_62)
        exp_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_7);  neg_7 = None
        add_63: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_7, 1);  exp_7 = None
        reciprocal_26: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.reciprocal.default(add_63);  add_63 = None
        mul_706: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        mul_707: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(getitem_274, mul_706);  getitem_274 = None
        sub_226: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_706);  mul_706 = None
        mul_708: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(add_62, sub_226);  add_62 = sub_226 = None
        add_330: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_708, 1);  mul_708 = None
        mul_709: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_707, add_330);  mul_707 = add_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_30: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_21, [0, 2, 3]);  getitem_21 = None
        unsqueeze_380: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_30, 0);  squeeze_30 = None
        unsqueeze_381: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
        unsqueeze_382: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
        sum_164: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_709, [0, 2, 3])
        sub_227: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_382);  convolution_10 = unsqueeze_382 = None
        mul_710: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_709, sub_227)
        sum_165: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_710, [0, 2, 3]);  mul_710 = None
        mul_711: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_164, 1.9073486328125e-06)
        unsqueeze_383: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
        unsqueeze_384: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
        unsqueeze_385: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
        mul_712: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, 1.9073486328125e-06)
        squeeze_31: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_10, [0, 2, 3]);  rsqrt_10 = None
        mul_713: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
        mul_714: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_712, mul_713);  mul_712 = mul_713 = None
        unsqueeze_386: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
        unsqueeze_387: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
        unsqueeze_388: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
        mul_715: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_31, primals_66);  primals_66 = None
        unsqueeze_389: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
        unsqueeze_390: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
        unsqueeze_391: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
        mul_716: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_388);  sub_227 = unsqueeze_388 = None
        sub_229: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(mul_709, mul_716);  mul_709 = mul_716 = None
        sub_230: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_229, unsqueeze_385);  sub_229 = unsqueeze_385 = None
        mul_717: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_230, unsqueeze_391);  sub_230 = unsqueeze_391 = None
        mul_718: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_165, squeeze_31);  sum_165 = squeeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_717, add_57, primals_62, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_717 = add_57 = primals_62 = None
        getitem_277: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = convolution_backward_24[0]
        getitem_278: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_24[1];  convolution_backward_24 = None
        add_331: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(getitem_268, getitem_277);  getitem_268 = getitem_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_166: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_331, [0, 2, 3])
        sub_231: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_394);  convolution_9 = unsqueeze_394 = None
        mul_719: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(add_331, sub_231)
        sum_167: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_719, [0, 2, 3]);  mul_719 = None
        mul_720: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_166, 1.9073486328125e-06)
        unsqueeze_395: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
        unsqueeze_396: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
        unsqueeze_397: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
        mul_721: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, 1.9073486328125e-06)
        mul_722: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
        mul_723: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_721, mul_722);  mul_721 = mul_722 = None
        unsqueeze_398: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
        unsqueeze_399: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
        unsqueeze_400: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
        mul_724: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_28, primals_60);  primals_60 = None
        unsqueeze_401: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
        unsqueeze_402: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
        unsqueeze_403: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
        mul_725: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_400);  sub_231 = unsqueeze_400 = None
        sub_233: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(add_331, mul_725);  mul_725 = None
        sub_234: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_233, unsqueeze_397);  sub_233 = unsqueeze_397 = None
        mul_726: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_234, unsqueeze_403);  sub_234 = unsqueeze_403 = None
        mul_727: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_167, squeeze_28);  sum_167 = squeeze_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_726, div_6, primals_56, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_726 = div_6 = primals_56 = None
        getitem_280: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = convolution_backward_25[0]
        getitem_281: "f32[64, 256, 1, 1][256, 1, 256, 256]cuda:0" = convolution_backward_25[1];  convolution_backward_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_8: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, getitem_17)
        mul_56: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
        unsqueeze_32: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_54, -1)
        unsqueeze_33: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        mul_62: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, unsqueeze_33);  mul_56 = unsqueeze_33 = None
        unsqueeze_34: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_55, -1);  primals_55 = None
        unsqueeze_35: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        add_50: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_35);  mul_62 = unsqueeze_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_6: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(add_50)
        exp_6: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_6);  neg_6 = None
        add_51: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_6, 1);  exp_6 = None
        reciprocal_27: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.reciprocal.default(add_51);  add_51 = None
        mul_728: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        mul_729: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(getitem_280, mul_728);  getitem_280 = None
        sub_235: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_728);  mul_728 = None
        mul_730: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(add_50, sub_235);  add_50 = sub_235 = None
        add_333: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_730, 1);  mul_730 = None
        mul_731: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_729, add_333);  mul_729 = add_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_24: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_17, [0, 2, 3]);  getitem_17 = None
        unsqueeze_404: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_24, 0);  squeeze_24 = None
        unsqueeze_405: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
        unsqueeze_406: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
        sum_168: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_731, [0, 2, 3])
        sub_236: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_406);  convolution_8 = unsqueeze_406 = None
        mul_732: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_731, sub_236)
        sum_169: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2, 3]);  mul_732 = None
        mul_733: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_168, 1.9073486328125e-06)
        unsqueeze_407: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
        unsqueeze_408: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
        unsqueeze_409: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
        mul_734: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, 1.9073486328125e-06)
        squeeze_25: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_8, [0, 2, 3]);  rsqrt_8 = None
        mul_735: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
        mul_736: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_734, mul_735);  mul_734 = mul_735 = None
        unsqueeze_410: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
        unsqueeze_411: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
        unsqueeze_412: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
        mul_737: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_25, primals_54);  primals_54 = None
        unsqueeze_413: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
        unsqueeze_414: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
        unsqueeze_415: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
        mul_738: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_412);  sub_236 = unsqueeze_412 = None
        sub_238: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(mul_731, mul_738);  mul_731 = mul_738 = None
        sub_239: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_409);  sub_238 = unsqueeze_409 = None
        mul_739: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_415);  sub_239 = unsqueeze_415 = None
        mul_740: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_169, squeeze_25);  sum_169 = squeeze_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_739, div_5, primals_50, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 256, [True, True, False]);  mul_739 = div_5 = primals_50 = None
        getitem_283: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = convolution_backward_26[0]
        getitem_284: "f32[256, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_26[1];  convolution_backward_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_7: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, getitem_15)
        mul_49: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
        unsqueeze_28: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_48, -1)
        unsqueeze_29: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_55: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_29);  mul_49 = unsqueeze_29 = None
        unsqueeze_30: "f32[256, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_49, -1);  primals_49 = None
        unsqueeze_31: "f32[256, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_44: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_55, unsqueeze_31);  mul_55 = unsqueeze_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_5: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.neg.default(add_44)
        exp_5: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.exp.default(neg_5);  neg_5 = None
        add_45: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(exp_5, 1);  exp_5 = None
        reciprocal_28: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.reciprocal.default(add_45);  add_45 = None
        mul_741: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        mul_742: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(getitem_283, mul_741);  getitem_283 = None
        sub_240: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_741);  mul_741 = None
        mul_743: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(add_44, sub_240);  add_44 = sub_240 = None
        add_335: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.add.Tensor(mul_743, 1);  mul_743 = None
        mul_744: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_742, add_335);  mul_742 = add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_21: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_15, [0, 2, 3]);  getitem_15 = None
        unsqueeze_416: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_21, 0);  squeeze_21 = None
        unsqueeze_417: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
        unsqueeze_418: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
        sum_170: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 2, 3])
        sub_241: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_418);  convolution_7 = unsqueeze_418 = None
        mul_745: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(mul_744, sub_241)
        sum_171: "f32[256][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_745, [0, 2, 3]);  mul_745 = None
        mul_746: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_170, 1.9073486328125e-06)
        unsqueeze_419: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
        unsqueeze_420: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
        unsqueeze_421: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
        mul_747: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, 1.9073486328125e-06)
        squeeze_22: "f32[256][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_7, [0, 2, 3]);  rsqrt_7 = None
        mul_748: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
        mul_749: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_747, mul_748);  mul_747 = mul_748 = None
        unsqueeze_422: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
        unsqueeze_423: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
        unsqueeze_424: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
        mul_750: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_22, primals_48);  primals_48 = None
        unsqueeze_425: "f32[1, 256][256, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
        unsqueeze_426: "f32[1, 256, 1][256, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
        unsqueeze_427: "f32[1, 256, 1, 1][256, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
        mul_751: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_424);  sub_241 = unsqueeze_424 = None
        sub_243: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(mul_744, mul_751);  mul_744 = mul_751 = None
        sub_244: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_421);  sub_243 = unsqueeze_421 = None
        mul_752: "f32[128, 256, 64, 64][1048576, 1, 16384, 256]cuda:0" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_427);  sub_244 = unsqueeze_427 = None
        mul_753: "f32[256][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_171, squeeze_22);  sum_171 = squeeze_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_752, add_39, primals_44, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_752 = add_39 = primals_44 = None
        getitem_286: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = convolution_backward_27[0]
        getitem_287: "f32[256, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_27[1];  convolution_backward_27 = None
        add_336: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.add.Tensor(add_331, getitem_286);  add_331 = getitem_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_172: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_336, [0, 2, 3])
        sub_245: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_430);  convolution_6 = unsqueeze_430 = None
        mul_754: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(add_336, sub_245)
        sum_173: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 2, 3]);  mul_754 = None
        mul_755: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_172, 1.9073486328125e-06)
        unsqueeze_431: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
        unsqueeze_432: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
        unsqueeze_433: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
        mul_756: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, 1.9073486328125e-06)
        mul_757: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
        mul_758: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_756, mul_757);  mul_756 = mul_757 = None
        unsqueeze_434: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
        unsqueeze_435: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
        unsqueeze_436: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
        mul_759: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_19, primals_42);  primals_42 = None
        unsqueeze_437: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
        unsqueeze_438: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
        unsqueeze_439: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
        mul_760: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_436);  sub_245 = unsqueeze_436 = None
        sub_247: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(add_336, mul_760);  add_336 = mul_760 = None
        sub_248: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_433);  sub_247 = unsqueeze_433 = None
        mul_761: "f32[128, 64, 64, 64][262144, 1, 4096, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_439);  sub_248 = unsqueeze_439 = None
        mul_762: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_173, squeeze_19);  sum_173 = squeeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_761, div_4, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_761 = div_4 = primals_38 = None
        getitem_289: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = convolution_backward_28[0]
        getitem_290: "f32[64, 128, 1, 1][128, 1, 128, 128]cuda:0" = convolution_backward_28[1];  convolution_backward_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_5: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, getitem_11)
        mul_35: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
        unsqueeze_20: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_36, -1)
        unsqueeze_21: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_41: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, unsqueeze_21);  mul_35 = unsqueeze_21 = None
        unsqueeze_22: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_37, -1);  primals_37 = None
        unsqueeze_23: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_33: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_23);  mul_41 = unsqueeze_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_4: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.neg.default(add_33)
        exp_4: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.exp.default(neg_4);  neg_4 = None
        add_34: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_4, 1);  exp_4 = None
        reciprocal_29: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_34);  add_34 = None
        mul_763: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        mul_764: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(getitem_289, mul_763);  getitem_289 = None
        sub_249: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_763);  mul_763 = None
        mul_765: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(add_33, sub_249);  add_33 = sub_249 = None
        add_338: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_765, 1);  mul_765 = None
        mul_766: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_764, add_338);  mul_764 = add_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_15: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_11, [0, 2, 3]);  getitem_11 = None
        unsqueeze_440: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_15, 0);  squeeze_15 = None
        unsqueeze_441: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
        unsqueeze_442: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
        sum_174: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 2, 3])
        sub_250: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_442);  convolution_5 = unsqueeze_442 = None
        mul_767: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_766, sub_250)
        sum_175: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_767, [0, 2, 3]);  mul_767 = None
        mul_768: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_174, 1.9073486328125e-06)
        unsqueeze_443: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
        unsqueeze_444: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
        unsqueeze_445: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
        mul_769: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, 1.9073486328125e-06)
        squeeze_16: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_5, [0, 2, 3]);  rsqrt_5 = None
        mul_770: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
        mul_771: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_769, mul_770);  mul_769 = mul_770 = None
        unsqueeze_446: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
        unsqueeze_447: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
        unsqueeze_448: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
        mul_772: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_16, primals_36);  primals_36 = None
        unsqueeze_449: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
        unsqueeze_450: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
        unsqueeze_451: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
        mul_773: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_250, unsqueeze_448);  sub_250 = unsqueeze_448 = None
        sub_252: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(mul_766, mul_773);  mul_766 = mul_773 = None
        sub_253: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_252, unsqueeze_445);  sub_252 = unsqueeze_445 = None
        mul_774: "f32[128, 128, 64, 64][524288, 1, 8192, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_451);  sub_253 = unsqueeze_451 = None
        mul_775: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_175, squeeze_16);  sum_175 = squeeze_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_774, div_3, primals_32, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 128, [True, True, False]);  mul_774 = div_3 = primals_32 = None
        getitem_292: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = convolution_backward_29[0]
        getitem_293: "f32[128, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_29[1];  convolution_backward_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_4: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, getitem_9)
        mul_28: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
        unsqueeze_16: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_30, -1)
        unsqueeze_17: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        mul_34: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_17);  mul_28 = unsqueeze_17 = None
        unsqueeze_18: "f32[128, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_31, -1);  primals_31 = None
        unsqueeze_19: "f32[128, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        add_27: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_34, unsqueeze_19);  mul_34 = unsqueeze_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_3: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.neg.default(add_27)
        exp_3: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.exp.default(neg_3);  neg_3 = None
        add_28: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.add.Tensor(exp_3, 1);  exp_3 = None
        reciprocal_30: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.reciprocal.default(add_28);  add_28 = None
        mul_776: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        mul_777: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(getitem_292, mul_776);  getitem_292 = None
        sub_254: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_776);  mul_776 = None
        mul_778: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(add_27, sub_254);  add_27 = sub_254 = None
        add_340: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.add.Tensor(mul_778, 1);  mul_778 = None
        mul_779: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_777, add_340);  mul_777 = add_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_12: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_9, [0, 2, 3]);  getitem_9 = None
        unsqueeze_452: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_12, 0);  squeeze_12 = None
        unsqueeze_453: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
        unsqueeze_454: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
        sum_176: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_779, [0, 2, 3])
        sub_255: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_454);  convolution_4 = unsqueeze_454 = None
        mul_780: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(mul_779, sub_255)
        sum_177: "f32[128][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 2, 3]);  mul_780 = None
        mul_781: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_176, 4.76837158203125e-07)
        unsqueeze_455: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
        unsqueeze_456: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
        unsqueeze_457: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
        mul_782: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, 4.76837158203125e-07)
        squeeze_13: "f32[128][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_4, [0, 2, 3]);  rsqrt_4 = None
        mul_783: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
        mul_784: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_782, mul_783);  mul_782 = mul_783 = None
        unsqueeze_458: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
        unsqueeze_459: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
        unsqueeze_460: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
        mul_785: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_13, primals_30);  primals_30 = None
        unsqueeze_461: "f32[1, 128][128, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
        unsqueeze_462: "f32[1, 128, 1][128, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
        unsqueeze_463: "f32[1, 128, 1, 1][128, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
        mul_786: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_460);  sub_255 = unsqueeze_460 = None
        sub_257: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(mul_779, mul_786);  mul_779 = mul_786 = None
        sub_258: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.sub.Tensor(sub_257, unsqueeze_457);  sub_257 = unsqueeze_457 = None
        mul_787: "f32[128, 128, 128, 128][2097152, 1, 16384, 128]cuda:0" = torch.ops.aten.mul.Tensor(sub_258, unsqueeze_463);  sub_258 = unsqueeze_463 = None
        mul_788: "f32[128][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_177, squeeze_13);  sum_177 = squeeze_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_787, add_22, primals_26, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_787 = add_22 = primals_26 = None
        getitem_295: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = convolution_backward_30[0]
        getitem_296: "f32[128, 32, 1, 1][32, 1, 32, 32]cuda:0" = convolution_backward_30[1];  convolution_backward_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sum_178: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(getitem_295, [0, 2, 3])
        sub_259: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_466);  convolution_3 = unsqueeze_466 = None
        mul_789: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.mul.Tensor(getitem_295, sub_259)
        sum_179: "f32[32][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_789, [0, 2, 3]);  mul_789 = None
        mul_790: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_178, 4.76837158203125e-07)
        unsqueeze_467: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
        unsqueeze_468: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
        unsqueeze_469: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
        mul_791: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, 4.76837158203125e-07)
        mul_792: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
        mul_793: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_791, mul_792);  mul_791 = mul_792 = None
        unsqueeze_470: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
        unsqueeze_471: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
        unsqueeze_472: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
        mul_794: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_10, primals_24);  primals_24 = None
        unsqueeze_473: "f32[1, 32][32, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
        unsqueeze_474: "f32[1, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
        unsqueeze_475: "f32[1, 32, 1, 1][32, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
        mul_795: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_472);  sub_259 = unsqueeze_472 = None
        sub_261: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.sub.Tensor(getitem_295, mul_795);  getitem_295 = mul_795 = None
        sub_262: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.sub.Tensor(sub_261, unsqueeze_469);  sub_261 = unsqueeze_469 = None
        mul_796: "f32[128, 32, 128, 128][524288, 1, 4096, 32]cuda:0" = torch.ops.aten.mul.Tensor(sub_262, unsqueeze_475);  sub_262 = unsqueeze_475 = None
        mul_797: "f32[32][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_179, squeeze_10);  sum_179 = squeeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_796, div_2, primals_20, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_796 = div_2 = primals_20 = None
        getitem_298: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = convolution_backward_31[0]
        getitem_299: "f32[32, 64, 1, 1][64, 1, 64, 64]cuda:0" = convolution_backward_31[1];  convolution_backward_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, getitem_5)
        mul_14: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
        unsqueeze_8: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_18, -1)
        unsqueeze_9: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        mul_20: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, unsqueeze_9);  mul_14 = unsqueeze_9 = None
        unsqueeze_10: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_19, -1);  primals_19 = None
        unsqueeze_11: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        add_16: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_11);  mul_20 = unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.neg.default(add_16)
        exp_2: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.exp.default(neg_2);  neg_2 = None
        add_17: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_2, 1);  exp_2 = None
        reciprocal_31: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.reciprocal.default(add_17);  add_17 = None
        mul_798: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        mul_799: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(getitem_298, mul_798);  getitem_298 = None
        sub_263: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_798);  mul_798 = None
        mul_800: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(add_16, sub_263);  add_16 = sub_263 = None
        add_342: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_800, 1);  mul_800 = None
        mul_801: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_799, add_342);  mul_799 = add_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_6: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_5, [0, 2, 3]);  getitem_5 = None
        unsqueeze_476: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_6, 0);  squeeze_6 = None
        unsqueeze_477: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
        unsqueeze_478: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
        sum_180: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_801, [0, 2, 3])
        sub_264: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_478);  convolution_2 = unsqueeze_478 = None
        mul_802: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_801, sub_264)
        sum_181: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_802, [0, 2, 3]);  mul_802 = None
        mul_803: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_180, 4.76837158203125e-07)
        unsqueeze_479: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
        unsqueeze_480: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
        unsqueeze_481: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
        mul_804: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, 4.76837158203125e-07)
        squeeze_7: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_2, [0, 2, 3]);  rsqrt_2 = None
        mul_805: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
        mul_806: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_804, mul_805);  mul_804 = mul_805 = None
        unsqueeze_482: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
        unsqueeze_483: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
        unsqueeze_484: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
        mul_807: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_7, primals_18);  primals_18 = None
        unsqueeze_485: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
        unsqueeze_486: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
        unsqueeze_487: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
        mul_808: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_484);  sub_264 = unsqueeze_484 = None
        sub_266: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(mul_801, mul_808);  mul_801 = mul_808 = None
        sub_267: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_481);  sub_266 = unsqueeze_481 = None
        mul_809: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_487);  sub_267 = unsqueeze_487 = None
        mul_810: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_181, squeeze_7);  sum_181 = squeeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_809, div_1, primals_14, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_809 = div_1 = primals_14 = None
        getitem_301: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = convolution_backward_32[0]
        getitem_302: "f32[64, 1, 3, 3][9, 1, 3, 1]cuda:0" = convolution_backward_32[1];  convolution_backward_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, getitem_3)
        mul_7: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        unsqueeze_4: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_12, -1)
        unsqueeze_5: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_13: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_5);  mul_7 = unsqueeze_5 = None
        unsqueeze_6: "f32[64, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_13, -1);  primals_13 = None
        unsqueeze_7: "f32[64, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_10: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_13, unsqueeze_7);  mul_13 = unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.neg.default(add_10)
        exp_1: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.exp.default(neg_1);  neg_1 = None
        add_11: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(exp_1, 1);  exp_1 = None
        reciprocal_32: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.reciprocal.default(add_11);  add_11 = None
        mul_811: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        mul_812: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(getitem_301, mul_811);  getitem_301 = None
        sub_268: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_811);  mul_811 = None
        mul_813: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(add_10, sub_268);  add_10 = sub_268 = None
        add_344: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.add.Tensor(mul_813, 1);  mul_813 = None
        mul_814: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_812, add_344);  mul_812 = add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze_3: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_3, [0, 2, 3]);  getitem_3 = None
        unsqueeze_488: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze_3, 0);  squeeze_3 = None
        unsqueeze_489: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
        unsqueeze_490: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
        sum_182: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3])
        sub_269: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_490);  convolution_1 = unsqueeze_490 = None
        mul_815: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(mul_814, sub_269)
        sum_183: "f32[64][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 2, 3]);  mul_815 = None
        mul_816: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_182, 4.76837158203125e-07)
        unsqueeze_491: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
        unsqueeze_492: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
        unsqueeze_493: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
        mul_817: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, 4.76837158203125e-07)
        squeeze_4: "f32[64][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt_1, [0, 2, 3]);  rsqrt_1 = None
        mul_818: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
        mul_819: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_817, mul_818);  mul_817 = mul_818 = None
        unsqueeze_494: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
        unsqueeze_495: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
        unsqueeze_496: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
        mul_820: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_4, primals_12);  primals_12 = None
        unsqueeze_497: "f32[1, 64][64, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
        unsqueeze_498: "f32[1, 64, 1][64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
        unsqueeze_499: "f32[1, 64, 1, 1][64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
        mul_821: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_496);  sub_269 = unsqueeze_496 = None
        sub_271: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(mul_814, mul_821);  mul_814 = mul_821 = None
        sub_272: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_493);  sub_271 = unsqueeze_493 = None
        mul_822: "f32[128, 64, 128, 128][1048576, 1, 8192, 64]cuda:0" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_499);  sub_272 = unsqueeze_499 = None
        mul_823: "f32[64][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_183, squeeze_4);  sum_183 = squeeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_822, div, primals_8, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_822 = div = primals_8 = None
        getitem_304: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = convolution_backward_33[0]
        getitem_305: "f32[64, 16, 1, 1][16, 1, 16, 16]cuda:0" = convolution_backward_33[1];  convolution_backward_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        sub: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution, getitem_1)
        mul: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        unsqueeze: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_6, -1)
        unsqueeze_1: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        mul_6: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul, unsqueeze_1);  mul = unsqueeze_1 = None
        unsqueeze_2: "f32[16, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(primals_7, -1);  primals_7 = None
        unsqueeze_3: "f32[16, 1, 1][1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        add_4: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_6, unsqueeze_3);  mul_6 = unsqueeze_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:148 in forward, code: x = self.act(x)
        neg: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.neg.default(add_4)
        exp: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.exp.default(neg);  neg = None
        add_5: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.add.Tensor(exp, 1);  exp = None
        reciprocal_33: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.reciprocal.default(add_5);  add_5 = None
        mul_824: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        mul_825: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(getitem_304, mul_824);  getitem_304 = None
        sub_273: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(1, mul_824);  mul_824 = None
        mul_826: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(add_4, sub_273);  add_4 = sub_273 = None
        add_346: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.add.Tensor(mul_826, 1);  mul_826 = None
        mul_827: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul_825, add_346);  mul_825 = add_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm_act.py:136 in forward, code: x = F.batch_norm(
        squeeze: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(getitem_1, [0, 2, 3]);  getitem_1 = None
        unsqueeze_500: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
        unsqueeze_501: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
        unsqueeze_502: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
        sum_184: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_827, [0, 2, 3])
        sub_274: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_502);  convolution = unsqueeze_502 = None
        mul_828: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(mul_827, sub_274)
        sum_185: "f32[16][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_828, [0, 2, 3]);  mul_828 = None
        mul_829: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_184, 4.76837158203125e-07)
        unsqueeze_503: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
        unsqueeze_504: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
        unsqueeze_505: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
        mul_830: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, 4.76837158203125e-07)
        squeeze_1: "f32[16][1]cuda:0" = torch.ops.aten.squeeze.dims(rsqrt, [0, 2, 3]);  rsqrt = None
        mul_831: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
        mul_832: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(mul_830, mul_831);  mul_830 = mul_831 = None
        unsqueeze_506: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
        unsqueeze_507: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
        unsqueeze_508: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
        mul_833: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(squeeze_1, primals_6);  primals_6 = None
        unsqueeze_509: "f32[1, 16][16, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mul_833, 0);  mul_833 = None
        unsqueeze_510: "f32[1, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
        unsqueeze_511: "f32[1, 16, 1, 1][16, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
        mul_834: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_274, unsqueeze_508);  sub_274 = unsqueeze_508 = None
        sub_276: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(mul_827, mul_834);  mul_827 = mul_834 = None
        sub_277: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.sub.Tensor(sub_276, unsqueeze_505);  sub_276 = unsqueeze_505 = None
        mul_835: "f32[128, 16, 128, 128][262144, 1, 2048, 16]cuda:0" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_511);  sub_277 = unsqueeze_511 = None
        mul_836: "f32[16][1]cuda:0" = torch.ops.aten.mul.Tensor(sum_185, squeeze_1);  sum_185 = squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/conv_bn_act.py:93 in forward, code: x = self.conv(x)
        convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_835, primals_2, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_835 = primals_2 = primals_1 = None
        getitem_308: "f32[16, 3, 3, 3][27, 1, 9, 3]cuda:0" = convolution_backward_34[1];  convolution_backward_34 = None
        return (getitem_308, None, None, None, None, mul_836, sum_184, getitem_305, None, None, None, mul_823, sum_182, getitem_302, None, None, None, mul_810, sum_180, getitem_299, None, None, None, mul_797, sum_178, getitem_296, None, None, None, mul_788, sum_176, getitem_293, None, None, None, mul_775, sum_174, getitem_290, None, None, None, mul_762, sum_172, getitem_287, None, None, None, mul_753, sum_170, getitem_284, None, None, None, mul_740, sum_168, getitem_281, None, None, None, mul_727, sum_166, getitem_278, None, None, None, mul_718, sum_164, getitem_275, None, None, None, mul_705, sum_162, getitem_272, None, None, None, mul_692, sum_160, getitem_269, None, None, None, mul_683, sum_158, getitem_266, None, None, None, mul_670, sum_156, getitem_263, None, None, None, mul_657, sum_154, getitem_260, None, None, None, mul_648, sum_152, getitem_257, sum_150, sum_151, mm_73, view_259, mm_71, view_253, sum_144, sum_145, mm_69, view_250, mm_67, view_247, sum_138, sum_139, mm_65, view_244, mm_63, view_238, sum_132, sum_133, mm_61, view_235, mm_59, view_232, sum_126, sum_127, getitem_246, None, None, None, mul_592, sum_122, getitem_243, None, None, None, mul_579, sum_120, getitem_240, None, None, None, mul_566, sum_118, getitem_237, None, None, None, mul_553, sum_116, getitem_234, None, None, None, mul_540, sum_114, getitem_231, None, None, None, mul_531, sum_112, getitem_228, sum_110, sum_111, mm_57, view_223, mm_55, view_217, sum_104, sum_105, mm_53, view_214, mm_51, view_211, sum_98, sum_99, mm_49, view_208, mm_47, view_202, sum_92, sum_93, mm_45, view_199, mm_43, view_196, sum_86, sum_87, mm_41, view_193, mm_39, view_187, sum_80, sum_81, mm_37, view_184, mm_35, view_181, sum_74, sum_75, mm_33, view_178, mm_31, view_172, sum_68, sum_69, mm_29, view_169, mm_27, view_166, sum_62, sum_63, getitem_209, None, None, None, mul_439, sum_58, getitem_206, None, None, None, mul_426, sum_56, getitem_203, None, None, None, mul_413, sum_54, getitem_200, None, None, None, mul_400, sum_52, getitem_197, None, None, None, mul_387, sum_50, getitem_194, None, None, None, mul_378, sum_48, getitem_191, sum_46, sum_47, mm_25, view_157, mm_23, view_151, sum_40, sum_41, mm_21, view_148, mm_19, view_145, sum_34, sum_35, mm_17, view_142, mm_15, view_136, sum_28, sum_29, mm_13, view_133, mm_11, view_130, sum_22, sum_23, mm_9, view_127, mm_7, view_121, sum_16, sum_17, mm_5, view_118, mm_3, view_115, sum_10, sum_11, getitem_176, None, None, None, mul_304, sum_6, getitem_173, None, None, None, mul_291, sum_4, getitem_170, None, None, None, mul_278, sum_2, mm_1, view_109)
