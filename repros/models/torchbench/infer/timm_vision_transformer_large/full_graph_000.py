class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[32, 3, 224, 224]", arg1_1: "f16[1408, 3, 14, 14]", arg2_1: "f16[1408]", arg3_1: "f16[1, 1, 1408]", arg4_1: "f16[1, 257, 1408]", arg5_1: "f16[1408]", arg6_1: "f16[1408]", arg7_1: "f16[4224, 1408]", arg8_1: "f16[4224]", arg9_1: "f16[1408, 1408]", arg10_1: "f16[1408]", arg11_1: "f16[1408]", arg12_1: "f16[1408]", arg13_1: "f16[6144, 1408]", arg14_1: "f16[6144]", arg15_1: "f16[1408, 6144]", arg16_1: "f16[1408]", arg17_1: "f16[1408]", arg18_1: "f16[1408]", arg19_1: "f16[4224, 1408]", arg20_1: "f16[4224]", arg21_1: "f16[1408, 1408]", arg22_1: "f16[1408]", arg23_1: "f16[1408]", arg24_1: "f16[1408]", arg25_1: "f16[6144, 1408]", arg26_1: "f16[6144]", arg27_1: "f16[1408, 6144]", arg28_1: "f16[1408]", arg29_1: "f16[1408]", arg30_1: "f16[1408]", arg31_1: "f16[4224, 1408]", arg32_1: "f16[4224]", arg33_1: "f16[1408, 1408]", arg34_1: "f16[1408]", arg35_1: "f16[1408]", arg36_1: "f16[1408]", arg37_1: "f16[6144, 1408]", arg38_1: "f16[6144]", arg39_1: "f16[1408, 6144]", arg40_1: "f16[1408]", arg41_1: "f16[1408]", arg42_1: "f16[1408]", arg43_1: "f16[4224, 1408]", arg44_1: "f16[4224]", arg45_1: "f16[1408, 1408]", arg46_1: "f16[1408]", arg47_1: "f16[1408]", arg48_1: "f16[1408]", arg49_1: "f16[6144, 1408]", arg50_1: "f16[6144]", arg51_1: "f16[1408, 6144]", arg52_1: "f16[1408]", arg53_1: "f16[1408]", arg54_1: "f16[1408]", arg55_1: "f16[4224, 1408]", arg56_1: "f16[4224]", arg57_1: "f16[1408, 1408]", arg58_1: "f16[1408]", arg59_1: "f16[1408]", arg60_1: "f16[1408]", arg61_1: "f16[6144, 1408]", arg62_1: "f16[6144]", arg63_1: "f16[1408, 6144]", arg64_1: "f16[1408]", arg65_1: "f16[1408]", arg66_1: "f16[1408]", arg67_1: "f16[4224, 1408]", arg68_1: "f16[4224]", arg69_1: "f16[1408, 1408]", arg70_1: "f16[1408]", arg71_1: "f16[1408]", arg72_1: "f16[1408]", arg73_1: "f16[6144, 1408]", arg74_1: "f16[6144]", arg75_1: "f16[1408, 6144]", arg76_1: "f16[1408]", arg77_1: "f16[1408]", arg78_1: "f16[1408]", arg79_1: "f16[4224, 1408]", arg80_1: "f16[4224]", arg81_1: "f16[1408, 1408]", arg82_1: "f16[1408]", arg83_1: "f16[1408]", arg84_1: "f16[1408]", arg85_1: "f16[6144, 1408]", arg86_1: "f16[6144]", arg87_1: "f16[1408, 6144]", arg88_1: "f16[1408]", arg89_1: "f16[1408]", arg90_1: "f16[1408]", arg91_1: "f16[4224, 1408]", arg92_1: "f16[4224]", arg93_1: "f16[1408, 1408]", arg94_1: "f16[1408]", arg95_1: "f16[1408]", arg96_1: "f16[1408]", arg97_1: "f16[6144, 1408]", arg98_1: "f16[6144]", arg99_1: "f16[1408, 6144]", arg100_1: "f16[1408]", arg101_1: "f16[1408]", arg102_1: "f16[1408]", arg103_1: "f16[4224, 1408]", arg104_1: "f16[4224]", arg105_1: "f16[1408, 1408]", arg106_1: "f16[1408]", arg107_1: "f16[1408]", arg108_1: "f16[1408]", arg109_1: "f16[6144, 1408]", arg110_1: "f16[6144]", arg111_1: "f16[1408, 6144]", arg112_1: "f16[1408]", arg113_1: "f16[1408]", arg114_1: "f16[1408]", arg115_1: "f16[4224, 1408]", arg116_1: "f16[4224]", arg117_1: "f16[1408, 1408]", arg118_1: "f16[1408]", arg119_1: "f16[1408]", arg120_1: "f16[1408]", arg121_1: "f16[6144, 1408]", arg122_1: "f16[6144]", arg123_1: "f16[1408, 6144]", arg124_1: "f16[1408]", arg125_1: "f16[1408]", arg126_1: "f16[1408]", arg127_1: "f16[4224, 1408]", arg128_1: "f16[4224]", arg129_1: "f16[1408, 1408]", arg130_1: "f16[1408]", arg131_1: "f16[1408]", arg132_1: "f16[1408]", arg133_1: "f16[6144, 1408]", arg134_1: "f16[6144]", arg135_1: "f16[1408, 6144]", arg136_1: "f16[1408]", arg137_1: "f16[1408]", arg138_1: "f16[1408]", arg139_1: "f16[4224, 1408]", arg140_1: "f16[4224]", arg141_1: "f16[1408, 1408]", arg142_1: "f16[1408]", arg143_1: "f16[1408]", arg144_1: "f16[1408]", arg145_1: "f16[6144, 1408]", arg146_1: "f16[6144]", arg147_1: "f16[1408, 6144]", arg148_1: "f16[1408]", arg149_1: "f16[1408]", arg150_1: "f16[1408]", arg151_1: "f16[4224, 1408]", arg152_1: "f16[4224]", arg153_1: "f16[1408, 1408]", arg154_1: "f16[1408]", arg155_1: "f16[1408]", arg156_1: "f16[1408]", arg157_1: "f16[6144, 1408]", arg158_1: "f16[6144]", arg159_1: "f16[1408, 6144]", arg160_1: "f16[1408]", arg161_1: "f16[1408]", arg162_1: "f16[1408]", arg163_1: "f16[4224, 1408]", arg164_1: "f16[4224]", arg165_1: "f16[1408, 1408]", arg166_1: "f16[1408]", arg167_1: "f16[1408]", arg168_1: "f16[1408]", arg169_1: "f16[6144, 1408]", arg170_1: "f16[6144]", arg171_1: "f16[1408, 6144]", arg172_1: "f16[1408]", arg173_1: "f16[1408]", arg174_1: "f16[1408]", arg175_1: "f16[4224, 1408]", arg176_1: "f16[4224]", arg177_1: "f16[1408, 1408]", arg178_1: "f16[1408]", arg179_1: "f16[1408]", arg180_1: "f16[1408]", arg181_1: "f16[6144, 1408]", arg182_1: "f16[6144]", arg183_1: "f16[1408, 6144]", arg184_1: "f16[1408]", arg185_1: "f16[1408]", arg186_1: "f16[1408]", arg187_1: "f16[4224, 1408]", arg188_1: "f16[4224]", arg189_1: "f16[1408, 1408]", arg190_1: "f16[1408]", arg191_1: "f16[1408]", arg192_1: "f16[1408]", arg193_1: "f16[6144, 1408]", arg194_1: "f16[6144]", arg195_1: "f16[1408, 6144]", arg196_1: "f16[1408]", arg197_1: "f16[1408]", arg198_1: "f16[1408]", arg199_1: "f16[4224, 1408]", arg200_1: "f16[4224]", arg201_1: "f16[1408, 1408]", arg202_1: "f16[1408]", arg203_1: "f16[1408]", arg204_1: "f16[1408]", arg205_1: "f16[6144, 1408]", arg206_1: "f16[6144]", arg207_1: "f16[1408, 6144]", arg208_1: "f16[1408]", arg209_1: "f16[1408]", arg210_1: "f16[1408]", arg211_1: "f16[4224, 1408]", arg212_1: "f16[4224]", arg213_1: "f16[1408, 1408]", arg214_1: "f16[1408]", arg215_1: "f16[1408]", arg216_1: "f16[1408]", arg217_1: "f16[6144, 1408]", arg218_1: "f16[6144]", arg219_1: "f16[1408, 6144]", arg220_1: "f16[1408]", arg221_1: "f16[1408]", arg222_1: "f16[1408]", arg223_1: "f16[4224, 1408]", arg224_1: "f16[4224]", arg225_1: "f16[1408, 1408]", arg226_1: "f16[1408]", arg227_1: "f16[1408]", arg228_1: "f16[1408]", arg229_1: "f16[6144, 1408]", arg230_1: "f16[6144]", arg231_1: "f16[1408, 6144]", arg232_1: "f16[1408]", arg233_1: "f16[1408]", arg234_1: "f16[1408]", arg235_1: "f16[4224, 1408]", arg236_1: "f16[4224]", arg237_1: "f16[1408, 1408]", arg238_1: "f16[1408]", arg239_1: "f16[1408]", arg240_1: "f16[1408]", arg241_1: "f16[6144, 1408]", arg242_1: "f16[6144]", arg243_1: "f16[1408, 6144]", arg244_1: "f16[1408]", arg245_1: "f16[1408]", arg246_1: "f16[1408]", arg247_1: "f16[4224, 1408]", arg248_1: "f16[4224]", arg249_1: "f16[1408, 1408]", arg250_1: "f16[1408]", arg251_1: "f16[1408]", arg252_1: "f16[1408]", arg253_1: "f16[6144, 1408]", arg254_1: "f16[6144]", arg255_1: "f16[1408, 6144]", arg256_1: "f16[1408]", arg257_1: "f16[1408]", arg258_1: "f16[1408]", arg259_1: "f16[4224, 1408]", arg260_1: "f16[4224]", arg261_1: "f16[1408, 1408]", arg262_1: "f16[1408]", arg263_1: "f16[1408]", arg264_1: "f16[1408]", arg265_1: "f16[6144, 1408]", arg266_1: "f16[6144]", arg267_1: "f16[1408, 6144]", arg268_1: "f16[1408]", arg269_1: "f16[1408]", arg270_1: "f16[1408]", arg271_1: "f16[4224, 1408]", arg272_1: "f16[4224]", arg273_1: "f16[1408, 1408]", arg274_1: "f16[1408]", arg275_1: "f16[1408]", arg276_1: "f16[1408]", arg277_1: "f16[6144, 1408]", arg278_1: "f16[6144]", arg279_1: "f16[1408, 6144]", arg280_1: "f16[1408]", arg281_1: "f16[1408]", arg282_1: "f16[1408]", arg283_1: "f16[4224, 1408]", arg284_1: "f16[4224]", arg285_1: "f16[1408, 1408]", arg286_1: "f16[1408]", arg287_1: "f16[1408]", arg288_1: "f16[1408]", arg289_1: "f16[6144, 1408]", arg290_1: "f16[6144]", arg291_1: "f16[1408, 6144]", arg292_1: "f16[1408]", arg293_1: "f16[1408]", arg294_1: "f16[1408]", arg295_1: "f16[4224, 1408]", arg296_1: "f16[4224]", arg297_1: "f16[1408, 1408]", arg298_1: "f16[1408]", arg299_1: "f16[1408]", arg300_1: "f16[1408]", arg301_1: "f16[6144, 1408]", arg302_1: "f16[6144]", arg303_1: "f16[1408, 6144]", arg304_1: "f16[1408]", arg305_1: "f16[1408]", arg306_1: "f16[1408]", arg307_1: "f16[4224, 1408]", arg308_1: "f16[4224]", arg309_1: "f16[1408, 1408]", arg310_1: "f16[1408]", arg311_1: "f16[1408]", arg312_1: "f16[1408]", arg313_1: "f16[6144, 1408]", arg314_1: "f16[6144]", arg315_1: "f16[1408, 6144]", arg316_1: "f16[1408]", arg317_1: "f16[1408]", arg318_1: "f16[1408]", arg319_1: "f16[4224, 1408]", arg320_1: "f16[4224]", arg321_1: "f16[1408, 1408]", arg322_1: "f16[1408]", arg323_1: "f16[1408]", arg324_1: "f16[1408]", arg325_1: "f16[6144, 1408]", arg326_1: "f16[6144]", arg327_1: "f16[1408, 6144]", arg328_1: "f16[1408]", arg329_1: "f16[1408]", arg330_1: "f16[1408]", arg331_1: "f16[4224, 1408]", arg332_1: "f16[4224]", arg333_1: "f16[1408, 1408]", arg334_1: "f16[1408]", arg335_1: "f16[1408]", arg336_1: "f16[1408]", arg337_1: "f16[6144, 1408]", arg338_1: "f16[6144]", arg339_1: "f16[1408, 6144]", arg340_1: "f16[1408]", arg341_1: "f16[1408]", arg342_1: "f16[1408]", arg343_1: "f16[4224, 1408]", arg344_1: "f16[4224]", arg345_1: "f16[1408, 1408]", arg346_1: "f16[1408]", arg347_1: "f16[1408]", arg348_1: "f16[1408]", arg349_1: "f16[6144, 1408]", arg350_1: "f16[6144]", arg351_1: "f16[1408, 6144]", arg352_1: "f16[1408]", arg353_1: "f16[1408]", arg354_1: "f16[1408]", arg355_1: "f16[4224, 1408]", arg356_1: "f16[4224]", arg357_1: "f16[1408, 1408]", arg358_1: "f16[1408]", arg359_1: "f16[1408]", arg360_1: "f16[1408]", arg361_1: "f16[6144, 1408]", arg362_1: "f16[6144]", arg363_1: "f16[1408, 6144]", arg364_1: "f16[1408]", arg365_1: "f16[1408]", arg366_1: "f16[1408]", arg367_1: "f16[4224, 1408]", arg368_1: "f16[4224]", arg369_1: "f16[1408, 1408]", arg370_1: "f16[1408]", arg371_1: "f16[1408]", arg372_1: "f16[1408]", arg373_1: "f16[6144, 1408]", arg374_1: "f16[6144]", arg375_1: "f16[1408, 6144]", arg376_1: "f16[1408]", arg377_1: "f16[1408]", arg378_1: "f16[1408]", arg379_1: "f16[4224, 1408]", arg380_1: "f16[4224]", arg381_1: "f16[1408, 1408]", arg382_1: "f16[1408]", arg383_1: "f16[1408]", arg384_1: "f16[1408]", arg385_1: "f16[6144, 1408]", arg386_1: "f16[6144]", arg387_1: "f16[1408, 6144]", arg388_1: "f16[1408]", arg389_1: "f16[1408]", arg390_1: "f16[1408]", arg391_1: "f16[4224, 1408]", arg392_1: "f16[4224]", arg393_1: "f16[1408, 1408]", arg394_1: "f16[1408]", arg395_1: "f16[1408]", arg396_1: "f16[1408]", arg397_1: "f16[6144, 1408]", arg398_1: "f16[6144]", arg399_1: "f16[1408, 6144]", arg400_1: "f16[1408]", arg401_1: "f16[1408]", arg402_1: "f16[1408]", arg403_1: "f16[4224, 1408]", arg404_1: "f16[4224]", arg405_1: "f16[1408, 1408]", arg406_1: "f16[1408]", arg407_1: "f16[1408]", arg408_1: "f16[1408]", arg409_1: "f16[6144, 1408]", arg410_1: "f16[6144]", arg411_1: "f16[1408, 6144]", arg412_1: "f16[1408]", arg413_1: "f16[1408]", arg414_1: "f16[1408]", arg415_1: "f16[4224, 1408]", arg416_1: "f16[4224]", arg417_1: "f16[1408, 1408]", arg418_1: "f16[1408]", arg419_1: "f16[1408]", arg420_1: "f16[1408]", arg421_1: "f16[6144, 1408]", arg422_1: "f16[6144]", arg423_1: "f16[1408, 6144]", arg424_1: "f16[1408]", arg425_1: "f16[1408]", arg426_1: "f16[1408]", arg427_1: "f16[4224, 1408]", arg428_1: "f16[4224]", arg429_1: "f16[1408, 1408]", arg430_1: "f16[1408]", arg431_1: "f16[1408]", arg432_1: "f16[1408]", arg433_1: "f16[6144, 1408]", arg434_1: "f16[6144]", arg435_1: "f16[1408, 6144]", arg436_1: "f16[1408]", arg437_1: "f16[1408]", arg438_1: "f16[1408]", arg439_1: "f16[4224, 1408]", arg440_1: "f16[4224]", arg441_1: "f16[1408, 1408]", arg442_1: "f16[1408]", arg443_1: "f16[1408]", arg444_1: "f16[1408]", arg445_1: "f16[6144, 1408]", arg446_1: "f16[6144]", arg447_1: "f16[1408, 6144]", arg448_1: "f16[1408]", arg449_1: "f16[1408]", arg450_1: "f16[1408]", arg451_1: "f16[4224, 1408]", arg452_1: "f16[4224]", arg453_1: "f16[1408, 1408]", arg454_1: "f16[1408]", arg455_1: "f16[1408]", arg456_1: "f16[1408]", arg457_1: "f16[6144, 1408]", arg458_1: "f16[6144]", arg459_1: "f16[1408, 6144]", arg460_1: "f16[1408]", arg461_1: "f16[1408]", arg462_1: "f16[1408]", arg463_1: "f16[4224, 1408]", arg464_1: "f16[4224]", arg465_1: "f16[1408, 1408]", arg466_1: "f16[1408]", arg467_1: "f16[1408]", arg468_1: "f16[1408]", arg469_1: "f16[6144, 1408]", arg470_1: "f16[6144]", arg471_1: "f16[1408, 6144]", arg472_1: "f16[1408]", arg473_1: "f16[1408]", arg474_1: "f16[1408]", arg475_1: "f16[4224, 1408]", arg476_1: "f16[4224]", arg477_1: "f16[1408, 1408]", arg478_1: "f16[1408]", arg479_1: "f16[1408]", arg480_1: "f16[1408]", arg481_1: "f16[6144, 1408]", arg482_1: "f16[6144]", arg483_1: "f16[1408, 6144]", arg484_1: "f16[1408]", arg485_1: "f16[1408]", arg486_1: "f16[1408]", arg487_1: "f16[1000, 1408]", arg488_1: "f16[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1042 in _pos_embed, code: to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        expand: "f16[32, 1, 1408]" = torch.ops.aten.expand.default(arg3_1, [32, -1, -1]);  arg3_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        convolution: "f16[32, 1408, 16, 16]" = torch.ops.aten.convolution.default(arg0_1, arg1_1, arg2_1, [14, 14], [0, 0], [1, 1], False, [0, 0], 1);  arg0_1 = arg1_1 = arg2_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:138 in forward, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        view: "f16[32, 1408, 256]" = torch.ops.aten.reshape.default(convolution, [32, 1408, 256]);  convolution = None
        permute: "f16[32, 256, 1408]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1072 in _pos_embed, code: x = torch.cat(to_cat + [x], dim=1)
        cat: "f16[32, 257, 1408]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1073 in _pos_embed, code: x = x + pos_embed
        add: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(cat, arg4_1);  cat = arg4_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add, torch.float32)
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type, [2], correction = 0, keepdim = True)
        getitem: "f32[32, 257, 1]" = var_mean[0]
        getitem_1: "f32[32, 257, 1]" = var_mean[1];  var_mean = None
        sub: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type, getitem_1);  convert_element_type = getitem_1 = None
        add_1: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
        rsqrt: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_1: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul, arg5_1);  mul = arg5_1 = None
        add_2: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
        convert_element_type_1: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_2, torch.float16);  add_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_1: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_1, [8224, 1408]);  convert_element_type_1 = None
        permute_1: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
        addmm: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg8_1, view_1, permute_1);  arg8_1 = view_1 = permute_1 = None
        view_2: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm, [32, 257, 4224]);  addmm = None
        view_3: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_2, [32, 257, 3, 16, 88]);  view_2 = None
        permute_2: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
        getitem_2: "f16[32, 16, 257, 88]" = unbind[0]
        getitem_3: "f16[32, 16, 257, 88]" = unbind[1]
        getitem_4: "f16[32, 16, 257, 88]" = unbind[2];  unbind = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_2, getitem_3, getitem_4, None, False);  getitem_2 = getitem_3 = getitem_4 = None
        getitem_5: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention[0];  _scaled_dot_product_cudnn_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_3: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
        view_4: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_3, [32, 257, 1408]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_5: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_4, [8224, 1408]);  view_4 = None
        permute_4: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg10_1, view_5, permute_4);  arg10_1 = view_5 = permute_4 = None
        view_6: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_1, [32, 257, 1408]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_3: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add, view_6);  add = view_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_8: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        var_mean_1 = torch.ops.aten.var_mean.correction(convert_element_type_8, [2], correction = 0, keepdim = True)
        getitem_14: "f32[32, 257, 1]" = var_mean_1[0]
        getitem_15: "f32[32, 257, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_1: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_8, getitem_15);  convert_element_type_8 = getitem_15 = None
        add_4: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
        rsqrt_1: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_2: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
        mul_3: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_2, arg11_1);  mul_2 = arg11_1 = None
        add_5: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_3, arg12_1);  mul_3 = arg12_1 = None
        convert_element_type_9: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_5, torch.float16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_7: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_9, [8224, 1408]);  convert_element_type_9 = None
        permute_5: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg14_1, view_7, permute_5);  arg14_1 = view_7 = permute_5 = None
        view_8: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_2, [32, 257, 6144]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_13: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        mul_4: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.5)
        mul_5: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 0.7071067811865476);  convert_element_type_13 = None
        erf: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_6: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
        convert_element_type_14: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_6, torch.float16);  mul_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_9: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_14, [8224, 6144]);  convert_element_type_14 = None
        permute_6: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_3: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg16_1, view_9, permute_6);  arg16_1 = view_9 = permute_6 = None
        view_10: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_3, [32, 257, 1408]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_7: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_3, view_10);  add_3 = view_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_18: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_7, torch.float32)
        var_mean_2 = torch.ops.aten.var_mean.correction(convert_element_type_18, [2], correction = 0, keepdim = True)
        getitem_16: "f32[32, 257, 1]" = var_mean_2[0]
        getitem_17: "f32[32, 257, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_2: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_18, getitem_17);  convert_element_type_18 = getitem_17 = None
        add_8: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
        rsqrt_2: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
        mul_7: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
        mul_8: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_7, arg17_1);  mul_7 = arg17_1 = None
        add_9: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_8, arg18_1);  mul_8 = arg18_1 = None
        convert_element_type_19: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_11: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_19, [8224, 1408]);  convert_element_type_19 = None
        permute_7: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_4: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg20_1, view_11, permute_7);  arg20_1 = view_11 = permute_7 = None
        view_12: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_4, [32, 257, 4224]);  addmm_4 = None
        view_13: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_12, [32, 257, 3, 16, 88]);  view_12 = None
        permute_8: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_13, [2, 0, 3, 1, 4]);  view_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_1 = torch.ops.aten.unbind.int(permute_8);  permute_8 = None
        getitem_18: "f16[32, 16, 257, 88]" = unbind_1[0]
        getitem_19: "f16[32, 16, 257, 88]" = unbind_1[1]
        getitem_20: "f16[32, 16, 257, 88]" = unbind_1[2];  unbind_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_1 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_18, getitem_19, getitem_20, None, False);  getitem_18 = getitem_19 = getitem_20 = None
        getitem_21: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_1[0];  _scaled_dot_product_cudnn_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_9: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
        view_14: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_9, [32, 257, 1408]);  permute_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_15: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_14, [8224, 1408]);  view_14 = None
        permute_10: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_5: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg22_1, view_15, permute_10);  arg22_1 = view_15 = permute_10 = None
        view_16: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_5, [32, 257, 1408]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_10: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_7, view_16);  add_7 = view_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_26: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_10, torch.float32)
        var_mean_3 = torch.ops.aten.var_mean.correction(convert_element_type_26, [2], correction = 0, keepdim = True)
        getitem_30: "f32[32, 257, 1]" = var_mean_3[0]
        getitem_31: "f32[32, 257, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_3: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_26, getitem_31);  convert_element_type_26 = getitem_31 = None
        add_11: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
        rsqrt_3: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_9: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = rsqrt_3 = None
        mul_10: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_9, arg23_1);  mul_9 = arg23_1 = None
        add_12: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_10, arg24_1);  mul_10 = arg24_1 = None
        convert_element_type_27: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_12, torch.float16);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_17: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_27, [8224, 1408]);  convert_element_type_27 = None
        permute_11: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
        addmm_6: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg26_1, view_17, permute_11);  arg26_1 = view_17 = permute_11 = None
        view_18: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_6, [32, 257, 6144]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_31: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_18, torch.float32);  view_18 = None
        mul_11: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 0.5)
        mul_12: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 0.7071067811865476);  convert_element_type_31 = None
        erf_1: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_13: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_13: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
        convert_element_type_32: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_13, torch.float16);  mul_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_19: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_32, [8224, 6144]);  convert_element_type_32 = None
        permute_12: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_7: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg28_1, view_19, permute_12);  arg28_1 = view_19 = permute_12 = None
        view_20: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_7, [32, 257, 1408]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_14: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_10, view_20);  add_10 = view_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_36: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        var_mean_4 = torch.ops.aten.var_mean.correction(convert_element_type_36, [2], correction = 0, keepdim = True)
        getitem_32: "f32[32, 257, 1]" = var_mean_4[0]
        getitem_33: "f32[32, 257, 1]" = var_mean_4[1];  var_mean_4 = None
        sub_4: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_36, getitem_33);  convert_element_type_36 = getitem_33 = None
        add_15: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
        rsqrt_4: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_14: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = rsqrt_4 = None
        mul_15: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_14, arg29_1);  mul_14 = arg29_1 = None
        add_16: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_15, arg30_1);  mul_15 = arg30_1 = None
        convert_element_type_37: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_16, torch.float16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_21: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_37, [8224, 1408]);  convert_element_type_37 = None
        permute_13: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
        addmm_8: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg32_1, view_21, permute_13);  arg32_1 = view_21 = permute_13 = None
        view_22: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_8, [32, 257, 4224]);  addmm_8 = None
        view_23: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_22, [32, 257, 3, 16, 88]);  view_22 = None
        permute_14: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_2 = torch.ops.aten.unbind.int(permute_14);  permute_14 = None
        getitem_34: "f16[32, 16, 257, 88]" = unbind_2[0]
        getitem_35: "f16[32, 16, 257, 88]" = unbind_2[1]
        getitem_36: "f16[32, 16, 257, 88]" = unbind_2[2];  unbind_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_2 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_34, getitem_35, getitem_36, None, False);  getitem_34 = getitem_35 = getitem_36 = None
        getitem_37: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_2[0];  _scaled_dot_product_cudnn_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_15: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3]);  getitem_37 = None
        view_24: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_15, [32, 257, 1408]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_25: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_24, [8224, 1408]);  view_24 = None
        permute_16: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_9: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg34_1, view_25, permute_16);  arg34_1 = view_25 = permute_16 = None
        view_26: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_9, [32, 257, 1408]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_17: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_14, view_26);  add_14 = view_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_44: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_17, torch.float32)
        var_mean_5 = torch.ops.aten.var_mean.correction(convert_element_type_44, [2], correction = 0, keepdim = True)
        getitem_46: "f32[32, 257, 1]" = var_mean_5[0]
        getitem_47: "f32[32, 257, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_5: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_44, getitem_47);  convert_element_type_44 = getitem_47 = None
        add_18: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
        rsqrt_5: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
        mul_16: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = rsqrt_5 = None
        mul_17: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_16, arg35_1);  mul_16 = arg35_1 = None
        add_19: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_17, arg36_1);  mul_17 = arg36_1 = None
        convert_element_type_45: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_19, torch.float16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_27: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_45, [8224, 1408]);  convert_element_type_45 = None
        permute_17: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
        addmm_10: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg38_1, view_27, permute_17);  arg38_1 = view_27 = permute_17 = None
        view_28: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_10, [32, 257, 6144]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_49: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        mul_18: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 0.5)
        mul_19: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 0.7071067811865476);  convert_element_type_49 = None
        erf_2: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
        add_20: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_20: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
        convert_element_type_50: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_20, torch.float16);  mul_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_29: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_50, [8224, 6144]);  convert_element_type_50 = None
        permute_18: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_11: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg40_1, view_29, permute_18);  arg40_1 = view_29 = permute_18 = None
        view_30: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_11, [32, 257, 1408]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_21: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_17, view_30);  add_17 = view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_54: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_21, torch.float32)
        var_mean_6 = torch.ops.aten.var_mean.correction(convert_element_type_54, [2], correction = 0, keepdim = True)
        getitem_48: "f32[32, 257, 1]" = var_mean_6[0]
        getitem_49: "f32[32, 257, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_6: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_54, getitem_49);  convert_element_type_54 = getitem_49 = None
        add_22: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
        rsqrt_6: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
        mul_21: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = rsqrt_6 = None
        mul_22: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_21, arg41_1);  mul_21 = arg41_1 = None
        add_23: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_22, arg42_1);  mul_22 = arg42_1 = None
        convert_element_type_55: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_23, torch.float16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_31: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_55, [8224, 1408]);  convert_element_type_55 = None
        permute_19: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_12: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg44_1, view_31, permute_19);  arg44_1 = view_31 = permute_19 = None
        view_32: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_12, [32, 257, 4224]);  addmm_12 = None
        view_33: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_32, [32, 257, 3, 16, 88]);  view_32 = None
        permute_20: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_3 = torch.ops.aten.unbind.int(permute_20);  permute_20 = None
        getitem_50: "f16[32, 16, 257, 88]" = unbind_3[0]
        getitem_51: "f16[32, 16, 257, 88]" = unbind_3[1]
        getitem_52: "f16[32, 16, 257, 88]" = unbind_3[2];  unbind_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_3 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_50, getitem_51, getitem_52, None, False);  getitem_50 = getitem_51 = getitem_52 = None
        getitem_53: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_3[0];  _scaled_dot_product_cudnn_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_21: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
        view_34: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_21, [32, 257, 1408]);  permute_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_35: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_34, [8224, 1408]);  view_34 = None
        permute_22: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_13: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg46_1, view_35, permute_22);  arg46_1 = view_35 = permute_22 = None
        view_36: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_13, [32, 257, 1408]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_24: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_21, view_36);  add_21 = view_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_62: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_24, torch.float32)
        var_mean_7 = torch.ops.aten.var_mean.correction(convert_element_type_62, [2], correction = 0, keepdim = True)
        getitem_62: "f32[32, 257, 1]" = var_mean_7[0]
        getitem_63: "f32[32, 257, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_7: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_62, getitem_63);  convert_element_type_62 = getitem_63 = None
        add_25: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
        rsqrt_7: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
        mul_23: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = rsqrt_7 = None
        mul_24: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_23, arg47_1);  mul_23 = arg47_1 = None
        add_26: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_24, arg48_1);  mul_24 = arg48_1 = None
        convert_element_type_63: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_26, torch.float16);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_37: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_63, [8224, 1408]);  convert_element_type_63 = None
        permute_23: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
        addmm_14: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg50_1, view_37, permute_23);  arg50_1 = view_37 = permute_23 = None
        view_38: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_14, [32, 257, 6144]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_67: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None
        mul_25: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.5)
        mul_26: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 0.7071067811865476);  convert_element_type_67 = None
        erf_3: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
        add_27: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_27: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
        convert_element_type_68: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_27, torch.float16);  mul_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_39: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_68, [8224, 6144]);  convert_element_type_68 = None
        permute_24: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_15: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg52_1, view_39, permute_24);  arg52_1 = view_39 = permute_24 = None
        view_40: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_15, [32, 257, 1408]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_28: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_24, view_40);  add_24 = view_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_72: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_28, torch.float32)
        var_mean_8 = torch.ops.aten.var_mean.correction(convert_element_type_72, [2], correction = 0, keepdim = True)
        getitem_64: "f32[32, 257, 1]" = var_mean_8[0]
        getitem_65: "f32[32, 257, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_8: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_72, getitem_65);  convert_element_type_72 = getitem_65 = None
        add_29: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
        rsqrt_8: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
        mul_28: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = rsqrt_8 = None
        mul_29: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_28, arg53_1);  mul_28 = arg53_1 = None
        add_30: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_29, arg54_1);  mul_29 = arg54_1 = None
        convert_element_type_73: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_30, torch.float16);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_41: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_73, [8224, 1408]);  convert_element_type_73 = None
        permute_25: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_16: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg56_1, view_41, permute_25);  arg56_1 = view_41 = permute_25 = None
        view_42: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_16, [32, 257, 4224]);  addmm_16 = None
        view_43: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_42, [32, 257, 3, 16, 88]);  view_42 = None
        permute_26: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_43, [2, 0, 3, 1, 4]);  view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_4 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
        getitem_66: "f16[32, 16, 257, 88]" = unbind_4[0]
        getitem_67: "f16[32, 16, 257, 88]" = unbind_4[1]
        getitem_68: "f16[32, 16, 257, 88]" = unbind_4[2];  unbind_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_4 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_66, getitem_67, getitem_68, None, False);  getitem_66 = getitem_67 = getitem_68 = None
        getitem_69: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_4[0];  _scaled_dot_product_cudnn_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_27: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3]);  getitem_69 = None
        view_44: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_27, [32, 257, 1408]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_45: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_44, [8224, 1408]);  view_44 = None
        permute_28: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_17: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg58_1, view_45, permute_28);  arg58_1 = view_45 = permute_28 = None
        view_46: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_17, [32, 257, 1408]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_31: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_28, view_46);  add_28 = view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_80: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_31, torch.float32)
        var_mean_9 = torch.ops.aten.var_mean.correction(convert_element_type_80, [2], correction = 0, keepdim = True)
        getitem_78: "f32[32, 257, 1]" = var_mean_9[0]
        getitem_79: "f32[32, 257, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_9: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_80, getitem_79);  convert_element_type_80 = getitem_79 = None
        add_32: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
        rsqrt_9: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
        mul_30: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = rsqrt_9 = None
        mul_31: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_30, arg59_1);  mul_30 = arg59_1 = None
        add_33: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_31, arg60_1);  mul_31 = arg60_1 = None
        convert_element_type_81: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_33, torch.float16);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_47: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_81, [8224, 1408]);  convert_element_type_81 = None
        permute_29: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_18: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg62_1, view_47, permute_29);  arg62_1 = view_47 = permute_29 = None
        view_48: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_18, [32, 257, 6144]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_85: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_48, torch.float32);  view_48 = None
        mul_32: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.5)
        mul_33: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 0.7071067811865476);  convert_element_type_85 = None
        erf_4: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
        add_34: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_34: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
        convert_element_type_86: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_34, torch.float16);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_49: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_86, [8224, 6144]);  convert_element_type_86 = None
        permute_30: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_19: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg64_1, view_49, permute_30);  arg64_1 = view_49 = permute_30 = None
        view_50: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_19, [32, 257, 1408]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_35: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_31, view_50);  add_31 = view_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_90: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_35, torch.float32)
        var_mean_10 = torch.ops.aten.var_mean.correction(convert_element_type_90, [2], correction = 0, keepdim = True)
        getitem_80: "f32[32, 257, 1]" = var_mean_10[0]
        getitem_81: "f32[32, 257, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_10: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_90, getitem_81);  convert_element_type_90 = getitem_81 = None
        add_36: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
        rsqrt_10: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
        mul_35: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = rsqrt_10 = None
        mul_36: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_35, arg65_1);  mul_35 = arg65_1 = None
        add_37: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_36, arg66_1);  mul_36 = arg66_1 = None
        convert_element_type_91: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_51: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_91, [8224, 1408]);  convert_element_type_91 = None
        permute_31: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_20: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg68_1, view_51, permute_31);  arg68_1 = view_51 = permute_31 = None
        view_52: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_20, [32, 257, 4224]);  addmm_20 = None
        view_53: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_52, [32, 257, 3, 16, 88]);  view_52 = None
        permute_32: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_53, [2, 0, 3, 1, 4]);  view_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_5 = torch.ops.aten.unbind.int(permute_32);  permute_32 = None
        getitem_82: "f16[32, 16, 257, 88]" = unbind_5[0]
        getitem_83: "f16[32, 16, 257, 88]" = unbind_5[1]
        getitem_84: "f16[32, 16, 257, 88]" = unbind_5[2];  unbind_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_5 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_82, getitem_83, getitem_84, None, False);  getitem_82 = getitem_83 = getitem_84 = None
        getitem_85: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_5[0];  _scaled_dot_product_cudnn_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_33: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
        view_54: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_33, [32, 257, 1408]);  permute_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_55: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_54, [8224, 1408]);  view_54 = None
        permute_34: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_21: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg70_1, view_55, permute_34);  arg70_1 = view_55 = permute_34 = None
        view_56: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_21, [32, 257, 1408]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_38: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_35, view_56);  add_35 = view_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_98: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_38, torch.float32)
        var_mean_11 = torch.ops.aten.var_mean.correction(convert_element_type_98, [2], correction = 0, keepdim = True)
        getitem_94: "f32[32, 257, 1]" = var_mean_11[0]
        getitem_95: "f32[32, 257, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_11: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_98, getitem_95);  convert_element_type_98 = getitem_95 = None
        add_39: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
        rsqrt_11: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
        mul_37: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = rsqrt_11 = None
        mul_38: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_37, arg71_1);  mul_37 = arg71_1 = None
        add_40: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_38, arg72_1);  mul_38 = arg72_1 = None
        convert_element_type_99: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_40, torch.float16);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_57: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_99, [8224, 1408]);  convert_element_type_99 = None
        permute_35: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_22: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg74_1, view_57, permute_35);  arg74_1 = view_57 = permute_35 = None
        view_58: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_22, [32, 257, 6144]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_103: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_58, torch.float32);  view_58 = None
        mul_39: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_103, 0.5)
        mul_40: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_103, 0.7071067811865476);  convert_element_type_103 = None
        erf_5: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
        add_41: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_41: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
        convert_element_type_104: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_41, torch.float16);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_59: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_104, [8224, 6144]);  convert_element_type_104 = None
        permute_36: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_23: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg76_1, view_59, permute_36);  arg76_1 = view_59 = permute_36 = None
        view_60: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_23, [32, 257, 1408]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_42: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_38, view_60);  add_38 = view_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_108: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_42, torch.float32)
        var_mean_12 = torch.ops.aten.var_mean.correction(convert_element_type_108, [2], correction = 0, keepdim = True)
        getitem_96: "f32[32, 257, 1]" = var_mean_12[0]
        getitem_97: "f32[32, 257, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_12: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_108, getitem_97);  convert_element_type_108 = getitem_97 = None
        add_43: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
        rsqrt_12: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
        mul_42: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = rsqrt_12 = None
        mul_43: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_42, arg77_1);  mul_42 = arg77_1 = None
        add_44: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_43, arg78_1);  mul_43 = arg78_1 = None
        convert_element_type_109: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_44, torch.float16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_61: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_109, [8224, 1408]);  convert_element_type_109 = None
        permute_37: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_24: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg80_1, view_61, permute_37);  arg80_1 = view_61 = permute_37 = None
        view_62: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_24, [32, 257, 4224]);  addmm_24 = None
        view_63: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_62, [32, 257, 3, 16, 88]);  view_62 = None
        permute_38: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_6 = torch.ops.aten.unbind.int(permute_38);  permute_38 = None
        getitem_98: "f16[32, 16, 257, 88]" = unbind_6[0]
        getitem_99: "f16[32, 16, 257, 88]" = unbind_6[1]
        getitem_100: "f16[32, 16, 257, 88]" = unbind_6[2];  unbind_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_6 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_98, getitem_99, getitem_100, None, False);  getitem_98 = getitem_99 = getitem_100 = None
        getitem_101: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_6[0];  _scaled_dot_product_cudnn_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_39: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
        view_64: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_39, [32, 257, 1408]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_65: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_64, [8224, 1408]);  view_64 = None
        permute_40: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_25: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg82_1, view_65, permute_40);  arg82_1 = view_65 = permute_40 = None
        view_66: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_25, [32, 257, 1408]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_45: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_42, view_66);  add_42 = view_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_116: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_45, torch.float32)
        var_mean_13 = torch.ops.aten.var_mean.correction(convert_element_type_116, [2], correction = 0, keepdim = True)
        getitem_110: "f32[32, 257, 1]" = var_mean_13[0]
        getitem_111: "f32[32, 257, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_13: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_116, getitem_111);  convert_element_type_116 = getitem_111 = None
        add_46: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
        rsqrt_13: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
        mul_44: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = rsqrt_13 = None
        mul_45: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_44, arg83_1);  mul_44 = arg83_1 = None
        add_47: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_45, arg84_1);  mul_45 = arg84_1 = None
        convert_element_type_117: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_47, torch.float16);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_67: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_117, [8224, 1408]);  convert_element_type_117 = None
        permute_41: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_26: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg86_1, view_67, permute_41);  arg86_1 = view_67 = permute_41 = None
        view_68: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_26, [32, 257, 6144]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_121: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_68, torch.float32);  view_68 = None
        mul_46: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_121, 0.5)
        mul_47: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_121, 0.7071067811865476);  convert_element_type_121 = None
        erf_6: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
        add_48: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_48: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None
        convert_element_type_122: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_48, torch.float16);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_69: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_122, [8224, 6144]);  convert_element_type_122 = None
        permute_42: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_27: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg88_1, view_69, permute_42);  arg88_1 = view_69 = permute_42 = None
        view_70: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_27, [32, 257, 1408]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_49: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_45, view_70);  add_45 = view_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_126: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_49, torch.float32)
        var_mean_14 = torch.ops.aten.var_mean.correction(convert_element_type_126, [2], correction = 0, keepdim = True)
        getitem_112: "f32[32, 257, 1]" = var_mean_14[0]
        getitem_113: "f32[32, 257, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_14: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_126, getitem_113);  convert_element_type_126 = getitem_113 = None
        add_50: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
        rsqrt_14: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
        mul_49: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = rsqrt_14 = None
        mul_50: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_49, arg89_1);  mul_49 = arg89_1 = None
        add_51: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_50, arg90_1);  mul_50 = arg90_1 = None
        convert_element_type_127: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_51, torch.float16);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_71: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_127, [8224, 1408]);  convert_element_type_127 = None
        permute_43: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_28: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg92_1, view_71, permute_43);  arg92_1 = view_71 = permute_43 = None
        view_72: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_28, [32, 257, 4224]);  addmm_28 = None
        view_73: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_72, [32, 257, 3, 16, 88]);  view_72 = None
        permute_44: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_73, [2, 0, 3, 1, 4]);  view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_7 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
        getitem_114: "f16[32, 16, 257, 88]" = unbind_7[0]
        getitem_115: "f16[32, 16, 257, 88]" = unbind_7[1]
        getitem_116: "f16[32, 16, 257, 88]" = unbind_7[2];  unbind_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_7 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_114, getitem_115, getitem_116, None, False);  getitem_114 = getitem_115 = getitem_116 = None
        getitem_117: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_7[0];  _scaled_dot_product_cudnn_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_45: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
        view_74: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_45, [32, 257, 1408]);  permute_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_75: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_74, [8224, 1408]);  view_74 = None
        permute_46: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_29: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg94_1, view_75, permute_46);  arg94_1 = view_75 = permute_46 = None
        view_76: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_29, [32, 257, 1408]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_52: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_49, view_76);  add_49 = view_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_134: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_52, torch.float32)
        var_mean_15 = torch.ops.aten.var_mean.correction(convert_element_type_134, [2], correction = 0, keepdim = True)
        getitem_126: "f32[32, 257, 1]" = var_mean_15[0]
        getitem_127: "f32[32, 257, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_15: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_134, getitem_127);  convert_element_type_134 = getitem_127 = None
        add_53: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
        rsqrt_15: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        mul_51: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = rsqrt_15 = None
        mul_52: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_51, arg95_1);  mul_51 = arg95_1 = None
        add_54: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_52, arg96_1);  mul_52 = arg96_1 = None
        convert_element_type_135: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_54, torch.float16);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_77: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_135, [8224, 1408]);  convert_element_type_135 = None
        permute_47: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_30: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg98_1, view_77, permute_47);  arg98_1 = view_77 = permute_47 = None
        view_78: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_30, [32, 257, 6144]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_139: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_78, torch.float32);  view_78 = None
        mul_53: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.5)
        mul_54: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 0.7071067811865476);  convert_element_type_139 = None
        erf_7: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
        add_55: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_55: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_53, add_55);  mul_53 = add_55 = None
        convert_element_type_140: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_55, torch.float16);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_79: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_140, [8224, 6144]);  convert_element_type_140 = None
        permute_48: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_31: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg100_1, view_79, permute_48);  arg100_1 = view_79 = permute_48 = None
        view_80: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_31, [32, 257, 1408]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_56: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_52, view_80);  add_52 = view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_144: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_56, torch.float32)
        var_mean_16 = torch.ops.aten.var_mean.correction(convert_element_type_144, [2], correction = 0, keepdim = True)
        getitem_128: "f32[32, 257, 1]" = var_mean_16[0]
        getitem_129: "f32[32, 257, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_16: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_144, getitem_129);  convert_element_type_144 = getitem_129 = None
        add_57: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
        rsqrt_16: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
        mul_56: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = rsqrt_16 = None
        mul_57: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_56, arg101_1);  mul_56 = arg101_1 = None
        add_58: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_57, arg102_1);  mul_57 = arg102_1 = None
        convert_element_type_145: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_58, torch.float16);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_81: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_145, [8224, 1408]);  convert_element_type_145 = None
        permute_49: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_32: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg104_1, view_81, permute_49);  arg104_1 = view_81 = permute_49 = None
        view_82: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_32, [32, 257, 4224]);  addmm_32 = None
        view_83: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_82, [32, 257, 3, 16, 88]);  view_82 = None
        permute_50: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_83, [2, 0, 3, 1, 4]);  view_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_8 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
        getitem_130: "f16[32, 16, 257, 88]" = unbind_8[0]
        getitem_131: "f16[32, 16, 257, 88]" = unbind_8[1]
        getitem_132: "f16[32, 16, 257, 88]" = unbind_8[2];  unbind_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_8 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_130, getitem_131, getitem_132, None, False);  getitem_130 = getitem_131 = getitem_132 = None
        getitem_133: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_8[0];  _scaled_dot_product_cudnn_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_51: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
        view_84: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_51, [32, 257, 1408]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_85: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_84, [8224, 1408]);  view_84 = None
        permute_52: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_33: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg106_1, view_85, permute_52);  arg106_1 = view_85 = permute_52 = None
        view_86: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_33, [32, 257, 1408]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_59: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_56, view_86);  add_56 = view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_152: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_59, torch.float32)
        var_mean_17 = torch.ops.aten.var_mean.correction(convert_element_type_152, [2], correction = 0, keepdim = True)
        getitem_142: "f32[32, 257, 1]" = var_mean_17[0]
        getitem_143: "f32[32, 257, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_17: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_152, getitem_143);  convert_element_type_152 = getitem_143 = None
        add_60: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
        rsqrt_17: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_58: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = rsqrt_17 = None
        mul_59: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_58, arg107_1);  mul_58 = arg107_1 = None
        add_61: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_59, arg108_1);  mul_59 = arg108_1 = None
        convert_element_type_153: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_61, torch.float16);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_87: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_153, [8224, 1408]);  convert_element_type_153 = None
        permute_53: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_34: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg110_1, view_87, permute_53);  arg110_1 = view_87 = permute_53 = None
        view_88: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_34, [32, 257, 6144]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_157: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_88, torch.float32);  view_88 = None
        mul_60: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.5)
        mul_61: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.7071067811865476);  convert_element_type_157 = None
        erf_8: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
        add_62: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_62: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_60, add_62);  mul_60 = add_62 = None
        convert_element_type_158: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_62, torch.float16);  mul_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_89: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_158, [8224, 6144]);  convert_element_type_158 = None
        permute_54: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_35: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg112_1, view_89, permute_54);  arg112_1 = view_89 = permute_54 = None
        view_90: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_35, [32, 257, 1408]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_63: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_59, view_90);  add_59 = view_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_162: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_63, torch.float32)
        var_mean_18 = torch.ops.aten.var_mean.correction(convert_element_type_162, [2], correction = 0, keepdim = True)
        getitem_144: "f32[32, 257, 1]" = var_mean_18[0]
        getitem_145: "f32[32, 257, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_18: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_162, getitem_145);  convert_element_type_162 = getitem_145 = None
        add_64: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
        rsqrt_18: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        mul_63: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = rsqrt_18 = None
        mul_64: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_63, arg113_1);  mul_63 = arg113_1 = None
        add_65: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_64, arg114_1);  mul_64 = arg114_1 = None
        convert_element_type_163: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_91: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_163, [8224, 1408]);  convert_element_type_163 = None
        permute_55: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_36: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg116_1, view_91, permute_55);  arg116_1 = view_91 = permute_55 = None
        view_92: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_36, [32, 257, 4224]);  addmm_36 = None
        view_93: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_92, [32, 257, 3, 16, 88]);  view_92 = None
        permute_56: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_93, [2, 0, 3, 1, 4]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_9 = torch.ops.aten.unbind.int(permute_56);  permute_56 = None
        getitem_146: "f16[32, 16, 257, 88]" = unbind_9[0]
        getitem_147: "f16[32, 16, 257, 88]" = unbind_9[1]
        getitem_148: "f16[32, 16, 257, 88]" = unbind_9[2];  unbind_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_9 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_146, getitem_147, getitem_148, None, False);  getitem_146 = getitem_147 = getitem_148 = None
        getitem_149: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_9[0];  _scaled_dot_product_cudnn_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_57: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3]);  getitem_149 = None
        view_94: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_57, [32, 257, 1408]);  permute_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_95: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_94, [8224, 1408]);  view_94 = None
        permute_58: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
        addmm_37: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg118_1, view_95, permute_58);  arg118_1 = view_95 = permute_58 = None
        view_96: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_37, [32, 257, 1408]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_66: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_63, view_96);  add_63 = view_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_170: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_66, torch.float32)
        var_mean_19 = torch.ops.aten.var_mean.correction(convert_element_type_170, [2], correction = 0, keepdim = True)
        getitem_158: "f32[32, 257, 1]" = var_mean_19[0]
        getitem_159: "f32[32, 257, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_19: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_170, getitem_159);  convert_element_type_170 = getitem_159 = None
        add_67: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
        rsqrt_19: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_65: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = rsqrt_19 = None
        mul_66: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_65, arg119_1);  mul_65 = arg119_1 = None
        add_68: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_66, arg120_1);  mul_66 = arg120_1 = None
        convert_element_type_171: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_68, torch.float16);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_97: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_171, [8224, 1408]);  convert_element_type_171 = None
        permute_59: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_38: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg122_1, view_97, permute_59);  arg122_1 = view_97 = permute_59 = None
        view_98: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_38, [32, 257, 6144]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_175: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_98, torch.float32);  view_98 = None
        mul_67: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.5)
        mul_68: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_175, 0.7071067811865476);  convert_element_type_175 = None
        erf_9: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
        add_69: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_69: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_67, add_69);  mul_67 = add_69 = None
        convert_element_type_176: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_69, torch.float16);  mul_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_99: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_176, [8224, 6144]);  convert_element_type_176 = None
        permute_60: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_39: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg124_1, view_99, permute_60);  arg124_1 = view_99 = permute_60 = None
        view_100: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_39, [32, 257, 1408]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_70: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_66, view_100);  add_66 = view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_180: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_70, torch.float32)
        var_mean_20 = torch.ops.aten.var_mean.correction(convert_element_type_180, [2], correction = 0, keepdim = True)
        getitem_160: "f32[32, 257, 1]" = var_mean_20[0]
        getitem_161: "f32[32, 257, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_20: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_180, getitem_161);  convert_element_type_180 = getitem_161 = None
        add_71: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
        rsqrt_20: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_70: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = rsqrt_20 = None
        mul_71: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_70, arg125_1);  mul_70 = arg125_1 = None
        add_72: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_71, arg126_1);  mul_71 = arg126_1 = None
        convert_element_type_181: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_72, torch.float16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_101: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_181, [8224, 1408]);  convert_element_type_181 = None
        permute_61: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_40: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg128_1, view_101, permute_61);  arg128_1 = view_101 = permute_61 = None
        view_102: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_40, [32, 257, 4224]);  addmm_40 = None
        view_103: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_102, [32, 257, 3, 16, 88]);  view_102 = None
        permute_62: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_103, [2, 0, 3, 1, 4]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_10 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
        getitem_162: "f16[32, 16, 257, 88]" = unbind_10[0]
        getitem_163: "f16[32, 16, 257, 88]" = unbind_10[1]
        getitem_164: "f16[32, 16, 257, 88]" = unbind_10[2];  unbind_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_10 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_162, getitem_163, getitem_164, None, False);  getitem_162 = getitem_163 = getitem_164 = None
        getitem_165: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_10[0];  _scaled_dot_product_cudnn_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_63: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
        view_104: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_63, [32, 257, 1408]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_105: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_104, [8224, 1408]);  view_104 = None
        permute_64: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_41: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg130_1, view_105, permute_64);  arg130_1 = view_105 = permute_64 = None
        view_106: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_41, [32, 257, 1408]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_73: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_70, view_106);  add_70 = view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_188: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_73, torch.float32)
        var_mean_21 = torch.ops.aten.var_mean.correction(convert_element_type_188, [2], correction = 0, keepdim = True)
        getitem_174: "f32[32, 257, 1]" = var_mean_21[0]
        getitem_175: "f32[32, 257, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_21: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_188, getitem_175);  convert_element_type_188 = getitem_175 = None
        add_74: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
        rsqrt_21: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_72: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = rsqrt_21 = None
        mul_73: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_72, arg131_1);  mul_72 = arg131_1 = None
        add_75: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_73, arg132_1);  mul_73 = arg132_1 = None
        convert_element_type_189: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_75, torch.float16);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_107: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_189, [8224, 1408]);  convert_element_type_189 = None
        permute_65: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_42: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg134_1, view_107, permute_65);  arg134_1 = view_107 = permute_65 = None
        view_108: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_42, [32, 257, 6144]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_193: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        mul_74: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_193, 0.5)
        mul_75: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_193, 0.7071067811865476);  convert_element_type_193 = None
        erf_10: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_76: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_76: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_74, add_76);  mul_74 = add_76 = None
        convert_element_type_194: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_76, torch.float16);  mul_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_109: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_194, [8224, 6144]);  convert_element_type_194 = None
        permute_66: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_43: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg136_1, view_109, permute_66);  arg136_1 = view_109 = permute_66 = None
        view_110: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_43, [32, 257, 1408]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_77: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_73, view_110);  add_73 = view_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_198: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_77, torch.float32)
        var_mean_22 = torch.ops.aten.var_mean.correction(convert_element_type_198, [2], correction = 0, keepdim = True)
        getitem_176: "f32[32, 257, 1]" = var_mean_22[0]
        getitem_177: "f32[32, 257, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_22: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_198, getitem_177);  convert_element_type_198 = getitem_177 = None
        add_78: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
        rsqrt_22: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_77: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = rsqrt_22 = None
        mul_78: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_77, arg137_1);  mul_77 = arg137_1 = None
        add_79: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_78, arg138_1);  mul_78 = arg138_1 = None
        convert_element_type_199: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_79, torch.float16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_111: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_199, [8224, 1408]);  convert_element_type_199 = None
        permute_67: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_44: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg140_1, view_111, permute_67);  arg140_1 = view_111 = permute_67 = None
        view_112: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_44, [32, 257, 4224]);  addmm_44 = None
        view_113: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_112, [32, 257, 3, 16, 88]);  view_112 = None
        permute_68: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_113, [2, 0, 3, 1, 4]);  view_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_11 = torch.ops.aten.unbind.int(permute_68);  permute_68 = None
        getitem_178: "f16[32, 16, 257, 88]" = unbind_11[0]
        getitem_179: "f16[32, 16, 257, 88]" = unbind_11[1]
        getitem_180: "f16[32, 16, 257, 88]" = unbind_11[2];  unbind_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_11 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_178, getitem_179, getitem_180, None, False);  getitem_178 = getitem_179 = getitem_180 = None
        getitem_181: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_11[0];  _scaled_dot_product_cudnn_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_69: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
        view_114: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_69, [32, 257, 1408]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_115: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_114, [8224, 1408]);  view_114 = None
        permute_70: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_45: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg142_1, view_115, permute_70);  arg142_1 = view_115 = permute_70 = None
        view_116: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_45, [32, 257, 1408]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_80: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_77, view_116);  add_77 = view_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_206: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_80, torch.float32)
        var_mean_23 = torch.ops.aten.var_mean.correction(convert_element_type_206, [2], correction = 0, keepdim = True)
        getitem_190: "f32[32, 257, 1]" = var_mean_23[0]
        getitem_191: "f32[32, 257, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_23: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_206, getitem_191);  convert_element_type_206 = getitem_191 = None
        add_81: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-06);  getitem_190 = None
        rsqrt_23: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
        mul_79: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = rsqrt_23 = None
        mul_80: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_79, arg143_1);  mul_79 = arg143_1 = None
        add_82: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_80, arg144_1);  mul_80 = arg144_1 = None
        convert_element_type_207: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_82, torch.float16);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_117: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_207, [8224, 1408]);  convert_element_type_207 = None
        permute_71: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_46: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg146_1, view_117, permute_71);  arg146_1 = view_117 = permute_71 = None
        view_118: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_46, [32, 257, 6144]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_211: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        mul_81: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.5)
        mul_82: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_211, 0.7071067811865476);  convert_element_type_211 = None
        erf_11: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
        add_83: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_83: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_81, add_83);  mul_81 = add_83 = None
        convert_element_type_212: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_83, torch.float16);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_119: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_212, [8224, 6144]);  convert_element_type_212 = None
        permute_72: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_47: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg148_1, view_119, permute_72);  arg148_1 = view_119 = permute_72 = None
        view_120: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_47, [32, 257, 1408]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_84: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_80, view_120);  add_80 = view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_216: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_84, torch.float32)
        var_mean_24 = torch.ops.aten.var_mean.correction(convert_element_type_216, [2], correction = 0, keepdim = True)
        getitem_192: "f32[32, 257, 1]" = var_mean_24[0]
        getitem_193: "f32[32, 257, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_24: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_216, getitem_193);  convert_element_type_216 = getitem_193 = None
        add_85: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
        rsqrt_24: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_84: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = rsqrt_24 = None
        mul_85: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_84, arg149_1);  mul_84 = arg149_1 = None
        add_86: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_85, arg150_1);  mul_85 = arg150_1 = None
        convert_element_type_217: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_86, torch.float16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_121: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_217, [8224, 1408]);  convert_element_type_217 = None
        permute_73: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_48: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg152_1, view_121, permute_73);  arg152_1 = view_121 = permute_73 = None
        view_122: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_48, [32, 257, 4224]);  addmm_48 = None
        view_123: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_122, [32, 257, 3, 16, 88]);  view_122 = None
        permute_74: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_123, [2, 0, 3, 1, 4]);  view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_12 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
        getitem_194: "f16[32, 16, 257, 88]" = unbind_12[0]
        getitem_195: "f16[32, 16, 257, 88]" = unbind_12[1]
        getitem_196: "f16[32, 16, 257, 88]" = unbind_12[2];  unbind_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_12 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_194, getitem_195, getitem_196, None, False);  getitem_194 = getitem_195 = getitem_196 = None
        getitem_197: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_12[0];  _scaled_dot_product_cudnn_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_75: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_197, [0, 2, 1, 3]);  getitem_197 = None
        view_124: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_75, [32, 257, 1408]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_125: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_124, [8224, 1408]);  view_124 = None
        permute_76: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_49: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg154_1, view_125, permute_76);  arg154_1 = view_125 = permute_76 = None
        view_126: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_49, [32, 257, 1408]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_87: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_84, view_126);  add_84 = view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_224: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_87, torch.float32)
        var_mean_25 = torch.ops.aten.var_mean.correction(convert_element_type_224, [2], correction = 0, keepdim = True)
        getitem_206: "f32[32, 257, 1]" = var_mean_25[0]
        getitem_207: "f32[32, 257, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_25: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_224, getitem_207);  convert_element_type_224 = getitem_207 = None
        add_88: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-06);  getitem_206 = None
        rsqrt_25: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
        mul_86: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = rsqrt_25 = None
        mul_87: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_86, arg155_1);  mul_86 = arg155_1 = None
        add_89: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_87, arg156_1);  mul_87 = arg156_1 = None
        convert_element_type_225: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_89, torch.float16);  add_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_127: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_225, [8224, 1408]);  convert_element_type_225 = None
        permute_77: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_50: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg158_1, view_127, permute_77);  arg158_1 = view_127 = permute_77 = None
        view_128: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_50, [32, 257, 6144]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_229: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_128, torch.float32);  view_128 = None
        mul_88: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_229, 0.5)
        mul_89: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_229, 0.7071067811865476);  convert_element_type_229 = None
        erf_12: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
        add_90: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_90: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_88, add_90);  mul_88 = add_90 = None
        convert_element_type_230: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_90, torch.float16);  mul_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_129: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_230, [8224, 6144]);  convert_element_type_230 = None
        permute_78: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_51: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg160_1, view_129, permute_78);  arg160_1 = view_129 = permute_78 = None
        view_130: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_51, [32, 257, 1408]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_91: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_87, view_130);  add_87 = view_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_234: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_91, torch.float32)
        var_mean_26 = torch.ops.aten.var_mean.correction(convert_element_type_234, [2], correction = 0, keepdim = True)
        getitem_208: "f32[32, 257, 1]" = var_mean_26[0]
        getitem_209: "f32[32, 257, 1]" = var_mean_26[1];  var_mean_26 = None
        sub_26: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_234, getitem_209);  convert_element_type_234 = getitem_209 = None
        add_92: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-06);  getitem_208 = None
        rsqrt_26: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
        mul_91: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = rsqrt_26 = None
        mul_92: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_91, arg161_1);  mul_91 = arg161_1 = None
        add_93: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_92, arg162_1);  mul_92 = arg162_1 = None
        convert_element_type_235: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_93, torch.float16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_131: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_235, [8224, 1408]);  convert_element_type_235 = None
        permute_79: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
        addmm_52: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg164_1, view_131, permute_79);  arg164_1 = view_131 = permute_79 = None
        view_132: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_52, [32, 257, 4224]);  addmm_52 = None
        view_133: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_132, [32, 257, 3, 16, 88]);  view_132 = None
        permute_80: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_133, [2, 0, 3, 1, 4]);  view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_13 = torch.ops.aten.unbind.int(permute_80);  permute_80 = None
        getitem_210: "f16[32, 16, 257, 88]" = unbind_13[0]
        getitem_211: "f16[32, 16, 257, 88]" = unbind_13[1]
        getitem_212: "f16[32, 16, 257, 88]" = unbind_13[2];  unbind_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_13 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_210, getitem_211, getitem_212, None, False);  getitem_210 = getitem_211 = getitem_212 = None
        getitem_213: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_13[0];  _scaled_dot_product_cudnn_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_81: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_213, [0, 2, 1, 3]);  getitem_213 = None
        view_134: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_81, [32, 257, 1408]);  permute_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_135: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_134, [8224, 1408]);  view_134 = None
        permute_82: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_53: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg166_1, view_135, permute_82);  arg166_1 = view_135 = permute_82 = None
        view_136: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_53, [32, 257, 1408]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_94: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_91, view_136);  add_91 = view_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_242: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_94, torch.float32)
        var_mean_27 = torch.ops.aten.var_mean.correction(convert_element_type_242, [2], correction = 0, keepdim = True)
        getitem_222: "f32[32, 257, 1]" = var_mean_27[0]
        getitem_223: "f32[32, 257, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_27: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_242, getitem_223);  convert_element_type_242 = getitem_223 = None
        add_95: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
        rsqrt_27: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
        mul_93: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_27);  sub_27 = rsqrt_27 = None
        mul_94: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_93, arg167_1);  mul_93 = arg167_1 = None
        add_96: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_94, arg168_1);  mul_94 = arg168_1 = None
        convert_element_type_243: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_96, torch.float16);  add_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_137: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_243, [8224, 1408]);  convert_element_type_243 = None
        permute_83: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
        addmm_54: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg170_1, view_137, permute_83);  arg170_1 = view_137 = permute_83 = None
        view_138: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_54, [32, 257, 6144]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_247: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_138, torch.float32);  view_138 = None
        mul_95: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_247, 0.5)
        mul_96: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_247, 0.7071067811865476);  convert_element_type_247 = None
        erf_13: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
        add_97: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_97: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_95, add_97);  mul_95 = add_97 = None
        convert_element_type_248: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_97, torch.float16);  mul_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_139: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_248, [8224, 6144]);  convert_element_type_248 = None
        permute_84: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_55: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg172_1, view_139, permute_84);  arg172_1 = view_139 = permute_84 = None
        view_140: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_55, [32, 257, 1408]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_98: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_94, view_140);  add_94 = view_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_252: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_98, torch.float32)
        var_mean_28 = torch.ops.aten.var_mean.correction(convert_element_type_252, [2], correction = 0, keepdim = True)
        getitem_224: "f32[32, 257, 1]" = var_mean_28[0]
        getitem_225: "f32[32, 257, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_28: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_252, getitem_225);  convert_element_type_252 = getitem_225 = None
        add_99: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-06);  getitem_224 = None
        rsqrt_28: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
        mul_98: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_28);  sub_28 = rsqrt_28 = None
        mul_99: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_98, arg173_1);  mul_98 = arg173_1 = None
        add_100: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_99, arg174_1);  mul_99 = arg174_1 = None
        convert_element_type_253: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_100, torch.float16);  add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_141: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_253, [8224, 1408]);  convert_element_type_253 = None
        permute_85: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_56: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg176_1, view_141, permute_85);  arg176_1 = view_141 = permute_85 = None
        view_142: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_56, [32, 257, 4224]);  addmm_56 = None
        view_143: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_142, [32, 257, 3, 16, 88]);  view_142 = None
        permute_86: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_143, [2, 0, 3, 1, 4]);  view_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_14 = torch.ops.aten.unbind.int(permute_86);  permute_86 = None
        getitem_226: "f16[32, 16, 257, 88]" = unbind_14[0]
        getitem_227: "f16[32, 16, 257, 88]" = unbind_14[1]
        getitem_228: "f16[32, 16, 257, 88]" = unbind_14[2];  unbind_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_14 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_226, getitem_227, getitem_228, None, False);  getitem_226 = getitem_227 = getitem_228 = None
        getitem_229: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_14[0];  _scaled_dot_product_cudnn_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_87: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_229, [0, 2, 1, 3]);  getitem_229 = None
        view_144: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_87, [32, 257, 1408]);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_145: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_144, [8224, 1408]);  view_144 = None
        permute_88: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_57: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg178_1, view_145, permute_88);  arg178_1 = view_145 = permute_88 = None
        view_146: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_57, [32, 257, 1408]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_101: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_98, view_146);  add_98 = view_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_260: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_101, torch.float32)
        var_mean_29 = torch.ops.aten.var_mean.correction(convert_element_type_260, [2], correction = 0, keepdim = True)
        getitem_238: "f32[32, 257, 1]" = var_mean_29[0]
        getitem_239: "f32[32, 257, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_29: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_260, getitem_239);  convert_element_type_260 = getitem_239 = None
        add_102: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-06);  getitem_238 = None
        rsqrt_29: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_102);  add_102 = None
        mul_100: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_29);  sub_29 = rsqrt_29 = None
        mul_101: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_100, arg179_1);  mul_100 = arg179_1 = None
        add_103: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_101, arg180_1);  mul_101 = arg180_1 = None
        convert_element_type_261: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_103, torch.float16);  add_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_147: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_261, [8224, 1408]);  convert_element_type_261 = None
        permute_89: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_58: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg182_1, view_147, permute_89);  arg182_1 = view_147 = permute_89 = None
        view_148: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_58, [32, 257, 6144]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_265: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_148, torch.float32);  view_148 = None
        mul_102: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_265, 0.5)
        mul_103: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_265, 0.7071067811865476);  convert_element_type_265 = None
        erf_14: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_104: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_104: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_102, add_104);  mul_102 = add_104 = None
        convert_element_type_266: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_104, torch.float16);  mul_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_149: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_266, [8224, 6144]);  convert_element_type_266 = None
        permute_90: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_59: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg184_1, view_149, permute_90);  arg184_1 = view_149 = permute_90 = None
        view_150: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_59, [32, 257, 1408]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_105: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_101, view_150);  add_101 = view_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_270: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_105, torch.float32)
        var_mean_30 = torch.ops.aten.var_mean.correction(convert_element_type_270, [2], correction = 0, keepdim = True)
        getitem_240: "f32[32, 257, 1]" = var_mean_30[0]
        getitem_241: "f32[32, 257, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_30: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_270, getitem_241);  convert_element_type_270 = getitem_241 = None
        add_106: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-06);  getitem_240 = None
        rsqrt_30: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
        mul_105: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_30);  sub_30 = rsqrt_30 = None
        mul_106: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_105, arg185_1);  mul_105 = arg185_1 = None
        add_107: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_106, arg186_1);  mul_106 = arg186_1 = None
        convert_element_type_271: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_107, torch.float16);  add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_151: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_271, [8224, 1408]);  convert_element_type_271 = None
        permute_91: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_60: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg188_1, view_151, permute_91);  arg188_1 = view_151 = permute_91 = None
        view_152: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_60, [32, 257, 4224]);  addmm_60 = None
        view_153: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_152, [32, 257, 3, 16, 88]);  view_152 = None
        permute_92: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_153, [2, 0, 3, 1, 4]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_15 = torch.ops.aten.unbind.int(permute_92);  permute_92 = None
        getitem_242: "f16[32, 16, 257, 88]" = unbind_15[0]
        getitem_243: "f16[32, 16, 257, 88]" = unbind_15[1]
        getitem_244: "f16[32, 16, 257, 88]" = unbind_15[2];  unbind_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_15 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_242, getitem_243, getitem_244, None, False);  getitem_242 = getitem_243 = getitem_244 = None
        getitem_245: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_15[0];  _scaled_dot_product_cudnn_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_93: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_245, [0, 2, 1, 3]);  getitem_245 = None
        view_154: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_93, [32, 257, 1408]);  permute_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_155: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_154, [8224, 1408]);  view_154 = None
        permute_94: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_61: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg190_1, view_155, permute_94);  arg190_1 = view_155 = permute_94 = None
        view_156: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_61, [32, 257, 1408]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_108: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_105, view_156);  add_105 = view_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_278: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_108, torch.float32)
        var_mean_31 = torch.ops.aten.var_mean.correction(convert_element_type_278, [2], correction = 0, keepdim = True)
        getitem_254: "f32[32, 257, 1]" = var_mean_31[0]
        getitem_255: "f32[32, 257, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_31: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_278, getitem_255);  convert_element_type_278 = getitem_255 = None
        add_109: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_254, 1e-06);  getitem_254 = None
        rsqrt_31: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
        mul_107: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_31);  sub_31 = rsqrt_31 = None
        mul_108: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_107, arg191_1);  mul_107 = arg191_1 = None
        add_110: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_108, arg192_1);  mul_108 = arg192_1 = None
        convert_element_type_279: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_110, torch.float16);  add_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_157: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_279, [8224, 1408]);  convert_element_type_279 = None
        permute_95: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_62: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg194_1, view_157, permute_95);  arg194_1 = view_157 = permute_95 = None
        view_158: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_62, [32, 257, 6144]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_283: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None
        mul_109: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_283, 0.5)
        mul_110: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_283, 0.7071067811865476);  convert_element_type_283 = None
        erf_15: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
        add_111: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_111: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_109, add_111);  mul_109 = add_111 = None
        convert_element_type_284: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_111, torch.float16);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_159: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_284, [8224, 6144]);  convert_element_type_284 = None
        permute_96: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
        addmm_63: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg196_1, view_159, permute_96);  arg196_1 = view_159 = permute_96 = None
        view_160: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_63, [32, 257, 1408]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_112: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_108, view_160);  add_108 = view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_288: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_112, torch.float32)
        var_mean_32 = torch.ops.aten.var_mean.correction(convert_element_type_288, [2], correction = 0, keepdim = True)
        getitem_256: "f32[32, 257, 1]" = var_mean_32[0]
        getitem_257: "f32[32, 257, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_32: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_288, getitem_257);  convert_element_type_288 = getitem_257 = None
        add_113: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_256, 1e-06);  getitem_256 = None
        rsqrt_32: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
        mul_112: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_32);  sub_32 = rsqrt_32 = None
        mul_113: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_112, arg197_1);  mul_112 = arg197_1 = None
        add_114: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_113, arg198_1);  mul_113 = arg198_1 = None
        convert_element_type_289: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_114, torch.float16);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_161: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_289, [8224, 1408]);  convert_element_type_289 = None
        permute_97: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_64: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg200_1, view_161, permute_97);  arg200_1 = view_161 = permute_97 = None
        view_162: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_64, [32, 257, 4224]);  addmm_64 = None
        view_163: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_162, [32, 257, 3, 16, 88]);  view_162 = None
        permute_98: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_163, [2, 0, 3, 1, 4]);  view_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_16 = torch.ops.aten.unbind.int(permute_98);  permute_98 = None
        getitem_258: "f16[32, 16, 257, 88]" = unbind_16[0]
        getitem_259: "f16[32, 16, 257, 88]" = unbind_16[1]
        getitem_260: "f16[32, 16, 257, 88]" = unbind_16[2];  unbind_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_16 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_258, getitem_259, getitem_260, None, False);  getitem_258 = getitem_259 = getitem_260 = None
        getitem_261: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_16[0];  _scaled_dot_product_cudnn_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_99: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_261, [0, 2, 1, 3]);  getitem_261 = None
        view_164: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_99, [32, 257, 1408]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_165: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_164, [8224, 1408]);  view_164 = None
        permute_100: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_65: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg202_1, view_165, permute_100);  arg202_1 = view_165 = permute_100 = None
        view_166: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_65, [32, 257, 1408]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_115: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_112, view_166);  add_112 = view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_296: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_115, torch.float32)
        var_mean_33 = torch.ops.aten.var_mean.correction(convert_element_type_296, [2], correction = 0, keepdim = True)
        getitem_270: "f32[32, 257, 1]" = var_mean_33[0]
        getitem_271: "f32[32, 257, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_33: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_296, getitem_271);  convert_element_type_296 = getitem_271 = None
        add_116: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_270, 1e-06);  getitem_270 = None
        rsqrt_33: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
        mul_114: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_33);  sub_33 = rsqrt_33 = None
        mul_115: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_114, arg203_1);  mul_114 = arg203_1 = None
        add_117: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_115, arg204_1);  mul_115 = arg204_1 = None
        convert_element_type_297: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_117, torch.float16);  add_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_167: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_297, [8224, 1408]);  convert_element_type_297 = None
        permute_101: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_66: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg206_1, view_167, permute_101);  arg206_1 = view_167 = permute_101 = None
        view_168: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_66, [32, 257, 6144]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_301: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_168, torch.float32);  view_168 = None
        mul_116: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_301, 0.5)
        mul_117: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_301, 0.7071067811865476);  convert_element_type_301 = None
        erf_16: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_117);  mul_117 = None
        add_118: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_118: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_116, add_118);  mul_116 = add_118 = None
        convert_element_type_302: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_118, torch.float16);  mul_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_169: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_302, [8224, 6144]);  convert_element_type_302 = None
        permute_102: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_67: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg208_1, view_169, permute_102);  arg208_1 = view_169 = permute_102 = None
        view_170: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_67, [32, 257, 1408]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_119: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_115, view_170);  add_115 = view_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_306: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_119, torch.float32)
        var_mean_34 = torch.ops.aten.var_mean.correction(convert_element_type_306, [2], correction = 0, keepdim = True)
        getitem_272: "f32[32, 257, 1]" = var_mean_34[0]
        getitem_273: "f32[32, 257, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_34: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_306, getitem_273);  convert_element_type_306 = getitem_273 = None
        add_120: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_272, 1e-06);  getitem_272 = None
        rsqrt_34: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
        mul_119: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_34);  sub_34 = rsqrt_34 = None
        mul_120: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_119, arg209_1);  mul_119 = arg209_1 = None
        add_121: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_120, arg210_1);  mul_120 = arg210_1 = None
        convert_element_type_307: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_121, torch.float16);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_171: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_307, [8224, 1408]);  convert_element_type_307 = None
        permute_103: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_68: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg212_1, view_171, permute_103);  arg212_1 = view_171 = permute_103 = None
        view_172: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_68, [32, 257, 4224]);  addmm_68 = None
        view_173: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_172, [32, 257, 3, 16, 88]);  view_172 = None
        permute_104: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_173, [2, 0, 3, 1, 4]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_17 = torch.ops.aten.unbind.int(permute_104);  permute_104 = None
        getitem_274: "f16[32, 16, 257, 88]" = unbind_17[0]
        getitem_275: "f16[32, 16, 257, 88]" = unbind_17[1]
        getitem_276: "f16[32, 16, 257, 88]" = unbind_17[2];  unbind_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_17 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_274, getitem_275, getitem_276, None, False);  getitem_274 = getitem_275 = getitem_276 = None
        getitem_277: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_17[0];  _scaled_dot_product_cudnn_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_105: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_277, [0, 2, 1, 3]);  getitem_277 = None
        view_174: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_105, [32, 257, 1408]);  permute_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_175: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_174, [8224, 1408]);  view_174 = None
        permute_106: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_69: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg214_1, view_175, permute_106);  arg214_1 = view_175 = permute_106 = None
        view_176: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_69, [32, 257, 1408]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_122: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_119, view_176);  add_119 = view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_314: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_122, torch.float32)
        var_mean_35 = torch.ops.aten.var_mean.correction(convert_element_type_314, [2], correction = 0, keepdim = True)
        getitem_286: "f32[32, 257, 1]" = var_mean_35[0]
        getitem_287: "f32[32, 257, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_35: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_314, getitem_287);  convert_element_type_314 = getitem_287 = None
        add_123: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_286, 1e-06);  getitem_286 = None
        rsqrt_35: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
        mul_121: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_35);  sub_35 = rsqrt_35 = None
        mul_122: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_121, arg215_1);  mul_121 = arg215_1 = None
        add_124: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_122, arg216_1);  mul_122 = arg216_1 = None
        convert_element_type_315: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_124, torch.float16);  add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_177: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_315, [8224, 1408]);  convert_element_type_315 = None
        permute_107: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_70: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg218_1, view_177, permute_107);  arg218_1 = view_177 = permute_107 = None
        view_178: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_70, [32, 257, 6144]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_319: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_178, torch.float32);  view_178 = None
        mul_123: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_319, 0.5)
        mul_124: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_319, 0.7071067811865476);  convert_element_type_319 = None
        erf_17: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_124);  mul_124 = None
        add_125: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_125: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_123, add_125);  mul_123 = add_125 = None
        convert_element_type_320: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_125, torch.float16);  mul_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_179: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_320, [8224, 6144]);  convert_element_type_320 = None
        permute_108: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_71: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg220_1, view_179, permute_108);  arg220_1 = view_179 = permute_108 = None
        view_180: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_71, [32, 257, 1408]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_126: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_122, view_180);  add_122 = view_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_324: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_126, torch.float32)
        var_mean_36 = torch.ops.aten.var_mean.correction(convert_element_type_324, [2], correction = 0, keepdim = True)
        getitem_288: "f32[32, 257, 1]" = var_mean_36[0]
        getitem_289: "f32[32, 257, 1]" = var_mean_36[1];  var_mean_36 = None
        sub_36: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_324, getitem_289);  convert_element_type_324 = getitem_289 = None
        add_127: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_288, 1e-06);  getitem_288 = None
        rsqrt_36: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
        mul_126: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_36);  sub_36 = rsqrt_36 = None
        mul_127: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_126, arg221_1);  mul_126 = arg221_1 = None
        add_128: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_127, arg222_1);  mul_127 = arg222_1 = None
        convert_element_type_325: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_128, torch.float16);  add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_181: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_325, [8224, 1408]);  convert_element_type_325 = None
        permute_109: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_72: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg224_1, view_181, permute_109);  arg224_1 = view_181 = permute_109 = None
        view_182: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_72, [32, 257, 4224]);  addmm_72 = None
        view_183: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_182, [32, 257, 3, 16, 88]);  view_182 = None
        permute_110: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_183, [2, 0, 3, 1, 4]);  view_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_18 = torch.ops.aten.unbind.int(permute_110);  permute_110 = None
        getitem_290: "f16[32, 16, 257, 88]" = unbind_18[0]
        getitem_291: "f16[32, 16, 257, 88]" = unbind_18[1]
        getitem_292: "f16[32, 16, 257, 88]" = unbind_18[2];  unbind_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_18 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_290, getitem_291, getitem_292, None, False);  getitem_290 = getitem_291 = getitem_292 = None
        getitem_293: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_18[0];  _scaled_dot_product_cudnn_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_111: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_293, [0, 2, 1, 3]);  getitem_293 = None
        view_184: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_111, [32, 257, 1408]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_185: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_184, [8224, 1408]);  view_184 = None
        permute_112: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_73: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg226_1, view_185, permute_112);  arg226_1 = view_185 = permute_112 = None
        view_186: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_73, [32, 257, 1408]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_129: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_126, view_186);  add_126 = view_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_332: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_129, torch.float32)
        var_mean_37 = torch.ops.aten.var_mean.correction(convert_element_type_332, [2], correction = 0, keepdim = True)
        getitem_302: "f32[32, 257, 1]" = var_mean_37[0]
        getitem_303: "f32[32, 257, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_37: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_332, getitem_303);  convert_element_type_332 = getitem_303 = None
        add_130: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_302, 1e-06);  getitem_302 = None
        rsqrt_37: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        mul_128: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_37);  sub_37 = rsqrt_37 = None
        mul_129: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_128, arg227_1);  mul_128 = arg227_1 = None
        add_131: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_129, arg228_1);  mul_129 = arg228_1 = None
        convert_element_type_333: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_131, torch.float16);  add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_187: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_333, [8224, 1408]);  convert_element_type_333 = None
        permute_113: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_74: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg230_1, view_187, permute_113);  arg230_1 = view_187 = permute_113 = None
        view_188: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_74, [32, 257, 6144]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_337: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_188, torch.float32);  view_188 = None
        mul_130: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_337, 0.5)
        mul_131: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_337, 0.7071067811865476);  convert_element_type_337 = None
        erf_18: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_131);  mul_131 = None
        add_132: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_132: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_130, add_132);  mul_130 = add_132 = None
        convert_element_type_338: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_132, torch.float16);  mul_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_189: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_338, [8224, 6144]);  convert_element_type_338 = None
        permute_114: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_75: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg232_1, view_189, permute_114);  arg232_1 = view_189 = permute_114 = None
        view_190: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_75, [32, 257, 1408]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_133: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_129, view_190);  add_129 = view_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_342: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_133, torch.float32)
        var_mean_38 = torch.ops.aten.var_mean.correction(convert_element_type_342, [2], correction = 0, keepdim = True)
        getitem_304: "f32[32, 257, 1]" = var_mean_38[0]
        getitem_305: "f32[32, 257, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_38: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_342, getitem_305);  convert_element_type_342 = getitem_305 = None
        add_134: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_304, 1e-06);  getitem_304 = None
        rsqrt_38: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
        mul_133: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_38);  sub_38 = rsqrt_38 = None
        mul_134: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_133, arg233_1);  mul_133 = arg233_1 = None
        add_135: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_134, arg234_1);  mul_134 = arg234_1 = None
        convert_element_type_343: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_135, torch.float16);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_191: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_343, [8224, 1408]);  convert_element_type_343 = None
        permute_115: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_76: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg236_1, view_191, permute_115);  arg236_1 = view_191 = permute_115 = None
        view_192: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_76, [32, 257, 4224]);  addmm_76 = None
        view_193: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_192, [32, 257, 3, 16, 88]);  view_192 = None
        permute_116: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_193, [2, 0, 3, 1, 4]);  view_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_19 = torch.ops.aten.unbind.int(permute_116);  permute_116 = None
        getitem_306: "f16[32, 16, 257, 88]" = unbind_19[0]
        getitem_307: "f16[32, 16, 257, 88]" = unbind_19[1]
        getitem_308: "f16[32, 16, 257, 88]" = unbind_19[2];  unbind_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_19 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_306, getitem_307, getitem_308, None, False);  getitem_306 = getitem_307 = getitem_308 = None
        getitem_309: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_19[0];  _scaled_dot_product_cudnn_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_117: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_309, [0, 2, 1, 3]);  getitem_309 = None
        view_194: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_117, [32, 257, 1408]);  permute_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_195: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_194, [8224, 1408]);  view_194 = None
        permute_118: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_77: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg238_1, view_195, permute_118);  arg238_1 = view_195 = permute_118 = None
        view_196: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_77, [32, 257, 1408]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_136: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_133, view_196);  add_133 = view_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_350: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_136, torch.float32)
        var_mean_39 = torch.ops.aten.var_mean.correction(convert_element_type_350, [2], correction = 0, keepdim = True)
        getitem_318: "f32[32, 257, 1]" = var_mean_39[0]
        getitem_319: "f32[32, 257, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_39: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_350, getitem_319);  convert_element_type_350 = getitem_319 = None
        add_137: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_318, 1e-06);  getitem_318 = None
        rsqrt_39: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        mul_135: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_39);  sub_39 = rsqrt_39 = None
        mul_136: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_135, arg239_1);  mul_135 = arg239_1 = None
        add_138: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_136, arg240_1);  mul_136 = arg240_1 = None
        convert_element_type_351: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_138, torch.float16);  add_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_197: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_351, [8224, 1408]);  convert_element_type_351 = None
        permute_119: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
        addmm_78: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg242_1, view_197, permute_119);  arg242_1 = view_197 = permute_119 = None
        view_198: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_78, [32, 257, 6144]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_355: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_198, torch.float32);  view_198 = None
        mul_137: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.5)
        mul_138: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_355, 0.7071067811865476);  convert_element_type_355 = None
        erf_19: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_138);  mul_138 = None
        add_139: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_139: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_137, add_139);  mul_137 = add_139 = None
        convert_element_type_356: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_139, torch.float16);  mul_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_199: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_356, [8224, 6144]);  convert_element_type_356 = None
        permute_120: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_79: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg244_1, view_199, permute_120);  arg244_1 = view_199 = permute_120 = None
        view_200: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_79, [32, 257, 1408]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_140: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_136, view_200);  add_136 = view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_360: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_140, torch.float32)
        var_mean_40 = torch.ops.aten.var_mean.correction(convert_element_type_360, [2], correction = 0, keepdim = True)
        getitem_320: "f32[32, 257, 1]" = var_mean_40[0]
        getitem_321: "f32[32, 257, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_40: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_360, getitem_321);  convert_element_type_360 = getitem_321 = None
        add_141: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_320, 1e-06);  getitem_320 = None
        rsqrt_40: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        mul_140: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_40);  sub_40 = rsqrt_40 = None
        mul_141: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_140, arg245_1);  mul_140 = arg245_1 = None
        add_142: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_141, arg246_1);  mul_141 = arg246_1 = None
        convert_element_type_361: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_142, torch.float16);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_201: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_361, [8224, 1408]);  convert_element_type_361 = None
        permute_121: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_80: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg248_1, view_201, permute_121);  arg248_1 = view_201 = permute_121 = None
        view_202: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_80, [32, 257, 4224]);  addmm_80 = None
        view_203: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_202, [32, 257, 3, 16, 88]);  view_202 = None
        permute_122: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_203, [2, 0, 3, 1, 4]);  view_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_20 = torch.ops.aten.unbind.int(permute_122);  permute_122 = None
        getitem_322: "f16[32, 16, 257, 88]" = unbind_20[0]
        getitem_323: "f16[32, 16, 257, 88]" = unbind_20[1]
        getitem_324: "f16[32, 16, 257, 88]" = unbind_20[2];  unbind_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_20 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_322, getitem_323, getitem_324, None, False);  getitem_322 = getitem_323 = getitem_324 = None
        getitem_325: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_20[0];  _scaled_dot_product_cudnn_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_123: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_325, [0, 2, 1, 3]);  getitem_325 = None
        view_204: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_123, [32, 257, 1408]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_205: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_204, [8224, 1408]);  view_204 = None
        permute_124: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_81: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg250_1, view_205, permute_124);  arg250_1 = view_205 = permute_124 = None
        view_206: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_81, [32, 257, 1408]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_143: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_140, view_206);  add_140 = view_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_368: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_143, torch.float32)
        var_mean_41 = torch.ops.aten.var_mean.correction(convert_element_type_368, [2], correction = 0, keepdim = True)
        getitem_334: "f32[32, 257, 1]" = var_mean_41[0]
        getitem_335: "f32[32, 257, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_41: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_368, getitem_335);  convert_element_type_368 = getitem_335 = None
        add_144: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_334, 1e-06);  getitem_334 = None
        rsqrt_41: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_142: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_41);  sub_41 = rsqrt_41 = None
        mul_143: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_142, arg251_1);  mul_142 = arg251_1 = None
        add_145: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_143, arg252_1);  mul_143 = arg252_1 = None
        convert_element_type_369: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_145, torch.float16);  add_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_207: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_369, [8224, 1408]);  convert_element_type_369 = None
        permute_125: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_82: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg254_1, view_207, permute_125);  arg254_1 = view_207 = permute_125 = None
        view_208: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_82, [32, 257, 6144]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_373: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_208, torch.float32);  view_208 = None
        mul_144: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_373, 0.5)
        mul_145: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_373, 0.7071067811865476);  convert_element_type_373 = None
        erf_20: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_145);  mul_145 = None
        add_146: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_146: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_144, add_146);  mul_144 = add_146 = None
        convert_element_type_374: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_146, torch.float16);  mul_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_209: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_374, [8224, 6144]);  convert_element_type_374 = None
        permute_126: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_83: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg256_1, view_209, permute_126);  arg256_1 = view_209 = permute_126 = None
        view_210: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_83, [32, 257, 1408]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_147: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_143, view_210);  add_143 = view_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_378: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_147, torch.float32)
        var_mean_42 = torch.ops.aten.var_mean.correction(convert_element_type_378, [2], correction = 0, keepdim = True)
        getitem_336: "f32[32, 257, 1]" = var_mean_42[0]
        getitem_337: "f32[32, 257, 1]" = var_mean_42[1];  var_mean_42 = None
        sub_42: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_378, getitem_337);  convert_element_type_378 = getitem_337 = None
        add_148: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_336, 1e-06);  getitem_336 = None
        rsqrt_42: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_147: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_42);  sub_42 = rsqrt_42 = None
        mul_148: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_147, arg257_1);  mul_147 = arg257_1 = None
        add_149: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_148, arg258_1);  mul_148 = arg258_1 = None
        convert_element_type_379: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_149, torch.float16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_211: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_379, [8224, 1408]);  convert_element_type_379 = None
        permute_127: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_84: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg260_1, view_211, permute_127);  arg260_1 = view_211 = permute_127 = None
        view_212: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_84, [32, 257, 4224]);  addmm_84 = None
        view_213: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_212, [32, 257, 3, 16, 88]);  view_212 = None
        permute_128: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_213, [2, 0, 3, 1, 4]);  view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_21 = torch.ops.aten.unbind.int(permute_128);  permute_128 = None
        getitem_338: "f16[32, 16, 257, 88]" = unbind_21[0]
        getitem_339: "f16[32, 16, 257, 88]" = unbind_21[1]
        getitem_340: "f16[32, 16, 257, 88]" = unbind_21[2];  unbind_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_21 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_338, getitem_339, getitem_340, None, False);  getitem_338 = getitem_339 = getitem_340 = None
        getitem_341: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_21[0];  _scaled_dot_product_cudnn_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_129: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_341, [0, 2, 1, 3]);  getitem_341 = None
        view_214: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_129, [32, 257, 1408]);  permute_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_215: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_214, [8224, 1408]);  view_214 = None
        permute_130: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_85: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg262_1, view_215, permute_130);  arg262_1 = view_215 = permute_130 = None
        view_216: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_85, [32, 257, 1408]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_150: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_147, view_216);  add_147 = view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_386: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_150, torch.float32)
        var_mean_43 = torch.ops.aten.var_mean.correction(convert_element_type_386, [2], correction = 0, keepdim = True)
        getitem_350: "f32[32, 257, 1]" = var_mean_43[0]
        getitem_351: "f32[32, 257, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_43: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_386, getitem_351);  convert_element_type_386 = getitem_351 = None
        add_151: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_350, 1e-06);  getitem_350 = None
        rsqrt_43: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_149: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_43);  sub_43 = rsqrt_43 = None
        mul_150: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_149, arg263_1);  mul_149 = arg263_1 = None
        add_152: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_150, arg264_1);  mul_150 = arg264_1 = None
        convert_element_type_387: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_152, torch.float16);  add_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_217: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_387, [8224, 1408]);  convert_element_type_387 = None
        permute_131: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_86: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg266_1, view_217, permute_131);  arg266_1 = view_217 = permute_131 = None
        view_218: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_86, [32, 257, 6144]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_391: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_218, torch.float32);  view_218 = None
        mul_151: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_391, 0.5)
        mul_152: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_391, 0.7071067811865476);  convert_element_type_391 = None
        erf_21: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_152);  mul_152 = None
        add_153: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_153: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_151, add_153);  mul_151 = add_153 = None
        convert_element_type_392: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_153, torch.float16);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_219: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_392, [8224, 6144]);  convert_element_type_392 = None
        permute_132: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
        addmm_87: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg268_1, view_219, permute_132);  arg268_1 = view_219 = permute_132 = None
        view_220: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_87, [32, 257, 1408]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_154: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_150, view_220);  add_150 = view_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_396: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_154, torch.float32)
        var_mean_44 = torch.ops.aten.var_mean.correction(convert_element_type_396, [2], correction = 0, keepdim = True)
        getitem_352: "f32[32, 257, 1]" = var_mean_44[0]
        getitem_353: "f32[32, 257, 1]" = var_mean_44[1];  var_mean_44 = None
        sub_44: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_396, getitem_353);  convert_element_type_396 = getitem_353 = None
        add_155: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_352, 1e-06);  getitem_352 = None
        rsqrt_44: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_154: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_44);  sub_44 = rsqrt_44 = None
        mul_155: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_154, arg269_1);  mul_154 = arg269_1 = None
        add_156: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_155, arg270_1);  mul_155 = arg270_1 = None
        convert_element_type_397: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_156, torch.float16);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_221: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_397, [8224, 1408]);  convert_element_type_397 = None
        permute_133: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_88: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg272_1, view_221, permute_133);  arg272_1 = view_221 = permute_133 = None
        view_222: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_88, [32, 257, 4224]);  addmm_88 = None
        view_223: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_222, [32, 257, 3, 16, 88]);  view_222 = None
        permute_134: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_223, [2, 0, 3, 1, 4]);  view_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_22 = torch.ops.aten.unbind.int(permute_134);  permute_134 = None
        getitem_354: "f16[32, 16, 257, 88]" = unbind_22[0]
        getitem_355: "f16[32, 16, 257, 88]" = unbind_22[1]
        getitem_356: "f16[32, 16, 257, 88]" = unbind_22[2];  unbind_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_22 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_354, getitem_355, getitem_356, None, False);  getitem_354 = getitem_355 = getitem_356 = None
        getitem_357: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_22[0];  _scaled_dot_product_cudnn_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_135: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_357, [0, 2, 1, 3]);  getitem_357 = None
        view_224: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_135, [32, 257, 1408]);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_225: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_224, [8224, 1408]);  view_224 = None
        permute_136: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
        addmm_89: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg274_1, view_225, permute_136);  arg274_1 = view_225 = permute_136 = None
        view_226: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_89, [32, 257, 1408]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_157: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_154, view_226);  add_154 = view_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_404: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_157, torch.float32)
        var_mean_45 = torch.ops.aten.var_mean.correction(convert_element_type_404, [2], correction = 0, keepdim = True)
        getitem_366: "f32[32, 257, 1]" = var_mean_45[0]
        getitem_367: "f32[32, 257, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_45: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_404, getitem_367);  convert_element_type_404 = getitem_367 = None
        add_158: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_366, 1e-06);  getitem_366 = None
        rsqrt_45: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
        mul_156: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_45);  sub_45 = rsqrt_45 = None
        mul_157: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_156, arg275_1);  mul_156 = arg275_1 = None
        add_159: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_157, arg276_1);  mul_157 = arg276_1 = None
        convert_element_type_405: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_159, torch.float16);  add_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_227: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_405, [8224, 1408]);  convert_element_type_405 = None
        permute_137: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_90: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg278_1, view_227, permute_137);  arg278_1 = view_227 = permute_137 = None
        view_228: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_90, [32, 257, 6144]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_409: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_228, torch.float32);  view_228 = None
        mul_158: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_409, 0.5)
        mul_159: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_409, 0.7071067811865476);  convert_element_type_409 = None
        erf_22: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
        add_160: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_160: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_158, add_160);  mul_158 = add_160 = None
        convert_element_type_410: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_160, torch.float16);  mul_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_229: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_410, [8224, 6144]);  convert_element_type_410 = None
        permute_138: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_91: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg280_1, view_229, permute_138);  arg280_1 = view_229 = permute_138 = None
        view_230: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_91, [32, 257, 1408]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_161: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_157, view_230);  add_157 = view_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_414: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_161, torch.float32)
        var_mean_46 = torch.ops.aten.var_mean.correction(convert_element_type_414, [2], correction = 0, keepdim = True)
        getitem_368: "f32[32, 257, 1]" = var_mean_46[0]
        getitem_369: "f32[32, 257, 1]" = var_mean_46[1];  var_mean_46 = None
        sub_46: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_414, getitem_369);  convert_element_type_414 = getitem_369 = None
        add_162: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_368, 1e-06);  getitem_368 = None
        rsqrt_46: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_161: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_46);  sub_46 = rsqrt_46 = None
        mul_162: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_161, arg281_1);  mul_161 = arg281_1 = None
        add_163: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_162, arg282_1);  mul_162 = arg282_1 = None
        convert_element_type_415: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_163, torch.float16);  add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_231: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_415, [8224, 1408]);  convert_element_type_415 = None
        permute_139: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
        addmm_92: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg284_1, view_231, permute_139);  arg284_1 = view_231 = permute_139 = None
        view_232: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_92, [32, 257, 4224]);  addmm_92 = None
        view_233: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_232, [32, 257, 3, 16, 88]);  view_232 = None
        permute_140: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_233, [2, 0, 3, 1, 4]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_23 = torch.ops.aten.unbind.int(permute_140);  permute_140 = None
        getitem_370: "f16[32, 16, 257, 88]" = unbind_23[0]
        getitem_371: "f16[32, 16, 257, 88]" = unbind_23[1]
        getitem_372: "f16[32, 16, 257, 88]" = unbind_23[2];  unbind_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_23 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_370, getitem_371, getitem_372, None, False);  getitem_370 = getitem_371 = getitem_372 = None
        getitem_373: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_23[0];  _scaled_dot_product_cudnn_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_141: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_373, [0, 2, 1, 3]);  getitem_373 = None
        view_234: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_141, [32, 257, 1408]);  permute_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_235: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_234, [8224, 1408]);  view_234 = None
        permute_142: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_93: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg286_1, view_235, permute_142);  arg286_1 = view_235 = permute_142 = None
        view_236: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_93, [32, 257, 1408]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_164: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_161, view_236);  add_161 = view_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_422: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_164, torch.float32)
        var_mean_47 = torch.ops.aten.var_mean.correction(convert_element_type_422, [2], correction = 0, keepdim = True)
        getitem_382: "f32[32, 257, 1]" = var_mean_47[0]
        getitem_383: "f32[32, 257, 1]" = var_mean_47[1];  var_mean_47 = None
        sub_47: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_422, getitem_383);  convert_element_type_422 = getitem_383 = None
        add_165: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_382, 1e-06);  getitem_382 = None
        rsqrt_47: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
        mul_163: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_47);  sub_47 = rsqrt_47 = None
        mul_164: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_163, arg287_1);  mul_163 = arg287_1 = None
        add_166: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_164, arg288_1);  mul_164 = arg288_1 = None
        convert_element_type_423: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_166, torch.float16);  add_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_237: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_423, [8224, 1408]);  convert_element_type_423 = None
        permute_143: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_94: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg290_1, view_237, permute_143);  arg290_1 = view_237 = permute_143 = None
        view_238: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_94, [32, 257, 6144]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_427: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_238, torch.float32);  view_238 = None
        mul_165: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_427, 0.5)
        mul_166: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_427, 0.7071067811865476);  convert_element_type_427 = None
        erf_23: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_166);  mul_166 = None
        add_167: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_167: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_165, add_167);  mul_165 = add_167 = None
        convert_element_type_428: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_167, torch.float16);  mul_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_239: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_428, [8224, 6144]);  convert_element_type_428 = None
        permute_144: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_95: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg292_1, view_239, permute_144);  arg292_1 = view_239 = permute_144 = None
        view_240: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_95, [32, 257, 1408]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_168: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_164, view_240);  add_164 = view_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_432: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_168, torch.float32)
        var_mean_48 = torch.ops.aten.var_mean.correction(convert_element_type_432, [2], correction = 0, keepdim = True)
        getitem_384: "f32[32, 257, 1]" = var_mean_48[0]
        getitem_385: "f32[32, 257, 1]" = var_mean_48[1];  var_mean_48 = None
        sub_48: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_432, getitem_385);  convert_element_type_432 = getitem_385 = None
        add_169: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_384, 1e-06);  getitem_384 = None
        rsqrt_48: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
        mul_168: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_48);  sub_48 = rsqrt_48 = None
        mul_169: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_168, arg293_1);  mul_168 = arg293_1 = None
        add_170: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_169, arg294_1);  mul_169 = arg294_1 = None
        convert_element_type_433: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_170, torch.float16);  add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_241: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_433, [8224, 1408]);  convert_element_type_433 = None
        permute_145: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_96: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg296_1, view_241, permute_145);  arg296_1 = view_241 = permute_145 = None
        view_242: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_96, [32, 257, 4224]);  addmm_96 = None
        view_243: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_242, [32, 257, 3, 16, 88]);  view_242 = None
        permute_146: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_243, [2, 0, 3, 1, 4]);  view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_24 = torch.ops.aten.unbind.int(permute_146);  permute_146 = None
        getitem_386: "f16[32, 16, 257, 88]" = unbind_24[0]
        getitem_387: "f16[32, 16, 257, 88]" = unbind_24[1]
        getitem_388: "f16[32, 16, 257, 88]" = unbind_24[2];  unbind_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_24 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_386, getitem_387, getitem_388, None, False);  getitem_386 = getitem_387 = getitem_388 = None
        getitem_389: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_24[0];  _scaled_dot_product_cudnn_attention_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_147: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_389, [0, 2, 1, 3]);  getitem_389 = None
        view_244: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_147, [32, 257, 1408]);  permute_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_245: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_244, [8224, 1408]);  view_244 = None
        permute_148: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_97: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg298_1, view_245, permute_148);  arg298_1 = view_245 = permute_148 = None
        view_246: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_97, [32, 257, 1408]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_171: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_168, view_246);  add_168 = view_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_440: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_171, torch.float32)
        var_mean_49 = torch.ops.aten.var_mean.correction(convert_element_type_440, [2], correction = 0, keepdim = True)
        getitem_398: "f32[32, 257, 1]" = var_mean_49[0]
        getitem_399: "f32[32, 257, 1]" = var_mean_49[1];  var_mean_49 = None
        sub_49: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_440, getitem_399);  convert_element_type_440 = getitem_399 = None
        add_172: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_398, 1e-06);  getitem_398 = None
        rsqrt_49: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
        mul_170: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_49);  sub_49 = rsqrt_49 = None
        mul_171: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_170, arg299_1);  mul_170 = arg299_1 = None
        add_173: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_171, arg300_1);  mul_171 = arg300_1 = None
        convert_element_type_441: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_173, torch.float16);  add_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_247: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_441, [8224, 1408]);  convert_element_type_441 = None
        permute_149: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        addmm_98: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg302_1, view_247, permute_149);  arg302_1 = view_247 = permute_149 = None
        view_248: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_98, [32, 257, 6144]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_445: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_248, torch.float32);  view_248 = None
        mul_172: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_445, 0.5)
        mul_173: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_445, 0.7071067811865476);  convert_element_type_445 = None
        erf_24: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_173);  mul_173 = None
        add_174: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_174: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_172, add_174);  mul_172 = add_174 = None
        convert_element_type_446: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_174, torch.float16);  mul_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_249: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_446, [8224, 6144]);  convert_element_type_446 = None
        permute_150: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_99: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg304_1, view_249, permute_150);  arg304_1 = view_249 = permute_150 = None
        view_250: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_99, [32, 257, 1408]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_175: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_171, view_250);  add_171 = view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_450: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_175, torch.float32)
        var_mean_50 = torch.ops.aten.var_mean.correction(convert_element_type_450, [2], correction = 0, keepdim = True)
        getitem_400: "f32[32, 257, 1]" = var_mean_50[0]
        getitem_401: "f32[32, 257, 1]" = var_mean_50[1];  var_mean_50 = None
        sub_50: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_450, getitem_401);  convert_element_type_450 = getitem_401 = None
        add_176: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_400, 1e-06);  getitem_400 = None
        rsqrt_50: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
        mul_175: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_50);  sub_50 = rsqrt_50 = None
        mul_176: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_175, arg305_1);  mul_175 = arg305_1 = None
        add_177: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_176, arg306_1);  mul_176 = arg306_1 = None
        convert_element_type_451: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_177, torch.float16);  add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_251: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_451, [8224, 1408]);  convert_element_type_451 = None
        permute_151: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_100: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg308_1, view_251, permute_151);  arg308_1 = view_251 = permute_151 = None
        view_252: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_100, [32, 257, 4224]);  addmm_100 = None
        view_253: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_252, [32, 257, 3, 16, 88]);  view_252 = None
        permute_152: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_253, [2, 0, 3, 1, 4]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_25 = torch.ops.aten.unbind.int(permute_152);  permute_152 = None
        getitem_402: "f16[32, 16, 257, 88]" = unbind_25[0]
        getitem_403: "f16[32, 16, 257, 88]" = unbind_25[1]
        getitem_404: "f16[32, 16, 257, 88]" = unbind_25[2];  unbind_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_25 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_402, getitem_403, getitem_404, None, False);  getitem_402 = getitem_403 = getitem_404 = None
        getitem_405: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_25[0];  _scaled_dot_product_cudnn_attention_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_153: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_405, [0, 2, 1, 3]);  getitem_405 = None
        view_254: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_153, [32, 257, 1408]);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_255: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_254, [8224, 1408]);  view_254 = None
        permute_154: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_101: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg310_1, view_255, permute_154);  arg310_1 = view_255 = permute_154 = None
        view_256: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_101, [32, 257, 1408]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_178: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_175, view_256);  add_175 = view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_458: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_178, torch.float32)
        var_mean_51 = torch.ops.aten.var_mean.correction(convert_element_type_458, [2], correction = 0, keepdim = True)
        getitem_414: "f32[32, 257, 1]" = var_mean_51[0]
        getitem_415: "f32[32, 257, 1]" = var_mean_51[1];  var_mean_51 = None
        sub_51: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_458, getitem_415);  convert_element_type_458 = getitem_415 = None
        add_179: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_414, 1e-06);  getitem_414 = None
        rsqrt_51: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
        mul_177: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_51);  sub_51 = rsqrt_51 = None
        mul_178: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_177, arg311_1);  mul_177 = arg311_1 = None
        add_180: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_178, arg312_1);  mul_178 = arg312_1 = None
        convert_element_type_459: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_180, torch.float16);  add_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_257: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_459, [8224, 1408]);  convert_element_type_459 = None
        permute_155: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_102: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg314_1, view_257, permute_155);  arg314_1 = view_257 = permute_155 = None
        view_258: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_102, [32, 257, 6144]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_463: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_258, torch.float32);  view_258 = None
        mul_179: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_463, 0.5)
        mul_180: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_463, 0.7071067811865476);  convert_element_type_463 = None
        erf_25: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
        add_181: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_181: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_179, add_181);  mul_179 = add_181 = None
        convert_element_type_464: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_181, torch.float16);  mul_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_259: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_464, [8224, 6144]);  convert_element_type_464 = None
        permute_156: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_103: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg316_1, view_259, permute_156);  arg316_1 = view_259 = permute_156 = None
        view_260: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_103, [32, 257, 1408]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_182: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_178, view_260);  add_178 = view_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_468: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_182, torch.float32)
        var_mean_52 = torch.ops.aten.var_mean.correction(convert_element_type_468, [2], correction = 0, keepdim = True)
        getitem_416: "f32[32, 257, 1]" = var_mean_52[0]
        getitem_417: "f32[32, 257, 1]" = var_mean_52[1];  var_mean_52 = None
        sub_52: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_468, getitem_417);  convert_element_type_468 = getitem_417 = None
        add_183: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_416, 1e-06);  getitem_416 = None
        rsqrt_52: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
        mul_182: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_52);  sub_52 = rsqrt_52 = None
        mul_183: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_182, arg317_1);  mul_182 = arg317_1 = None
        add_184: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_183, arg318_1);  mul_183 = arg318_1 = None
        convert_element_type_469: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_184, torch.float16);  add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_261: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_469, [8224, 1408]);  convert_element_type_469 = None
        permute_157: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
        addmm_104: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg320_1, view_261, permute_157);  arg320_1 = view_261 = permute_157 = None
        view_262: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_104, [32, 257, 4224]);  addmm_104 = None
        view_263: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_262, [32, 257, 3, 16, 88]);  view_262 = None
        permute_158: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_263, [2, 0, 3, 1, 4]);  view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_26 = torch.ops.aten.unbind.int(permute_158);  permute_158 = None
        getitem_418: "f16[32, 16, 257, 88]" = unbind_26[0]
        getitem_419: "f16[32, 16, 257, 88]" = unbind_26[1]
        getitem_420: "f16[32, 16, 257, 88]" = unbind_26[2];  unbind_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_26 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_418, getitem_419, getitem_420, None, False);  getitem_418 = getitem_419 = getitem_420 = None
        getitem_421: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_26[0];  _scaled_dot_product_cudnn_attention_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_159: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_421, [0, 2, 1, 3]);  getitem_421 = None
        view_264: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_159, [32, 257, 1408]);  permute_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_265: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_264, [8224, 1408]);  view_264 = None
        permute_160: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_105: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg322_1, view_265, permute_160);  arg322_1 = view_265 = permute_160 = None
        view_266: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_105, [32, 257, 1408]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_185: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_182, view_266);  add_182 = view_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_476: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_185, torch.float32)
        var_mean_53 = torch.ops.aten.var_mean.correction(convert_element_type_476, [2], correction = 0, keepdim = True)
        getitem_430: "f32[32, 257, 1]" = var_mean_53[0]
        getitem_431: "f32[32, 257, 1]" = var_mean_53[1];  var_mean_53 = None
        sub_53: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_476, getitem_431);  convert_element_type_476 = getitem_431 = None
        add_186: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_430, 1e-06);  getitem_430 = None
        rsqrt_53: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_186);  add_186 = None
        mul_184: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_53);  sub_53 = rsqrt_53 = None
        mul_185: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_184, arg323_1);  mul_184 = arg323_1 = None
        add_187: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_185, arg324_1);  mul_185 = arg324_1 = None
        convert_element_type_477: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_187, torch.float16);  add_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_267: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_477, [8224, 1408]);  convert_element_type_477 = None
        permute_161: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
        addmm_106: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg326_1, view_267, permute_161);  arg326_1 = view_267 = permute_161 = None
        view_268: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_106, [32, 257, 6144]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_481: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_268, torch.float32);  view_268 = None
        mul_186: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_481, 0.5)
        mul_187: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_481, 0.7071067811865476);  convert_element_type_481 = None
        erf_26: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_187);  mul_187 = None
        add_188: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
        mul_188: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_186, add_188);  mul_186 = add_188 = None
        convert_element_type_482: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_188, torch.float16);  mul_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_269: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_482, [8224, 6144]);  convert_element_type_482 = None
        permute_162: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_107: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg328_1, view_269, permute_162);  arg328_1 = view_269 = permute_162 = None
        view_270: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_107, [32, 257, 1408]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_189: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_185, view_270);  add_185 = view_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_486: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_189, torch.float32)
        var_mean_54 = torch.ops.aten.var_mean.correction(convert_element_type_486, [2], correction = 0, keepdim = True)
        getitem_432: "f32[32, 257, 1]" = var_mean_54[0]
        getitem_433: "f32[32, 257, 1]" = var_mean_54[1];  var_mean_54 = None
        sub_54: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_486, getitem_433);  convert_element_type_486 = getitem_433 = None
        add_190: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_432, 1e-06);  getitem_432 = None
        rsqrt_54: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
        mul_189: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_54);  sub_54 = rsqrt_54 = None
        mul_190: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_189, arg329_1);  mul_189 = arg329_1 = None
        add_191: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_190, arg330_1);  mul_190 = arg330_1 = None
        convert_element_type_487: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_191, torch.float16);  add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_271: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_487, [8224, 1408]);  convert_element_type_487 = None
        permute_163: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_108: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg332_1, view_271, permute_163);  arg332_1 = view_271 = permute_163 = None
        view_272: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_108, [32, 257, 4224]);  addmm_108 = None
        view_273: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_272, [32, 257, 3, 16, 88]);  view_272 = None
        permute_164: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_273, [2, 0, 3, 1, 4]);  view_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_27 = torch.ops.aten.unbind.int(permute_164);  permute_164 = None
        getitem_434: "f16[32, 16, 257, 88]" = unbind_27[0]
        getitem_435: "f16[32, 16, 257, 88]" = unbind_27[1]
        getitem_436: "f16[32, 16, 257, 88]" = unbind_27[2];  unbind_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_27 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_434, getitem_435, getitem_436, None, False);  getitem_434 = getitem_435 = getitem_436 = None
        getitem_437: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_27[0];  _scaled_dot_product_cudnn_attention_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_165: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_437, [0, 2, 1, 3]);  getitem_437 = None
        view_274: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_165, [32, 257, 1408]);  permute_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_275: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_274, [8224, 1408]);  view_274 = None
        permute_166: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_109: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg334_1, view_275, permute_166);  arg334_1 = view_275 = permute_166 = None
        view_276: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_109, [32, 257, 1408]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_192: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_189, view_276);  add_189 = view_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_494: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_192, torch.float32)
        var_mean_55 = torch.ops.aten.var_mean.correction(convert_element_type_494, [2], correction = 0, keepdim = True)
        getitem_446: "f32[32, 257, 1]" = var_mean_55[0]
        getitem_447: "f32[32, 257, 1]" = var_mean_55[1];  var_mean_55 = None
        sub_55: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_494, getitem_447);  convert_element_type_494 = getitem_447 = None
        add_193: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_446, 1e-06);  getitem_446 = None
        rsqrt_55: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
        mul_191: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_55);  sub_55 = rsqrt_55 = None
        mul_192: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_191, arg335_1);  mul_191 = arg335_1 = None
        add_194: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_192, arg336_1);  mul_192 = arg336_1 = None
        convert_element_type_495: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_194, torch.float16);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_277: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_495, [8224, 1408]);  convert_element_type_495 = None
        permute_167: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_110: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg338_1, view_277, permute_167);  arg338_1 = view_277 = permute_167 = None
        view_278: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_110, [32, 257, 6144]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_499: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_278, torch.float32);  view_278 = None
        mul_193: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_499, 0.5)
        mul_194: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_499, 0.7071067811865476);  convert_element_type_499 = None
        erf_27: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_194);  mul_194 = None
        add_195: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
        mul_195: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_193, add_195);  mul_193 = add_195 = None
        convert_element_type_500: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_195, torch.float16);  mul_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_279: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_500, [8224, 6144]);  convert_element_type_500 = None
        permute_168: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_111: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg340_1, view_279, permute_168);  arg340_1 = view_279 = permute_168 = None
        view_280: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_111, [32, 257, 1408]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_196: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_192, view_280);  add_192 = view_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_504: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_196, torch.float32)
        var_mean_56 = torch.ops.aten.var_mean.correction(convert_element_type_504, [2], correction = 0, keepdim = True)
        getitem_448: "f32[32, 257, 1]" = var_mean_56[0]
        getitem_449: "f32[32, 257, 1]" = var_mean_56[1];  var_mean_56 = None
        sub_56: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_504, getitem_449);  convert_element_type_504 = getitem_449 = None
        add_197: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_448, 1e-06);  getitem_448 = None
        rsqrt_56: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
        mul_196: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_56);  sub_56 = rsqrt_56 = None
        mul_197: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_196, arg341_1);  mul_196 = arg341_1 = None
        add_198: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_197, arg342_1);  mul_197 = arg342_1 = None
        convert_element_type_505: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_198, torch.float16);  add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_281: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_505, [8224, 1408]);  convert_element_type_505 = None
        permute_169: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_112: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg344_1, view_281, permute_169);  arg344_1 = view_281 = permute_169 = None
        view_282: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_112, [32, 257, 4224]);  addmm_112 = None
        view_283: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_282, [32, 257, 3, 16, 88]);  view_282 = None
        permute_170: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_283, [2, 0, 3, 1, 4]);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_28 = torch.ops.aten.unbind.int(permute_170);  permute_170 = None
        getitem_450: "f16[32, 16, 257, 88]" = unbind_28[0]
        getitem_451: "f16[32, 16, 257, 88]" = unbind_28[1]
        getitem_452: "f16[32, 16, 257, 88]" = unbind_28[2];  unbind_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_28 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_450, getitem_451, getitem_452, None, False);  getitem_450 = getitem_451 = getitem_452 = None
        getitem_453: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_28[0];  _scaled_dot_product_cudnn_attention_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_171: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_453, [0, 2, 1, 3]);  getitem_453 = None
        view_284: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_171, [32, 257, 1408]);  permute_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_285: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_284, [8224, 1408]);  view_284 = None
        permute_172: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
        addmm_113: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg346_1, view_285, permute_172);  arg346_1 = view_285 = permute_172 = None
        view_286: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_113, [32, 257, 1408]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_199: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_196, view_286);  add_196 = view_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_512: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_199, torch.float32)
        var_mean_57 = torch.ops.aten.var_mean.correction(convert_element_type_512, [2], correction = 0, keepdim = True)
        getitem_462: "f32[32, 257, 1]" = var_mean_57[0]
        getitem_463: "f32[32, 257, 1]" = var_mean_57[1];  var_mean_57 = None
        sub_57: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_512, getitem_463);  convert_element_type_512 = getitem_463 = None
        add_200: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_462, 1e-06);  getitem_462 = None
        rsqrt_57: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
        mul_198: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_57);  sub_57 = rsqrt_57 = None
        mul_199: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_198, arg347_1);  mul_198 = arg347_1 = None
        add_201: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_199, arg348_1);  mul_199 = arg348_1 = None
        convert_element_type_513: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_201, torch.float16);  add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_287: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_513, [8224, 1408]);  convert_element_type_513 = None
        permute_173: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_114: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg350_1, view_287, permute_173);  arg350_1 = view_287 = permute_173 = None
        view_288: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_114, [32, 257, 6144]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_517: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_288, torch.float32);  view_288 = None
        mul_200: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_517, 0.5)
        mul_201: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_517, 0.7071067811865476);  convert_element_type_517 = None
        erf_28: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
        add_202: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
        mul_202: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_200, add_202);  mul_200 = add_202 = None
        convert_element_type_518: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_202, torch.float16);  mul_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_289: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_518, [8224, 6144]);  convert_element_type_518 = None
        permute_174: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_115: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg352_1, view_289, permute_174);  arg352_1 = view_289 = permute_174 = None
        view_290: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_115, [32, 257, 1408]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_203: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_199, view_290);  add_199 = view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_522: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_203, torch.float32)
        var_mean_58 = torch.ops.aten.var_mean.correction(convert_element_type_522, [2], correction = 0, keepdim = True)
        getitem_464: "f32[32, 257, 1]" = var_mean_58[0]
        getitem_465: "f32[32, 257, 1]" = var_mean_58[1];  var_mean_58 = None
        sub_58: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_522, getitem_465);  convert_element_type_522 = getitem_465 = None
        add_204: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_464, 1e-06);  getitem_464 = None
        rsqrt_58: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
        mul_203: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_58);  sub_58 = rsqrt_58 = None
        mul_204: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_203, arg353_1);  mul_203 = arg353_1 = None
        add_205: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_204, arg354_1);  mul_204 = arg354_1 = None
        convert_element_type_523: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_205, torch.float16);  add_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_291: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_523, [8224, 1408]);  convert_element_type_523 = None
        permute_175: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
        addmm_116: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg356_1, view_291, permute_175);  arg356_1 = view_291 = permute_175 = None
        view_292: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_116, [32, 257, 4224]);  addmm_116 = None
        view_293: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_292, [32, 257, 3, 16, 88]);  view_292 = None
        permute_176: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_293, [2, 0, 3, 1, 4]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_29 = torch.ops.aten.unbind.int(permute_176);  permute_176 = None
        getitem_466: "f16[32, 16, 257, 88]" = unbind_29[0]
        getitem_467: "f16[32, 16, 257, 88]" = unbind_29[1]
        getitem_468: "f16[32, 16, 257, 88]" = unbind_29[2];  unbind_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_29 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_466, getitem_467, getitem_468, None, False);  getitem_466 = getitem_467 = getitem_468 = None
        getitem_469: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_29[0];  _scaled_dot_product_cudnn_attention_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_177: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_469, [0, 2, 1, 3]);  getitem_469 = None
        view_294: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_177, [32, 257, 1408]);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_295: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_294, [8224, 1408]);  view_294 = None
        permute_178: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_117: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg358_1, view_295, permute_178);  arg358_1 = view_295 = permute_178 = None
        view_296: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_117, [32, 257, 1408]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_206: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_203, view_296);  add_203 = view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_530: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_206, torch.float32)
        var_mean_59 = torch.ops.aten.var_mean.correction(convert_element_type_530, [2], correction = 0, keepdim = True)
        getitem_478: "f32[32, 257, 1]" = var_mean_59[0]
        getitem_479: "f32[32, 257, 1]" = var_mean_59[1];  var_mean_59 = None
        sub_59: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_530, getitem_479);  convert_element_type_530 = getitem_479 = None
        add_207: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_478, 1e-06);  getitem_478 = None
        rsqrt_59: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        mul_205: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_59);  sub_59 = rsqrt_59 = None
        mul_206: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_205, arg359_1);  mul_205 = arg359_1 = None
        add_208: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_206, arg360_1);  mul_206 = arg360_1 = None
        convert_element_type_531: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_208, torch.float16);  add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_297: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_531, [8224, 1408]);  convert_element_type_531 = None
        permute_179: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_118: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg362_1, view_297, permute_179);  arg362_1 = view_297 = permute_179 = None
        view_298: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_118, [32, 257, 6144]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_535: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_298, torch.float32);  view_298 = None
        mul_207: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_535, 0.5)
        mul_208: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_535, 0.7071067811865476);  convert_element_type_535 = None
        erf_29: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
        add_209: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
        mul_209: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_207, add_209);  mul_207 = add_209 = None
        convert_element_type_536: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_209, torch.float16);  mul_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_299: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_536, [8224, 6144]);  convert_element_type_536 = None
        permute_180: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_119: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg364_1, view_299, permute_180);  arg364_1 = view_299 = permute_180 = None
        view_300: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_119, [32, 257, 1408]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_210: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_206, view_300);  add_206 = view_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_540: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_210, torch.float32)
        var_mean_60 = torch.ops.aten.var_mean.correction(convert_element_type_540, [2], correction = 0, keepdim = True)
        getitem_480: "f32[32, 257, 1]" = var_mean_60[0]
        getitem_481: "f32[32, 257, 1]" = var_mean_60[1];  var_mean_60 = None
        sub_60: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_540, getitem_481);  convert_element_type_540 = getitem_481 = None
        add_211: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_480, 1e-06);  getitem_480 = None
        rsqrt_60: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
        mul_210: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_60);  sub_60 = rsqrt_60 = None
        mul_211: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_210, arg365_1);  mul_210 = arg365_1 = None
        add_212: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_211, arg366_1);  mul_211 = arg366_1 = None
        convert_element_type_541: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_212, torch.float16);  add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_301: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_541, [8224, 1408]);  convert_element_type_541 = None
        permute_181: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_120: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg368_1, view_301, permute_181);  arg368_1 = view_301 = permute_181 = None
        view_302: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_120, [32, 257, 4224]);  addmm_120 = None
        view_303: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_302, [32, 257, 3, 16, 88]);  view_302 = None
        permute_182: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_303, [2, 0, 3, 1, 4]);  view_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_30 = torch.ops.aten.unbind.int(permute_182);  permute_182 = None
        getitem_482: "f16[32, 16, 257, 88]" = unbind_30[0]
        getitem_483: "f16[32, 16, 257, 88]" = unbind_30[1]
        getitem_484: "f16[32, 16, 257, 88]" = unbind_30[2];  unbind_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_30 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_482, getitem_483, getitem_484, None, False);  getitem_482 = getitem_483 = getitem_484 = None
        getitem_485: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_30[0];  _scaled_dot_product_cudnn_attention_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_183: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_485, [0, 2, 1, 3]);  getitem_485 = None
        view_304: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_183, [32, 257, 1408]);  permute_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_305: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_304, [8224, 1408]);  view_304 = None
        permute_184: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_121: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg370_1, view_305, permute_184);  arg370_1 = view_305 = permute_184 = None
        view_306: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_121, [32, 257, 1408]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_213: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_210, view_306);  add_210 = view_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_548: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_213, torch.float32)
        var_mean_61 = torch.ops.aten.var_mean.correction(convert_element_type_548, [2], correction = 0, keepdim = True)
        getitem_494: "f32[32, 257, 1]" = var_mean_61[0]
        getitem_495: "f32[32, 257, 1]" = var_mean_61[1];  var_mean_61 = None
        sub_61: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_548, getitem_495);  convert_element_type_548 = getitem_495 = None
        add_214: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_494, 1e-06);  getitem_494 = None
        rsqrt_61: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        mul_212: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_61);  sub_61 = rsqrt_61 = None
        mul_213: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_212, arg371_1);  mul_212 = arg371_1 = None
        add_215: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_213, arg372_1);  mul_213 = arg372_1 = None
        convert_element_type_549: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_215, torch.float16);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_307: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_549, [8224, 1408]);  convert_element_type_549 = None
        permute_185: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_122: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg374_1, view_307, permute_185);  arg374_1 = view_307 = permute_185 = None
        view_308: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_122, [32, 257, 6144]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_553: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_308, torch.float32);  view_308 = None
        mul_214: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_553, 0.5)
        mul_215: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_553, 0.7071067811865476);  convert_element_type_553 = None
        erf_30: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_215);  mul_215 = None
        add_216: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
        mul_216: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_214, add_216);  mul_214 = add_216 = None
        convert_element_type_554: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_216, torch.float16);  mul_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_309: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_554, [8224, 6144]);  convert_element_type_554 = None
        permute_186: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
        addmm_123: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg376_1, view_309, permute_186);  arg376_1 = view_309 = permute_186 = None
        view_310: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_123, [32, 257, 1408]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_217: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_213, view_310);  add_213 = view_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_558: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_217, torch.float32)
        var_mean_62 = torch.ops.aten.var_mean.correction(convert_element_type_558, [2], correction = 0, keepdim = True)
        getitem_496: "f32[32, 257, 1]" = var_mean_62[0]
        getitem_497: "f32[32, 257, 1]" = var_mean_62[1];  var_mean_62 = None
        sub_62: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_558, getitem_497);  convert_element_type_558 = getitem_497 = None
        add_218: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_496, 1e-06);  getitem_496 = None
        rsqrt_62: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        mul_217: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_62);  sub_62 = rsqrt_62 = None
        mul_218: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_217, arg377_1);  mul_217 = arg377_1 = None
        add_219: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_218, arg378_1);  mul_218 = arg378_1 = None
        convert_element_type_559: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_219, torch.float16);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_311: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_559, [8224, 1408]);  convert_element_type_559 = None
        permute_187: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_124: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg380_1, view_311, permute_187);  arg380_1 = view_311 = permute_187 = None
        view_312: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_124, [32, 257, 4224]);  addmm_124 = None
        view_313: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_312, [32, 257, 3, 16, 88]);  view_312 = None
        permute_188: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_313, [2, 0, 3, 1, 4]);  view_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_31 = torch.ops.aten.unbind.int(permute_188);  permute_188 = None
        getitem_498: "f16[32, 16, 257, 88]" = unbind_31[0]
        getitem_499: "f16[32, 16, 257, 88]" = unbind_31[1]
        getitem_500: "f16[32, 16, 257, 88]" = unbind_31[2];  unbind_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_31 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_498, getitem_499, getitem_500, None, False);  getitem_498 = getitem_499 = getitem_500 = None
        getitem_501: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_31[0];  _scaled_dot_product_cudnn_attention_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_189: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_501, [0, 2, 1, 3]);  getitem_501 = None
        view_314: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_189, [32, 257, 1408]);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_315: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_314, [8224, 1408]);  view_314 = None
        permute_190: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
        addmm_125: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg382_1, view_315, permute_190);  arg382_1 = view_315 = permute_190 = None
        view_316: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_125, [32, 257, 1408]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_220: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_217, view_316);  add_217 = view_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_566: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_220, torch.float32)
        var_mean_63 = torch.ops.aten.var_mean.correction(convert_element_type_566, [2], correction = 0, keepdim = True)
        getitem_510: "f32[32, 257, 1]" = var_mean_63[0]
        getitem_511: "f32[32, 257, 1]" = var_mean_63[1];  var_mean_63 = None
        sub_63: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_566, getitem_511);  convert_element_type_566 = getitem_511 = None
        add_221: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_510, 1e-06);  getitem_510 = None
        rsqrt_63: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        mul_219: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_63);  sub_63 = rsqrt_63 = None
        mul_220: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_219, arg383_1);  mul_219 = arg383_1 = None
        add_222: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_220, arg384_1);  mul_220 = arg384_1 = None
        convert_element_type_567: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_222, torch.float16);  add_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_317: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_567, [8224, 1408]);  convert_element_type_567 = None
        permute_191: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_126: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg386_1, view_317, permute_191);  arg386_1 = view_317 = permute_191 = None
        view_318: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_126, [32, 257, 6144]);  addmm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_571: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_318, torch.float32);  view_318 = None
        mul_221: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_571, 0.5)
        mul_222: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_571, 0.7071067811865476);  convert_element_type_571 = None
        erf_31: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_222);  mul_222 = None
        add_223: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
        mul_223: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_221, add_223);  mul_221 = add_223 = None
        convert_element_type_572: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_223, torch.float16);  mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_319: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_572, [8224, 6144]);  convert_element_type_572 = None
        permute_192: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        addmm_127: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg388_1, view_319, permute_192);  arg388_1 = view_319 = permute_192 = None
        view_320: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_127, [32, 257, 1408]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_224: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_220, view_320);  add_220 = view_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_576: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_224, torch.float32)
        var_mean_64 = torch.ops.aten.var_mean.correction(convert_element_type_576, [2], correction = 0, keepdim = True)
        getitem_512: "f32[32, 257, 1]" = var_mean_64[0]
        getitem_513: "f32[32, 257, 1]" = var_mean_64[1];  var_mean_64 = None
        sub_64: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_576, getitem_513);  convert_element_type_576 = getitem_513 = None
        add_225: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_512, 1e-06);  getitem_512 = None
        rsqrt_64: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        mul_224: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_64);  sub_64 = rsqrt_64 = None
        mul_225: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_224, arg389_1);  mul_224 = arg389_1 = None
        add_226: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_225, arg390_1);  mul_225 = arg390_1 = None
        convert_element_type_577: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_226, torch.float16);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_321: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_577, [8224, 1408]);  convert_element_type_577 = None
        permute_193: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg391_1, [1, 0]);  arg391_1 = None
        addmm_128: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg392_1, view_321, permute_193);  arg392_1 = view_321 = permute_193 = None
        view_322: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_128, [32, 257, 4224]);  addmm_128 = None
        view_323: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_322, [32, 257, 3, 16, 88]);  view_322 = None
        permute_194: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_323, [2, 0, 3, 1, 4]);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_32 = torch.ops.aten.unbind.int(permute_194);  permute_194 = None
        getitem_514: "f16[32, 16, 257, 88]" = unbind_32[0]
        getitem_515: "f16[32, 16, 257, 88]" = unbind_32[1]
        getitem_516: "f16[32, 16, 257, 88]" = unbind_32[2];  unbind_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_32 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_514, getitem_515, getitem_516, None, False);  getitem_514 = getitem_515 = getitem_516 = None
        getitem_517: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_32[0];  _scaled_dot_product_cudnn_attention_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_195: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_517, [0, 2, 1, 3]);  getitem_517 = None
        view_324: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_195, [32, 257, 1408]);  permute_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_325: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_324, [8224, 1408]);  view_324 = None
        permute_196: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
        addmm_129: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg394_1, view_325, permute_196);  arg394_1 = view_325 = permute_196 = None
        view_326: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_129, [32, 257, 1408]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_227: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_224, view_326);  add_224 = view_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_584: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_227, torch.float32)
        var_mean_65 = torch.ops.aten.var_mean.correction(convert_element_type_584, [2], correction = 0, keepdim = True)
        getitem_526: "f32[32, 257, 1]" = var_mean_65[0]
        getitem_527: "f32[32, 257, 1]" = var_mean_65[1];  var_mean_65 = None
        sub_65: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_584, getitem_527);  convert_element_type_584 = getitem_527 = None
        add_228: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_526, 1e-06);  getitem_526 = None
        rsqrt_65: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        mul_226: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_65);  sub_65 = rsqrt_65 = None
        mul_227: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_226, arg395_1);  mul_226 = arg395_1 = None
        add_229: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_227, arg396_1);  mul_227 = arg396_1 = None
        convert_element_type_585: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_229, torch.float16);  add_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_327: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_585, [8224, 1408]);  convert_element_type_585 = None
        permute_197: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg397_1, [1, 0]);  arg397_1 = None
        addmm_130: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg398_1, view_327, permute_197);  arg398_1 = view_327 = permute_197 = None
        view_328: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_130, [32, 257, 6144]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_589: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        mul_228: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_589, 0.5)
        mul_229: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_589, 0.7071067811865476);  convert_element_type_589 = None
        erf_32: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_229);  mul_229 = None
        add_230: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
        mul_230: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_228, add_230);  mul_228 = add_230 = None
        convert_element_type_590: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_230, torch.float16);  mul_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_329: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_590, [8224, 6144]);  convert_element_type_590 = None
        permute_198: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg399_1, [1, 0]);  arg399_1 = None
        addmm_131: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg400_1, view_329, permute_198);  arg400_1 = view_329 = permute_198 = None
        view_330: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_131, [32, 257, 1408]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_231: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_227, view_330);  add_227 = view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_594: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_231, torch.float32)
        var_mean_66 = torch.ops.aten.var_mean.correction(convert_element_type_594, [2], correction = 0, keepdim = True)
        getitem_528: "f32[32, 257, 1]" = var_mean_66[0]
        getitem_529: "f32[32, 257, 1]" = var_mean_66[1];  var_mean_66 = None
        sub_66: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_594, getitem_529);  convert_element_type_594 = getitem_529 = None
        add_232: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_528, 1e-06);  getitem_528 = None
        rsqrt_66: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        mul_231: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_66);  sub_66 = rsqrt_66 = None
        mul_232: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_231, arg401_1);  mul_231 = arg401_1 = None
        add_233: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_232, arg402_1);  mul_232 = arg402_1 = None
        convert_element_type_595: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_233, torch.float16);  add_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_331: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_595, [8224, 1408]);  convert_element_type_595 = None
        permute_199: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
        addmm_132: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg404_1, view_331, permute_199);  arg404_1 = view_331 = permute_199 = None
        view_332: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_132, [32, 257, 4224]);  addmm_132 = None
        view_333: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_332, [32, 257, 3, 16, 88]);  view_332 = None
        permute_200: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_333, [2, 0, 3, 1, 4]);  view_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_33 = torch.ops.aten.unbind.int(permute_200);  permute_200 = None
        getitem_530: "f16[32, 16, 257, 88]" = unbind_33[0]
        getitem_531: "f16[32, 16, 257, 88]" = unbind_33[1]
        getitem_532: "f16[32, 16, 257, 88]" = unbind_33[2];  unbind_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_33 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_530, getitem_531, getitem_532, None, False);  getitem_530 = getitem_531 = getitem_532 = None
        getitem_533: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_33[0];  _scaled_dot_product_cudnn_attention_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_201: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_533, [0, 2, 1, 3]);  getitem_533 = None
        view_334: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_201, [32, 257, 1408]);  permute_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_335: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_334, [8224, 1408]);  view_334 = None
        permute_202: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
        addmm_133: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg406_1, view_335, permute_202);  arg406_1 = view_335 = permute_202 = None
        view_336: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_133, [32, 257, 1408]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_234: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_231, view_336);  add_231 = view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_602: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_234, torch.float32)
        var_mean_67 = torch.ops.aten.var_mean.correction(convert_element_type_602, [2], correction = 0, keepdim = True)
        getitem_542: "f32[32, 257, 1]" = var_mean_67[0]
        getitem_543: "f32[32, 257, 1]" = var_mean_67[1];  var_mean_67 = None
        sub_67: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_602, getitem_543);  convert_element_type_602 = getitem_543 = None
        add_235: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_542, 1e-06);  getitem_542 = None
        rsqrt_67: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_235);  add_235 = None
        mul_233: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_67);  sub_67 = rsqrt_67 = None
        mul_234: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_233, arg407_1);  mul_233 = arg407_1 = None
        add_236: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_234, arg408_1);  mul_234 = arg408_1 = None
        convert_element_type_603: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_236, torch.float16);  add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_337: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_603, [8224, 1408]);  convert_element_type_603 = None
        permute_203: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
        addmm_134: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg410_1, view_337, permute_203);  arg410_1 = view_337 = permute_203 = None
        view_338: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_134, [32, 257, 6144]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_607: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_338, torch.float32);  view_338 = None
        mul_235: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_607, 0.5)
        mul_236: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_607, 0.7071067811865476);  convert_element_type_607 = None
        erf_33: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_236);  mul_236 = None
        add_237: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
        mul_237: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_235, add_237);  mul_235 = add_237 = None
        convert_element_type_608: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_237, torch.float16);  mul_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_339: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_608, [8224, 6144]);  convert_element_type_608 = None
        permute_204: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
        addmm_135: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg412_1, view_339, permute_204);  arg412_1 = view_339 = permute_204 = None
        view_340: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_135, [32, 257, 1408]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_238: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_234, view_340);  add_234 = view_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_612: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_238, torch.float32)
        var_mean_68 = torch.ops.aten.var_mean.correction(convert_element_type_612, [2], correction = 0, keepdim = True)
        getitem_544: "f32[32, 257, 1]" = var_mean_68[0]
        getitem_545: "f32[32, 257, 1]" = var_mean_68[1];  var_mean_68 = None
        sub_68: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_612, getitem_545);  convert_element_type_612 = getitem_545 = None
        add_239: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_544, 1e-06);  getitem_544 = None
        rsqrt_68: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        mul_238: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_68);  sub_68 = rsqrt_68 = None
        mul_239: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_238, arg413_1);  mul_238 = arg413_1 = None
        add_240: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_239, arg414_1);  mul_239 = arg414_1 = None
        convert_element_type_613: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_240, torch.float16);  add_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_341: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_613, [8224, 1408]);  convert_element_type_613 = None
        permute_205: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg415_1, [1, 0]);  arg415_1 = None
        addmm_136: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg416_1, view_341, permute_205);  arg416_1 = view_341 = permute_205 = None
        view_342: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_136, [32, 257, 4224]);  addmm_136 = None
        view_343: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_342, [32, 257, 3, 16, 88]);  view_342 = None
        permute_206: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_343, [2, 0, 3, 1, 4]);  view_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_34 = torch.ops.aten.unbind.int(permute_206);  permute_206 = None
        getitem_546: "f16[32, 16, 257, 88]" = unbind_34[0]
        getitem_547: "f16[32, 16, 257, 88]" = unbind_34[1]
        getitem_548: "f16[32, 16, 257, 88]" = unbind_34[2];  unbind_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_34 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_546, getitem_547, getitem_548, None, False);  getitem_546 = getitem_547 = getitem_548 = None
        getitem_549: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_34[0];  _scaled_dot_product_cudnn_attention_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_207: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_549, [0, 2, 1, 3]);  getitem_549 = None
        view_344: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_207, [32, 257, 1408]);  permute_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_345: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_344, [8224, 1408]);  view_344 = None
        permute_208: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg417_1, [1, 0]);  arg417_1 = None
        addmm_137: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg418_1, view_345, permute_208);  arg418_1 = view_345 = permute_208 = None
        view_346: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_137, [32, 257, 1408]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_241: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_238, view_346);  add_238 = view_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_620: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_241, torch.float32)
        var_mean_69 = torch.ops.aten.var_mean.correction(convert_element_type_620, [2], correction = 0, keepdim = True)
        getitem_558: "f32[32, 257, 1]" = var_mean_69[0]
        getitem_559: "f32[32, 257, 1]" = var_mean_69[1];  var_mean_69 = None
        sub_69: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_620, getitem_559);  convert_element_type_620 = getitem_559 = None
        add_242: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_558, 1e-06);  getitem_558 = None
        rsqrt_69: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_242);  add_242 = None
        mul_240: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_69);  sub_69 = rsqrt_69 = None
        mul_241: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_240, arg419_1);  mul_240 = arg419_1 = None
        add_243: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_241, arg420_1);  mul_241 = arg420_1 = None
        convert_element_type_621: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_243, torch.float16);  add_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_347: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_621, [8224, 1408]);  convert_element_type_621 = None
        permute_209: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg421_1, [1, 0]);  arg421_1 = None
        addmm_138: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg422_1, view_347, permute_209);  arg422_1 = view_347 = permute_209 = None
        view_348: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_138, [32, 257, 6144]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_625: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_348, torch.float32);  view_348 = None
        mul_242: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_625, 0.5)
        mul_243: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_625, 0.7071067811865476);  convert_element_type_625 = None
        erf_34: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_243);  mul_243 = None
        add_244: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
        mul_244: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_242, add_244);  mul_242 = add_244 = None
        convert_element_type_626: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_244, torch.float16);  mul_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_349: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_626, [8224, 6144]);  convert_element_type_626 = None
        permute_210: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
        addmm_139: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg424_1, view_349, permute_210);  arg424_1 = view_349 = permute_210 = None
        view_350: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_139, [32, 257, 1408]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_245: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_241, view_350);  add_241 = view_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_630: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_245, torch.float32)
        var_mean_70 = torch.ops.aten.var_mean.correction(convert_element_type_630, [2], correction = 0, keepdim = True)
        getitem_560: "f32[32, 257, 1]" = var_mean_70[0]
        getitem_561: "f32[32, 257, 1]" = var_mean_70[1];  var_mean_70 = None
        sub_70: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_630, getitem_561);  convert_element_type_630 = getitem_561 = None
        add_246: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_560, 1e-06);  getitem_560 = None
        rsqrt_70: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_246);  add_246 = None
        mul_245: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_70);  sub_70 = rsqrt_70 = None
        mul_246: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_245, arg425_1);  mul_245 = arg425_1 = None
        add_247: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_246, arg426_1);  mul_246 = arg426_1 = None
        convert_element_type_631: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_247, torch.float16);  add_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_351: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_631, [8224, 1408]);  convert_element_type_631 = None
        permute_211: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg427_1, [1, 0]);  arg427_1 = None
        addmm_140: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg428_1, view_351, permute_211);  arg428_1 = view_351 = permute_211 = None
        view_352: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_140, [32, 257, 4224]);  addmm_140 = None
        view_353: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_352, [32, 257, 3, 16, 88]);  view_352 = None
        permute_212: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_353, [2, 0, 3, 1, 4]);  view_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_35 = torch.ops.aten.unbind.int(permute_212);  permute_212 = None
        getitem_562: "f16[32, 16, 257, 88]" = unbind_35[0]
        getitem_563: "f16[32, 16, 257, 88]" = unbind_35[1]
        getitem_564: "f16[32, 16, 257, 88]" = unbind_35[2];  unbind_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_35 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_562, getitem_563, getitem_564, None, False);  getitem_562 = getitem_563 = getitem_564 = None
        getitem_565: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_35[0];  _scaled_dot_product_cudnn_attention_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_213: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_565, [0, 2, 1, 3]);  getitem_565 = None
        view_354: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_213, [32, 257, 1408]);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_355: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_354, [8224, 1408]);  view_354 = None
        permute_214: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg429_1, [1, 0]);  arg429_1 = None
        addmm_141: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg430_1, view_355, permute_214);  arg430_1 = view_355 = permute_214 = None
        view_356: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_141, [32, 257, 1408]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_248: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_245, view_356);  add_245 = view_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_638: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_248, torch.float32)
        var_mean_71 = torch.ops.aten.var_mean.correction(convert_element_type_638, [2], correction = 0, keepdim = True)
        getitem_574: "f32[32, 257, 1]" = var_mean_71[0]
        getitem_575: "f32[32, 257, 1]" = var_mean_71[1];  var_mean_71 = None
        sub_71: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_638, getitem_575);  convert_element_type_638 = getitem_575 = None
        add_249: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_574, 1e-06);  getitem_574 = None
        rsqrt_71: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_249);  add_249 = None
        mul_247: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_71);  sub_71 = rsqrt_71 = None
        mul_248: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_247, arg431_1);  mul_247 = arg431_1 = None
        add_250: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_248, arg432_1);  mul_248 = arg432_1 = None
        convert_element_type_639: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_250, torch.float16);  add_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_357: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_639, [8224, 1408]);  convert_element_type_639 = None
        permute_215: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg433_1, [1, 0]);  arg433_1 = None
        addmm_142: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg434_1, view_357, permute_215);  arg434_1 = view_357 = permute_215 = None
        view_358: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_142, [32, 257, 6144]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_643: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_358, torch.float32);  view_358 = None
        mul_249: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_643, 0.5)
        mul_250: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_643, 0.7071067811865476);  convert_element_type_643 = None
        erf_35: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_250);  mul_250 = None
        add_251: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
        mul_251: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_249, add_251);  mul_249 = add_251 = None
        convert_element_type_644: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_251, torch.float16);  mul_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_359: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_644, [8224, 6144]);  convert_element_type_644 = None
        permute_216: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
        addmm_143: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg436_1, view_359, permute_216);  arg436_1 = view_359 = permute_216 = None
        view_360: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_143, [32, 257, 1408]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_252: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_248, view_360);  add_248 = view_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_648: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_252, torch.float32)
        var_mean_72 = torch.ops.aten.var_mean.correction(convert_element_type_648, [2], correction = 0, keepdim = True)
        getitem_576: "f32[32, 257, 1]" = var_mean_72[0]
        getitem_577: "f32[32, 257, 1]" = var_mean_72[1];  var_mean_72 = None
        sub_72: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_648, getitem_577);  convert_element_type_648 = getitem_577 = None
        add_253: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_576, 1e-06);  getitem_576 = None
        rsqrt_72: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_253);  add_253 = None
        mul_252: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_72);  sub_72 = rsqrt_72 = None
        mul_253: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_252, arg437_1);  mul_252 = arg437_1 = None
        add_254: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_253, arg438_1);  mul_253 = arg438_1 = None
        convert_element_type_649: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_254, torch.float16);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_361: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_649, [8224, 1408]);  convert_element_type_649 = None
        permute_217: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg439_1, [1, 0]);  arg439_1 = None
        addmm_144: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg440_1, view_361, permute_217);  arg440_1 = view_361 = permute_217 = None
        view_362: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_144, [32, 257, 4224]);  addmm_144 = None
        view_363: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_362, [32, 257, 3, 16, 88]);  view_362 = None
        permute_218: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_363, [2, 0, 3, 1, 4]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_36 = torch.ops.aten.unbind.int(permute_218);  permute_218 = None
        getitem_578: "f16[32, 16, 257, 88]" = unbind_36[0]
        getitem_579: "f16[32, 16, 257, 88]" = unbind_36[1]
        getitem_580: "f16[32, 16, 257, 88]" = unbind_36[2];  unbind_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_36 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_578, getitem_579, getitem_580, None, False);  getitem_578 = getitem_579 = getitem_580 = None
        getitem_581: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_36[0];  _scaled_dot_product_cudnn_attention_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_219: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_581, [0, 2, 1, 3]);  getitem_581 = None
        view_364: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_219, [32, 257, 1408]);  permute_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_365: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_364, [8224, 1408]);  view_364 = None
        permute_220: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        addmm_145: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg442_1, view_365, permute_220);  arg442_1 = view_365 = permute_220 = None
        view_366: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_145, [32, 257, 1408]);  addmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_255: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_252, view_366);  add_252 = view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_656: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_255, torch.float32)
        var_mean_73 = torch.ops.aten.var_mean.correction(convert_element_type_656, [2], correction = 0, keepdim = True)
        getitem_590: "f32[32, 257, 1]" = var_mean_73[0]
        getitem_591: "f32[32, 257, 1]" = var_mean_73[1];  var_mean_73 = None
        sub_73: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_656, getitem_591);  convert_element_type_656 = getitem_591 = None
        add_256: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_590, 1e-06);  getitem_590 = None
        rsqrt_73: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_256);  add_256 = None
        mul_254: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_73);  sub_73 = rsqrt_73 = None
        mul_255: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_254, arg443_1);  mul_254 = arg443_1 = None
        add_257: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_255, arg444_1);  mul_255 = arg444_1 = None
        convert_element_type_657: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_257, torch.float16);  add_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_367: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_657, [8224, 1408]);  convert_element_type_657 = None
        permute_221: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
        addmm_146: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg446_1, view_367, permute_221);  arg446_1 = view_367 = permute_221 = None
        view_368: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_146, [32, 257, 6144]);  addmm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_661: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_368, torch.float32);  view_368 = None
        mul_256: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_661, 0.5)
        mul_257: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_661, 0.7071067811865476);  convert_element_type_661 = None
        erf_36: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_257);  mul_257 = None
        add_258: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
        mul_258: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_256, add_258);  mul_256 = add_258 = None
        convert_element_type_662: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_258, torch.float16);  mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_369: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_662, [8224, 6144]);  convert_element_type_662 = None
        permute_222: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
        addmm_147: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg448_1, view_369, permute_222);  arg448_1 = view_369 = permute_222 = None
        view_370: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_147, [32, 257, 1408]);  addmm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_259: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_255, view_370);  add_255 = view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_666: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_259, torch.float32)
        var_mean_74 = torch.ops.aten.var_mean.correction(convert_element_type_666, [2], correction = 0, keepdim = True)
        getitem_592: "f32[32, 257, 1]" = var_mean_74[0]
        getitem_593: "f32[32, 257, 1]" = var_mean_74[1];  var_mean_74 = None
        sub_74: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_666, getitem_593);  convert_element_type_666 = getitem_593 = None
        add_260: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_592, 1e-06);  getitem_592 = None
        rsqrt_74: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_260);  add_260 = None
        mul_259: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_74);  sub_74 = rsqrt_74 = None
        mul_260: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_259, arg449_1);  mul_259 = arg449_1 = None
        add_261: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_260, arg450_1);  mul_260 = arg450_1 = None
        convert_element_type_667: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_261, torch.float16);  add_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_371: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_667, [8224, 1408]);  convert_element_type_667 = None
        permute_223: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg451_1, [1, 0]);  arg451_1 = None
        addmm_148: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg452_1, view_371, permute_223);  arg452_1 = view_371 = permute_223 = None
        view_372: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_148, [32, 257, 4224]);  addmm_148 = None
        view_373: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_372, [32, 257, 3, 16, 88]);  view_372 = None
        permute_224: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_373, [2, 0, 3, 1, 4]);  view_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_37 = torch.ops.aten.unbind.int(permute_224);  permute_224 = None
        getitem_594: "f16[32, 16, 257, 88]" = unbind_37[0]
        getitem_595: "f16[32, 16, 257, 88]" = unbind_37[1]
        getitem_596: "f16[32, 16, 257, 88]" = unbind_37[2];  unbind_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_37 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_594, getitem_595, getitem_596, None, False);  getitem_594 = getitem_595 = getitem_596 = None
        getitem_597: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_37[0];  _scaled_dot_product_cudnn_attention_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_225: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_597, [0, 2, 1, 3]);  getitem_597 = None
        view_374: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_225, [32, 257, 1408]);  permute_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_375: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_374, [8224, 1408]);  view_374 = None
        permute_226: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg453_1, [1, 0]);  arg453_1 = None
        addmm_149: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg454_1, view_375, permute_226);  arg454_1 = view_375 = permute_226 = None
        view_376: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_149, [32, 257, 1408]);  addmm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_262: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_259, view_376);  add_259 = view_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_674: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_262, torch.float32)
        var_mean_75 = torch.ops.aten.var_mean.correction(convert_element_type_674, [2], correction = 0, keepdim = True)
        getitem_606: "f32[32, 257, 1]" = var_mean_75[0]
        getitem_607: "f32[32, 257, 1]" = var_mean_75[1];  var_mean_75 = None
        sub_75: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_674, getitem_607);  convert_element_type_674 = getitem_607 = None
        add_263: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_606, 1e-06);  getitem_606 = None
        rsqrt_75: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_263);  add_263 = None
        mul_261: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_75);  sub_75 = rsqrt_75 = None
        mul_262: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_261, arg455_1);  mul_261 = arg455_1 = None
        add_264: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_262, arg456_1);  mul_262 = arg456_1 = None
        convert_element_type_675: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_264, torch.float16);  add_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_377: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_675, [8224, 1408]);  convert_element_type_675 = None
        permute_227: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg457_1, [1, 0]);  arg457_1 = None
        addmm_150: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg458_1, view_377, permute_227);  arg458_1 = view_377 = permute_227 = None
        view_378: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_150, [32, 257, 6144]);  addmm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_679: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_378, torch.float32);  view_378 = None
        mul_263: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_679, 0.5)
        mul_264: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_679, 0.7071067811865476);  convert_element_type_679 = None
        erf_37: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_264);  mul_264 = None
        add_265: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
        mul_265: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_263, add_265);  mul_263 = add_265 = None
        convert_element_type_680: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_265, torch.float16);  mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_379: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_680, [8224, 6144]);  convert_element_type_680 = None
        permute_228: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
        addmm_151: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg460_1, view_379, permute_228);  arg460_1 = view_379 = permute_228 = None
        view_380: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_151, [32, 257, 1408]);  addmm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_266: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_262, view_380);  add_262 = view_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_684: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_266, torch.float32)
        var_mean_76 = torch.ops.aten.var_mean.correction(convert_element_type_684, [2], correction = 0, keepdim = True)
        getitem_608: "f32[32, 257, 1]" = var_mean_76[0]
        getitem_609: "f32[32, 257, 1]" = var_mean_76[1];  var_mean_76 = None
        sub_76: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_684, getitem_609);  convert_element_type_684 = getitem_609 = None
        add_267: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_608, 1e-06);  getitem_608 = None
        rsqrt_76: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_267);  add_267 = None
        mul_266: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_76);  sub_76 = rsqrt_76 = None
        mul_267: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_266, arg461_1);  mul_266 = arg461_1 = None
        add_268: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_267, arg462_1);  mul_267 = arg462_1 = None
        convert_element_type_685: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_268, torch.float16);  add_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_381: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_685, [8224, 1408]);  convert_element_type_685 = None
        permute_229: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
        addmm_152: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg464_1, view_381, permute_229);  arg464_1 = view_381 = permute_229 = None
        view_382: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_152, [32, 257, 4224]);  addmm_152 = None
        view_383: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_382, [32, 257, 3, 16, 88]);  view_382 = None
        permute_230: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_383, [2, 0, 3, 1, 4]);  view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_38 = torch.ops.aten.unbind.int(permute_230);  permute_230 = None
        getitem_610: "f16[32, 16, 257, 88]" = unbind_38[0]
        getitem_611: "f16[32, 16, 257, 88]" = unbind_38[1]
        getitem_612: "f16[32, 16, 257, 88]" = unbind_38[2];  unbind_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_38 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_610, getitem_611, getitem_612, None, False);  getitem_610 = getitem_611 = getitem_612 = None
        getitem_613: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_38[0];  _scaled_dot_product_cudnn_attention_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_231: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_613, [0, 2, 1, 3]);  getitem_613 = None
        view_384: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_231, [32, 257, 1408]);  permute_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_385: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_384, [8224, 1408]);  view_384 = None
        permute_232: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
        addmm_153: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg466_1, view_385, permute_232);  arg466_1 = view_385 = permute_232 = None
        view_386: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_153, [32, 257, 1408]);  addmm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_269: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_266, view_386);  add_266 = view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_692: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_269, torch.float32)
        var_mean_77 = torch.ops.aten.var_mean.correction(convert_element_type_692, [2], correction = 0, keepdim = True)
        getitem_622: "f32[32, 257, 1]" = var_mean_77[0]
        getitem_623: "f32[32, 257, 1]" = var_mean_77[1];  var_mean_77 = None
        sub_77: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_692, getitem_623);  convert_element_type_692 = getitem_623 = None
        add_270: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_622, 1e-06);  getitem_622 = None
        rsqrt_77: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_270);  add_270 = None
        mul_268: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_77);  sub_77 = rsqrt_77 = None
        mul_269: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_268, arg467_1);  mul_268 = arg467_1 = None
        add_271: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_269, arg468_1);  mul_269 = arg468_1 = None
        convert_element_type_693: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_271, torch.float16);  add_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_387: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_693, [8224, 1408]);  convert_element_type_693 = None
        permute_233: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg469_1, [1, 0]);  arg469_1 = None
        addmm_154: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg470_1, view_387, permute_233);  arg470_1 = view_387 = permute_233 = None
        view_388: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_154, [32, 257, 6144]);  addmm_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_697: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_388, torch.float32);  view_388 = None
        mul_270: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_697, 0.5)
        mul_271: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_697, 0.7071067811865476);  convert_element_type_697 = None
        erf_38: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_271);  mul_271 = None
        add_272: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
        mul_272: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_270, add_272);  mul_270 = add_272 = None
        convert_element_type_698: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_272, torch.float16);  mul_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_389: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_698, [8224, 6144]);  convert_element_type_698 = None
        permute_234: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg471_1, [1, 0]);  arg471_1 = None
        addmm_155: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg472_1, view_389, permute_234);  arg472_1 = view_389 = permute_234 = None
        view_390: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_155, [32, 257, 1408]);  addmm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_273: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_269, view_390);  add_269 = view_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_702: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_273, torch.float32)
        var_mean_78 = torch.ops.aten.var_mean.correction(convert_element_type_702, [2], correction = 0, keepdim = True)
        getitem_624: "f32[32, 257, 1]" = var_mean_78[0]
        getitem_625: "f32[32, 257, 1]" = var_mean_78[1];  var_mean_78 = None
        sub_78: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_702, getitem_625);  convert_element_type_702 = getitem_625 = None
        add_274: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_624, 1e-06);  getitem_624 = None
        rsqrt_78: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_274);  add_274 = None
        mul_273: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_78);  sub_78 = rsqrt_78 = None
        mul_274: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_273, arg473_1);  mul_273 = arg473_1 = None
        add_275: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_274, arg474_1);  mul_274 = arg474_1 = None
        convert_element_type_703: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_275, torch.float16);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:115 in forward, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        view_391: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_703, [8224, 1408]);  convert_element_type_703 = None
        permute_235: "f16[1408, 4224]" = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
        addmm_156: "f16[8224, 4224]" = torch.ops.aten.addmm.default(arg476_1, view_391, permute_235);  arg476_1 = view_391 = permute_235 = None
        view_392: "f16[32, 257, 4224]" = torch.ops.aten.reshape.default(addmm_156, [32, 257, 4224]);  addmm_156 = None
        view_393: "f16[32, 257, 3, 16, 88]" = torch.ops.aten.reshape.default(view_392, [32, 257, 3, 16, 88]);  view_392 = None
        permute_236: "f16[3, 32, 16, 257, 88]" = torch.ops.aten.permute.default(view_393, [2, 0, 3, 1, 4]);  view_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:116 in forward, code: q, k, v = qkv.unbind(0)
        unbind_39 = torch.ops.aten.unbind.int(permute_236);  permute_236 = None
        getitem_626: "f16[32, 16, 257, 88]" = unbind_39[0]
        getitem_627: "f16[32, 16, 257, 88]" = unbind_39[1]
        getitem_628: "f16[32, 16, 257, 88]" = unbind_39[2];  unbind_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:120 in forward, code: x = F.scaled_dot_product_attention(
        _scaled_dot_product_cudnn_attention_39 = torch.ops.aten._scaled_dot_product_cudnn_attention.default(getitem_626, getitem_627, getitem_628, None, False);  getitem_626 = getitem_627 = getitem_628 = None
        getitem_629: "f16[32, 16, 257, 88]" = _scaled_dot_product_cudnn_attention_39[0];  _scaled_dot_product_cudnn_attention_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:135 in forward, code: x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        permute_237: "f16[32, 257, 16, 88]" = torch.ops.aten.permute.default(getitem_629, [0, 2, 1, 3]);  getitem_629 = None
        view_394: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(permute_237, [32, 257, 1408]);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/attention.py:137 in forward, code: x = self.proj(x)
        view_395: "f16[8224, 1408]" = torch.ops.aten.reshape.default(view_394, [8224, 1408]);  view_394 = None
        permute_238: "f16[1408, 1408]" = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        addmm_157: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg478_1, view_395, permute_238);  arg478_1 = view_395 = permute_238 = None
        view_396: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_157, [32, 257, 1408]);  addmm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:212 in forward, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask=attn_mask, is_causal=is_causal)))
        add_276: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_273, view_396);  add_273 = view_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_710: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_276, torch.float32)
        var_mean_79 = torch.ops.aten.var_mean.correction(convert_element_type_710, [2], correction = 0, keepdim = True)
        getitem_638: "f32[32, 257, 1]" = var_mean_79[0]
        getitem_639: "f32[32, 257, 1]" = var_mean_79[1];  var_mean_79 = None
        sub_79: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_710, getitem_639);  convert_element_type_710 = getitem_639 = None
        add_277: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_638, 1e-06);  getitem_638 = None
        rsqrt_79: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_277);  add_277 = None
        mul_275: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_79);  sub_79 = rsqrt_79 = None
        mul_276: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_275, arg479_1);  mul_275 = arg479_1 = None
        add_278: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_276, arg480_1);  mul_276 = arg480_1 = None
        convert_element_type_711: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_278, torch.float16);  add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        view_397: "f16[8224, 1408]" = torch.ops.aten.reshape.default(convert_element_type_711, [8224, 1408]);  convert_element_type_711 = None
        permute_239: "f16[1408, 6144]" = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
        addmm_158: "f16[8224, 6144]" = torch.ops.aten.addmm.default(arg482_1, view_397, permute_239);  arg482_1 = view_397 = permute_239 = None
        view_398: "f16[32, 257, 6144]" = torch.ops.aten.reshape.default(addmm_158, [32, 257, 6144]);  addmm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:49 in forward, code: x = self.act(x)
        convert_element_type_715: "f32[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        mul_277: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_715, 0.5)
        mul_278: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(convert_element_type_715, 0.7071067811865476);  convert_element_type_715 = None
        erf_39: "f32[32, 257, 6144]" = torch.ops.aten.erf.default(mul_278);  mul_278 = None
        add_279: "f32[32, 257, 6144]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
        mul_279: "f32[32, 257, 6144]" = torch.ops.aten.mul.Tensor(mul_277, add_279);  mul_277 = add_279 = None
        convert_element_type_716: "f16[32, 257, 6144]" = torch.ops.prims.convert_element_type.default(mul_279, torch.float16);  mul_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        view_399: "f16[8224, 6144]" = torch.ops.aten.reshape.default(convert_element_type_716, [8224, 6144]);  convert_element_type_716 = None
        permute_240: "f16[6144, 1408]" = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
        addmm_159: "f16[8224, 1408]" = torch.ops.aten.addmm.default(arg484_1, view_399, permute_240);  arg484_1 = view_399 = permute_240 = None
        view_400: "f16[32, 257, 1408]" = torch.ops.aten.reshape.default(addmm_159, [32, 257, 1408]);  addmm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:213 in forward, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        add_280: "f16[32, 257, 1408]" = torch.ops.aten.add.Tensor(add_276, view_400);  add_276 = view_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/norm.py:89 in forward, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        convert_element_type_720: "f32[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_280, torch.float32);  add_280 = None
        var_mean_80 = torch.ops.aten.var_mean.correction(convert_element_type_720, [2], correction = 0, keepdim = True)
        getitem_640: "f32[32, 257, 1]" = var_mean_80[0]
        getitem_641: "f32[32, 257, 1]" = var_mean_80[1];  var_mean_80 = None
        sub_80: "f32[32, 257, 1408]" = torch.ops.aten.sub.Tensor(convert_element_type_720, getitem_641);  convert_element_type_720 = getitem_641 = None
        add_281: "f32[32, 257, 1]" = torch.ops.aten.add.Tensor(getitem_640, 1e-06);  getitem_640 = None
        rsqrt_80: "f32[32, 257, 1]" = torch.ops.aten.rsqrt.default(add_281);  add_281 = None
        mul_280: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_80);  sub_80 = rsqrt_80 = None
        mul_281: "f32[32, 257, 1408]" = torch.ops.aten.mul.Tensor(mul_280, arg485_1);  mul_280 = arg485_1 = None
        add_282: "f32[32, 257, 1408]" = torch.ops.aten.add.Tensor(mul_281, arg486_1);  mul_281 = arg486_1 = None
        convert_element_type_721: "f16[32, 257, 1408]" = torch.ops.prims.convert_element_type.default(add_282, torch.float16);  add_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:696 in global_pool_nlc, code: x = x[:, 0]  # class token
        select: "f16[32, 1408]" = torch.ops.aten.select.int(convert_element_type_721, 1, 0);  convert_element_type_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1291 in forward_head, code: x = self.head_drop(x)
        clone_121: "f16[32, 1408]" = torch.ops.aten.clone.default(select);  select = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/vision_transformer.py:1292 in forward_head, code: return x if pre_logits else self.head(x)
        permute_241: "f16[1408, 1000]" = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
        addmm_160: "f16[32, 1000]" = torch.ops.aten.addmm.default(arg488_1, clone_121, permute_241);  arg488_1 = clone_121 = permute_241 = None
        return (addmm_160,)
