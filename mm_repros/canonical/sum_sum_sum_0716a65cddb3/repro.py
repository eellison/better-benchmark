"""
Standalone repro captured via capture_hook.
Label: MT5ForConditionalGeneration
Pattern hash: 0716a65cddb3
Shape hash: 49968b40
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[32, 128]", arg3_1: "bf16[32, 128, 250112]", arg4_1: "f32[4096, 1]", arg5_1: "f32[4096, 1]", arg6_1: "bf16[32, 128, 250112]", arg7_1: "bf16[4096, 512]", arg8_1: "bf16[250112, 512]", arg9_1: "b8[32, 128, 512]", arg10_1: "f32[512]", arg11_1: "f32[32, 128, 512]", arg12_1: "f32[32, 128, 1]", arg13_1: "b8[32, 128, 512]", arg14_1: "bf16[4096, 1024]", arg15_1: "bf16[512, 1024]", arg16_1: "b8[32, 128, 1024]", arg17_1: "bf16[4096, 1024]", arg18_1: "bf16[4096, 1024]", arg19_1: "bf16[4096, 512]", arg20_1: "bf16[1024, 512]", arg21_1: "bf16[1024, 512]", arg22_1: "f32[512]", arg23_1: "f32[32, 128, 512]", arg24_1: "f32[32, 128, 1]", arg25_1: "b8[32, 128, 512]", arg26_1: "bf16[4096, 384]", arg27_1: "bf16[512, 384]", arg28_1: "bf16[192, 128, 128]", arg29_1: "bf16[192, 64, 128]", arg30_1: "b8[32, 6, 128, 128]", arg31_1: "bf16[192, 128, 128]", arg32_1: "f32[32, 6, 128, 1]", arg33_1: "f32[32, 6, 128, 1]", arg34_1: "bf16[192, 64, 128]", arg35_1: "bf16[192, 128, 64]", arg36_1: "bf16[4096, 512]", arg37_1: "bf16[384, 512]", arg38_1: "f32[32, 128, 512]", arg39_1: "bf16[384, 512]", arg40_1: "bf16[4096, 512]", arg41_1: "bf16[384, 512]", arg42_1: "f32[512]", arg43_1: "f32[32, 128, 512]", arg44_1: "f32[32, 128, 1]", arg45_1: "b8[32, 128, 512]", arg46_1: "bf16[4096, 384]", arg47_1: "bf16[512, 384]", arg48_1: "bf16[192, 128, 128]", arg49_1: "bf16[192, 64, 128]", arg50_1: "b8[32, 6, 128, 128]", arg51_1: "bf16[32, 6, 128, 128]", arg52_1: "f32[32, 6, 128, 1]", arg53_1: "f32[32, 6, 128, 1]", arg54_1: "bf16[192, 64, 128]", arg55_1: "bf16[192, 128, 64]", arg56_1: "bf16[4096, 512]", arg57_1: "bf16[384, 512]", arg58_1: "bf16[384, 512]", arg59_1: "bf16[384, 512]", arg60_1: "f32[512]", arg61_1: "f32[32, 128, 512]", arg62_1: "f32[32, 128, 1]", arg63_1: "b8[32, 128, 512]", arg64_1: "bf16[4096, 1024]", arg65_1: "bf16[512, 1024]", arg66_1: "b8[32, 128, 1024]", arg67_1: "bf16[4096, 1024]", arg68_1: "bf16[4096, 1024]", arg69_1: "bf16[4096, 512]", arg70_1: "bf16[1024, 512]", arg71_1: "bf16[1024, 512]", arg72_1: "f32[512]", arg73_1: "f32[32, 128, 512]", arg74_1: "f32[32, 128, 1]", arg75_1: "b8[32, 128, 512]", arg76_1: "bf16[4096, 384]", arg77_1: "bf16[512, 384]", arg78_1: "bf16[192, 128, 128]", arg79_1: "bf16[192, 64, 128]", arg80_1: "b8[32, 6, 128, 128]", arg81_1: "bf16[192, 128, 128]", arg82_1: "f32[32, 6, 128, 1]", arg83_1: "f32[32, 6, 128, 1]", arg84_1: "bf16[192, 64, 128]", arg85_1: "bf16[192, 128, 64]", arg86_1: "bf16[384, 512]", arg87_1: "bf16[384, 512]", arg88_1: "bf16[4096, 512]", arg89_1: "bf16[384, 512]", arg90_1: "f32[512]", arg91_1: "f32[32, 128, 512]", arg92_1: "f32[32, 128, 1]", arg93_1: "b8[32, 128, 512]", arg94_1: "bf16[4096, 384]", arg95_1: "bf16[512, 384]", arg96_1: "bf16[192, 128, 128]", arg97_1: "bf16[192, 64, 128]", arg98_1: "b8[32, 6, 128, 128]", arg99_1: "bf16[32, 6, 128, 128]", arg100_1: "f32[32, 6, 128, 1]", arg101_1: "f32[32, 6, 128, 1]", arg102_1: "bf16[192, 64, 128]", arg103_1: "bf16[192, 128, 64]", arg104_1: "bf16[4096, 512]", arg105_1: "bf16[384, 512]", arg106_1: "bf16[384, 512]", arg107_1: "bf16[384, 512]", arg108_1: "f32[512]", arg109_1: "f32[32, 128, 512]", arg110_1: "f32[32, 128, 1]", arg111_1: "b8[32, 128, 512]", arg112_1: "bf16[4096, 1024]", arg113_1: "bf16[512, 1024]", arg114_1: "b8[32, 128, 1024]", arg115_1: "bf16[4096, 1024]", arg116_1: "bf16[4096, 1024]", arg117_1: "bf16[4096, 512]", arg118_1: "bf16[1024, 512]", arg119_1: "bf16[1024, 512]", arg120_1: "f32[512]", arg121_1: "f32[32, 128, 512]", arg122_1: "f32[32, 128, 1]", arg123_1: "b8[32, 128, 512]", arg124_1: "bf16[4096, 384]", arg125_1: "bf16[512, 384]", arg126_1: "bf16[192, 128, 128]", arg127_1: "bf16[192, 64, 128]", arg128_1: "b8[32, 6, 128, 128]", arg129_1: "bf16[192, 128, 128]", arg130_1: "f32[32, 6, 128, 1]", arg131_1: "f32[32, 6, 128, 1]", arg132_1: "bf16[192, 64, 128]", arg133_1: "bf16[192, 128, 64]", arg134_1: "bf16[384, 512]", arg135_1: "bf16[384, 512]", arg136_1: "bf16[4096, 512]", arg137_1: "bf16[384, 512]", arg138_1: "f32[512]", arg139_1: "f32[32, 128, 512]", arg140_1: "f32[32, 128, 1]", arg141_1: "b8[32, 128, 512]", arg142_1: "bf16[4096, 384]", arg143_1: "bf16[512, 384]", arg144_1: "bf16[192, 128, 128]", arg145_1: "bf16[192, 64, 128]", arg146_1: "b8[32, 6, 128, 128]", arg147_1: "bf16[32, 6, 128, 128]", arg148_1: "f32[32, 6, 128, 1]", arg149_1: "f32[32, 6, 128, 1]", arg150_1: "bf16[192, 64, 128]", arg151_1: "bf16[192, 128, 64]", arg152_1: "bf16[4096, 512]", arg153_1: "bf16[384, 512]", arg154_1: "bf16[384, 512]", arg155_1: "bf16[384, 512]", arg156_1: "f32[512]", arg157_1: "f32[32, 128, 512]", arg158_1: "f32[32, 128, 1]", arg159_1: "b8[32, 128, 512]", arg160_1: "bf16[4096, 1024]", arg161_1: "bf16[512, 1024]", arg162_1: "b8[32, 128, 1024]", arg163_1: "bf16[4096, 1024]", arg164_1: "bf16[4096, 1024]", arg165_1: "bf16[4096, 512]", arg166_1: "bf16[1024, 512]", arg167_1: "bf16[1024, 512]", arg168_1: "f32[512]", arg169_1: "f32[32, 128, 512]", arg170_1: "f32[32, 128, 1]", arg171_1: "b8[32, 128, 512]", arg172_1: "bf16[4096, 384]", arg173_1: "bf16[512, 384]", arg174_1: "bf16[192, 128, 128]", arg175_1: "bf16[192, 64, 128]", arg176_1: "b8[32, 6, 128, 128]", arg177_1: "bf16[192, 128, 128]", arg178_1: "f32[32, 6, 128, 1]", arg179_1: "f32[32, 6, 128, 1]", arg180_1: "bf16[192, 64, 128]", arg181_1: "bf16[192, 128, 64]", arg182_1: "bf16[384, 512]", arg183_1: "bf16[384, 512]", arg184_1: "bf16[4096, 512]", arg185_1: "bf16[384, 512]", arg186_1: "f32[512]", arg187_1: "f32[32, 128, 512]", arg188_1: "f32[32, 128, 1]", arg189_1: "b8[32, 128, 512]", arg190_1: "bf16[4096, 384]", arg191_1: "bf16[512, 384]", arg192_1: "bf16[192, 128, 128]", arg193_1: "bf16[192, 64, 128]", arg194_1: "b8[32, 6, 128, 128]", arg195_1: "bf16[32, 6, 128, 128]", arg196_1: "f32[32, 6, 128, 1]", arg197_1: "f32[32, 6, 128, 1]", arg198_1: "bf16[192, 64, 128]", arg199_1: "bf16[192, 128, 64]", arg200_1: "bf16[4096, 512]", arg201_1: "bf16[384, 512]", arg202_1: "bf16[384, 512]", arg203_1: "bf16[384, 512]", arg204_1: "f32[512]", arg205_1: "f32[32, 128, 512]", arg206_1: "f32[32, 128, 1]", arg207_1: "b8[32, 128, 512]", arg208_1: "bf16[4096, 1024]", arg209_1: "bf16[512, 1024]", arg210_1: "b8[32, 128, 1024]", arg211_1: "bf16[4096, 1024]", arg212_1: "bf16[4096, 1024]", arg213_1: "bf16[4096, 512]", arg214_1: "bf16[1024, 512]", arg215_1: "bf16[1024, 512]", arg216_1: "f32[512]", arg217_1: "f32[32, 128, 512]", arg218_1: "f32[32, 128, 1]", arg219_1: "b8[32, 128, 512]", arg220_1: "bf16[4096, 384]", arg221_1: "bf16[512, 384]", arg222_1: "bf16[192, 128, 128]", arg223_1: "bf16[192, 64, 128]", arg224_1: "b8[32, 6, 128, 128]", arg225_1: "bf16[192, 128, 128]", arg226_1: "f32[32, 6, 128, 1]", arg227_1: "f32[32, 6, 128, 1]", arg228_1: "bf16[192, 64, 128]", arg229_1: "bf16[192, 128, 64]", arg230_1: "bf16[384, 512]", arg231_1: "bf16[384, 512]", arg232_1: "bf16[4096, 512]", arg233_1: "bf16[384, 512]", arg234_1: "f32[512]", arg235_1: "f32[32, 128, 512]", arg236_1: "f32[32, 128, 1]", arg237_1: "b8[32, 128, 512]", arg238_1: "bf16[4096, 384]", arg239_1: "bf16[512, 384]", arg240_1: "bf16[192, 128, 128]", arg241_1: "bf16[192, 64, 128]", arg242_1: "b8[32, 6, 128, 128]", arg243_1: "bf16[32, 6, 128, 128]", arg244_1: "f32[32, 6, 128, 1]", arg245_1: "f32[32, 6, 128, 1]", arg246_1: "bf16[192, 64, 128]", arg247_1: "bf16[192, 128, 64]", arg248_1: "bf16[4096, 512]", arg249_1: "bf16[384, 512]", arg250_1: "bf16[384, 512]", arg251_1: "bf16[384, 512]", arg252_1: "f32[512]", arg253_1: "f32[32, 128, 512]", arg254_1: "f32[32, 128, 1]", arg255_1: "b8[32, 128, 512]", arg256_1: "bf16[4096, 1024]", arg257_1: "bf16[512, 1024]", arg258_1: "b8[32, 128, 1024]", arg259_1: "bf16[4096, 1024]", arg260_1: "bf16[4096, 1024]", arg261_1: "bf16[4096, 512]", arg262_1: "bf16[1024, 512]", arg263_1: "bf16[1024, 512]", arg264_1: "f32[512]", arg265_1: "f32[32, 128, 512]", arg266_1: "f32[32, 128, 1]", arg267_1: "b8[32, 128, 512]", arg268_1: "bf16[4096, 384]", arg269_1: "bf16[512, 384]", arg270_1: "bf16[192, 128, 128]", arg271_1: "bf16[192, 64, 128]", arg272_1: "b8[32, 6, 128, 128]", arg273_1: "bf16[192, 128, 128]", arg274_1: "f32[32, 6, 128, 1]", arg275_1: "f32[32, 6, 128, 1]", arg276_1: "bf16[192, 64, 128]", arg277_1: "bf16[192, 128, 64]", arg278_1: "bf16[384, 512]", arg279_1: "bf16[384, 512]", arg280_1: "bf16[4096, 512]", arg281_1: "bf16[384, 512]", arg282_1: "f32[512]", arg283_1: "f32[32, 128, 512]", arg284_1: "f32[32, 128, 1]", arg285_1: "b8[32, 128, 512]", arg286_1: "bf16[4096, 384]", arg287_1: "bf16[512, 384]", arg288_1: "bf16[192, 128, 128]", arg289_1: "bf16[192, 64, 128]", arg290_1: "b8[32, 6, 128, 128]", arg291_1: "bf16[32, 6, 128, 128]", arg292_1: "f32[32, 6, 128, 1]", arg293_1: "f32[32, 6, 128, 1]", arg294_1: "bf16[192, 64, 128]", arg295_1: "bf16[192, 128, 64]", arg296_1: "bf16[4096, 512]", arg297_1: "bf16[384, 512]", arg298_1: "bf16[384, 512]", arg299_1: "bf16[384, 512]", arg300_1: "f32[512]", arg301_1: "f32[32, 128, 512]", arg302_1: "f32[32, 128, 1]", arg303_1: "b8[32, 128, 512]", arg304_1: "bf16[4096, 1024]", arg305_1: "bf16[512, 1024]", arg306_1: "b8[32, 128, 1024]", arg307_1: "bf16[4096, 1024]", arg308_1: "bf16[4096, 1024]", arg309_1: "bf16[4096, 512]", arg310_1: "bf16[1024, 512]", arg311_1: "bf16[1024, 512]", arg312_1: "f32[512]", arg313_1: "f32[32, 128, 512]", arg314_1: "f32[32, 128, 1]", arg315_1: "b8[32, 128, 512]", arg316_1: "bf16[4096, 384]", arg317_1: "bf16[512, 384]", arg318_1: "bf16[192, 128, 128]", arg319_1: "bf16[192, 64, 128]", arg320_1: "b8[32, 6, 128, 128]", arg321_1: "bf16[192, 128, 128]", arg322_1: "f32[32, 6, 128, 1]", arg323_1: "f32[32, 6, 128, 1]", arg324_1: "bf16[192, 64, 128]", arg325_1: "bf16[192, 128, 64]", arg326_1: "bf16[384, 512]", arg327_1: "bf16[384, 512]", arg328_1: "bf16[4096, 512]", arg329_1: "bf16[384, 512]", arg330_1: "f32[512]", arg331_1: "f32[32, 128, 512]", arg332_1: "f32[32, 128, 1]", arg333_1: "b8[32, 128, 512]", arg334_1: "bf16[4096, 384]", arg335_1: "bf16[512, 384]", arg336_1: "bf16[192, 128, 128]", arg337_1: "bf16[192, 64, 128]", arg338_1: "b8[32, 6, 128, 128]", arg339_1: "bf16[32, 6, 128, 128]", arg340_1: "f32[32, 6, 128, 1]", arg341_1: "f32[32, 6, 128, 1]", arg342_1: "bf16[192, 64, 128]", arg343_1: "bf16[192, 128, 64]", arg344_1: "bf16[4096, 512]", arg345_1: "bf16[384, 512]", arg346_1: "bf16[384, 512]", arg347_1: "bf16[384, 512]", arg348_1: "f32[512]", arg349_1: "f32[32, 128, 512]", arg350_1: "f32[32, 128, 1]", arg351_1: "b8[32, 128, 512]", arg352_1: "bf16[4096, 1024]", arg353_1: "bf16[512, 1024]", arg354_1: "b8[32, 128, 1024]", arg355_1: "bf16[4096, 1024]", arg356_1: "bf16[4096, 1024]", arg357_1: "bf16[4096, 512]", arg358_1: "bf16[1024, 512]", arg359_1: "bf16[1024, 512]", arg360_1: "f32[512]", arg361_1: "f32[32, 128, 512]", arg362_1: "f32[32, 128, 1]", arg363_1: "b8[32, 128, 512]", arg364_1: "bf16[4096, 384]", arg365_1: "bf16[512, 384]", arg366_1: "bf16[192, 128, 128]", arg367_1: "bf16[192, 64, 128]", arg368_1: "b8[32, 6, 128, 128]", arg369_1: "bf16[192, 128, 128]", arg370_1: "f32[32, 6, 128, 1]", arg371_1: "f32[32, 6, 128, 1]", arg372_1: "bf16[192, 64, 128]", arg373_1: "bf16[192, 128, 64]", arg374_1: "bf16[384, 512]", arg375_1: "bf16[384, 512]", arg376_1: "bf16[4096, 512]", arg377_1: "bf16[384, 512]", arg378_1: "f32[512]", arg379_1: "f32[32, 128, 512]", arg380_1: "f32[32, 128, 1]", arg381_1: "b8[32, 128, 512]", arg382_1: "bf16[4096, 384]", arg383_1: "bf16[512, 384]", arg384_1: "bf16[192, 128, 128]", arg385_1: "bf16[192, 64, 128]", arg386_1: "b8[32, 6, 128, 128]", arg387_1: "i64[1, 1, 128]", arg388_1: "bf16[192, 128, 128]", arg389_1: "f32[128, 128, 6]", arg390_1: "f32[32, 6, 128, 1]", arg391_1: "f32[32, 6, 128, 1]", arg392_1: "i64[128, 128]", arg393_1: "bf16[192, 64, 128]", arg394_1: "bf16[192, 128, 64]", arg395_1: "bf16[4096, 512]", arg396_1: "bf16[384, 512]", arg397_1: "bf16[384, 512]", arg398_1: "bf16[384, 512]", arg399_1: "f32[512]", arg400_1: "b8[32, 128, 512]", arg401_1: "f32[32, 128, 512]", arg402_1: "f32[32, 128, 1]", arg403_1: "i64[32, 128]", arg404_1: "b8[32, 128, 512]", arg405_1: "f32[512]", arg406_1: "f32[32, 128, 512]", arg407_1: "f32[32, 128, 1]", arg408_1: "b8[32, 128, 512]", arg409_1: "bf16[4096, 1024]", arg410_1: "bf16[512, 1024]", arg411_1: "b8[32, 128, 1024]", arg412_1: "bf16[4096, 1024]", arg413_1: "bf16[4096, 1024]", arg414_1: "bf16[4096, 512]", arg415_1: "bf16[1024, 512]", arg416_1: "bf16[1024, 512]", arg417_1: "f32[512]", arg418_1: "f32[32, 128, 512]", arg419_1: "f32[32, 128, 1]", arg420_1: "b8[32, 128, 512]", arg421_1: "bf16[4096, 384]", arg422_1: "bf16[512, 384]", arg423_1: "bf16[192, 128, 128]", arg424_1: "bf16[192, 64, 128]", arg425_1: "b8[32, 6, 128, 128]", arg426_1: "bf16[32, 6, 128, 128]", arg427_1: "f32[32, 6, 128, 1]", arg428_1: "f32[32, 6, 128, 1]", arg429_1: "bf16[192, 64, 128]", arg430_1: "bf16[192, 128, 64]", arg431_1: "bf16[4096, 512]", arg432_1: "bf16[384, 512]", arg433_1: "bf16[384, 512]", arg434_1: "bf16[384, 512]", arg435_1: "f32[512]", arg436_1: "f32[32, 128, 512]", arg437_1: "f32[32, 128, 1]", arg438_1: "b8[32, 128, 512]", arg439_1: "bf16[4096, 1024]", arg440_1: "bf16[512, 1024]", arg441_1: "b8[32, 128, 1024]", arg442_1: "bf16[4096, 1024]", arg443_1: "bf16[4096, 1024]", arg444_1: "bf16[4096, 512]", arg445_1: "bf16[1024, 512]", arg446_1: "bf16[1024, 512]", arg447_1: "f32[512]", arg448_1: "f32[32, 128, 512]", arg449_1: "f32[32, 128, 1]", arg450_1: "b8[32, 128, 512]", arg451_1: "bf16[4096, 384]", arg452_1: "bf16[512, 384]", arg453_1: "bf16[192, 128, 128]", arg454_1: "bf16[192, 64, 128]", arg455_1: "b8[32, 6, 128, 128]", arg456_1: "bf16[32, 6, 128, 128]", arg457_1: "f32[32, 6, 128, 1]", arg458_1: "f32[32, 6, 128, 1]", arg459_1: "bf16[192, 64, 128]", arg460_1: "bf16[192, 128, 64]", arg461_1: "bf16[4096, 512]", arg462_1: "bf16[384, 512]", arg463_1: "bf16[384, 512]", arg464_1: "bf16[384, 512]", arg465_1: "f32[512]", arg466_1: "f32[32, 128, 512]", arg467_1: "f32[32, 128, 1]", arg468_1: "b8[32, 128, 512]", arg469_1: "bf16[4096, 1024]", arg470_1: "bf16[512, 1024]", arg471_1: "b8[32, 128, 1024]", arg472_1: "bf16[4096, 1024]", arg473_1: "bf16[4096, 1024]", arg474_1: "bf16[4096, 512]", arg475_1: "bf16[1024, 512]", arg476_1: "bf16[1024, 512]", arg477_1: "f32[512]", arg478_1: "f32[32, 128, 512]", arg479_1: "f32[32, 128, 1]", arg480_1: "b8[32, 128, 512]", arg481_1: "bf16[4096, 384]", arg482_1: "bf16[512, 384]", arg483_1: "bf16[192, 128, 128]", arg484_1: "bf16[192, 64, 128]", arg485_1: "b8[32, 6, 128, 128]", arg486_1: "bf16[32, 6, 128, 128]", arg487_1: "f32[32, 6, 128, 1]", arg488_1: "f32[32, 6, 128, 1]", arg489_1: "bf16[192, 64, 128]", arg490_1: "bf16[192, 128, 64]", arg491_1: "bf16[4096, 512]", arg492_1: "bf16[384, 512]", arg493_1: "bf16[384, 512]", arg494_1: "bf16[384, 512]", arg495_1: "f32[512]", arg496_1: "f32[32, 128, 512]", arg497_1: "f32[32, 128, 1]", arg498_1: "b8[32, 128, 512]", arg499_1: "bf16[4096, 1024]", arg500_1: "bf16[512, 1024]", arg501_1: "b8[32, 128, 1024]", arg502_1: "bf16[4096, 1024]", arg503_1: "bf16[4096, 1024]", arg504_1: "bf16[4096, 512]", arg505_1: "bf16[1024, 512]", arg506_1: "bf16[1024, 512]", arg507_1: "f32[512]", arg508_1: "f32[32, 128, 512]", arg509_1: "f32[32, 128, 1]", arg510_1: "b8[32, 128, 512]", arg511_1: "bf16[4096, 384]", arg512_1: "bf16[512, 384]", arg513_1: "bf16[192, 128, 128]", arg514_1: "bf16[192, 64, 128]", arg515_1: "b8[32, 6, 128, 128]", arg516_1: "bf16[32, 6, 128, 128]", arg517_1: "f32[32, 6, 128, 1]", arg518_1: "f32[32, 6, 128, 1]", arg519_1: "bf16[192, 64, 128]", arg520_1: "bf16[192, 128, 64]", arg521_1: "bf16[4096, 512]", arg522_1: "bf16[384, 512]", arg523_1: "bf16[384, 512]", arg524_1: "bf16[384, 512]", arg525_1: "f32[512]", arg526_1: "f32[32, 128, 512]", arg527_1: "f32[32, 128, 1]", arg528_1: "b8[32, 128, 512]", arg529_1: "bf16[4096, 1024]", arg530_1: "bf16[512, 1024]", arg531_1: "b8[32, 128, 1024]", arg532_1: "bf16[4096, 1024]", arg533_1: "bf16[4096, 1024]", arg534_1: "bf16[4096, 512]", arg535_1: "bf16[1024, 512]", arg536_1: "bf16[1024, 512]", arg537_1: "f32[512]", arg538_1: "f32[32, 128, 512]", arg539_1: "f32[32, 128, 1]", arg540_1: "b8[32, 128, 512]", arg541_1: "bf16[4096, 384]", arg542_1: "bf16[512, 384]", arg543_1: "bf16[192, 128, 128]", arg544_1: "bf16[192, 64, 128]", arg545_1: "b8[32, 6, 128, 128]", arg546_1: "bf16[32, 6, 128, 128]", arg547_1: "f32[32, 6, 128, 1]", arg548_1: "f32[32, 6, 128, 1]", arg549_1: "bf16[192, 64, 128]", arg550_1: "bf16[192, 128, 64]", arg551_1: "bf16[4096, 512]", arg552_1: "bf16[384, 512]", arg553_1: "bf16[384, 512]", arg554_1: "bf16[384, 512]", arg555_1: "f32[512]", arg556_1: "f32[32, 128, 512]", arg557_1: "f32[32, 128, 1]", arg558_1: "b8[32, 128, 512]", arg559_1: "bf16[4096, 1024]", arg560_1: "bf16[512, 1024]", arg561_1: "b8[32, 128, 1024]", arg562_1: "bf16[4096, 1024]", arg563_1: "bf16[4096, 1024]", arg564_1: "bf16[4096, 512]", arg565_1: "bf16[1024, 512]", arg566_1: "bf16[1024, 512]", arg567_1: "f32[512]", arg568_1: "f32[32, 128, 512]", arg569_1: "f32[32, 128, 1]", arg570_1: "b8[32, 128, 512]", arg571_1: "bf16[4096, 384]", arg572_1: "bf16[512, 384]", arg573_1: "bf16[192, 128, 128]", arg574_1: "bf16[192, 64, 128]", arg575_1: "b8[32, 6, 128, 128]", arg576_1: "bf16[32, 6, 128, 128]", arg577_1: "f32[32, 6, 128, 1]", arg578_1: "f32[32, 6, 128, 1]", arg579_1: "bf16[192, 64, 128]", arg580_1: "bf16[192, 128, 64]", arg581_1: "bf16[4096, 512]", arg582_1: "bf16[384, 512]", arg583_1: "bf16[384, 512]", arg584_1: "bf16[384, 512]", arg585_1: "f32[512]", arg586_1: "f32[32, 128, 512]", arg587_1: "f32[32, 128, 1]", arg588_1: "b8[32, 128, 512]", arg589_1: "bf16[4096, 1024]", arg590_1: "bf16[512, 1024]", arg591_1: "b8[32, 128, 1024]", arg592_1: "bf16[4096, 1024]", arg593_1: "bf16[4096, 1024]", arg594_1: "bf16[4096, 512]", arg595_1: "bf16[1024, 512]", arg596_1: "bf16[1024, 512]", arg597_1: "f32[512]", arg598_1: "f32[32, 128, 512]", arg599_1: "f32[32, 128, 1]", arg600_1: "b8[32, 128, 512]", arg601_1: "bf16[4096, 384]", arg602_1: "bf16[512, 384]", arg603_1: "bf16[192, 128, 128]", arg604_1: "bf16[192, 64, 128]", arg605_1: "b8[32, 6, 128, 128]", arg606_1: "bf16[32, 6, 128, 128]", arg607_1: "f32[32, 6, 128, 1]", arg608_1: "f32[32, 6, 128, 1]", arg609_1: "bf16[192, 64, 128]", arg610_1: "bf16[192, 128, 64]", arg611_1: "bf16[4096, 512]", arg612_1: "bf16[384, 512]", arg613_1: "bf16[384, 512]", arg614_1: "bf16[384, 512]", arg615_1: "f32[512]", arg616_1: "f32[32, 128, 512]", arg617_1: "f32[32, 128, 1]", arg618_1: "b8[32, 128, 512]", arg619_1: "bf16[4096, 1024]", arg620_1: "bf16[512, 1024]", arg621_1: "b8[32, 128, 1024]", arg622_1: "bf16[4096, 1024]", arg623_1: "bf16[4096, 1024]", arg624_1: "bf16[4096, 512]", arg625_1: "bf16[1024, 512]", arg626_1: "bf16[1024, 512]", arg627_1: "f32[512]", arg628_1: "f32[32, 128, 512]", arg629_1: "f32[32, 128, 1]", arg630_1: "b8[32, 128, 512]", arg631_1: "bf16[4096, 384]", arg632_1: "bf16[512, 384]", arg633_1: "bf16[192, 128, 128]", arg634_1: "bf16[192, 64, 128]", arg635_1: "b8[32, 6, 128, 128]", arg636_1: "bf16[192, 128, 128]", arg637_1: "f32[128, 128, 6]", arg638_1: "f32[32, 6, 128, 1]", arg639_1: "f32[32, 6, 128, 1]", arg640_1: "i64[128, 128]", arg641_1: "bf16[192, 64, 128]", arg642_1: "bf16[192, 128, 64]", arg643_1: "bf16[4096, 512]", arg644_1: "bf16[384, 512]", arg645_1: "bf16[384, 512]", arg646_1: "bf16[384, 512]", arg647_1: "f32[512]", arg648_1: "b8[32, 128, 512]", arg649_1: "f32[32, 128, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[4096]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[4096, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[250112]" = torch.ops.prims.iota.default(250112, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 250112]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[4096, 250112]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[4096, 250112]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[4096, 250112]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[4096, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = None
        mul: "f32[4096, 250112]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "bf16[4096, 250112]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[4096, 250112]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_2: "bf16[4096, 250112]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[4096, 250112]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        sub_1: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_3: "bf16[4096, 250112]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[4096, 250112]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp: "f32[4096, 250112]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_1: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[4096, 250112]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[4096, 250112]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_3: "bf16[32, 128, 250112]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        add: "bf16[32, 128, 250112]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "bf16[4096, 250112]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        permute: "bf16[250112, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm: "bf16[250112, 512]" = torch.ops.aten.mm.default(permute, arg7_1);  permute = arg7_1 = None
        mm_1: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_4, arg8_1);  view_4 = arg8_1 = None
        view_5: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_1, _shape_param_5);  mm_1 = _shape_param_5 = None
        convert_element_type_6: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        convert_element_type_7: "f32[250112, 512]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        convert_element_type_8: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        mul_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_3: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_2);  convert_element_type_6 = mul_2 = None
        mul_4: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_3, arg10_1);  arg10_1 = None
        mul_5: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg11_1, arg12_1)
        mul_6: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_3, mul_5);  mul_3 = mul_5 = None
        sum_2: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_6, [0, 1], True, dtype = torch.float32);  mul_6 = None
        view_6: "f32[512]" = torch.ops.aten.view.default(sum_2, _shape_param_6);  sum_2 = _shape_param_6 = None
        mul_7: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_4, arg11_1)
        mul_8: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_4, arg12_1);  mul_4 = None
        sum_3: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_7, [2], True, dtype = torch.float32);  mul_7 = None
        pow_1: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg12_1, 3);  arg12_1 = None
        mul_9: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_3, -0.5);  sum_3 = None
        mul_10: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_9, pow_1);  mul_9 = pow_1 = None
        expand_1: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_10, _shape_param_7);  mul_10 = _shape_param_7 = None
        div_1: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_1, 512);  expand_1 = None
        pow_2: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg11_1, 1.0);  arg11_1 = None
        mul_11: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        mul_12: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_1, mul_11);  div_1 = mul_11 = None
        add_1: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_8, mul_12);  mul_8 = mul_12 = None
        convert_element_type_9: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16)
        convert_element_type_10: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        mul_13: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_14: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_13);  convert_element_type_9 = mul_13 = None
        view_7: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_14, _shape_param_8);  mul_14 = _shape_param_8 = None
        permute_1: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_7, [1, 0])
        mm_2: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_1, arg14_1);  permute_1 = arg14_1 = None
        mm_3: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_7, arg15_1);  view_7 = arg15_1 = None
        view_8: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_3, _shape_param_9);  mm_3 = _shape_param_9 = None
        convert_element_type_11: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        convert_element_type_12: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_13: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.float32);  arg16_1 = None
        mul_15: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_16: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, mul_15);  convert_element_type_11 = mul_15 = None
        view_9: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg17_1, _shape_param_10);  arg17_1 = _shape_param_10 = None
        mul_17: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_9, 0.5)
        convert_element_type_14: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32)
        pow_3: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_14, 3.0)
        mul_18: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_2: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_9, mul_18);  view_9 = mul_18 = None
        mul_19: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_2, 0.7978845608028654);  add_2 = None
        tanh: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_19);  mul_19 = None
        add_3: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_20: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_17, add_3)
        mul_21: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_16, mul_20);  mul_20 = None
        view_10: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg18_1, _shape_param_11);  arg18_1 = _shape_param_11 = None
        mul_22: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_16, view_10);  mul_16 = view_10 = None
        convert_element_type_15: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_21, torch.bfloat16);  mul_21 = None
        view_11: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_15, _shape_param_12);  convert_element_type_15 = _shape_param_12 = None
        permute_2: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_4: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_2, arg19_1);  permute_2 = None
        mm_5: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_11, arg20_1);  view_11 = arg20_1 = None
        view_12: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_5, _shape_param_13);  mm_5 = _shape_param_13 = None
        convert_element_type_16: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        convert_element_type_17: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        mul_23: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_22, mul_17);  mul_17 = None
        mul_24: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_22, add_3);  mul_22 = add_3 = None
        convert_element_type_18: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        mul_25: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_3: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_25);  mul_25 = None
        mul_26: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_23, sub_3);  mul_23 = sub_3 = None
        mul_27: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_26, 0.7978845608028654);  mul_26 = None
        convert_element_type_19: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_27, torch.bfloat16)
        mul_28: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_27, 0.044715);  mul_27 = None
        pow_4: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_14, 2.0);  convert_element_type_14 = None
        mul_29: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_4, 3.0);  pow_4 = None
        mul_30: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_28, mul_29);  mul_28 = mul_29 = None
        convert_element_type_20: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_30, torch.bfloat16);  mul_30 = None
        add_4: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_19, convert_element_type_20);  convert_element_type_19 = convert_element_type_20 = None
        mul_31: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 0.5);  convert_element_type_18 = None
        add_5: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_4, mul_31);  add_4 = mul_31 = None
        view_13: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_5, _shape_param_14);  add_5 = _shape_param_14 = None
        permute_3: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_13, [1, 0])
        mm_6: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_3, arg19_1);  permute_3 = arg19_1 = None
        mm_7: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_13, arg21_1);  view_13 = arg21_1 = None
        view_14: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_7, _shape_param_15);  mm_7 = _shape_param_15 = None
        convert_element_type_21: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32);  view_14 = None
        add_6: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_16, convert_element_type_21);  convert_element_type_16 = convert_element_type_21 = None
        convert_element_type_22: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        mul_32: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_6, arg22_1);  arg22_1 = None
        mul_33: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg23_1, arg24_1)
        mul_34: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_6, mul_33);  add_6 = mul_33 = None
        sum_4: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_34, [0, 1], True, dtype = torch.float32);  mul_34 = None
        view_15: "f32[512]" = torch.ops.aten.view.default(sum_4, _shape_param_16);  sum_4 = _shape_param_16 = None
        mul_35: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_32, arg23_1)
        mul_36: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_32, arg24_1);  mul_32 = None
        sum_5: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True, dtype = torch.float32);  mul_35 = None
        add_7: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_1, mul_36);  add_1 = mul_36 = None
        pow_5: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg24_1, 3);  arg24_1 = None
        mul_37: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_5, -0.5);  sum_5 = None
        mul_38: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_37, pow_5);  mul_37 = pow_5 = None
        expand_2: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_38, _shape_param_17);  mul_38 = _shape_param_17 = None
        div_2: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_2, 512);  expand_2 = None
        pow_6: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg23_1, 1.0);  arg23_1 = None
        mul_39: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_6, 2.0);  pow_6 = None
        mul_40: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_2, mul_39);  div_2 = mul_39 = None
        add_8: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_7, mul_40);  add_7 = mul_40 = None
        convert_element_type_23: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16)
        convert_element_type_24: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg25_1, torch.bfloat16);  arg25_1 = None
        mul_41: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_42: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_23, mul_41);  convert_element_type_23 = mul_41 = None
        view_16: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_42, _shape_param_18);  mul_42 = _shape_param_18 = None
        permute_4: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_16, [1, 0])
        mm_8: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_4, arg26_1);  permute_4 = arg26_1 = None
        mm_9: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_16, arg27_1);  view_16 = arg27_1 = None
        view_17: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_9, _shape_param_19);  mm_9 = _shape_param_19 = None
        convert_element_type_25: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        view_18: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_17, _shape_param_20);  view_17 = _shape_param_20 = None
        permute_5: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_19: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone, _shape_param_21);  clone = _shape_param_21 = None
        bmm: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg28_1, view_19);  arg28_1 = None
        bmm_1: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_19, arg29_1);  view_19 = arg29_1 = None
        view_20: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm, _shape_param_22);  bmm = _shape_param_22 = None
        view_21: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_23);  bmm_1 = _shape_param_23 = None
        convert_element_type_26: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.bfloat16);  arg30_1 = None
        mul_43: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_44: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_21, mul_43);  view_21 = mul_43 = None
        convert_element_type_27: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_44, torch.float32);  mul_44 = None
        view_22: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg31_1, _shape_param_24);  arg31_1 = _shape_param_24 = None
        convert_element_type_28: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_22, torch.bfloat16);  view_22 = None
        convert_element_type_29: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_28, torch.float32);  convert_element_type_28 = None
        sub_4: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_29, arg32_1);  convert_element_type_29 = arg32_1 = None
        exp_1: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div_3: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, arg33_1);  exp_1 = arg33_1 = None
        mul_45: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_27, div_3);  convert_element_type_27 = None
        sum_6: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_45, [-1], True)
        neg: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg, sum_6, mul_45);  neg = sum_6 = mul_45 = None
        convert_element_type_30: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_23: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_30, _shape_param_25);  convert_element_type_30 = _shape_param_25 = None
        bmm_2: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg34_1, view_23);  arg34_1 = None
        bmm_3: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_23, arg35_1);  view_23 = arg35_1 = None
        view_24: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_26);  bmm_2 = _shape_param_26 = None
        view_25: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_27);  bmm_3 = _shape_param_27 = None
        permute_6: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_24, [0, 1, 3, 2]);  view_24 = None
        permute_7: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_20, [0, 2, 1, 3]);  view_20 = None
        clone_1: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_26: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_1, _shape_param_28);  clone_1 = _shape_param_28 = None
        view_27: "bf16[4096, 384]" = torch.ops.aten.view.default(view_26, _shape_param_29);  view_26 = _shape_param_29 = None
        permute_8: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_27, [1, 0])
        mm_10: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_8, arg36_1);  permute_8 = None
        mm_11: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_27, arg37_1);  view_27 = arg37_1 = None
        view_28: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_11, _shape_param_30);  mm_11 = _shape_param_30 = None
        convert_element_type_31: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_28, torch.float32);  view_28 = None
        add_9: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(arg38_1, convert_element_type_31);  arg38_1 = convert_element_type_31 = None
        convert_element_type_32: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        permute_9: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_6, [0, 2, 1, 3]);  permute_6 = None
        view_29: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_9, _shape_param_31);  permute_9 = _shape_param_31 = None
        clone_2: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_29, memory_format = torch.contiguous_format);  view_29 = None
        view_30: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_2, _shape_param_32);  clone_2 = _shape_param_32 = None
        permute_10: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_30, [1, 0])
        mm_12: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_10, arg36_1);  permute_10 = None
        mm_13: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_30, arg39_1);  view_30 = arg39_1 = None
        view_31: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_13, _shape_param_33);  mm_13 = _shape_param_33 = None
        convert_element_type_33: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        add_10: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_9, convert_element_type_33);  add_9 = convert_element_type_33 = None
        convert_element_type_34: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        permute_11: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
        clone_3: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
        view_32: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_3, _shape_param_34);  clone_3 = _shape_param_34 = None
        view_33: "bf16[4096, 384]" = torch.ops.aten.view.default(view_32, _shape_param_35);  view_32 = _shape_param_35 = None
        permute_12: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_33, [1, 0])
        mm_14: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_12, arg40_1);  permute_12 = arg40_1 = None
        mm_15: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_33, arg41_1);  view_33 = arg41_1 = None
        view_34: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_15, _shape_param_36);  mm_15 = _shape_param_36 = None
        convert_element_type_35: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        convert_element_type_36: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        mul_46: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_35, arg42_1);  arg42_1 = None
        mul_47: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg43_1, arg44_1)
        mul_48: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_35, mul_47);  convert_element_type_35 = mul_47 = None
        sum_7: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_48, [0, 1], True, dtype = torch.float32);  mul_48 = None
        view_35: "f32[512]" = torch.ops.aten.view.default(sum_7, _shape_param_37);  sum_7 = _shape_param_37 = None
        mul_49: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_46, arg43_1)
        mul_50: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_46, arg44_1);  mul_46 = None
        sum_8: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_49, [2], True, dtype = torch.float32);  mul_49 = None
        add_11: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_8, mul_50);  add_8 = mul_50 = None
        pow_7: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg44_1, 3);  arg44_1 = None
        mul_51: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_8, -0.5);  sum_8 = None
        mul_52: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_51, pow_7);  mul_51 = pow_7 = None
        expand_3: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_52, _shape_param_38);  mul_52 = _shape_param_38 = None
        div_4: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_3, 512);  expand_3 = None
        pow_8: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg43_1, 1.0);  arg43_1 = None
        mul_53: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_8, 2.0);  pow_8 = None
        mul_54: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_4, mul_53);  div_4 = mul_53 = None
        add_12: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_11, mul_54);  add_11 = mul_54 = None
        convert_element_type_37: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16)
        convert_element_type_38: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        mul_55: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
        mul_56: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_37, mul_55);  convert_element_type_37 = mul_55 = None
        view_36: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_56, _shape_param_39);  mul_56 = _shape_param_39 = None
        permute_13: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_36, [1, 0])
        mm_16: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_13, arg46_1);  permute_13 = arg46_1 = None
        mm_17: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_36, arg47_1);  view_36 = arg47_1 = None
        view_37: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_17, _shape_param_40);  mm_17 = _shape_param_40 = None
        convert_element_type_39: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_16, torch.float32);  mm_16 = None
        view_38: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_37, _shape_param_41);  view_37 = _shape_param_41 = None
        permute_14: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_4: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_14, memory_format = torch.contiguous_format);  permute_14 = None
        view_39: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_42);  clone_4 = _shape_param_42 = None
        bmm_4: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg48_1, view_39);  arg48_1 = None
        bmm_5: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_39, arg49_1);  view_39 = arg49_1 = None
        view_40: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_4, _shape_param_43);  bmm_4 = _shape_param_43 = None
        view_41: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_5, _shape_param_44);  bmm_5 = _shape_param_44 = None
        convert_element_type_40: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg50_1, torch.bfloat16);  arg50_1 = None
        mul_57: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
        mul_58: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_41, mul_57);  view_41 = mul_57 = None
        convert_element_type_41: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_58, torch.float32);  mul_58 = None
        convert_element_type_42: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.float32);  arg51_1 = None
        sub_5: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_42, arg52_1);  convert_element_type_42 = arg52_1 = None
        exp_2: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        div_5: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, arg53_1);  exp_2 = arg53_1 = None
        mul_59: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_41, div_5);  convert_element_type_41 = None
        sum_9: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_59, [-1], True)
        neg_1: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_1: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_1, sum_9, mul_59);  neg_1 = sum_9 = mul_59 = None
        convert_element_type_43: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        view_42: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_43, _shape_param_45);  convert_element_type_43 = _shape_param_45 = None
        view_43: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_42, _shape_param_46);  view_42 = _shape_param_46 = None
        view_44: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_43, _shape_param_47);  _shape_param_47 = None
        convert_element_type_44: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        bmm_6: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg54_1, view_44);  arg54_1 = None
        bmm_7: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_44, arg55_1);  view_44 = arg55_1 = None
        view_45: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_48);  bmm_6 = _shape_param_48 = None
        view_46: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_49);  bmm_7 = _shape_param_49 = None
        permute_15: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_45, [0, 1, 3, 2]);  view_45 = None
        permute_16: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        clone_5: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
        view_47: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_5, _shape_param_50);  clone_5 = _shape_param_50 = None
        view_48: "bf16[4096, 384]" = torch.ops.aten.view.default(view_47, _shape_param_51);  view_47 = _shape_param_51 = None
        permute_17: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_48, [1, 0])
        mm_18: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_17, arg56_1);  permute_17 = None
        mm_19: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_48, arg57_1);  view_48 = arg57_1 = None
        view_49: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_19, _shape_param_52);  mm_19 = _shape_param_52 = None
        convert_element_type_45: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None
        convert_element_type_46: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        permute_18: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_15, [0, 2, 1, 3]);  permute_15 = None
        view_50: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_18, _shape_param_53);  permute_18 = _shape_param_53 = None
        clone_6: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_50, memory_format = torch.contiguous_format);  view_50 = None
        view_51: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_6, _shape_param_54);  clone_6 = _shape_param_54 = None
        permute_19: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_51, [1, 0])
        mm_20: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_19, arg56_1);  permute_19 = None
        mm_21: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_51, arg58_1);  view_51 = arg58_1 = None
        view_52: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_21, _shape_param_55);  mm_21 = _shape_param_55 = None
        convert_element_type_47: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_52, torch.float32);  view_52 = None
        add_13: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_45, convert_element_type_47);  convert_element_type_45 = convert_element_type_47 = None
        convert_element_type_48: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        permute_20: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        clone_7: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
        view_53: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_7, _shape_param_56);  clone_7 = _shape_param_56 = None
        view_54: "bf16[4096, 384]" = torch.ops.aten.view.default(view_53, _shape_param_57);  view_53 = _shape_param_57 = None
        permute_21: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_54, [1, 0])
        mm_22: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_21, arg56_1);  permute_21 = arg56_1 = None
        mm_23: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_54, arg59_1);  view_54 = arg59_1 = None
        view_55: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_23, _shape_param_58);  mm_23 = _shape_param_58 = None
        convert_element_type_49: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        add_14: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_13, convert_element_type_49);  add_13 = convert_element_type_49 = None
        convert_element_type_50: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        mul_60: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_14, arg60_1);  arg60_1 = None
        mul_61: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg61_1, arg62_1)
        mul_62: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_14, mul_61);  add_14 = mul_61 = None
        sum_10: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_62, [0, 1], True, dtype = torch.float32);  mul_62 = None
        view_56: "f32[512]" = torch.ops.aten.view.default(sum_10, _shape_param_59);  sum_10 = _shape_param_59 = None
        mul_63: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_60, arg61_1)
        mul_64: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_60, arg62_1);  mul_60 = None
        sum_11: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_63, [2], True, dtype = torch.float32);  mul_63 = None
        add_15: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_12, mul_64);  add_12 = mul_64 = None
        pow_9: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg62_1, 3);  arg62_1 = None
        mul_65: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_11, -0.5);  sum_11 = None
        mul_66: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_65, pow_9);  mul_65 = pow_9 = None
        expand_4: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_66, _shape_param_60);  mul_66 = _shape_param_60 = None
        div_6: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_4, 512);  expand_4 = None
        pow_10: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg61_1, 1.0);  arg61_1 = None
        mul_67: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_10, 2.0);  pow_10 = None
        mul_68: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_6, mul_67);  div_6 = mul_67 = None
        add_16: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_15, mul_68);  add_15 = mul_68 = None
        convert_element_type_51: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16)
        convert_element_type_52: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg63_1, torch.bfloat16);  arg63_1 = None
        mul_69: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_70: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_51, mul_69);  convert_element_type_51 = mul_69 = None
        view_57: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_70, _shape_param_61);  mul_70 = _shape_param_61 = None
        permute_22: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_57, [1, 0])
        mm_24: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_22, arg64_1);  permute_22 = arg64_1 = None
        mm_25: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_57, arg65_1);  view_57 = arg65_1 = None
        view_58: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_25, _shape_param_62);  mm_25 = _shape_param_62 = None
        convert_element_type_53: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_58, torch.float32);  view_58 = None
        convert_element_type_54: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        convert_element_type_55: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg66_1, torch.float32);  arg66_1 = None
        mul_71: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
        mul_72: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_53, mul_71);  convert_element_type_53 = mul_71 = None
        view_59: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg67_1, _shape_param_63);  arg67_1 = _shape_param_63 = None
        mul_73: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_59, 0.5)
        convert_element_type_56: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_59, torch.float32)
        pow_11: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_56, 3.0)
        mul_74: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_17: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_59, mul_74);  view_59 = mul_74 = None
        mul_75: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_75);  mul_75 = None
        add_18: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_76: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_73, add_18)
        mul_77: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_72, mul_76);  mul_76 = None
        view_60: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg68_1, _shape_param_64);  arg68_1 = _shape_param_64 = None
        mul_78: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_72, view_60);  mul_72 = view_60 = None
        convert_element_type_57: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_77, torch.bfloat16);  mul_77 = None
        view_61: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_57, _shape_param_65);  convert_element_type_57 = _shape_param_65 = None
        permute_23: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_61, [1, 0])
        mm_26: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_23, arg69_1);  permute_23 = None
        mm_27: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_61, arg70_1);  view_61 = arg70_1 = None
        view_62: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_27, _shape_param_66);  mm_27 = _shape_param_66 = None
        convert_element_type_58: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_62, torch.float32);  view_62 = None
        convert_element_type_59: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_26, torch.float32);  mm_26 = None
        mul_79: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_78, mul_73);  mul_73 = None
        mul_80: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_78, add_18);  mul_78 = add_18 = None
        convert_element_type_60: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_80, torch.bfloat16);  mul_80 = None
        mul_81: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_6: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_81);  mul_81 = None
        mul_82: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_79, sub_6);  mul_79 = sub_6 = None
        mul_83: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_82, 0.7978845608028654);  mul_82 = None
        convert_element_type_61: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16)
        mul_84: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_83, 0.044715);  mul_83 = None
        pow_12: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_56, 2.0);  convert_element_type_56 = None
        mul_85: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_12, 3.0);  pow_12 = None
        mul_86: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
        convert_element_type_62: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16);  mul_86 = None
        add_19: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_61, convert_element_type_62);  convert_element_type_61 = convert_element_type_62 = None
        mul_87: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 0.5);  convert_element_type_60 = None
        add_20: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_19, mul_87);  add_19 = mul_87 = None
        view_63: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_20, _shape_param_67);  add_20 = _shape_param_67 = None
        permute_24: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_63, [1, 0])
        mm_28: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_24, arg69_1);  permute_24 = arg69_1 = None
        mm_29: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_63, arg71_1);  view_63 = arg71_1 = None
        view_64: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_29, _shape_param_68);  mm_29 = _shape_param_68 = None
        convert_element_type_63: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_64, torch.float32);  view_64 = None
        add_21: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_58, convert_element_type_63);  convert_element_type_58 = convert_element_type_63 = None
        convert_element_type_64: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_28, torch.float32);  mm_28 = None
        mul_88: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_21, arg72_1);  arg72_1 = None
        mul_89: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg73_1, arg74_1)
        mul_90: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_21, mul_89);  add_21 = mul_89 = None
        sum_12: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_90, [0, 1], True, dtype = torch.float32);  mul_90 = None
        view_65: "f32[512]" = torch.ops.aten.view.default(sum_12, _shape_param_69);  sum_12 = _shape_param_69 = None
        mul_91: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_88, arg73_1)
        mul_92: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_88, arg74_1);  mul_88 = None
        sum_13: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_91, [2], True, dtype = torch.float32);  mul_91 = None
        add_22: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_16, mul_92);  add_16 = mul_92 = None
        pow_13: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg74_1, 3);  arg74_1 = None
        mul_93: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_13, -0.5);  sum_13 = None
        mul_94: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_93, pow_13);  mul_93 = pow_13 = None
        expand_5: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_94, _shape_param_70);  mul_94 = _shape_param_70 = None
        div_7: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_5, 512);  expand_5 = None
        pow_14: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg73_1, 1.0);  arg73_1 = None
        mul_95: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_14, 2.0);  pow_14 = None
        mul_96: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_7, mul_95);  div_7 = mul_95 = None
        add_23: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_22, mul_96);  add_22 = mul_96 = None
        convert_element_type_65: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16)
        convert_element_type_66: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg75_1, torch.bfloat16);  arg75_1 = None
        mul_97: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_98: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_65, mul_97);  convert_element_type_65 = mul_97 = None
        view_66: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_98, _shape_param_71);  mul_98 = _shape_param_71 = None
        permute_25: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_66, [1, 0])
        mm_30: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_25, arg76_1);  permute_25 = arg76_1 = None
        mm_31: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_66, arg77_1);  view_66 = arg77_1 = None
        view_67: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_31, _shape_param_72);  mm_31 = _shape_param_72 = None
        convert_element_type_67: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_30, torch.float32);  mm_30 = None
        view_68: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_67, _shape_param_73);  view_67 = _shape_param_73 = None
        permute_26: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_8: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_69: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_74);  clone_8 = _shape_param_74 = None
        bmm_8: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg78_1, view_69);  arg78_1 = None
        bmm_9: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_69, arg79_1);  view_69 = arg79_1 = None
        view_70: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_8, _shape_param_75);  bmm_8 = _shape_param_75 = None
        view_71: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_9, _shape_param_76);  bmm_9 = _shape_param_76 = None
        convert_element_type_68: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.bfloat16);  arg80_1 = None
        mul_99: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_100: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_71, mul_99);  view_71 = mul_99 = None
        convert_element_type_69: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_100, torch.float32);  mul_100 = None
        view_72: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg81_1, _shape_param_77);  arg81_1 = _shape_param_77 = None
        convert_element_type_70: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_72, torch.bfloat16);  view_72 = None
        convert_element_type_71: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_70, torch.float32);  convert_element_type_70 = None
        sub_7: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_71, arg82_1);  convert_element_type_71 = arg82_1 = None
        exp_3: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        div_8: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, arg83_1);  exp_3 = arg83_1 = None
        mul_101: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_69, div_8);  convert_element_type_69 = None
        sum_14: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [-1], True)
        neg_2: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_2: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_2, sum_14, mul_101);  neg_2 = sum_14 = mul_101 = None
        convert_element_type_72: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        view_73: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_72, _shape_param_78);  convert_element_type_72 = _shape_param_78 = None
        bmm_10: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg84_1, view_73);  arg84_1 = None
        bmm_11: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_73, arg85_1);  view_73 = arg85_1 = None
        view_74: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_79);  bmm_10 = _shape_param_79 = None
        view_75: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_80);  bmm_11 = _shape_param_80 = None
        permute_27: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 1, 3, 2]);  view_74 = None
        permute_28: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_9: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_76: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_9, _shape_param_81);  clone_9 = _shape_param_81 = None
        view_77: "bf16[4096, 384]" = torch.ops.aten.view.default(view_76, _shape_param_82);  view_76 = _shape_param_82 = None
        permute_29: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_77, [1, 0])
        mm_32: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_29, arg36_1);  permute_29 = None
        mm_33: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_77, arg86_1);  view_77 = arg86_1 = None
        view_78: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_33, _shape_param_83);  mm_33 = _shape_param_83 = None
        convert_element_type_73: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_78, torch.float32);  view_78 = None
        add_24: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_10, convert_element_type_73);  add_10 = convert_element_type_73 = None
        convert_element_type_74: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_32, torch.float32);  mm_32 = None
        permute_30: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_27, [0, 2, 1, 3]);  permute_27 = None
        view_79: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_30, _shape_param_84);  permute_30 = _shape_param_84 = None
        clone_10: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_79, memory_format = torch.contiguous_format);  view_79 = None
        view_80: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_10, _shape_param_85);  clone_10 = _shape_param_85 = None
        permute_31: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_80, [1, 0])
        mm_34: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_31, arg36_1);  permute_31 = None
        mm_35: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_80, arg87_1);  view_80 = arg87_1 = None
        view_81: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_35, _shape_param_86);  mm_35 = _shape_param_86 = None
        convert_element_type_75: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_81, torch.float32);  view_81 = None
        add_25: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_24, convert_element_type_75);  add_24 = convert_element_type_75 = None
        convert_element_type_76: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        permute_32: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
        clone_11: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
        view_82: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_11, _shape_param_87);  clone_11 = _shape_param_87 = None
        view_83: "bf16[4096, 384]" = torch.ops.aten.view.default(view_82, _shape_param_88);  view_82 = _shape_param_88 = None
        permute_33: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_83, [1, 0])
        mm_36: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_33, arg88_1);  permute_33 = arg88_1 = None
        mm_37: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_83, arg89_1);  view_83 = arg89_1 = None
        view_84: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_37, _shape_param_89);  mm_37 = _shape_param_89 = None
        convert_element_type_77: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_84, torch.float32);  view_84 = None
        convert_element_type_78: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        mul_102: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_77, arg90_1);  arg90_1 = None
        mul_103: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg91_1, arg92_1)
        mul_104: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_77, mul_103);  convert_element_type_77 = mul_103 = None
        sum_15: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_104, [0, 1], True, dtype = torch.float32);  mul_104 = None
        view_85: "f32[512]" = torch.ops.aten.view.default(sum_15, _shape_param_90);  sum_15 = _shape_param_90 = None
        mul_105: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_102, arg91_1)
        mul_106: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_102, arg92_1);  mul_102 = None
        sum_16: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_105, [2], True, dtype = torch.float32);  mul_105 = None
        add_26: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_23, mul_106);  add_23 = mul_106 = None
        pow_15: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg92_1, 3);  arg92_1 = None
        mul_107: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_16, -0.5);  sum_16 = None
        mul_108: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_107, pow_15);  mul_107 = pow_15 = None
        expand_6: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_108, _shape_param_91);  mul_108 = _shape_param_91 = None
        div_9: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_6, 512);  expand_6 = None
        pow_16: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg91_1, 1.0);  arg91_1 = None
        mul_109: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_16, 2.0);  pow_16 = None
        mul_110: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_9, mul_109);  div_9 = mul_109 = None
        add_27: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_26, mul_110);  add_26 = mul_110 = None
        convert_element_type_79: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_27, torch.bfloat16)
        convert_element_type_80: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        mul_111: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 1.1111111111111112);  convert_element_type_80 = None
        mul_112: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_79, mul_111);  convert_element_type_79 = mul_111 = None
        view_86: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_112, _shape_param_92);  mul_112 = _shape_param_92 = None
        permute_34: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_86, [1, 0])
        mm_38: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_34, arg94_1);  permute_34 = arg94_1 = None
        mm_39: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_86, arg95_1);  view_86 = arg95_1 = None
        view_87: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_39, _shape_param_93);  mm_39 = _shape_param_93 = None
        convert_element_type_81: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_38, torch.float32);  mm_38 = None
        view_88: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_87, _shape_param_94);  view_87 = _shape_param_94 = None
        permute_35: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        clone_12: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_89: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_95);  clone_12 = _shape_param_95 = None
        bmm_12: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg96_1, view_89);  arg96_1 = None
        bmm_13: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_89, arg97_1);  view_89 = arg97_1 = None
        view_90: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_12, _shape_param_96);  bmm_12 = _shape_param_96 = None
        view_91: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_13, _shape_param_97);  bmm_13 = _shape_param_97 = None
        convert_element_type_82: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.bfloat16);  arg98_1 = None
        mul_113: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 1.1111111111111112);  convert_element_type_82 = None
        mul_114: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_91, mul_113);  view_91 = mul_113 = None
        convert_element_type_83: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_114, torch.float32);  mul_114 = None
        convert_element_type_84: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg99_1, torch.float32);  arg99_1 = None
        sub_8: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_84, arg100_1);  convert_element_type_84 = arg100_1 = None
        exp_4: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        div_10: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, arg101_1);  exp_4 = arg101_1 = None
        mul_115: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_83, div_10);  convert_element_type_83 = None
        sum_17: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_115, [-1], True)
        neg_3: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_3: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_3, sum_17, mul_115);  neg_3 = sum_17 = mul_115 = None
        convert_element_type_85: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        view_92: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_85, _shape_param_98);  convert_element_type_85 = _shape_param_98 = None
        view_93: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_92, _shape_param_99);  view_92 = _shape_param_99 = None
        view_94: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_93, _shape_param_100);  _shape_param_100 = None
        convert_element_type_86: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        add_28: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_44, convert_element_type_86);  convert_element_type_44 = convert_element_type_86 = None
        bmm_14: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg102_1, view_94);  arg102_1 = None
        bmm_15: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_94, arg103_1);  view_94 = arg103_1 = None
        view_95: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_101);  bmm_14 = _shape_param_101 = None
        view_96: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_102);  bmm_15 = _shape_param_102 = None
        permute_36: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_95, [0, 1, 3, 2]);  view_95 = None
        permute_37: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        clone_13: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_97: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_13, _shape_param_103);  clone_13 = _shape_param_103 = None
        view_98: "bf16[4096, 384]" = torch.ops.aten.view.default(view_97, _shape_param_104);  view_97 = _shape_param_104 = None
        permute_38: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_98, [1, 0])
        mm_40: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_38, arg104_1);  permute_38 = None
        mm_41: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_98, arg105_1);  view_98 = arg105_1 = None
        view_99: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_41, _shape_param_105);  mm_41 = _shape_param_105 = None
        convert_element_type_87: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        convert_element_type_88: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_40, torch.float32);  mm_40 = None
        permute_39: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_36, [0, 2, 1, 3]);  permute_36 = None
        view_100: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_39, _shape_param_106);  permute_39 = _shape_param_106 = None
        clone_14: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_100, memory_format = torch.contiguous_format);  view_100 = None
        view_101: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_14, _shape_param_107);  clone_14 = _shape_param_107 = None
        permute_40: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_101, [1, 0])
        mm_42: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_40, arg104_1);  permute_40 = None
        mm_43: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_101, arg106_1);  view_101 = arg106_1 = None
        view_102: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_43, _shape_param_108);  mm_43 = _shape_param_108 = None
        convert_element_type_89: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_102, torch.float32);  view_102 = None
        add_29: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_87, convert_element_type_89);  convert_element_type_87 = convert_element_type_89 = None
        convert_element_type_90: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_42, torch.float32);  mm_42 = None
        permute_41: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        clone_15: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_103: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_15, _shape_param_109);  clone_15 = _shape_param_109 = None
        view_104: "bf16[4096, 384]" = torch.ops.aten.view.default(view_103, _shape_param_110);  view_103 = _shape_param_110 = None
        permute_42: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_104, [1, 0])
        mm_44: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_42, arg104_1);  permute_42 = arg104_1 = None
        mm_45: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_104, arg107_1);  view_104 = arg107_1 = None
        view_105: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_45, _shape_param_111);  mm_45 = _shape_param_111 = None
        convert_element_type_91: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        add_30: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_29, convert_element_type_91);  add_29 = convert_element_type_91 = None
        convert_element_type_92: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        mul_116: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_30, arg108_1);  arg108_1 = None
        mul_117: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg109_1, arg110_1)
        mul_118: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_30, mul_117);  add_30 = mul_117 = None
        sum_18: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_118, [0, 1], True, dtype = torch.float32);  mul_118 = None
        view_106: "f32[512]" = torch.ops.aten.view.default(sum_18, _shape_param_112);  sum_18 = _shape_param_112 = None
        mul_119: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_116, arg109_1)
        mul_120: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_116, arg110_1);  mul_116 = None
        sum_19: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True, dtype = torch.float32);  mul_119 = None
        add_31: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_27, mul_120);  add_27 = mul_120 = None
        pow_17: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg110_1, 3);  arg110_1 = None
        mul_121: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_19, -0.5);  sum_19 = None
        mul_122: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_121, pow_17);  mul_121 = pow_17 = None
        expand_7: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_122, _shape_param_113);  mul_122 = _shape_param_113 = None
        div_11: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_7, 512);  expand_7 = None
        pow_18: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg109_1, 1.0);  arg109_1 = None
        mul_123: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_18, 2.0);  pow_18 = None
        mul_124: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_11, mul_123);  div_11 = mul_123 = None
        add_32: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_31, mul_124);  add_31 = mul_124 = None
        convert_element_type_93: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16)
        convert_element_type_94: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg111_1, torch.bfloat16);  arg111_1 = None
        mul_125: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 1.1111111111111112);  convert_element_type_94 = None
        mul_126: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_93, mul_125);  convert_element_type_93 = mul_125 = None
        view_107: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_126, _shape_param_114);  mul_126 = _shape_param_114 = None
        permute_43: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_107, [1, 0])
        mm_46: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_43, arg112_1);  permute_43 = arg112_1 = None
        mm_47: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_107, arg113_1);  view_107 = arg113_1 = None
        view_108: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_47, _shape_param_115);  mm_47 = _shape_param_115 = None
        convert_element_type_95: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        convert_element_type_96: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        convert_element_type_97: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.float32);  arg114_1 = None
        mul_127: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_97, 1.1111111111111112);  convert_element_type_97 = None
        mul_128: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_95, mul_127);  convert_element_type_95 = mul_127 = None
        view_109: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg115_1, _shape_param_116);  arg115_1 = _shape_param_116 = None
        mul_129: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        convert_element_type_98: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_109, torch.float32)
        pow_19: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_98, 3.0)
        mul_130: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_33: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_109, mul_130);  view_109 = mul_130 = None
        mul_131: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_2: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_131);  mul_131 = None
        add_34: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_132: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_129, add_34)
        mul_133: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_128, mul_132);  mul_132 = None
        view_110: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg116_1, _shape_param_117);  arg116_1 = _shape_param_117 = None
        mul_134: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_128, view_110);  mul_128 = view_110 = None
        convert_element_type_99: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None
        view_111: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_99, _shape_param_118);  convert_element_type_99 = _shape_param_118 = None
        permute_44: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_111, [1, 0])
        mm_48: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_44, arg117_1);  permute_44 = None
        mm_49: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_111, arg118_1);  view_111 = arg118_1 = None
        view_112: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_49, _shape_param_119);  mm_49 = _shape_param_119 = None
        convert_element_type_100: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_112, torch.float32);  view_112 = None
        convert_element_type_101: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_48, torch.float32);  mm_48 = None
        mul_135: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_134, mul_129);  mul_129 = None
        mul_136: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_134, add_34);  mul_134 = add_34 = None
        convert_element_type_102: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None
        mul_137: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_9: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_137);  mul_137 = None
        mul_138: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_135, sub_9);  mul_135 = sub_9 = None
        mul_139: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_138, 0.7978845608028654);  mul_138 = None
        convert_element_type_103: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_139, torch.bfloat16)
        mul_140: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_139, 0.044715);  mul_139 = None
        pow_20: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_98, 2.0);  convert_element_type_98 = None
        mul_141: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_142: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_140, mul_141);  mul_140 = mul_141 = None
        convert_element_type_104: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_142, torch.bfloat16);  mul_142 = None
        add_35: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_103, convert_element_type_104);  convert_element_type_103 = convert_element_type_104 = None
        mul_143: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_102, 0.5);  convert_element_type_102 = None
        add_36: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_35, mul_143);  add_35 = mul_143 = None
        view_113: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_36, _shape_param_120);  add_36 = _shape_param_120 = None
        permute_45: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_113, [1, 0])
        mm_50: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_45, arg117_1);  permute_45 = arg117_1 = None
        mm_51: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_113, arg119_1);  view_113 = arg119_1 = None
        view_114: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_51, _shape_param_121);  mm_51 = _shape_param_121 = None
        convert_element_type_105: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_114, torch.float32);  view_114 = None
        add_37: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_100, convert_element_type_105);  convert_element_type_100 = convert_element_type_105 = None
        convert_element_type_106: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_50, torch.float32);  mm_50 = None
        mul_144: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_37, arg120_1);  arg120_1 = None
        mul_145: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg121_1, arg122_1)
        mul_146: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_37, mul_145);  add_37 = mul_145 = None
        sum_20: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_146, [0, 1], True, dtype = torch.float32);  mul_146 = None
        view_115: "f32[512]" = torch.ops.aten.view.default(sum_20, _shape_param_122);  sum_20 = _shape_param_122 = None
        mul_147: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_144, arg121_1)
        mul_148: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_144, arg122_1);  mul_144 = None
        sum_21: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True, dtype = torch.float32);  mul_147 = None
        add_38: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_32, mul_148);  add_32 = mul_148 = None
        pow_21: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg122_1, 3);  arg122_1 = None
        mul_149: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_21, -0.5);  sum_21 = None
        mul_150: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_149, pow_21);  mul_149 = pow_21 = None
        expand_8: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_150, _shape_param_123);  mul_150 = _shape_param_123 = None
        div_12: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_8, 512);  expand_8 = None
        pow_22: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg121_1, 1.0);  arg121_1 = None
        mul_151: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_22, 2.0);  pow_22 = None
        mul_152: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_12, mul_151);  div_12 = mul_151 = None
        add_39: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_38, mul_152);  add_38 = mul_152 = None
        convert_element_type_107: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16)
        convert_element_type_108: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.bfloat16);  arg123_1 = None
        mul_153: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_108, 1.1111111111111112);  convert_element_type_108 = None
        mul_154: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_107, mul_153);  convert_element_type_107 = mul_153 = None
        view_116: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_154, _shape_param_124);  mul_154 = _shape_param_124 = None
        permute_46: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_116, [1, 0])
        mm_52: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_46, arg124_1);  permute_46 = arg124_1 = None
        mm_53: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_116, arg125_1);  view_116 = arg125_1 = None
        view_117: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_53, _shape_param_125);  mm_53 = _shape_param_125 = None
        convert_element_type_109: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_52, torch.float32);  mm_52 = None
        view_118: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_117, _shape_param_126);  view_117 = _shape_param_126 = None
        permute_47: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_16: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_119: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_127);  clone_16 = _shape_param_127 = None
        bmm_16: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg126_1, view_119);  arg126_1 = None
        bmm_17: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_119, arg127_1);  view_119 = arg127_1 = None
        view_120: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_16, _shape_param_128);  bmm_16 = _shape_param_128 = None
        view_121: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_17, _shape_param_129);  bmm_17 = _shape_param_129 = None
        convert_element_type_110: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.bfloat16);  arg128_1 = None
        mul_155: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_110, 1.1111111111111112);  convert_element_type_110 = None
        mul_156: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_121, mul_155);  view_121 = mul_155 = None
        convert_element_type_111: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_156, torch.float32);  mul_156 = None
        view_122: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg129_1, _shape_param_130);  arg129_1 = _shape_param_130 = None
        convert_element_type_112: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_122, torch.bfloat16);  view_122 = None
        convert_element_type_113: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_112, torch.float32);  convert_element_type_112 = None
        sub_10: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_113, arg130_1);  convert_element_type_113 = arg130_1 = None
        exp_5: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        div_13: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, arg131_1);  exp_5 = arg131_1 = None
        mul_157: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_111, div_13);  convert_element_type_111 = None
        sum_22: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [-1], True)
        neg_4: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_4: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_4, sum_22, mul_157);  neg_4 = sum_22 = mul_157 = None
        convert_element_type_114: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        view_123: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_114, _shape_param_131);  convert_element_type_114 = _shape_param_131 = None
        bmm_18: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg132_1, view_123);  arg132_1 = None
        bmm_19: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_123, arg133_1);  view_123 = arg133_1 = None
        view_124: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_132);  bmm_18 = _shape_param_132 = None
        view_125: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_133);  bmm_19 = _shape_param_133 = None
        permute_48: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_124, [0, 1, 3, 2]);  view_124 = None
        permute_49: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        clone_17: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_126: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_17, _shape_param_134);  clone_17 = _shape_param_134 = None
        view_127: "bf16[4096, 384]" = torch.ops.aten.view.default(view_126, _shape_param_135);  view_126 = _shape_param_135 = None
        permute_50: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_127, [1, 0])
        mm_54: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_50, arg36_1);  permute_50 = None
        mm_55: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_127, arg134_1);  view_127 = arg134_1 = None
        view_128: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_55, _shape_param_136);  mm_55 = _shape_param_136 = None
        convert_element_type_115: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_128, torch.float32);  view_128 = None
        add_40: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_25, convert_element_type_115);  add_25 = convert_element_type_115 = None
        convert_element_type_116: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        permute_51: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_48, [0, 2, 1, 3]);  permute_48 = None
        view_129: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_51, _shape_param_137);  permute_51 = _shape_param_137 = None
        clone_18: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_129, memory_format = torch.contiguous_format);  view_129 = None
        view_130: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_18, _shape_param_138);  clone_18 = _shape_param_138 = None
        permute_52: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_130, [1, 0])
        mm_56: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_52, arg36_1);  permute_52 = None
        mm_57: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_130, arg135_1);  view_130 = arg135_1 = None
        view_131: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_57, _shape_param_139);  mm_57 = _shape_param_139 = None
        convert_element_type_117: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None
        add_41: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_40, convert_element_type_117);  add_40 = convert_element_type_117 = None
        convert_element_type_118: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        permute_53: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
        clone_19: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
        view_132: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_19, _shape_param_140);  clone_19 = _shape_param_140 = None
        view_133: "bf16[4096, 384]" = torch.ops.aten.view.default(view_132, _shape_param_141);  view_132 = _shape_param_141 = None
        permute_54: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_133, [1, 0])
        mm_58: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_54, arg136_1);  permute_54 = arg136_1 = None
        mm_59: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_133, arg137_1);  view_133 = arg137_1 = None
        view_134: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_59, _shape_param_142);  mm_59 = _shape_param_142 = None
        convert_element_type_119: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_134, torch.float32);  view_134 = None
        convert_element_type_120: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        mul_158: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_119, arg138_1);  arg138_1 = None
        mul_159: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg139_1, arg140_1)
        mul_160: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_119, mul_159);  convert_element_type_119 = mul_159 = None
        sum_23: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1], True, dtype = torch.float32);  mul_160 = None
        view_135: "f32[512]" = torch.ops.aten.view.default(sum_23, _shape_param_143);  sum_23 = _shape_param_143 = None
        mul_161: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_158, arg139_1)
        mul_162: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_158, arg140_1);  mul_158 = None
        sum_24: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_161, [2], True, dtype = torch.float32);  mul_161 = None
        add_42: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_39, mul_162);  add_39 = mul_162 = None
        pow_23: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg140_1, 3);  arg140_1 = None
        mul_163: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_24, -0.5);  sum_24 = None
        mul_164: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_163, pow_23);  mul_163 = pow_23 = None
        expand_9: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_164, _shape_param_144);  mul_164 = _shape_param_144 = None
        div_14: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_9, 512);  expand_9 = None
        pow_24: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg139_1, 1.0);  arg139_1 = None
        mul_165: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_24, 2.0);  pow_24 = None
        mul_166: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_14, mul_165);  div_14 = mul_165 = None
        add_43: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_42, mul_166);  add_42 = mul_166 = None
        convert_element_type_121: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16)
        convert_element_type_122: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.bfloat16);  arg141_1 = None
        mul_167: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_122, 1.1111111111111112);  convert_element_type_122 = None
        mul_168: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_121, mul_167);  convert_element_type_121 = mul_167 = None
        view_136: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_168, _shape_param_145);  mul_168 = _shape_param_145 = None
        permute_55: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_136, [1, 0])
        mm_60: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_55, arg142_1);  permute_55 = arg142_1 = None
        mm_61: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_136, arg143_1);  view_136 = arg143_1 = None
        view_137: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_61, _shape_param_146);  mm_61 = _shape_param_146 = None
        convert_element_type_123: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        view_138: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_137, _shape_param_147);  view_137 = _shape_param_147 = None
        permute_56: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_20: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
        view_139: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_148);  clone_20 = _shape_param_148 = None
        bmm_20: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg144_1, view_139);  arg144_1 = None
        bmm_21: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_139, arg145_1);  view_139 = arg145_1 = None
        view_140: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_20, _shape_param_149);  bmm_20 = _shape_param_149 = None
        view_141: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_21, _shape_param_150);  bmm_21 = _shape_param_150 = None
        convert_element_type_124: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg146_1, torch.bfloat16);  arg146_1 = None
        mul_169: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 1.1111111111111112);  convert_element_type_124 = None
        mul_170: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_141, mul_169);  view_141 = mul_169 = None
        convert_element_type_125: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_170, torch.float32);  mul_170 = None
        convert_element_type_126: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.float32);  arg147_1 = None
        sub_11: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_126, arg148_1);  convert_element_type_126 = arg148_1 = None
        exp_6: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        div_15: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, arg149_1);  exp_6 = arg149_1 = None
        mul_171: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_125, div_15);  convert_element_type_125 = None
        sum_25: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [-1], True)
        neg_5: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_5: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_5, sum_25, mul_171);  neg_5 = sum_25 = mul_171 = None
        convert_element_type_127: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        view_142: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_127, _shape_param_151);  convert_element_type_127 = _shape_param_151 = None
        view_143: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_142, _shape_param_152);  view_142 = _shape_param_152 = None
        view_144: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_143, _shape_param_153);  _shape_param_153 = None
        convert_element_type_128: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        add_44: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_28, convert_element_type_128);  add_28 = convert_element_type_128 = None
        bmm_22: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg150_1, view_144);  arg150_1 = None
        bmm_23: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_144, arg151_1);  view_144 = arg151_1 = None
        view_145: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_154);  bmm_22 = _shape_param_154 = None
        view_146: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_155);  bmm_23 = _shape_param_155 = None
        permute_57: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_145, [0, 1, 3, 2]);  view_145 = None
        permute_58: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
        clone_21: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_58, memory_format = torch.contiguous_format);  permute_58 = None
        view_147: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_21, _shape_param_156);  clone_21 = _shape_param_156 = None
        view_148: "bf16[4096, 384]" = torch.ops.aten.view.default(view_147, _shape_param_157);  view_147 = _shape_param_157 = None
        permute_59: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_148, [1, 0])
        mm_62: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_59, arg152_1);  permute_59 = None
        mm_63: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_148, arg153_1);  view_148 = arg153_1 = None
        view_149: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_63, _shape_param_158);  mm_63 = _shape_param_158 = None
        convert_element_type_129: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_149, torch.float32);  view_149 = None
        convert_element_type_130: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_62, torch.float32);  mm_62 = None
        permute_60: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_57, [0, 2, 1, 3]);  permute_57 = None
        view_150: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_60, _shape_param_159);  permute_60 = _shape_param_159 = None
        clone_22: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_150, memory_format = torch.contiguous_format);  view_150 = None
        view_151: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_22, _shape_param_160);  clone_22 = _shape_param_160 = None
        permute_61: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_151, [1, 0])
        mm_64: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_61, arg152_1);  permute_61 = None
        mm_65: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_151, arg154_1);  view_151 = arg154_1 = None
        view_152: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_65, _shape_param_161);  mm_65 = _shape_param_161 = None
        convert_element_type_131: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_152, torch.float32);  view_152 = None
        add_45: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_129, convert_element_type_131);  convert_element_type_129 = convert_element_type_131 = None
        convert_element_type_132: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_64, torch.float32);  mm_64 = None
        permute_62: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_23: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
        view_153: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_23, _shape_param_162);  clone_23 = _shape_param_162 = None
        view_154: "bf16[4096, 384]" = torch.ops.aten.view.default(view_153, _shape_param_163);  view_153 = _shape_param_163 = None
        permute_63: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_154, [1, 0])
        mm_66: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_63, arg152_1);  permute_63 = arg152_1 = None
        mm_67: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_154, arg155_1);  view_154 = arg155_1 = None
        view_155: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_67, _shape_param_164);  mm_67 = _shape_param_164 = None
        convert_element_type_133: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_155, torch.float32);  view_155 = None
        add_46: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_45, convert_element_type_133);  add_45 = convert_element_type_133 = None
        convert_element_type_134: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        mul_172: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_46, arg156_1);  arg156_1 = None
        mul_173: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg157_1, arg158_1)
        mul_174: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_46, mul_173);  add_46 = mul_173 = None
        sum_26: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 1], True, dtype = torch.float32);  mul_174 = None
        view_156: "f32[512]" = torch.ops.aten.view.default(sum_26, _shape_param_165);  sum_26 = _shape_param_165 = None
        mul_175: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_172, arg157_1)
        mul_176: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_172, arg158_1);  mul_172 = None
        sum_27: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_175, [2], True, dtype = torch.float32);  mul_175 = None
        add_47: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_43, mul_176);  add_43 = mul_176 = None
        pow_25: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg158_1, 3);  arg158_1 = None
        mul_177: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_27, -0.5);  sum_27 = None
        mul_178: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_177, pow_25);  mul_177 = pow_25 = None
        expand_10: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_178, _shape_param_166);  mul_178 = _shape_param_166 = None
        div_16: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_10, 512);  expand_10 = None
        pow_26: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg157_1, 1.0);  arg157_1 = None
        mul_179: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_26, 2.0);  pow_26 = None
        mul_180: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_16, mul_179);  div_16 = mul_179 = None
        add_48: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_47, mul_180);  add_47 = mul_180 = None
        convert_element_type_135: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16)
        convert_element_type_136: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg159_1, torch.bfloat16);  arg159_1 = None
        mul_181: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_136, 1.1111111111111112);  convert_element_type_136 = None
        mul_182: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_135, mul_181);  convert_element_type_135 = mul_181 = None
        view_157: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_182, _shape_param_167);  mul_182 = _shape_param_167 = None
        permute_64: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_157, [1, 0])
        mm_68: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_64, arg160_1);  permute_64 = arg160_1 = None
        mm_69: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_157, arg161_1);  view_157 = arg161_1 = None
        view_158: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_69, _shape_param_168);  mm_69 = _shape_param_168 = None
        convert_element_type_137: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None
        convert_element_type_138: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        convert_element_type_139: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.float32);  arg162_1 = None
        mul_183: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_139, 1.1111111111111112);  convert_element_type_139 = None
        mul_184: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_137, mul_183);  convert_element_type_137 = mul_183 = None
        view_159: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg163_1, _shape_param_169);  arg163_1 = _shape_param_169 = None
        mul_185: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_159, 0.5)
        convert_element_type_140: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32)
        pow_27: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_140, 3.0)
        mul_186: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_49: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_159, mul_186);  view_159 = mul_186 = None
        mul_187: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_3: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_187);  mul_187 = None
        add_50: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_188: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_185, add_50)
        mul_189: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_184, mul_188);  mul_188 = None
        view_160: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg164_1, _shape_param_170);  arg164_1 = _shape_param_170 = None
        mul_190: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_184, view_160);  mul_184 = view_160 = None
        convert_element_type_141: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_189, torch.bfloat16);  mul_189 = None
        view_161: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_141, _shape_param_171);  convert_element_type_141 = _shape_param_171 = None
        permute_65: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_161, [1, 0])
        mm_70: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_65, arg165_1);  permute_65 = None
        mm_71: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_161, arg166_1);  view_161 = arg166_1 = None
        view_162: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_71, _shape_param_172);  mm_71 = _shape_param_172 = None
        convert_element_type_142: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_162, torch.float32);  view_162 = None
        convert_element_type_143: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        mul_191: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_190, mul_185);  mul_185 = None
        mul_192: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_190, add_50);  mul_190 = add_50 = None
        convert_element_type_144: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_192, torch.bfloat16);  mul_192 = None
        mul_193: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_12: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_193);  mul_193 = None
        mul_194: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_191, sub_12);  mul_191 = sub_12 = None
        mul_195: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_194, 0.7978845608028654);  mul_194 = None
        convert_element_type_145: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_195, torch.bfloat16)
        mul_196: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_195, 0.044715);  mul_195 = None
        pow_28: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_140, 2.0);  convert_element_type_140 = None
        mul_197: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_28, 3.0);  pow_28 = None
        mul_198: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_196, mul_197);  mul_196 = mul_197 = None
        convert_element_type_146: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_198, torch.bfloat16);  mul_198 = None
        add_51: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_145, convert_element_type_146);  convert_element_type_145 = convert_element_type_146 = None
        mul_199: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_144, 0.5);  convert_element_type_144 = None
        add_52: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_51, mul_199);  add_51 = mul_199 = None
        view_163: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_52, _shape_param_173);  add_52 = _shape_param_173 = None
        permute_66: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_163, [1, 0])
        mm_72: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_66, arg165_1);  permute_66 = arg165_1 = None
        mm_73: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_163, arg167_1);  view_163 = arg167_1 = None
        view_164: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_73, _shape_param_174);  mm_73 = _shape_param_174 = None
        convert_element_type_147: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_164, torch.float32);  view_164 = None
        add_53: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_142, convert_element_type_147);  convert_element_type_142 = convert_element_type_147 = None
        convert_element_type_148: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        mul_200: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_53, arg168_1);  arg168_1 = None
        mul_201: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg169_1, arg170_1)
        mul_202: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_53, mul_201);  add_53 = mul_201 = None
        sum_28: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_202, [0, 1], True, dtype = torch.float32);  mul_202 = None
        view_165: "f32[512]" = torch.ops.aten.view.default(sum_28, _shape_param_175);  sum_28 = _shape_param_175 = None
        mul_203: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_200, arg169_1)
        mul_204: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_200, arg170_1);  mul_200 = None
        sum_29: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True, dtype = torch.float32);  mul_203 = None
        add_54: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_48, mul_204);  add_48 = mul_204 = None
        pow_29: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg170_1, 3);  arg170_1 = None
        mul_205: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_29, -0.5);  sum_29 = None
        mul_206: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_205, pow_29);  mul_205 = pow_29 = None
        expand_11: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_206, _shape_param_176);  mul_206 = _shape_param_176 = None
        div_17: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_11, 512);  expand_11 = None
        pow_30: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg169_1, 1.0);  arg169_1 = None
        mul_207: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_30, 2.0);  pow_30 = None
        mul_208: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_17, mul_207);  div_17 = mul_207 = None
        add_55: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_54, mul_208);  add_54 = mul_208 = None
        convert_element_type_149: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16)
        convert_element_type_150: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg171_1, torch.bfloat16);  arg171_1 = None
        mul_209: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_150, 1.1111111111111112);  convert_element_type_150 = None
        mul_210: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_149, mul_209);  convert_element_type_149 = mul_209 = None
        view_166: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_210, _shape_param_177);  mul_210 = _shape_param_177 = None
        permute_67: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_166, [1, 0])
        mm_74: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_67, arg172_1);  permute_67 = arg172_1 = None
        mm_75: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_166, arg173_1);  view_166 = arg173_1 = None
        view_167: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_75, _shape_param_178);  mm_75 = _shape_param_178 = None
        convert_element_type_151: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_74, torch.float32);  mm_74 = None
        view_168: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_167, _shape_param_179);  view_167 = _shape_param_179 = None
        permute_68: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_24: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
        view_169: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_180);  clone_24 = _shape_param_180 = None
        bmm_24: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg174_1, view_169);  arg174_1 = None
        bmm_25: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_169, arg175_1);  view_169 = arg175_1 = None
        view_170: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_24, _shape_param_181);  bmm_24 = _shape_param_181 = None
        view_171: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_25, _shape_param_182);  bmm_25 = _shape_param_182 = None
        convert_element_type_152: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg176_1, torch.bfloat16);  arg176_1 = None
        mul_211: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_152, 1.1111111111111112);  convert_element_type_152 = None
        mul_212: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_171, mul_211);  view_171 = mul_211 = None
        convert_element_type_153: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_212, torch.float32);  mul_212 = None
        view_172: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg177_1, _shape_param_183);  arg177_1 = _shape_param_183 = None
        convert_element_type_154: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_172, torch.bfloat16);  view_172 = None
        convert_element_type_155: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_154, torch.float32);  convert_element_type_154 = None
        sub_13: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_155, arg178_1);  convert_element_type_155 = arg178_1 = None
        exp_7: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        div_18: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, arg179_1);  exp_7 = arg179_1 = None
        mul_213: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_153, div_18);  convert_element_type_153 = None
        sum_30: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [-1], True)
        neg_6: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_6: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_6, sum_30, mul_213);  neg_6 = sum_30 = mul_213 = None
        convert_element_type_156: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        view_173: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_156, _shape_param_184);  convert_element_type_156 = _shape_param_184 = None
        bmm_26: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg180_1, view_173);  arg180_1 = None
        bmm_27: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_173, arg181_1);  view_173 = arg181_1 = None
        view_174: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_185);  bmm_26 = _shape_param_185 = None
        view_175: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_186);  bmm_27 = _shape_param_186 = None
        permute_69: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_174, [0, 1, 3, 2]);  view_174 = None
        permute_70: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_25: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_176: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_25, _shape_param_187);  clone_25 = _shape_param_187 = None
        view_177: "bf16[4096, 384]" = torch.ops.aten.view.default(view_176, _shape_param_188);  view_176 = _shape_param_188 = None
        permute_71: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_177, [1, 0])
        mm_76: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_71, arg36_1);  permute_71 = None
        mm_77: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_177, arg182_1);  view_177 = arg182_1 = None
        view_178: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_77, _shape_param_189);  mm_77 = _shape_param_189 = None
        convert_element_type_157: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_178, torch.float32);  view_178 = None
        add_56: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_41, convert_element_type_157);  add_41 = convert_element_type_157 = None
        convert_element_type_158: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        permute_72: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_69, [0, 2, 1, 3]);  permute_69 = None
        view_179: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_72, _shape_param_190);  permute_72 = _shape_param_190 = None
        clone_26: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_179, memory_format = torch.contiguous_format);  view_179 = None
        view_180: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_26, _shape_param_191);  clone_26 = _shape_param_191 = None
        permute_73: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_180, [1, 0])
        mm_78: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_73, arg36_1);  permute_73 = None
        mm_79: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_180, arg183_1);  view_180 = arg183_1 = None
        view_181: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_79, _shape_param_192);  mm_79 = _shape_param_192 = None
        convert_element_type_159: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_181, torch.float32);  view_181 = None
        add_57: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_56, convert_element_type_159);  add_56 = convert_element_type_159 = None
        convert_element_type_160: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        permute_74: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
        clone_27: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
        view_182: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_27, _shape_param_193);  clone_27 = _shape_param_193 = None
        view_183: "bf16[4096, 384]" = torch.ops.aten.view.default(view_182, _shape_param_194);  view_182 = _shape_param_194 = None
        permute_75: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_183, [1, 0])
        mm_80: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_75, arg184_1);  permute_75 = arg184_1 = None
        mm_81: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_183, arg185_1);  view_183 = arg185_1 = None
        view_184: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_81, _shape_param_195);  mm_81 = _shape_param_195 = None
        convert_element_type_161: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_184, torch.float32);  view_184 = None
        convert_element_type_162: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        mul_214: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_161, arg186_1);  arg186_1 = None
        mul_215: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg187_1, arg188_1)
        mul_216: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_161, mul_215);  convert_element_type_161 = mul_215 = None
        sum_31: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1], True, dtype = torch.float32);  mul_216 = None
        view_185: "f32[512]" = torch.ops.aten.view.default(sum_31, _shape_param_196);  sum_31 = _shape_param_196 = None
        mul_217: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_214, arg187_1)
        mul_218: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_214, arg188_1);  mul_214 = None
        sum_32: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True, dtype = torch.float32);  mul_217 = None
        add_58: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_55, mul_218);  add_55 = mul_218 = None
        pow_31: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg188_1, 3);  arg188_1 = None
        mul_219: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_32, -0.5);  sum_32 = None
        mul_220: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_219, pow_31);  mul_219 = pow_31 = None
        expand_12: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_220, _shape_param_197);  mul_220 = _shape_param_197 = None
        div_19: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_12, 512);  expand_12 = None
        pow_32: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg187_1, 1.0);  arg187_1 = None
        mul_221: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_32, 2.0);  pow_32 = None
        mul_222: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_19, mul_221);  div_19 = mul_221 = None
        add_59: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_58, mul_222);  add_58 = mul_222 = None
        convert_element_type_163: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_59, torch.bfloat16)
        convert_element_type_164: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.bfloat16);  arg189_1 = None
        mul_223: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_164, 1.1111111111111112);  convert_element_type_164 = None
        mul_224: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_163, mul_223);  convert_element_type_163 = mul_223 = None
        view_186: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_224, _shape_param_198);  mul_224 = _shape_param_198 = None
        permute_76: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_186, [1, 0])
        mm_82: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_76, arg190_1);  permute_76 = arg190_1 = None
        mm_83: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_186, arg191_1);  view_186 = arg191_1 = None
        view_187: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_83, _shape_param_199);  mm_83 = _shape_param_199 = None
        convert_element_type_165: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        view_188: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_187, _shape_param_200);  view_187 = _shape_param_200 = None
        permute_77: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
        clone_28: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_189: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_201);  clone_28 = _shape_param_201 = None
        bmm_28: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg192_1, view_189);  arg192_1 = None
        bmm_29: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_189, arg193_1);  view_189 = arg193_1 = None
        view_190: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_28, _shape_param_202);  bmm_28 = _shape_param_202 = None
        view_191: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_29, _shape_param_203);  bmm_29 = _shape_param_203 = None
        convert_element_type_166: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.bfloat16);  arg194_1 = None
        mul_225: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_166, 1.1111111111111112);  convert_element_type_166 = None
        mul_226: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_191, mul_225);  view_191 = mul_225 = None
        convert_element_type_167: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_226, torch.float32);  mul_226 = None
        convert_element_type_168: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg195_1, torch.float32);  arg195_1 = None
        sub_14: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_168, arg196_1);  convert_element_type_168 = arg196_1 = None
        exp_8: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        div_20: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, arg197_1);  exp_8 = arg197_1 = None
        mul_227: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_167, div_20);  convert_element_type_167 = None
        sum_33: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [-1], True)
        neg_7: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_20);  div_20 = None
        fma_7: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_7, sum_33, mul_227);  neg_7 = sum_33 = mul_227 = None
        convert_element_type_169: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        view_192: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_169, _shape_param_204);  convert_element_type_169 = _shape_param_204 = None
        view_193: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_192, _shape_param_205);  view_192 = _shape_param_205 = None
        view_194: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_193, _shape_param_206);  _shape_param_206 = None
        convert_element_type_170: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_193, torch.float32);  view_193 = None
        add_60: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_44, convert_element_type_170);  add_44 = convert_element_type_170 = None
        bmm_30: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg198_1, view_194);  arg198_1 = None
        bmm_31: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_194, arg199_1);  view_194 = arg199_1 = None
        view_195: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_207);  bmm_30 = _shape_param_207 = None
        view_196: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_208);  bmm_31 = _shape_param_208 = None
        permute_78: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_195, [0, 1, 3, 2]);  view_195 = None
        permute_79: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
        clone_29: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        view_197: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_29, _shape_param_209);  clone_29 = _shape_param_209 = None
        view_198: "bf16[4096, 384]" = torch.ops.aten.view.default(view_197, _shape_param_210);  view_197 = _shape_param_210 = None
        permute_80: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_84: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_80, arg200_1);  permute_80 = None
        mm_85: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_198, arg201_1);  view_198 = arg201_1 = None
        view_199: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_85, _shape_param_211);  mm_85 = _shape_param_211 = None
        convert_element_type_171: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        convert_element_type_172: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        permute_81: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_78, [0, 2, 1, 3]);  permute_78 = None
        view_200: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_81, _shape_param_212);  permute_81 = _shape_param_212 = None
        clone_30: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_200, memory_format = torch.contiguous_format);  view_200 = None
        view_201: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_30, _shape_param_213);  clone_30 = _shape_param_213 = None
        permute_82: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_201, [1, 0])
        mm_86: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_82, arg200_1);  permute_82 = None
        mm_87: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_201, arg202_1);  view_201 = arg202_1 = None
        view_202: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_87, _shape_param_214);  mm_87 = _shape_param_214 = None
        convert_element_type_173: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_202, torch.float32);  view_202 = None
        add_61: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_171, convert_element_type_173);  convert_element_type_171 = convert_element_type_173 = None
        convert_element_type_174: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        permute_83: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
        clone_31: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_203: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_31, _shape_param_215);  clone_31 = _shape_param_215 = None
        view_204: "bf16[4096, 384]" = torch.ops.aten.view.default(view_203, _shape_param_216);  view_203 = _shape_param_216 = None
        permute_84: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_204, [1, 0])
        mm_88: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_84, arg200_1);  permute_84 = arg200_1 = None
        mm_89: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_204, arg203_1);  view_204 = arg203_1 = None
        view_205: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_89, _shape_param_217);  mm_89 = _shape_param_217 = None
        convert_element_type_175: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        add_62: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_61, convert_element_type_175);  add_61 = convert_element_type_175 = None
        convert_element_type_176: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        mul_228: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_62, arg204_1);  arg204_1 = None
        mul_229: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg205_1, arg206_1)
        mul_230: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_62, mul_229);  add_62 = mul_229 = None
        sum_34: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_230, [0, 1], True, dtype = torch.float32);  mul_230 = None
        view_206: "f32[512]" = torch.ops.aten.view.default(sum_34, _shape_param_218);  sum_34 = _shape_param_218 = None
        mul_231: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_228, arg205_1)
        mul_232: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_228, arg206_1);  mul_228 = None
        sum_35: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True, dtype = torch.float32);  mul_231 = None
        add_63: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_59, mul_232);  add_59 = mul_232 = None
        pow_33: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg206_1, 3);  arg206_1 = None
        mul_233: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_35, -0.5);  sum_35 = None
        mul_234: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_233, pow_33);  mul_233 = pow_33 = None
        expand_13: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_234, _shape_param_219);  mul_234 = _shape_param_219 = None
        div_21: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_13, 512);  expand_13 = None
        pow_34: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg205_1, 1.0);  arg205_1 = None
        mul_235: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_34, 2.0);  pow_34 = None
        mul_236: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_21, mul_235);  div_21 = mul_235 = None
        add_64: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_63, mul_236);  add_63 = mul_236 = None
        convert_element_type_177: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16)
        convert_element_type_178: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.bfloat16);  arg207_1 = None
        mul_237: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_178, 1.1111111111111112);  convert_element_type_178 = None
        mul_238: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_177, mul_237);  convert_element_type_177 = mul_237 = None
        view_207: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_238, _shape_param_220);  mul_238 = _shape_param_220 = None
        permute_85: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_90: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_85, arg208_1);  permute_85 = arg208_1 = None
        mm_91: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_207, arg209_1);  view_207 = arg209_1 = None
        view_208: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_91, _shape_param_221);  mm_91 = _shape_param_221 = None
        convert_element_type_179: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_208, torch.float32);  view_208 = None
        convert_element_type_180: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        convert_element_type_181: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg210_1, torch.float32);  arg210_1 = None
        mul_239: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_181, 1.1111111111111112);  convert_element_type_181 = None
        mul_240: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_179, mul_239);  convert_element_type_179 = mul_239 = None
        view_209: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg211_1, _shape_param_222);  arg211_1 = _shape_param_222 = None
        mul_241: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_209, 0.5)
        convert_element_type_182: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_209, torch.float32)
        pow_35: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_182, 3.0)
        mul_242: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_35, 0.044715);  pow_35 = None
        add_65: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_209, mul_242);  view_209 = mul_242 = None
        mul_243: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_4: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_243);  mul_243 = None
        add_66: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_244: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_241, add_66)
        mul_245: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_240, mul_244);  mul_244 = None
        view_210: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg212_1, _shape_param_223);  arg212_1 = _shape_param_223 = None
        mul_246: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_240, view_210);  mul_240 = view_210 = None
        convert_element_type_183: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_245, torch.bfloat16);  mul_245 = None
        view_211: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_183, _shape_param_224);  convert_element_type_183 = _shape_param_224 = None
        permute_86: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_211, [1, 0])
        mm_92: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_86, arg213_1);  permute_86 = None
        mm_93: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_211, arg214_1);  view_211 = arg214_1 = None
        view_212: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_93, _shape_param_225);  mm_93 = _shape_param_225 = None
        convert_element_type_184: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        convert_element_type_185: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        mul_247: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_246, mul_241);  mul_241 = None
        mul_248: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_246, add_66);  mul_246 = add_66 = None
        convert_element_type_186: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None
        mul_249: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_15: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_249);  mul_249 = None
        mul_250: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_247, sub_15);  mul_247 = sub_15 = None
        mul_251: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_250, 0.7978845608028654);  mul_250 = None
        convert_element_type_187: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_251, torch.bfloat16)
        mul_252: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_251, 0.044715);  mul_251 = None
        pow_36: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_182, 2.0);  convert_element_type_182 = None
        mul_253: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_254: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
        convert_element_type_188: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_254, torch.bfloat16);  mul_254 = None
        add_67: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_187, convert_element_type_188);  convert_element_type_187 = convert_element_type_188 = None
        mul_255: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_186, 0.5);  convert_element_type_186 = None
        add_68: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_67, mul_255);  add_67 = mul_255 = None
        view_213: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_68, _shape_param_226);  add_68 = _shape_param_226 = None
        permute_87: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_94: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_87, arg213_1);  permute_87 = arg213_1 = None
        mm_95: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_213, arg215_1);  view_213 = arg215_1 = None
        view_214: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_95, _shape_param_227);  mm_95 = _shape_param_227 = None
        convert_element_type_189: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_214, torch.float32);  view_214 = None
        add_69: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_184, convert_element_type_189);  convert_element_type_184 = convert_element_type_189 = None
        convert_element_type_190: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        mul_256: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_69, arg216_1);  arg216_1 = None
        mul_257: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg217_1, arg218_1)
        mul_258: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_69, mul_257);  add_69 = mul_257 = None
        sum_36: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1], True, dtype = torch.float32);  mul_258 = None
        view_215: "f32[512]" = torch.ops.aten.view.default(sum_36, _shape_param_228);  sum_36 = _shape_param_228 = None
        mul_259: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_256, arg217_1)
        mul_260: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_256, arg218_1);  mul_256 = None
        sum_37: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True, dtype = torch.float32);  mul_259 = None
        add_70: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_64, mul_260);  add_64 = mul_260 = None
        pow_37: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg218_1, 3);  arg218_1 = None
        mul_261: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_37, -0.5);  sum_37 = None
        mul_262: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_261, pow_37);  mul_261 = pow_37 = None
        expand_14: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_262, _shape_param_229);  mul_262 = _shape_param_229 = None
        div_22: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_14, 512);  expand_14 = None
        pow_38: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg217_1, 1.0);  arg217_1 = None
        mul_263: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
        mul_264: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_22, mul_263);  div_22 = mul_263 = None
        add_71: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_70, mul_264);  add_70 = mul_264 = None
        convert_element_type_191: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16)
        convert_element_type_192: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg219_1, torch.bfloat16);  arg219_1 = None
        mul_265: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_192, 1.1111111111111112);  convert_element_type_192 = None
        mul_266: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_191, mul_265);  convert_element_type_191 = mul_265 = None
        view_216: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_266, _shape_param_230);  mul_266 = _shape_param_230 = None
        permute_88: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_216, [1, 0])
        mm_96: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_88, arg220_1);  permute_88 = arg220_1 = None
        mm_97: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_216, arg221_1);  view_216 = arg221_1 = None
        view_217: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_97, _shape_param_231);  mm_97 = _shape_param_231 = None
        convert_element_type_193: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        view_218: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_217, _shape_param_232);  view_217 = _shape_param_232 = None
        permute_89: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        clone_32: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_219: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_233);  clone_32 = _shape_param_233 = None
        bmm_32: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg222_1, view_219);  arg222_1 = None
        bmm_33: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_219, arg223_1);  view_219 = arg223_1 = None
        view_220: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_32, _shape_param_234);  bmm_32 = _shape_param_234 = None
        view_221: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_33, _shape_param_235);  bmm_33 = _shape_param_235 = None
        convert_element_type_194: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg224_1, torch.bfloat16);  arg224_1 = None
        mul_267: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_194, 1.1111111111111112);  convert_element_type_194 = None
        mul_268: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_221, mul_267);  view_221 = mul_267 = None
        convert_element_type_195: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_268, torch.float32);  mul_268 = None
        view_222: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg225_1, _shape_param_236);  arg225_1 = _shape_param_236 = None
        convert_element_type_196: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_222, torch.bfloat16);  view_222 = None
        convert_element_type_197: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_196, torch.float32);  convert_element_type_196 = None
        sub_16: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_197, arg226_1);  convert_element_type_197 = arg226_1 = None
        exp_9: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        div_23: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, arg227_1);  exp_9 = arg227_1 = None
        mul_269: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_195, div_23);  convert_element_type_195 = None
        sum_38: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [-1], True)
        neg_8: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma_8: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_8, sum_38, mul_269);  neg_8 = sum_38 = mul_269 = None
        convert_element_type_198: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        view_223: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_198, _shape_param_237);  convert_element_type_198 = _shape_param_237 = None
        bmm_34: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg228_1, view_223);  arg228_1 = None
        bmm_35: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_223, arg229_1);  view_223 = arg229_1 = None
        view_224: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_238);  bmm_34 = _shape_param_238 = None
        view_225: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_239);  bmm_35 = _shape_param_239 = None
        permute_90: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_224, [0, 1, 3, 2]);  view_224 = None
        permute_91: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
        clone_33: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
        view_226: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_33, _shape_param_240);  clone_33 = _shape_param_240 = None
        view_227: "bf16[4096, 384]" = torch.ops.aten.view.default(view_226, _shape_param_241);  view_226 = _shape_param_241 = None
        permute_92: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_227, [1, 0])
        mm_98: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_92, arg36_1);  permute_92 = None
        mm_99: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_227, arg230_1);  view_227 = arg230_1 = None
        view_228: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_99, _shape_param_242);  mm_99 = _shape_param_242 = None
        convert_element_type_199: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_228, torch.float32);  view_228 = None
        add_72: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_57, convert_element_type_199);  add_57 = convert_element_type_199 = None
        convert_element_type_200: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None
        permute_93: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_90, [0, 2, 1, 3]);  permute_90 = None
        view_229: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_93, _shape_param_243);  permute_93 = _shape_param_243 = None
        clone_34: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_229, memory_format = torch.contiguous_format);  view_229 = None
        view_230: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_34, _shape_param_244);  clone_34 = _shape_param_244 = None
        permute_94: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_230, [1, 0])
        mm_100: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_94, arg36_1);  permute_94 = None
        mm_101: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_230, arg231_1);  view_230 = arg231_1 = None
        view_231: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_101, _shape_param_245);  mm_101 = _shape_param_245 = None
        convert_element_type_201: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_231, torch.float32);  view_231 = None
        add_73: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_72, convert_element_type_201);  add_72 = convert_element_type_201 = None
        convert_element_type_202: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        permute_95: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
        clone_35: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_232: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_35, _shape_param_246);  clone_35 = _shape_param_246 = None
        view_233: "bf16[4096, 384]" = torch.ops.aten.view.default(view_232, _shape_param_247);  view_232 = _shape_param_247 = None
        permute_96: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_233, [1, 0])
        mm_102: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_96, arg232_1);  permute_96 = arg232_1 = None
        mm_103: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_233, arg233_1);  view_233 = arg233_1 = None
        view_234: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_103, _shape_param_248);  mm_103 = _shape_param_248 = None
        convert_element_type_203: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_234, torch.float32);  view_234 = None
        convert_element_type_204: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        mul_270: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_203, arg234_1);  arg234_1 = None
        mul_271: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg235_1, arg236_1)
        mul_272: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_203, mul_271);  convert_element_type_203 = mul_271 = None
        sum_39: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1], True, dtype = torch.float32);  mul_272 = None
        view_235: "f32[512]" = torch.ops.aten.view.default(sum_39, _shape_param_249);  sum_39 = _shape_param_249 = None
        mul_273: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_270, arg235_1)
        mul_274: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_270, arg236_1);  mul_270 = None
        sum_40: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True, dtype = torch.float32);  mul_273 = None
        add_74: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_71, mul_274);  add_71 = mul_274 = None
        pow_39: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg236_1, 3);  arg236_1 = None
        mul_275: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_40, -0.5);  sum_40 = None
        mul_276: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_275, pow_39);  mul_275 = pow_39 = None
        expand_15: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_276, _shape_param_250);  mul_276 = _shape_param_250 = None
        div_24: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_15, 512);  expand_15 = None
        pow_40: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg235_1, 1.0);  arg235_1 = None
        mul_277: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
        mul_278: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_24, mul_277);  div_24 = mul_277 = None
        add_75: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_74, mul_278);  add_74 = mul_278 = None
        convert_element_type_205: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_75, torch.bfloat16)
        convert_element_type_206: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.bfloat16);  arg237_1 = None
        mul_279: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_206, 1.1111111111111112);  convert_element_type_206 = None
        mul_280: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_205, mul_279);  convert_element_type_205 = mul_279 = None
        view_236: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_280, _shape_param_251);  mul_280 = _shape_param_251 = None
        permute_97: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_236, [1, 0])
        mm_104: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_97, arg238_1);  permute_97 = arg238_1 = None
        mm_105: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_236, arg239_1);  view_236 = arg239_1 = None
        view_237: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_105, _shape_param_252);  mm_105 = _shape_param_252 = None
        convert_element_type_207: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        view_238: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_237, _shape_param_253);  view_237 = _shape_param_253 = None
        permute_98: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
        clone_36: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_98, memory_format = torch.contiguous_format);  permute_98 = None
        view_239: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_254);  clone_36 = _shape_param_254 = None
        bmm_36: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg240_1, view_239);  arg240_1 = None
        bmm_37: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_239, arg241_1);  view_239 = arg241_1 = None
        view_240: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_36, _shape_param_255);  bmm_36 = _shape_param_255 = None
        view_241: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_37, _shape_param_256);  bmm_37 = _shape_param_256 = None
        convert_element_type_208: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg242_1, torch.bfloat16);  arg242_1 = None
        mul_281: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_208, 1.1111111111111112);  convert_element_type_208 = None
        mul_282: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_241, mul_281);  view_241 = mul_281 = None
        convert_element_type_209: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_282, torch.float32);  mul_282 = None
        convert_element_type_210: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.float32);  arg243_1 = None
        sub_17: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_210, arg244_1);  convert_element_type_210 = arg244_1 = None
        exp_10: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        div_25: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, arg245_1);  exp_10 = arg245_1 = None
        mul_283: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_209, div_25);  convert_element_type_209 = None
        sum_41: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [-1], True)
        neg_9: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_9: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_9, sum_41, mul_283);  neg_9 = sum_41 = mul_283 = None
        convert_element_type_211: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        view_242: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_211, _shape_param_257);  convert_element_type_211 = _shape_param_257 = None
        view_243: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_242, _shape_param_258);  view_242 = _shape_param_258 = None
        view_244: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_243, _shape_param_259);  _shape_param_259 = None
        convert_element_type_212: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_243, torch.float32);  view_243 = None
        add_76: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_60, convert_element_type_212);  add_60 = convert_element_type_212 = None
        bmm_38: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg246_1, view_244);  arg246_1 = None
        bmm_39: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_244, arg247_1);  view_244 = arg247_1 = None
        view_245: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_260);  bmm_38 = _shape_param_260 = None
        view_246: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_261);  bmm_39 = _shape_param_261 = None
        permute_99: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_245, [0, 1, 3, 2]);  view_245 = None
        permute_100: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
        clone_37: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_100, memory_format = torch.contiguous_format);  permute_100 = None
        view_247: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_37, _shape_param_262);  clone_37 = _shape_param_262 = None
        view_248: "bf16[4096, 384]" = torch.ops.aten.view.default(view_247, _shape_param_263);  view_247 = _shape_param_263 = None
        permute_101: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_248, [1, 0])
        mm_106: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_101, arg248_1);  permute_101 = None
        mm_107: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_248, arg249_1);  view_248 = arg249_1 = None
        view_249: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_107, _shape_param_264);  mm_107 = _shape_param_264 = None
        convert_element_type_213: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        convert_element_type_214: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        permute_102: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_99, [0, 2, 1, 3]);  permute_99 = None
        view_250: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_102, _shape_param_265);  permute_102 = _shape_param_265 = None
        clone_38: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_250, memory_format = torch.contiguous_format);  view_250 = None
        view_251: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_38, _shape_param_266);  clone_38 = _shape_param_266 = None
        permute_103: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_251, [1, 0])
        mm_108: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_103, arg248_1);  permute_103 = None
        mm_109: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_251, arg250_1);  view_251 = arg250_1 = None
        view_252: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_109, _shape_param_267);  mm_109 = _shape_param_267 = None
        convert_element_type_215: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_252, torch.float32);  view_252 = None
        add_77: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_213, convert_element_type_215);  convert_element_type_213 = convert_element_type_215 = None
        convert_element_type_216: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        permute_104: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        clone_39: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_253: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_39, _shape_param_268);  clone_39 = _shape_param_268 = None
        view_254: "bf16[4096, 384]" = torch.ops.aten.view.default(view_253, _shape_param_269);  view_253 = _shape_param_269 = None
        permute_105: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_254, [1, 0])
        mm_110: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_105, arg248_1);  permute_105 = arg248_1 = None
        mm_111: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_254, arg251_1);  view_254 = arg251_1 = None
        view_255: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_111, _shape_param_270);  mm_111 = _shape_param_270 = None
        convert_element_type_217: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_255, torch.float32);  view_255 = None
        add_78: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_77, convert_element_type_217);  add_77 = convert_element_type_217 = None
        convert_element_type_218: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_110, torch.float32);  mm_110 = None
        mul_284: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_78, arg252_1);  arg252_1 = None
        mul_285: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg253_1, arg254_1)
        mul_286: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_78, mul_285);  add_78 = mul_285 = None
        sum_42: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 1], True, dtype = torch.float32);  mul_286 = None
        view_256: "f32[512]" = torch.ops.aten.view.default(sum_42, _shape_param_271);  sum_42 = _shape_param_271 = None
        mul_287: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_284, arg253_1)
        mul_288: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_284, arg254_1);  mul_284 = None
        sum_43: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True, dtype = torch.float32);  mul_287 = None
        add_79: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_75, mul_288);  add_75 = mul_288 = None
        pow_41: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg254_1, 3);  arg254_1 = None
        mul_289: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_43, -0.5);  sum_43 = None
        mul_290: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_289, pow_41);  mul_289 = pow_41 = None
        expand_16: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_290, _shape_param_272);  mul_290 = _shape_param_272 = None
        div_26: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_16, 512);  expand_16 = None
        pow_42: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg253_1, 1.0);  arg253_1 = None
        mul_291: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
        mul_292: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_26, mul_291);  div_26 = mul_291 = None
        add_80: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_79, mul_292);  add_79 = mul_292 = None
        convert_element_type_219: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16)
        convert_element_type_220: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg255_1, torch.bfloat16);  arg255_1 = None
        mul_293: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_220, 1.1111111111111112);  convert_element_type_220 = None
        mul_294: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_219, mul_293);  convert_element_type_219 = mul_293 = None
        view_257: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_294, _shape_param_273);  mul_294 = _shape_param_273 = None
        permute_106: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_257, [1, 0])
        mm_112: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_106, arg256_1);  permute_106 = arg256_1 = None
        mm_113: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_257, arg257_1);  view_257 = arg257_1 = None
        view_258: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_113, _shape_param_274);  mm_113 = _shape_param_274 = None
        convert_element_type_221: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_258, torch.float32);  view_258 = None
        convert_element_type_222: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None
        convert_element_type_223: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.float32);  arg258_1 = None
        mul_295: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_223, 1.1111111111111112);  convert_element_type_223 = None
        mul_296: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_221, mul_295);  convert_element_type_221 = mul_295 = None
        view_259: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg259_1, _shape_param_275);  arg259_1 = _shape_param_275 = None
        mul_297: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_259, 0.5)
        convert_element_type_224: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_259, torch.float32)
        pow_43: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_224, 3.0)
        mul_298: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_43, 0.044715);  pow_43 = None
        add_81: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_259, mul_298);  view_259 = mul_298 = None
        mul_299: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_5: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_299);  mul_299 = None
        add_82: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_300: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_297, add_82)
        mul_301: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_296, mul_300);  mul_300 = None
        view_260: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg260_1, _shape_param_276);  arg260_1 = _shape_param_276 = None
        mul_302: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_296, view_260);  mul_296 = view_260 = None
        convert_element_type_225: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_301, torch.bfloat16);  mul_301 = None
        view_261: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_225, _shape_param_277);  convert_element_type_225 = _shape_param_277 = None
        permute_107: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_261, [1, 0])
        mm_114: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_107, arg261_1);  permute_107 = None
        mm_115: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_261, arg262_1);  view_261 = arg262_1 = None
        view_262: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_115, _shape_param_278);  mm_115 = _shape_param_278 = None
        convert_element_type_226: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_262, torch.float32);  view_262 = None
        convert_element_type_227: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        mul_303: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_302, mul_297);  mul_297 = None
        mul_304: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_302, add_82);  mul_302 = add_82 = None
        convert_element_type_228: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_304, torch.bfloat16);  mul_304 = None
        mul_305: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_18: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_305);  mul_305 = None
        mul_306: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_303, sub_18);  mul_303 = sub_18 = None
        mul_307: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_306, 0.7978845608028654);  mul_306 = None
        convert_element_type_229: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_307, torch.bfloat16)
        mul_308: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_307, 0.044715);  mul_307 = None
        pow_44: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_224, 2.0);  convert_element_type_224 = None
        mul_309: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_310: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_308, mul_309);  mul_308 = mul_309 = None
        convert_element_type_230: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_310, torch.bfloat16);  mul_310 = None
        add_83: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_229, convert_element_type_230);  convert_element_type_229 = convert_element_type_230 = None
        mul_311: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_228, 0.5);  convert_element_type_228 = None
        add_84: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_83, mul_311);  add_83 = mul_311 = None
        view_263: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_84, _shape_param_279);  add_84 = _shape_param_279 = None
        permute_108: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_263, [1, 0])
        mm_116: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_108, arg261_1);  permute_108 = arg261_1 = None
        mm_117: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_263, arg263_1);  view_263 = arg263_1 = None
        view_264: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_117, _shape_param_280);  mm_117 = _shape_param_280 = None
        convert_element_type_231: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_264, torch.float32);  view_264 = None
        add_85: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_226, convert_element_type_231);  convert_element_type_226 = convert_element_type_231 = None
        convert_element_type_232: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        mul_312: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_85, arg264_1);  arg264_1 = None
        mul_313: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg265_1, arg266_1)
        mul_314: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_85, mul_313);  add_85 = mul_313 = None
        sum_44: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1], True, dtype = torch.float32);  mul_314 = None
        view_265: "f32[512]" = torch.ops.aten.view.default(sum_44, _shape_param_281);  sum_44 = _shape_param_281 = None
        mul_315: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_312, arg265_1)
        mul_316: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_312, arg266_1);  mul_312 = None
        sum_45: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_315, [2], True, dtype = torch.float32);  mul_315 = None
        add_86: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_80, mul_316);  add_80 = mul_316 = None
        pow_45: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg266_1, 3);  arg266_1 = None
        mul_317: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_45, -0.5);  sum_45 = None
        mul_318: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_317, pow_45);  mul_317 = pow_45 = None
        expand_17: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_318, _shape_param_282);  mul_318 = _shape_param_282 = None
        div_27: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_17, 512);  expand_17 = None
        pow_46: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg265_1, 1.0);  arg265_1 = None
        mul_319: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
        mul_320: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_27, mul_319);  div_27 = mul_319 = None
        add_87: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_86, mul_320);  add_86 = mul_320 = None
        convert_element_type_233: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16)
        convert_element_type_234: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.bfloat16);  arg267_1 = None
        mul_321: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_234, 1.1111111111111112);  convert_element_type_234 = None
        mul_322: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_233, mul_321);  convert_element_type_233 = mul_321 = None
        view_266: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_322, _shape_param_283);  mul_322 = _shape_param_283 = None
        permute_109: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_266, [1, 0])
        mm_118: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_109, arg268_1);  permute_109 = arg268_1 = None
        mm_119: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_266, arg269_1);  view_266 = arg269_1 = None
        view_267: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_119, _shape_param_284);  mm_119 = _shape_param_284 = None
        convert_element_type_235: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        view_268: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_267, _shape_param_285);  view_267 = _shape_param_285 = None
        permute_110: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        clone_40: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_269: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_286);  clone_40 = _shape_param_286 = None
        bmm_40: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg270_1, view_269);  arg270_1 = None
        bmm_41: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_269, arg271_1);  view_269 = arg271_1 = None
        view_270: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_40, _shape_param_287);  bmm_40 = _shape_param_287 = None
        view_271: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_41, _shape_param_288);  bmm_41 = _shape_param_288 = None
        convert_element_type_236: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg272_1, torch.bfloat16);  arg272_1 = None
        mul_323: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_236, 1.1111111111111112);  convert_element_type_236 = None
        mul_324: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_271, mul_323);  view_271 = mul_323 = None
        convert_element_type_237: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_324, torch.float32);  mul_324 = None
        view_272: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg273_1, _shape_param_289);  arg273_1 = _shape_param_289 = None
        convert_element_type_238: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_272, torch.bfloat16);  view_272 = None
        convert_element_type_239: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_238, torch.float32);  convert_element_type_238 = None
        sub_19: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_239, arg274_1);  convert_element_type_239 = arg274_1 = None
        exp_11: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        div_28: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, arg275_1);  exp_11 = arg275_1 = None
        mul_325: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_237, div_28);  convert_element_type_237 = None
        sum_46: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [-1], True)
        neg_10: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_28);  div_28 = None
        fma_10: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_10, sum_46, mul_325);  neg_10 = sum_46 = mul_325 = None
        convert_element_type_240: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        view_273: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_240, _shape_param_290);  convert_element_type_240 = _shape_param_290 = None
        bmm_42: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg276_1, view_273);  arg276_1 = None
        bmm_43: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_273, arg277_1);  view_273 = arg277_1 = None
        view_274: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_291);  bmm_42 = _shape_param_291 = None
        view_275: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_292);  bmm_43 = _shape_param_292 = None
        permute_111: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_274, [0, 1, 3, 2]);  view_274 = None
        permute_112: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        clone_41: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
        view_276: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_41, _shape_param_293);  clone_41 = _shape_param_293 = None
        view_277: "bf16[4096, 384]" = torch.ops.aten.view.default(view_276, _shape_param_294);  view_276 = _shape_param_294 = None
        permute_113: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_277, [1, 0])
        mm_120: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_113, arg36_1);  permute_113 = None
        mm_121: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_277, arg278_1);  view_277 = arg278_1 = None
        view_278: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_121, _shape_param_295);  mm_121 = _shape_param_295 = None
        convert_element_type_241: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_278, torch.float32);  view_278 = None
        add_88: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_73, convert_element_type_241);  add_73 = convert_element_type_241 = None
        convert_element_type_242: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        permute_114: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_111, [0, 2, 1, 3]);  permute_111 = None
        view_279: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_114, _shape_param_296);  permute_114 = _shape_param_296 = None
        clone_42: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_279, memory_format = torch.contiguous_format);  view_279 = None
        view_280: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_42, _shape_param_297);  clone_42 = _shape_param_297 = None
        permute_115: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_280, [1, 0])
        mm_122: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_115, arg36_1);  permute_115 = None
        mm_123: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_280, arg279_1);  view_280 = arg279_1 = None
        view_281: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_123, _shape_param_298);  mm_123 = _shape_param_298 = None
        convert_element_type_243: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_281, torch.float32);  view_281 = None
        add_89: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_88, convert_element_type_243);  add_88 = convert_element_type_243 = None
        convert_element_type_244: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_122, torch.float32);  mm_122 = None
        permute_116: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None
        clone_43: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_282: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_43, _shape_param_299);  clone_43 = _shape_param_299 = None
        view_283: "bf16[4096, 384]" = torch.ops.aten.view.default(view_282, _shape_param_300);  view_282 = _shape_param_300 = None
        permute_117: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_124: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_117, arg280_1);  permute_117 = arg280_1 = None
        mm_125: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_283, arg281_1);  view_283 = arg281_1 = None
        view_284: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_125, _shape_param_301);  mm_125 = _shape_param_301 = None
        convert_element_type_245: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_284, torch.float32);  view_284 = None
        convert_element_type_246: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None
        mul_326: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_245, arg282_1);  arg282_1 = None
        mul_327: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg283_1, arg284_1)
        mul_328: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_245, mul_327);  convert_element_type_245 = mul_327 = None
        sum_47: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 1], True, dtype = torch.float32);  mul_328 = None
        view_285: "f32[512]" = torch.ops.aten.view.default(sum_47, _shape_param_302);  sum_47 = _shape_param_302 = None
        mul_329: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_326, arg283_1)
        mul_330: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_326, arg284_1);  mul_326 = None
        sum_48: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True, dtype = torch.float32);  mul_329 = None
        add_90: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_87, mul_330);  add_87 = mul_330 = None
        pow_47: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg284_1, 3);  arg284_1 = None
        mul_331: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_48, -0.5);  sum_48 = None
        mul_332: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_331, pow_47);  mul_331 = pow_47 = None
        expand_18: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_332, _shape_param_303);  mul_332 = _shape_param_303 = None
        div_29: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_18, 512);  expand_18 = None
        pow_48: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg283_1, 1.0);  arg283_1 = None
        mul_333: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
        mul_334: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_29, mul_333);  div_29 = mul_333 = None
        add_91: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_90, mul_334);  add_90 = mul_334 = None
        convert_element_type_247: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_91, torch.bfloat16)
        convert_element_type_248: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg285_1, torch.bfloat16);  arg285_1 = None
        mul_335: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_248, 1.1111111111111112);  convert_element_type_248 = None
        mul_336: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_247, mul_335);  convert_element_type_247 = mul_335 = None
        view_286: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_336, _shape_param_304);  mul_336 = _shape_param_304 = None
        permute_118: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_286, [1, 0])
        mm_126: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_118, arg286_1);  permute_118 = arg286_1 = None
        mm_127: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_286, arg287_1);  view_286 = arg287_1 = None
        view_287: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_127, _shape_param_305);  mm_127 = _shape_param_305 = None
        convert_element_type_249: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        view_288: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_287, _shape_param_306);  view_287 = _shape_param_306 = None
        permute_119: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_288, [0, 2, 1, 3]);  view_288 = None
        clone_44: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
        view_289: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_307);  clone_44 = _shape_param_307 = None
        bmm_44: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg288_1, view_289);  arg288_1 = None
        bmm_45: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_289, arg289_1);  view_289 = arg289_1 = None
        view_290: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_44, _shape_param_308);  bmm_44 = _shape_param_308 = None
        view_291: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_45, _shape_param_309);  bmm_45 = _shape_param_309 = None
        convert_element_type_250: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg290_1, torch.bfloat16);  arg290_1 = None
        mul_337: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_250, 1.1111111111111112);  convert_element_type_250 = None
        mul_338: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_291, mul_337);  view_291 = mul_337 = None
        convert_element_type_251: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_338, torch.float32);  mul_338 = None
        convert_element_type_252: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg291_1, torch.float32);  arg291_1 = None
        sub_20: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_252, arg292_1);  convert_element_type_252 = arg292_1 = None
        exp_12: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        div_30: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, arg293_1);  exp_12 = arg293_1 = None
        mul_339: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_251, div_30);  convert_element_type_251 = None
        sum_49: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [-1], True)
        neg_11: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_30);  div_30 = None
        fma_11: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_11, sum_49, mul_339);  neg_11 = sum_49 = mul_339 = None
        convert_element_type_253: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        view_292: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_253, _shape_param_310);  convert_element_type_253 = _shape_param_310 = None
        view_293: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_292, _shape_param_311);  view_292 = _shape_param_311 = None
        view_294: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_293, _shape_param_312);  _shape_param_312 = None
        convert_element_type_254: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_293, torch.float32);  view_293 = None
        add_92: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_76, convert_element_type_254);  add_76 = convert_element_type_254 = None
        bmm_46: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg294_1, view_294);  arg294_1 = None
        bmm_47: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_294, arg295_1);  view_294 = arg295_1 = None
        view_295: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_313);  bmm_46 = _shape_param_313 = None
        view_296: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_314);  bmm_47 = _shape_param_314 = None
        permute_120: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_295, [0, 1, 3, 2]);  view_295 = None
        permute_121: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        clone_45: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_297: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_45, _shape_param_315);  clone_45 = _shape_param_315 = None
        view_298: "bf16[4096, 384]" = torch.ops.aten.view.default(view_297, _shape_param_316);  view_297 = _shape_param_316 = None
        permute_122: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_298, [1, 0])
        mm_128: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_122, arg296_1);  permute_122 = None
        mm_129: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_298, arg297_1);  view_298 = arg297_1 = None
        view_299: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_129, _shape_param_317);  mm_129 = _shape_param_317 = None
        convert_element_type_255: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_299, torch.float32);  view_299 = None
        convert_element_type_256: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        permute_123: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_120, [0, 2, 1, 3]);  permute_120 = None
        view_300: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_123, _shape_param_318);  permute_123 = _shape_param_318 = None
        clone_46: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_300, memory_format = torch.contiguous_format);  view_300 = None
        view_301: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_46, _shape_param_319);  clone_46 = _shape_param_319 = None
        permute_124: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_301, [1, 0])
        mm_130: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_124, arg296_1);  permute_124 = None
        mm_131: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_301, arg298_1);  view_301 = arg298_1 = None
        view_302: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_131, _shape_param_320);  mm_131 = _shape_param_320 = None
        convert_element_type_257: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_302, torch.float32);  view_302 = None
        add_93: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_255, convert_element_type_257);  convert_element_type_255 = convert_element_type_257 = None
        convert_element_type_258: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        permute_125: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        clone_47: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_303: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_47, _shape_param_321);  clone_47 = _shape_param_321 = None
        view_304: "bf16[4096, 384]" = torch.ops.aten.view.default(view_303, _shape_param_322);  view_303 = _shape_param_322 = None
        permute_126: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_304, [1, 0])
        mm_132: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_126, arg296_1);  permute_126 = arg296_1 = None
        mm_133: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_304, arg299_1);  view_304 = arg299_1 = None
        view_305: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_133, _shape_param_323);  mm_133 = _shape_param_323 = None
        convert_element_type_259: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None
        add_94: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_93, convert_element_type_259);  add_93 = convert_element_type_259 = None
        convert_element_type_260: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None
        mul_340: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_94, arg300_1);  arg300_1 = None
        mul_341: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg301_1, arg302_1)
        mul_342: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_94, mul_341);  add_94 = mul_341 = None
        sum_50: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_342, [0, 1], True, dtype = torch.float32);  mul_342 = None
        view_306: "f32[512]" = torch.ops.aten.view.default(sum_50, _shape_param_324);  sum_50 = _shape_param_324 = None
        mul_343: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_340, arg301_1)
        mul_344: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_340, arg302_1);  mul_340 = None
        sum_51: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_343, [2], True, dtype = torch.float32);  mul_343 = None
        add_95: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_91, mul_344);  add_91 = mul_344 = None
        pow_49: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg302_1, 3);  arg302_1 = None
        mul_345: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_51, -0.5);  sum_51 = None
        mul_346: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_345, pow_49);  mul_345 = pow_49 = None
        expand_19: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_346, _shape_param_325);  mul_346 = _shape_param_325 = None
        div_31: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_19, 512);  expand_19 = None
        pow_50: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg301_1, 1.0);  arg301_1 = None
        mul_347: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
        mul_348: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_31, mul_347);  div_31 = mul_347 = None
        add_96: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_95, mul_348);  add_95 = mul_348 = None
        convert_element_type_261: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16)
        convert_element_type_262: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.bfloat16);  arg303_1 = None
        mul_349: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_262, 1.1111111111111112);  convert_element_type_262 = None
        mul_350: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_261, mul_349);  convert_element_type_261 = mul_349 = None
        view_307: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_350, _shape_param_326);  mul_350 = _shape_param_326 = None
        permute_127: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_307, [1, 0])
        mm_134: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_127, arg304_1);  permute_127 = arg304_1 = None
        mm_135: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_307, arg305_1);  view_307 = arg305_1 = None
        view_308: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_135, _shape_param_327);  mm_135 = _shape_param_327 = None
        convert_element_type_263: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_308, torch.float32);  view_308 = None
        convert_element_type_264: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_134, torch.float32);  mm_134 = None
        convert_element_type_265: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg306_1, torch.float32);  arg306_1 = None
        mul_351: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_265, 1.1111111111111112);  convert_element_type_265 = None
        mul_352: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_263, mul_351);  convert_element_type_263 = mul_351 = None
        view_309: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg307_1, _shape_param_328);  arg307_1 = _shape_param_328 = None
        mul_353: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_309, 0.5)
        convert_element_type_266: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_309, torch.float32)
        pow_51: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 3.0)
        mul_354: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_51, 0.044715);  pow_51 = None
        add_97: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_309, mul_354);  view_309 = mul_354 = None
        mul_355: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_6: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_355);  mul_355 = None
        add_98: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_356: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_353, add_98)
        mul_357: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_352, mul_356);  mul_356 = None
        view_310: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg308_1, _shape_param_329);  arg308_1 = _shape_param_329 = None
        mul_358: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_352, view_310);  mul_352 = view_310 = None
        convert_element_type_267: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_357, torch.bfloat16);  mul_357 = None
        view_311: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_267, _shape_param_330);  convert_element_type_267 = _shape_param_330 = None
        permute_128: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_136: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_128, arg309_1);  permute_128 = None
        mm_137: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_311, arg310_1);  view_311 = arg310_1 = None
        view_312: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_137, _shape_param_331);  mm_137 = _shape_param_331 = None
        convert_element_type_268: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_312, torch.float32);  view_312 = None
        convert_element_type_269: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        mul_359: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_358, mul_353);  mul_353 = None
        mul_360: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_358, add_98);  mul_358 = add_98 = None
        convert_element_type_270: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16);  mul_360 = None
        mul_361: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_21: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_361);  mul_361 = None
        mul_362: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_359, sub_21);  mul_359 = sub_21 = None
        mul_363: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_362, 0.7978845608028654);  mul_362 = None
        convert_element_type_271: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_363, torch.bfloat16)
        mul_364: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_363, 0.044715);  mul_363 = None
        pow_52: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_266, 2.0);  convert_element_type_266 = None
        mul_365: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_52, 3.0);  pow_52 = None
        mul_366: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_364, mul_365);  mul_364 = mul_365 = None
        convert_element_type_272: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_366, torch.bfloat16);  mul_366 = None
        add_99: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_271, convert_element_type_272);  convert_element_type_271 = convert_element_type_272 = None
        mul_367: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_270, 0.5);  convert_element_type_270 = None
        add_100: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_99, mul_367);  add_99 = mul_367 = None
        view_313: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_100, _shape_param_332);  add_100 = _shape_param_332 = None
        permute_129: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_313, [1, 0])
        mm_138: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_129, arg309_1);  permute_129 = arg309_1 = None
        mm_139: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_313, arg311_1);  view_313 = arg311_1 = None
        view_314: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_139, _shape_param_333);  mm_139 = _shape_param_333 = None
        convert_element_type_273: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_314, torch.float32);  view_314 = None
        add_101: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_268, convert_element_type_273);  convert_element_type_268 = convert_element_type_273 = None
        convert_element_type_274: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        mul_368: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_101, arg312_1);  arg312_1 = None
        mul_369: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg313_1, arg314_1)
        mul_370: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_101, mul_369);  add_101 = mul_369 = None
        sum_52: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_370, [0, 1], True, dtype = torch.float32);  mul_370 = None
        view_315: "f32[512]" = torch.ops.aten.view.default(sum_52, _shape_param_334);  sum_52 = _shape_param_334 = None
        mul_371: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_368, arg313_1)
        mul_372: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_368, arg314_1);  mul_368 = None
        sum_53: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_371, [2], True, dtype = torch.float32);  mul_371 = None
        add_102: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_96, mul_372);  add_96 = mul_372 = None
        pow_53: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg314_1, 3);  arg314_1 = None
        mul_373: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_53, -0.5);  sum_53 = None
        mul_374: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_373, pow_53);  mul_373 = pow_53 = None
        expand_20: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_374, _shape_param_335);  mul_374 = _shape_param_335 = None
        div_32: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_20, 512);  expand_20 = None
        pow_54: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg313_1, 1.0);  arg313_1 = None
        mul_375: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_54, 2.0);  pow_54 = None
        mul_376: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_32, mul_375);  div_32 = mul_375 = None
        add_103: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_102, mul_376);  add_102 = mul_376 = None
        convert_element_type_275: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_103, torch.bfloat16)
        convert_element_type_276: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.bfloat16);  arg315_1 = None
        mul_377: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_276, 1.1111111111111112);  convert_element_type_276 = None
        mul_378: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_275, mul_377);  convert_element_type_275 = mul_377 = None
        view_316: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_378, _shape_param_336);  mul_378 = _shape_param_336 = None
        permute_130: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_316, [1, 0])
        mm_140: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_130, arg316_1);  permute_130 = arg316_1 = None
        mm_141: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_316, arg317_1);  view_316 = arg317_1 = None
        view_317: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_141, _shape_param_337);  mm_141 = _shape_param_337 = None
        convert_element_type_277: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        view_318: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_317, _shape_param_338);  view_317 = _shape_param_338 = None
        permute_131: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_48: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_131, memory_format = torch.contiguous_format);  permute_131 = None
        view_319: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_339);  clone_48 = _shape_param_339 = None
        bmm_48: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg318_1, view_319);  arg318_1 = None
        bmm_49: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_319, arg319_1);  view_319 = arg319_1 = None
        view_320: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_48, _shape_param_340);  bmm_48 = _shape_param_340 = None
        view_321: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_49, _shape_param_341);  bmm_49 = _shape_param_341 = None
        convert_element_type_278: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg320_1, torch.bfloat16);  arg320_1 = None
        mul_379: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_278, 1.1111111111111112);  convert_element_type_278 = None
        mul_380: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_321, mul_379);  view_321 = mul_379 = None
        convert_element_type_279: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_380, torch.float32);  mul_380 = None
        view_322: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg321_1, _shape_param_342);  arg321_1 = _shape_param_342 = None
        convert_element_type_280: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_322, torch.bfloat16);  view_322 = None
        convert_element_type_281: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_280, torch.float32);  convert_element_type_280 = None
        sub_22: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_281, arg322_1);  convert_element_type_281 = arg322_1 = None
        exp_13: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        div_33: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, arg323_1);  exp_13 = arg323_1 = None
        mul_381: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_279, div_33);  convert_element_type_279 = None
        sum_54: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_381, [-1], True)
        neg_12: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_33);  div_33 = None
        fma_12: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_12, sum_54, mul_381);  neg_12 = sum_54 = mul_381 = None
        convert_element_type_282: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None
        view_323: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_282, _shape_param_343);  convert_element_type_282 = _shape_param_343 = None
        bmm_50: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg324_1, view_323);  arg324_1 = None
        bmm_51: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_323, arg325_1);  view_323 = arg325_1 = None
        view_324: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_50, _shape_param_344);  bmm_50 = _shape_param_344 = None
        view_325: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_51, _shape_param_345);  bmm_51 = _shape_param_345 = None
        permute_132: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_324, [0, 1, 3, 2]);  view_324 = None
        permute_133: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
        clone_49: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
        view_326: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_49, _shape_param_346);  clone_49 = _shape_param_346 = None
        view_327: "bf16[4096, 384]" = torch.ops.aten.view.default(view_326, _shape_param_347);  view_326 = _shape_param_347 = None
        permute_134: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_327, [1, 0])
        mm_142: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_134, arg36_1);  permute_134 = None
        mm_143: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_327, arg326_1);  view_327 = arg326_1 = None
        view_328: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_143, _shape_param_348);  mm_143 = _shape_param_348 = None
        convert_element_type_283: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        add_104: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_89, convert_element_type_283);  add_89 = convert_element_type_283 = None
        convert_element_type_284: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        permute_135: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_132, [0, 2, 1, 3]);  permute_132 = None
        view_329: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_135, _shape_param_349);  permute_135 = _shape_param_349 = None
        clone_50: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_329, memory_format = torch.contiguous_format);  view_329 = None
        view_330: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_50, _shape_param_350);  clone_50 = _shape_param_350 = None
        permute_136: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_330, [1, 0])
        mm_144: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_136, arg36_1);  permute_136 = None
        mm_145: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_330, arg327_1);  view_330 = arg327_1 = None
        view_331: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_145, _shape_param_351);  mm_145 = _shape_param_351 = None
        convert_element_type_285: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_331, torch.float32);  view_331 = None
        add_105: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_104, convert_element_type_285);  add_104 = convert_element_type_285 = None
        convert_element_type_286: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        permute_137: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
        clone_51: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_332: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_51, _shape_param_352);  clone_51 = _shape_param_352 = None
        view_333: "bf16[4096, 384]" = torch.ops.aten.view.default(view_332, _shape_param_353);  view_332 = _shape_param_353 = None
        permute_138: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_333, [1, 0])
        mm_146: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_138, arg328_1);  permute_138 = arg328_1 = None
        mm_147: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_333, arg329_1);  view_333 = arg329_1 = None
        view_334: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_147, _shape_param_354);  mm_147 = _shape_param_354 = None
        convert_element_type_287: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_334, torch.float32);  view_334 = None
        convert_element_type_288: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None
        mul_382: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_287, arg330_1);  arg330_1 = None
        mul_383: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg331_1, arg332_1)
        mul_384: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_287, mul_383);  convert_element_type_287 = mul_383 = None
        sum_55: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_384, [0, 1], True, dtype = torch.float32);  mul_384 = None
        view_335: "f32[512]" = torch.ops.aten.view.default(sum_55, _shape_param_355);  sum_55 = _shape_param_355 = None
        mul_385: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_382, arg331_1)
        mul_386: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_382, arg332_1);  mul_382 = None
        sum_56: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True, dtype = torch.float32);  mul_385 = None
        add_106: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_103, mul_386);  add_103 = mul_386 = None
        pow_55: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg332_1, 3);  arg332_1 = None
        mul_387: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_56, -0.5);  sum_56 = None
        mul_388: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_387, pow_55);  mul_387 = pow_55 = None
        expand_21: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_388, _shape_param_356);  mul_388 = _shape_param_356 = None
        div_34: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_21, 512);  expand_21 = None
        pow_56: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg331_1, 1.0);  arg331_1 = None
        mul_389: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_56, 2.0);  pow_56 = None
        mul_390: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_34, mul_389);  div_34 = mul_389 = None
        add_107: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_106, mul_390);  add_106 = mul_390 = None
        convert_element_type_289: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_107, torch.bfloat16)
        convert_element_type_290: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.bfloat16);  arg333_1 = None
        mul_391: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_290, 1.1111111111111112);  convert_element_type_290 = None
        mul_392: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_289, mul_391);  convert_element_type_289 = mul_391 = None
        view_336: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_392, _shape_param_357);  mul_392 = _shape_param_357 = None
        permute_139: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_148: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_139, arg334_1);  permute_139 = arg334_1 = None
        mm_149: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_336, arg335_1);  view_336 = arg335_1 = None
        view_337: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_149, _shape_param_358);  mm_149 = _shape_param_358 = None
        convert_element_type_291: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None
        view_338: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_337, _shape_param_359);  view_337 = _shape_param_359 = None
        permute_140: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        clone_52: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_339: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_360);  clone_52 = _shape_param_360 = None
        bmm_52: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg336_1, view_339);  arg336_1 = None
        bmm_53: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_339, arg337_1);  view_339 = arg337_1 = None
        view_340: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_52, _shape_param_361);  bmm_52 = _shape_param_361 = None
        view_341: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_53, _shape_param_362);  bmm_53 = _shape_param_362 = None
        convert_element_type_292: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.bfloat16);  arg338_1 = None
        mul_393: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_292, 1.1111111111111112);  convert_element_type_292 = None
        mul_394: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_341, mul_393);  view_341 = mul_393 = None
        convert_element_type_293: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_394, torch.float32);  mul_394 = None
        convert_element_type_294: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg339_1, torch.float32);  arg339_1 = None
        sub_23: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_294, arg340_1);  convert_element_type_294 = arg340_1 = None
        exp_14: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        div_35: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, arg341_1);  exp_14 = arg341_1 = None
        mul_395: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_293, div_35);  convert_element_type_293 = None
        sum_57: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [-1], True)
        neg_13: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_35);  div_35 = None
        fma_13: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_13, sum_57, mul_395);  neg_13 = sum_57 = mul_395 = None
        convert_element_type_295: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None
        view_342: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_295, _shape_param_363);  convert_element_type_295 = _shape_param_363 = None
        view_343: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_342, _shape_param_364);  view_342 = _shape_param_364 = None
        view_344: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_343, _shape_param_365);  _shape_param_365 = None
        convert_element_type_296: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_343, torch.float32);  view_343 = None
        add_108: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_92, convert_element_type_296);  add_92 = convert_element_type_296 = None
        bmm_54: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg342_1, view_344);  arg342_1 = None
        bmm_55: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_344, arg343_1);  view_344 = arg343_1 = None
        view_345: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_54, _shape_param_366);  bmm_54 = _shape_param_366 = None
        view_346: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_55, _shape_param_367);  bmm_55 = _shape_param_367 = None
        permute_141: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_345, [0, 1, 3, 2]);  view_345 = None
        permute_142: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        clone_53: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
        view_347: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_53, _shape_param_368);  clone_53 = _shape_param_368 = None
        view_348: "bf16[4096, 384]" = torch.ops.aten.view.default(view_347, _shape_param_369);  view_347 = _shape_param_369 = None
        permute_143: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_348, [1, 0])
        mm_150: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_143, arg344_1);  permute_143 = None
        mm_151: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_348, arg345_1);  view_348 = arg345_1 = None
        view_349: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_151, _shape_param_370);  mm_151 = _shape_param_370 = None
        convert_element_type_297: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        convert_element_type_298: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        permute_144: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_141, [0, 2, 1, 3]);  permute_141 = None
        view_350: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_144, _shape_param_371);  permute_144 = _shape_param_371 = None
        clone_54: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_350, memory_format = torch.contiguous_format);  view_350 = None
        view_351: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_54, _shape_param_372);  clone_54 = _shape_param_372 = None
        permute_145: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_351, [1, 0])
        mm_152: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_145, arg344_1);  permute_145 = None
        mm_153: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_351, arg346_1);  view_351 = arg346_1 = None
        view_352: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_153, _shape_param_373);  mm_153 = _shape_param_373 = None
        convert_element_type_299: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_352, torch.float32);  view_352 = None
        add_109: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_297, convert_element_type_299);  convert_element_type_297 = convert_element_type_299 = None
        convert_element_type_300: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        permute_146: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        clone_55: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_146, memory_format = torch.contiguous_format);  permute_146 = None
        view_353: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_55, _shape_param_374);  clone_55 = _shape_param_374 = None
        view_354: "bf16[4096, 384]" = torch.ops.aten.view.default(view_353, _shape_param_375);  view_353 = _shape_param_375 = None
        permute_147: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_154: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_147, arg344_1);  permute_147 = arg344_1 = None
        mm_155: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_354, arg347_1);  view_354 = arg347_1 = None
        view_355: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_155, _shape_param_376);  mm_155 = _shape_param_376 = None
        convert_element_type_301: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_355, torch.float32);  view_355 = None
        add_110: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_109, convert_element_type_301);  add_109 = convert_element_type_301 = None
        convert_element_type_302: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        mul_396: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_110, arg348_1);  arg348_1 = None
        mul_397: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg349_1, arg350_1)
        mul_398: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_110, mul_397);  add_110 = mul_397 = None
        sum_58: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_398, [0, 1], True, dtype = torch.float32);  mul_398 = None
        view_356: "f32[512]" = torch.ops.aten.view.default(sum_58, _shape_param_377);  sum_58 = _shape_param_377 = None
        mul_399: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_396, arg349_1)
        mul_400: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_396, arg350_1);  mul_396 = None
        sum_59: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_399, [2], True, dtype = torch.float32);  mul_399 = None
        add_111: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_107, mul_400);  add_107 = mul_400 = None
        pow_57: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg350_1, 3);  arg350_1 = None
        mul_401: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_59, -0.5);  sum_59 = None
        mul_402: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_401, pow_57);  mul_401 = pow_57 = None
        expand_22: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_402, _shape_param_378);  mul_402 = _shape_param_378 = None
        div_36: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_22, 512);  expand_22 = None
        pow_58: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg349_1, 1.0);  arg349_1 = None
        mul_403: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_58, 2.0);  pow_58 = None
        mul_404: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_36, mul_403);  div_36 = mul_403 = None
        add_112: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_111, mul_404);  add_111 = mul_404 = None
        convert_element_type_303: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16)
        convert_element_type_304: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg351_1, torch.bfloat16);  arg351_1 = None
        mul_405: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_304, 1.1111111111111112);  convert_element_type_304 = None
        mul_406: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_303, mul_405);  convert_element_type_303 = mul_405 = None
        view_357: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_406, _shape_param_379);  mul_406 = _shape_param_379 = None
        permute_148: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_357, [1, 0])
        mm_156: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_148, arg352_1);  permute_148 = arg352_1 = None
        mm_157: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_357, arg353_1);  view_357 = arg353_1 = None
        view_358: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_157, _shape_param_380);  mm_157 = _shape_param_380 = None
        convert_element_type_305: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_358, torch.float32);  view_358 = None
        convert_element_type_306: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        convert_element_type_307: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg354_1, torch.float32);  arg354_1 = None
        mul_407: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_307, 1.1111111111111112);  convert_element_type_307 = None
        mul_408: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_305, mul_407);  convert_element_type_305 = mul_407 = None
        view_359: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg355_1, _shape_param_381);  arg355_1 = _shape_param_381 = None
        mul_409: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_359, 0.5)
        convert_element_type_308: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_359, torch.float32)
        pow_59: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_308, 3.0)
        mul_410: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_59, 0.044715);  pow_59 = None
        add_113: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_359, mul_410);  view_359 = mul_410 = None
        mul_411: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        tanh_7: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_411);  mul_411 = None
        add_114: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_412: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_409, add_114)
        mul_413: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_408, mul_412);  mul_412 = None
        view_360: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg356_1, _shape_param_382);  arg356_1 = _shape_param_382 = None
        mul_414: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_408, view_360);  mul_408 = view_360 = None
        convert_element_type_309: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_413, torch.bfloat16);  mul_413 = None
        view_361: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_309, _shape_param_383);  convert_element_type_309 = _shape_param_383 = None
        permute_149: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_361, [1, 0])
        mm_158: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_149, arg357_1);  permute_149 = None
        mm_159: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_361, arg358_1);  view_361 = arg358_1 = None
        view_362: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_159, _shape_param_384);  mm_159 = _shape_param_384 = None
        convert_element_type_310: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_362, torch.float32);  view_362 = None
        convert_element_type_311: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_158, torch.float32);  mm_158 = None
        mul_415: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_414, mul_409);  mul_409 = None
        mul_416: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_414, add_114);  mul_414 = add_114 = None
        convert_element_type_312: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_416, torch.bfloat16);  mul_416 = None
        mul_417: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_24: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_417);  mul_417 = None
        mul_418: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_415, sub_24);  mul_415 = sub_24 = None
        mul_419: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_418, 0.7978845608028654);  mul_418 = None
        convert_element_type_313: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_419, torch.bfloat16)
        mul_420: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_419, 0.044715);  mul_419 = None
        pow_60: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_308, 2.0);  convert_element_type_308 = None
        mul_421: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_60, 3.0);  pow_60 = None
        mul_422: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
        convert_element_type_314: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_422, torch.bfloat16);  mul_422 = None
        add_115: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_313, convert_element_type_314);  convert_element_type_313 = convert_element_type_314 = None
        mul_423: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_312, 0.5);  convert_element_type_312 = None
        add_116: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_115, mul_423);  add_115 = mul_423 = None
        view_363: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_116, _shape_param_385);  add_116 = _shape_param_385 = None
        permute_150: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_363, [1, 0])
        mm_160: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_150, arg357_1);  permute_150 = arg357_1 = None
        mm_161: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_363, arg359_1);  view_363 = arg359_1 = None
        view_364: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_161, _shape_param_386);  mm_161 = _shape_param_386 = None
        convert_element_type_315: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_364, torch.float32);  view_364 = None
        add_117: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_310, convert_element_type_315);  convert_element_type_310 = convert_element_type_315 = None
        convert_element_type_316: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        mul_424: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_117, arg360_1);  arg360_1 = None
        mul_425: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg361_1, arg362_1)
        mul_426: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_117, mul_425);  add_117 = mul_425 = None
        sum_60: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 1], True, dtype = torch.float32);  mul_426 = None
        view_365: "f32[512]" = torch.ops.aten.view.default(sum_60, _shape_param_387);  sum_60 = _shape_param_387 = None
        mul_427: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_424, arg361_1)
        mul_428: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_424, arg362_1);  mul_424 = None
        sum_61: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True, dtype = torch.float32);  mul_427 = None
        add_118: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_112, mul_428);  add_112 = mul_428 = None
        pow_61: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg362_1, 3);  arg362_1 = None
        mul_429: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_61, -0.5);  sum_61 = None
        mul_430: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_429, pow_61);  mul_429 = pow_61 = None
        expand_23: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_430, _shape_param_388);  mul_430 = _shape_param_388 = None
        div_37: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_23, 512);  expand_23 = None
        pow_62: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg361_1, 1.0);  arg361_1 = None
        mul_431: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_62, 2.0);  pow_62 = None
        mul_432: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_37, mul_431);  div_37 = mul_431 = None
        add_119: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_118, mul_432);  add_118 = mul_432 = None
        convert_element_type_317: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_119, torch.bfloat16)
        convert_element_type_318: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg363_1, torch.bfloat16);  arg363_1 = None
        mul_433: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_318, 1.1111111111111112);  convert_element_type_318 = None
        mul_434: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_317, mul_433);  convert_element_type_317 = mul_433 = None
        view_366: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_434, _shape_param_389);  mul_434 = _shape_param_389 = None
        permute_151: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_162: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_151, arg364_1);  permute_151 = arg364_1 = None
        mm_163: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_366, arg365_1);  view_366 = arg365_1 = None
        view_367: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_163, _shape_param_390);  mm_163 = _shape_param_390 = None
        convert_element_type_319: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        view_368: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_367, _shape_param_391);  view_367 = _shape_param_391 = None
        permute_152: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
        clone_56: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
        view_369: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_56, _shape_param_392);  clone_56 = _shape_param_392 = None
        bmm_56: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg366_1, view_369);  arg366_1 = None
        bmm_57: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_369, arg367_1);  view_369 = arg367_1 = None
        view_370: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_56, _shape_param_393);  bmm_56 = _shape_param_393 = None
        view_371: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_57, _shape_param_394);  bmm_57 = _shape_param_394 = None
        convert_element_type_320: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg368_1, torch.bfloat16);  arg368_1 = None
        mul_435: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_320, 1.1111111111111112);  convert_element_type_320 = None
        mul_436: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_371, mul_435);  view_371 = mul_435 = None
        convert_element_type_321: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_436, torch.float32);  mul_436 = None
        view_372: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg369_1, _shape_param_395);  arg369_1 = _shape_param_395 = None
        convert_element_type_322: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_372, torch.bfloat16);  view_372 = None
        convert_element_type_323: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_322, torch.float32);  convert_element_type_322 = None
        sub_25: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_323, arg370_1);  convert_element_type_323 = arg370_1 = None
        exp_15: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        div_38: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, arg371_1);  exp_15 = arg371_1 = None
        mul_437: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_321, div_38);  convert_element_type_321 = None
        sum_62: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [-1], True)
        neg_14: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_38);  div_38 = None
        fma_14: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_14, sum_62, mul_437);  neg_14 = sum_62 = mul_437 = None
        convert_element_type_324: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None
        view_373: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_324, _shape_param_396);  convert_element_type_324 = _shape_param_396 = None
        bmm_58: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg372_1, view_373);  arg372_1 = None
        bmm_59: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_373, arg373_1);  view_373 = arg373_1 = None
        view_374: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_58, _shape_param_397);  bmm_58 = _shape_param_397 = None
        view_375: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_59, _shape_param_398);  bmm_59 = _shape_param_398 = None
        permute_153: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_374, [0, 1, 3, 2]);  view_374 = None
        permute_154: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
        clone_57: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_376: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_57, _shape_param_399);  clone_57 = _shape_param_399 = None
        view_377: "bf16[4096, 384]" = torch.ops.aten.view.default(view_376, _shape_param_400);  view_376 = _shape_param_400 = None
        permute_155: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_377, [1, 0])
        mm_164: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_155, arg36_1);  permute_155 = None
        mm_165: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_377, arg374_1);  view_377 = arg374_1 = None
        view_378: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_165, _shape_param_401);  mm_165 = _shape_param_401 = None
        convert_element_type_325: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_378, torch.float32);  view_378 = None
        add_120: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_105, convert_element_type_325);  add_105 = convert_element_type_325 = None
        convert_element_type_326: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        permute_156: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_153, [0, 2, 1, 3]);  permute_153 = None
        view_379: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_156, _shape_param_402);  permute_156 = _shape_param_402 = None
        clone_58: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_379, memory_format = torch.contiguous_format);  view_379 = None
        view_380: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_58, _shape_param_403);  clone_58 = _shape_param_403 = None
        permute_157: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_166: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_157, arg36_1);  permute_157 = arg36_1 = None
        mm_167: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_380, arg375_1);  view_380 = arg375_1 = None
        view_381: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_167, _shape_param_404);  mm_167 = _shape_param_404 = None
        convert_element_type_327: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_381, torch.float32);  view_381 = None
        add_121: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_120, convert_element_type_327);  add_120 = convert_element_type_327 = None
        convert_element_type_328: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        permute_158: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
        clone_59: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_382: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_59, _shape_param_405);  clone_59 = _shape_param_405 = None
        view_383: "bf16[4096, 384]" = torch.ops.aten.view.default(view_382, _shape_param_406);  view_382 = _shape_param_406 = None
        permute_159: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_383, [1, 0])
        mm_168: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_159, arg376_1);  permute_159 = arg376_1 = None
        mm_169: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_383, arg377_1);  view_383 = arg377_1 = None
        view_384: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_169, _shape_param_407);  mm_169 = _shape_param_407 = None
        convert_element_type_329: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_384, torch.float32);  view_384 = None
        convert_element_type_330: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        mul_438: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_329, arg378_1);  arg378_1 = None
        mul_439: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg379_1, arg380_1)
        mul_440: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_329, mul_439);  convert_element_type_329 = mul_439 = None
        sum_63: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_440, [0, 1], True, dtype = torch.float32);  mul_440 = None
        view_385: "f32[512]" = torch.ops.aten.view.default(sum_63, _shape_param_408);  sum_63 = _shape_param_408 = None
        mul_441: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_438, arg379_1)
        mul_442: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_438, arg380_1);  mul_438 = None
        sum_64: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True, dtype = torch.float32);  mul_441 = None
        add_122: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_119, mul_442);  add_119 = mul_442 = None
        pow_63: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg380_1, 3);  arg380_1 = None
        mul_443: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_64, -0.5);  sum_64 = None
        mul_444: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_443, pow_63);  mul_443 = pow_63 = None
        expand_24: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_444, _shape_param_409);  mul_444 = _shape_param_409 = None
        div_39: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_24, 512);  expand_24 = None
        pow_64: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg379_1, 1.0);  arg379_1 = None
        mul_445: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
        mul_446: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_39, mul_445);  div_39 = mul_445 = None
        add_123: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_122, mul_446);  add_122 = mul_446 = None
        convert_element_type_331: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_123, torch.bfloat16)
        convert_element_type_332: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg381_1, torch.bfloat16);  arg381_1 = None
        mul_447: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_332, 1.1111111111111112);  convert_element_type_332 = None
        mul_448: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_331, mul_447);  convert_element_type_331 = mul_447 = None
        view_386: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_448, _shape_param_410);  mul_448 = _shape_param_410 = None
        permute_160: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_386, [1, 0])
        mm_170: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_160, arg382_1);  permute_160 = arg382_1 = None
        mm_171: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_386, arg383_1);  view_386 = arg383_1 = None
        view_387: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_171, _shape_param_411);  mm_171 = _shape_param_411 = None
        convert_element_type_333: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None
        view_388: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_387, _shape_param_412);  view_387 = _shape_param_412 = None
        permute_161: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_388, [0, 2, 1, 3]);  view_388 = None
        clone_60: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_389: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_413);  clone_60 = _shape_param_413 = None
        bmm_60: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg384_1, view_389);  arg384_1 = None
        bmm_61: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_389, arg385_1);  view_389 = arg385_1 = None
        view_390: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_60, _shape_param_414);  bmm_60 = _shape_param_414 = None
        view_391: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_61, _shape_param_415);  bmm_61 = _shape_param_415 = None
        convert_element_type_334: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg386_1, torch.bfloat16);  arg386_1 = None
        mul_449: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_334, 1.1111111111111112);  convert_element_type_334 = None
        mul_450: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_391, mul_449);  view_391 = mul_449 = None
        convert_element_type_335: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_450, torch.float32);  mul_450 = None
        unsqueeze_1: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(arg387_1, 3)
        unsqueeze_2: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(arg387_1, 2);  arg387_1 = None
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_1);  unsqueeze_2 = unsqueeze_1 = None
        expand_25: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(le, _shape_param_416);  le = _shape_param_416 = None
        full_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_25, full_1, full_2);  expand_25 = full_1 = full_2 = None
        view_392: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg388_1, _shape_param_417);  arg388_1 = _shape_param_417 = None
        permute_162: "f32[6, 128, 128]" = torch.ops.aten.permute.default(arg389_1, [2, 0, 1]);  arg389_1 = None
        unsqueeze_3: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_162, 0);  permute_162 = None
        add_124: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_3, where_3);  unsqueeze_3 = where_3 = None
        add_125: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_392, add_124);  view_392 = add_124 = None
        convert_element_type_336: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16);  add_125 = None
        convert_element_type_337: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_336, torch.float32);  convert_element_type_336 = None
        sub_26: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_337, arg390_1);  convert_element_type_337 = arg390_1 = None
        exp_16: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
        div_40: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, arg391_1);  exp_16 = arg391_1 = None
        mul_451: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_335, div_40);  convert_element_type_335 = None
        sum_65: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [-1], True)
        neg_15: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_40);  div_40 = None
        fma_15: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_15, sum_65, mul_451);  neg_15 = sum_65 = mul_451 = None
        convert_element_type_338: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None
        view_393: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_338, _shape_param_418);  convert_element_type_338 = _shape_param_418 = None
        view_394: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_393, _shape_param_419);  view_393 = _shape_param_419 = None
        view_395: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_394, _shape_param_420);  _shape_param_420 = None
        convert_element_type_339: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_394, torch.float32);  view_394 = None
        add_126: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_108, convert_element_type_339);  add_108 = convert_element_type_339 = None
        sum_66: "f32[1, 6, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_126, [0], True, dtype = torch.float32);  add_126 = None
        squeeze: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sum_66, 0);  sum_66 = None
        permute_163: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
        full_3: "b8[128, 128, 1]" = torch.ops.aten.full.default(_shape_param_421, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_421 = None
        full_4: "f32[32, 6]" = torch.ops.aten.full.default(_shape_param_422, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_422 = None
        _unsafe_masked_index_put_accumulate: "f32[32, 6]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, full_3, [arg392_1], permute_163);  arg392_1 = permute_163 = None
        bmm_62: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg393_1, view_395);  arg393_1 = None
        bmm_63: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_395, arg394_1);  view_395 = arg394_1 = None
        view_396: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_62, _shape_param_423);  bmm_62 = _shape_param_423 = None
        view_397: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_63, _shape_param_424);  bmm_63 = _shape_param_424 = None
        permute_164: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_396, [0, 1, 3, 2]);  view_396 = None
        permute_165: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
        clone_61: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_398: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_61, _shape_param_425);  clone_61 = _shape_param_425 = None
        view_399: "bf16[4096, 384]" = torch.ops.aten.view.default(view_398, _shape_param_426);  view_398 = _shape_param_426 = None
        permute_166: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_399, [1, 0])
        mm_172: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_166, arg395_1);  permute_166 = None
        mm_173: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_399, arg396_1);  view_399 = arg396_1 = None
        view_400: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_173, _shape_param_427);  mm_173 = _shape_param_427 = None
        convert_element_type_340: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_400, torch.float32);  view_400 = None
        convert_element_type_341: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None
        permute_167: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_164, [0, 2, 1, 3]);  permute_164 = None
        view_401: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_167, _shape_param_428);  permute_167 = _shape_param_428 = None
        clone_62: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_401, memory_format = torch.contiguous_format);  view_401 = None
        view_402: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_62, _shape_param_429);  clone_62 = _shape_param_429 = None
        permute_168: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_402, [1, 0])
        mm_174: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_168, arg395_1);  permute_168 = None
        mm_175: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_402, arg397_1);  view_402 = arg397_1 = None
        view_403: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_175, _shape_param_430);  mm_175 = _shape_param_430 = None
        convert_element_type_342: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_403, torch.float32);  view_403 = None
        add_127: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_340, convert_element_type_342);  convert_element_type_340 = convert_element_type_342 = None
        convert_element_type_343: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        permute_169: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
        clone_63: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_404: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_63, _shape_param_431);  clone_63 = _shape_param_431 = None
        view_405: "bf16[4096, 384]" = torch.ops.aten.view.default(view_404, _shape_param_432);  view_404 = _shape_param_432 = None
        permute_170: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_405, [1, 0])
        mm_176: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_170, arg395_1);  permute_170 = arg395_1 = None
        mm_177: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_405, arg398_1);  view_405 = arg398_1 = None
        view_406: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_177, _shape_param_433);  mm_177 = _shape_param_433 = None
        convert_element_type_344: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_406, torch.float32);  view_406 = None
        add_128: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_127, convert_element_type_344);  add_127 = convert_element_type_344 = None
        convert_element_type_345: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        mul_452: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_128, arg399_1);  arg399_1 = None
        mul_453: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg400_1, arg401_1)
        mul_454: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_453, 1.1111111111111112);  mul_453 = None
        mul_455: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_454, arg402_1)
        mul_456: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_128, mul_455);  add_128 = mul_455 = None
        sum_67: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_456, [0, 1], True, dtype = torch.float32);  mul_456 = None
        view_407: "f32[512]" = torch.ops.aten.view.default(sum_67, _shape_param_434);  sum_67 = _shape_param_434 = None
        mul_457: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_452, mul_454)
        mul_458: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_452, arg402_1);  mul_452 = None
        sum_68: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_457, [2], True, dtype = torch.float32);  mul_457 = None
        add_129: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_123, mul_458);  add_123 = mul_458 = None
        pow_65: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg402_1, 3);  arg402_1 = None
        mul_459: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_68, -0.5);  sum_68 = None
        mul_460: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_459, pow_65);  mul_459 = pow_65 = None
        expand_26: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_460, _shape_param_435);  mul_460 = _shape_param_435 = None
        div_41: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_26, 512);  expand_26 = None
        pow_66: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_454, 1.0);  mul_454 = None
        mul_461: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_66, 2.0);  pow_66 = None
        mul_462: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_41, mul_461);  div_41 = mul_461 = None
        add_130: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_129, mul_462);  add_129 = mul_462 = None
        convert_element_type_346: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg400_1, torch.float32);  arg400_1 = None
        mul_463: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_346, 1.1111111111111112);  convert_element_type_346 = None
        mul_464: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_130, mul_463);  add_130 = mul_463 = None
        ge: "b8[32, 128]" = torch.ops.aten.ge.Scalar(arg403_1, 0)
        lt: "b8[32, 128]" = torch.ops.aten.lt.Scalar(arg403_1, 250112)
        bitwise_and: "b8[32, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_1: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg403_1, -1)
        bitwise_and_1: "b8[32, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_1);  bitwise_and = ne_1 = None
        unsqueeze_4: "b8[32, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_5: "f32[250112, 512]" = torch.ops.aten.full.default(_shape_param_436, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_436 = None
        _unsafe_masked_index_put_accumulate_1: "f32[250112, 512]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_5, unsqueeze_4, [arg403_1], mul_464);  mul_464 = None
        add_131: "f32[250112, 512]" = torch.ops.aten.add.Tensor(convert_element_type_7, _unsafe_masked_index_put_accumulate_1);  convert_element_type_7 = _unsafe_masked_index_put_accumulate_1 = None
        convert_element_type_347: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg404_1, torch.float32);  arg404_1 = None
        mul_465: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_347, 1.1111111111111112);  convert_element_type_347 = None
        mul_466: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_121, mul_465);  add_121 = mul_465 = None
        mul_467: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_466, arg405_1);  arg405_1 = None
        mul_468: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg406_1, arg407_1)
        mul_469: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_466, mul_468);  mul_466 = mul_468 = None
        sum_69: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1], True, dtype = torch.float32);  mul_469 = None
        view_408: "f32[512]" = torch.ops.aten.view.default(sum_69, _shape_param_437);  sum_69 = _shape_param_437 = None
        mul_470: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_467, arg406_1)
        mul_471: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_467, arg407_1);  mul_467 = None
        sum_70: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [2], True, dtype = torch.float32);  mul_470 = None
        pow_67: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg407_1, 3);  arg407_1 = None
        mul_472: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_70, -0.5);  sum_70 = None
        mul_473: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_472, pow_67);  mul_472 = pow_67 = None
        expand_27: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_473, _shape_param_438);  mul_473 = _shape_param_438 = None
        div_42: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_27, 512);  expand_27 = None
        pow_68: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg406_1, 1.0);  arg406_1 = None
        mul_474: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_68, 2.0);  pow_68 = None
        mul_475: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_42, mul_474);  div_42 = mul_474 = None
        add_132: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_471, mul_475);  mul_471 = mul_475 = None
        convert_element_type_348: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_132, torch.bfloat16)
        convert_element_type_349: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg408_1, torch.bfloat16);  arg408_1 = None
        mul_476: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_349, 1.1111111111111112);  convert_element_type_349 = None
        mul_477: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_348, mul_476);  convert_element_type_348 = mul_476 = None
        view_409: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_477, _shape_param_439);  mul_477 = _shape_param_439 = None
        permute_171: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_409, [1, 0])
        mm_178: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_171, arg409_1);  permute_171 = arg409_1 = None
        mm_179: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_409, arg410_1);  view_409 = arg410_1 = None
        view_410: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_179, _shape_param_440);  mm_179 = _shape_param_440 = None
        convert_element_type_350: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_410, torch.float32);  view_410 = None
        convert_element_type_351: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        convert_element_type_352: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg411_1, torch.float32);  arg411_1 = None
        mul_478: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_352, 1.1111111111111112);  convert_element_type_352 = None
        mul_479: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_350, mul_478);  convert_element_type_350 = mul_478 = None
        view_411: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg412_1, _shape_param_441);  arg412_1 = _shape_param_441 = None
        mul_480: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_411, 0.5)
        convert_element_type_353: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_411, torch.float32)
        pow_69: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_353, 3.0)
        mul_481: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_69, 0.044715);  pow_69 = None
        add_133: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_411, mul_481);  view_411 = mul_481 = None
        mul_482: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_133, 0.7978845608028654);  add_133 = None
        tanh_8: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_482);  mul_482 = None
        add_134: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_483: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_480, add_134)
        mul_484: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_479, mul_483);  mul_483 = None
        view_412: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg413_1, _shape_param_442);  arg413_1 = _shape_param_442 = None
        mul_485: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_479, view_412);  mul_479 = view_412 = None
        convert_element_type_354: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_484, torch.bfloat16);  mul_484 = None
        view_413: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_354, _shape_param_443);  convert_element_type_354 = _shape_param_443 = None
        permute_172: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_413, [1, 0])
        mm_180: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_172, arg414_1);  permute_172 = None
        mm_181: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_413, arg415_1);  view_413 = arg415_1 = None
        view_414: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_181, _shape_param_444);  mm_181 = _shape_param_444 = None
        convert_element_type_355: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_414, torch.float32);  view_414 = None
        convert_element_type_356: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None
        mul_486: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_485, mul_480);  mul_480 = None
        mul_487: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_485, add_134);  mul_485 = add_134 = None
        convert_element_type_357: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_487, torch.bfloat16);  mul_487 = None
        mul_488: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_27: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_488);  mul_488 = None
        mul_489: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_486, sub_27);  mul_486 = sub_27 = None
        mul_490: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_489, 0.7978845608028654);  mul_489 = None
        convert_element_type_358: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_490, torch.bfloat16)
        mul_491: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_490, 0.044715);  mul_490 = None
        pow_70: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_353, 2.0);  convert_element_type_353 = None
        mul_492: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_70, 3.0);  pow_70 = None
        mul_493: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
        convert_element_type_359: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_493, torch.bfloat16);  mul_493 = None
        add_135: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_358, convert_element_type_359);  convert_element_type_358 = convert_element_type_359 = None
        mul_494: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_357, 0.5);  convert_element_type_357 = None
        add_136: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_135, mul_494);  add_135 = mul_494 = None
        view_415: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_136, _shape_param_445);  add_136 = _shape_param_445 = None
        permute_173: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_415, [1, 0])
        mm_182: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_173, arg414_1);  permute_173 = arg414_1 = None
        mm_183: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_415, arg416_1);  view_415 = arg416_1 = None
        view_416: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_183, _shape_param_446);  mm_183 = _shape_param_446 = None
        convert_element_type_360: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_416, torch.float32);  view_416 = None
        add_137: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_355, convert_element_type_360);  convert_element_type_355 = convert_element_type_360 = None
        convert_element_type_361: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_182, torch.float32);  mm_182 = None
        mul_495: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_137, arg417_1);  arg417_1 = None
        mul_496: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg418_1, arg419_1)
        mul_497: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_137, mul_496);  add_137 = mul_496 = None
        sum_71: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 1], True, dtype = torch.float32);  mul_497 = None
        view_417: "f32[512]" = torch.ops.aten.view.default(sum_71, _shape_param_447);  sum_71 = _shape_param_447 = None
        mul_498: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_495, arg418_1)
        mul_499: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_495, arg419_1);  mul_495 = None
        sum_72: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_498, [2], True, dtype = torch.float32);  mul_498 = None
        add_138: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_132, mul_499);  add_132 = mul_499 = None
        pow_71: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg419_1, 3);  arg419_1 = None
        mul_500: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_72, -0.5);  sum_72 = None
        mul_501: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_500, pow_71);  mul_500 = pow_71 = None
        expand_28: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_501, _shape_param_448);  mul_501 = _shape_param_448 = None
        div_43: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_28, 512);  expand_28 = None
        pow_72: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg418_1, 1.0);  arg418_1 = None
        mul_502: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
        mul_503: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_43, mul_502);  div_43 = mul_502 = None
        add_139: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_138, mul_503);  add_138 = mul_503 = None
        convert_element_type_362: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_139, torch.bfloat16)
        convert_element_type_363: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg420_1, torch.bfloat16);  arg420_1 = None
        mul_504: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_363, 1.1111111111111112);  convert_element_type_363 = None
        mul_505: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_362, mul_504);  convert_element_type_362 = mul_504 = None
        view_418: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_505, _shape_param_449);  mul_505 = _shape_param_449 = None
        permute_174: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_418, [1, 0])
        mm_184: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_174, arg421_1);  permute_174 = arg421_1 = None
        mm_185: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_418, arg422_1);  view_418 = arg422_1 = None
        view_419: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_185, _shape_param_450);  mm_185 = _shape_param_450 = None
        convert_element_type_364: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        view_420: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_419, _shape_param_451);  view_419 = _shape_param_451 = None
        permute_175: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None
        clone_64: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
        view_421: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_452);  clone_64 = _shape_param_452 = None
        bmm_64: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg423_1, view_421);  arg423_1 = None
        bmm_65: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_421, arg424_1);  view_421 = arg424_1 = None
        view_422: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_64, _shape_param_453);  bmm_64 = _shape_param_453 = None
        view_423: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_65, _shape_param_454);  bmm_65 = _shape_param_454 = None
        convert_element_type_365: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg425_1, torch.bfloat16);  arg425_1 = None
        mul_506: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_365, 1.1111111111111112);  convert_element_type_365 = None
        mul_507: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_423, mul_506);  view_423 = mul_506 = None
        convert_element_type_366: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_507, torch.float32);  mul_507 = None
        convert_element_type_367: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg426_1, torch.float32);  arg426_1 = None
        sub_28: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_367, arg427_1);  convert_element_type_367 = arg427_1 = None
        exp_17: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
        div_44: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, arg428_1);  exp_17 = arg428_1 = None
        mul_508: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_366, div_44);  convert_element_type_366 = None
        sum_73: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [-1], True)
        neg_16: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_44);  div_44 = None
        fma_16: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_16, sum_73, mul_508);  neg_16 = sum_73 = mul_508 = None
        convert_element_type_368: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None
        view_424: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_368, _shape_param_455);  convert_element_type_368 = _shape_param_455 = None
        view_425: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_424, _shape_param_456);  view_424 = _shape_param_456 = None
        view_426: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_425, _shape_param_457);  _shape_param_457 = None
        convert_element_type_369: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_425, torch.float32);  view_425 = None
        bmm_66: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg429_1, view_426);  arg429_1 = None
        bmm_67: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_426, arg430_1);  view_426 = arg430_1 = None
        view_427: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_66, _shape_param_458);  bmm_66 = _shape_param_458 = None
        view_428: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_67, _shape_param_459);  bmm_67 = _shape_param_459 = None
        permute_176: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_427, [0, 1, 3, 2]);  view_427 = None
        permute_177: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
        clone_65: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_429: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_65, _shape_param_460);  clone_65 = _shape_param_460 = None
        view_430: "bf16[4096, 384]" = torch.ops.aten.view.default(view_429, _shape_param_461);  view_429 = _shape_param_461 = None
        permute_178: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_430, [1, 0])
        mm_186: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_178, arg431_1);  permute_178 = None
        mm_187: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_430, arg432_1);  view_430 = arg432_1 = None
        view_431: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_187, _shape_param_462);  mm_187 = _shape_param_462 = None
        convert_element_type_370: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_431, torch.float32);  view_431 = None
        convert_element_type_371: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        permute_179: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_176, [0, 2, 1, 3]);  permute_176 = None
        view_432: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_179, _shape_param_463);  permute_179 = _shape_param_463 = None
        clone_66: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_432, memory_format = torch.contiguous_format);  view_432 = None
        view_433: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_66, _shape_param_464);  clone_66 = _shape_param_464 = None
        permute_180: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_188: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_180, arg431_1);  permute_180 = None
        mm_189: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_433, arg433_1);  view_433 = arg433_1 = None
        view_434: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_189, _shape_param_465);  mm_189 = _shape_param_465 = None
        convert_element_type_372: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_434, torch.float32);  view_434 = None
        add_140: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_370, convert_element_type_372);  convert_element_type_370 = convert_element_type_372 = None
        convert_element_type_373: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        permute_181: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
        clone_67: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_435: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_67, _shape_param_466);  clone_67 = _shape_param_466 = None
        view_436: "bf16[4096, 384]" = torch.ops.aten.view.default(view_435, _shape_param_467);  view_435 = _shape_param_467 = None
        permute_182: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_190: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_182, arg431_1);  permute_182 = arg431_1 = None
        mm_191: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_436, arg434_1);  view_436 = arg434_1 = None
        view_437: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_191, _shape_param_468);  mm_191 = _shape_param_468 = None
        convert_element_type_374: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        add_141: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_140, convert_element_type_374);  add_140 = convert_element_type_374 = None
        convert_element_type_375: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        mul_509: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_141, arg435_1);  arg435_1 = None
        mul_510: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg436_1, arg437_1)
        mul_511: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_141, mul_510);  add_141 = mul_510 = None
        sum_74: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1], True, dtype = torch.float32);  mul_511 = None
        view_438: "f32[512]" = torch.ops.aten.view.default(sum_74, _shape_param_469);  sum_74 = _shape_param_469 = None
        mul_512: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_509, arg436_1)
        mul_513: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_509, arg437_1);  mul_509 = None
        sum_75: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [2], True, dtype = torch.float32);  mul_512 = None
        add_142: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_139, mul_513);  add_139 = mul_513 = None
        pow_73: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg437_1, 3);  arg437_1 = None
        mul_514: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_75, -0.5);  sum_75 = None
        mul_515: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_514, pow_73);  mul_514 = pow_73 = None
        expand_29: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_515, _shape_param_470);  mul_515 = _shape_param_470 = None
        div_45: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_29, 512);  expand_29 = None
        pow_74: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg436_1, 1.0);  arg436_1 = None
        mul_516: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
        mul_517: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_45, mul_516);  div_45 = mul_516 = None
        add_143: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_142, mul_517);  add_142 = mul_517 = None
        convert_element_type_376: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16)
        convert_element_type_377: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg438_1, torch.bfloat16);  arg438_1 = None
        mul_518: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_377, 1.1111111111111112);  convert_element_type_377 = None
        mul_519: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_376, mul_518);  convert_element_type_376 = mul_518 = None
        view_439: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_519, _shape_param_471);  mul_519 = _shape_param_471 = None
        permute_183: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_439, [1, 0])
        mm_192: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_183, arg439_1);  permute_183 = arg439_1 = None
        mm_193: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_439, arg440_1);  view_439 = arg440_1 = None
        view_440: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_193, _shape_param_472);  mm_193 = _shape_param_472 = None
        convert_element_type_378: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_440, torch.float32);  view_440 = None
        convert_element_type_379: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        convert_element_type_380: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg441_1, torch.float32);  arg441_1 = None
        mul_520: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_380, 1.1111111111111112);  convert_element_type_380 = None
        mul_521: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_378, mul_520);  convert_element_type_378 = mul_520 = None
        view_441: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg442_1, _shape_param_473);  arg442_1 = _shape_param_473 = None
        mul_522: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_441, 0.5)
        convert_element_type_381: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_441, torch.float32)
        pow_75: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_381, 3.0)
        mul_523: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_75, 0.044715);  pow_75 = None
        add_144: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_441, mul_523);  view_441 = mul_523 = None
        mul_524: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_144, 0.7978845608028654);  add_144 = None
        tanh_9: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_524);  mul_524 = None
        add_145: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_525: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_522, add_145)
        mul_526: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_521, mul_525);  mul_525 = None
        view_442: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg443_1, _shape_param_474);  arg443_1 = _shape_param_474 = None
        mul_527: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_521, view_442);  mul_521 = view_442 = None
        convert_element_type_382: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_526, torch.bfloat16);  mul_526 = None
        view_443: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_382, _shape_param_475);  convert_element_type_382 = _shape_param_475 = None
        permute_184: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_443, [1, 0])
        mm_194: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_184, arg444_1);  permute_184 = None
        mm_195: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_443, arg445_1);  view_443 = arg445_1 = None
        view_444: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_195, _shape_param_476);  mm_195 = _shape_param_476 = None
        convert_element_type_383: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_444, torch.float32);  view_444 = None
        convert_element_type_384: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_194, torch.float32);  mm_194 = None
        mul_528: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_527, mul_522);  mul_522 = None
        mul_529: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_527, add_145);  mul_527 = add_145 = None
        convert_element_type_385: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_529, torch.bfloat16);  mul_529 = None
        mul_530: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_29: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_530);  mul_530 = None
        mul_531: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_528, sub_29);  mul_528 = sub_29 = None
        mul_532: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_531, 0.7978845608028654);  mul_531 = None
        convert_element_type_386: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_532, torch.bfloat16)
        mul_533: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_532, 0.044715);  mul_532 = None
        pow_76: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_381, 2.0);  convert_element_type_381 = None
        mul_534: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_76, 3.0);  pow_76 = None
        mul_535: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_533, mul_534);  mul_533 = mul_534 = None
        convert_element_type_387: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_535, torch.bfloat16);  mul_535 = None
        add_146: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_386, convert_element_type_387);  convert_element_type_386 = convert_element_type_387 = None
        mul_536: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_385, 0.5);  convert_element_type_385 = None
        add_147: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_146, mul_536);  add_146 = mul_536 = None
        view_445: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_147, _shape_param_477);  add_147 = _shape_param_477 = None
        permute_185: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_445, [1, 0])
        mm_196: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_185, arg444_1);  permute_185 = arg444_1 = None
        mm_197: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_445, arg446_1);  view_445 = arg446_1 = None
        view_446: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_197, _shape_param_478);  mm_197 = _shape_param_478 = None
        convert_element_type_388: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_446, torch.float32);  view_446 = None
        add_148: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_383, convert_element_type_388);  convert_element_type_383 = convert_element_type_388 = None
        convert_element_type_389: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_196, torch.float32);  mm_196 = None
        mul_537: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_148, arg447_1);  arg447_1 = None
        mul_538: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg448_1, arg449_1)
        mul_539: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_148, mul_538);  add_148 = mul_538 = None
        sum_76: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 1], True, dtype = torch.float32);  mul_539 = None
        view_447: "f32[512]" = torch.ops.aten.view.default(sum_76, _shape_param_479);  sum_76 = _shape_param_479 = None
        mul_540: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_537, arg448_1)
        mul_541: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_537, arg449_1);  mul_537 = None
        sum_77: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_540, [2], True, dtype = torch.float32);  mul_540 = None
        add_149: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_143, mul_541);  add_143 = mul_541 = None
        pow_77: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg449_1, 3);  arg449_1 = None
        mul_542: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_77, -0.5);  sum_77 = None
        mul_543: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_542, pow_77);  mul_542 = pow_77 = None
        expand_30: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_543, _shape_param_480);  mul_543 = _shape_param_480 = None
        div_46: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_30, 512);  expand_30 = None
        pow_78: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg448_1, 1.0);  arg448_1 = None
        mul_544: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_78, 2.0);  pow_78 = None
        mul_545: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_46, mul_544);  div_46 = mul_544 = None
        add_150: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_149, mul_545);  add_149 = mul_545 = None
        convert_element_type_390: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_150, torch.bfloat16)
        convert_element_type_391: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg450_1, torch.bfloat16);  arg450_1 = None
        mul_546: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_391, 1.1111111111111112);  convert_element_type_391 = None
        mul_547: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_390, mul_546);  convert_element_type_390 = mul_546 = None
        view_448: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_547, _shape_param_481);  mul_547 = _shape_param_481 = None
        permute_186: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_198: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_186, arg451_1);  permute_186 = arg451_1 = None
        mm_199: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_448, arg452_1);  view_448 = arg452_1 = None
        view_449: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_199, _shape_param_482);  mm_199 = _shape_param_482 = None
        convert_element_type_392: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_198, torch.float32);  mm_198 = None
        view_450: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_449, _shape_param_483);  view_449 = _shape_param_483 = None
        permute_187: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
        clone_68: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_451: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_68, _shape_param_484);  clone_68 = _shape_param_484 = None
        bmm_68: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg453_1, view_451);  arg453_1 = None
        bmm_69: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_451, arg454_1);  view_451 = arg454_1 = None
        view_452: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_68, _shape_param_485);  bmm_68 = _shape_param_485 = None
        view_453: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_69, _shape_param_486);  bmm_69 = _shape_param_486 = None
        convert_element_type_393: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg455_1, torch.bfloat16);  arg455_1 = None
        mul_548: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_393, 1.1111111111111112);  convert_element_type_393 = None
        mul_549: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_453, mul_548);  view_453 = mul_548 = None
        convert_element_type_394: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_549, torch.float32);  mul_549 = None
        convert_element_type_395: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg456_1, torch.float32);  arg456_1 = None
        sub_30: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_395, arg457_1);  convert_element_type_395 = arg457_1 = None
        exp_18: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_30);  sub_30 = None
        div_47: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, arg458_1);  exp_18 = arg458_1 = None
        mul_550: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_394, div_47);  convert_element_type_394 = None
        sum_78: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_550, [-1], True)
        neg_17: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_47);  div_47 = None
        fma_17: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_17, sum_78, mul_550);  neg_17 = sum_78 = mul_550 = None
        convert_element_type_396: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None
        view_454: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_396, _shape_param_487);  convert_element_type_396 = _shape_param_487 = None
        view_455: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_454, _shape_param_488);  view_454 = _shape_param_488 = None
        view_456: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_455, _shape_param_489);  _shape_param_489 = None
        convert_element_type_397: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_455, torch.float32);  view_455 = None
        add_151: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_369, convert_element_type_397);  convert_element_type_369 = convert_element_type_397 = None
        bmm_70: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg459_1, view_456);  arg459_1 = None
        bmm_71: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_456, arg460_1);  view_456 = arg460_1 = None
        view_457: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_70, _shape_param_490);  bmm_70 = _shape_param_490 = None
        view_458: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_71, _shape_param_491);  bmm_71 = _shape_param_491 = None
        permute_188: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_457, [0, 1, 3, 2]);  view_457 = None
        permute_189: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_452, [0, 2, 1, 3]);  view_452 = None
        clone_69: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_459: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_69, _shape_param_492);  clone_69 = _shape_param_492 = None
        view_460: "bf16[4096, 384]" = torch.ops.aten.view.default(view_459, _shape_param_493);  view_459 = _shape_param_493 = None
        permute_190: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_460, [1, 0])
        mm_200: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_190, arg461_1);  permute_190 = None
        mm_201: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_460, arg462_1);  view_460 = arg462_1 = None
        view_461: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_201, _shape_param_494);  mm_201 = _shape_param_494 = None
        convert_element_type_398: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_461, torch.float32);  view_461 = None
        convert_element_type_399: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        permute_191: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_188, [0, 2, 1, 3]);  permute_188 = None
        view_462: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_191, _shape_param_495);  permute_191 = _shape_param_495 = None
        clone_70: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_462, memory_format = torch.contiguous_format);  view_462 = None
        view_463: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_70, _shape_param_496);  clone_70 = _shape_param_496 = None
        permute_192: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_463, [1, 0])
        mm_202: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_192, arg461_1);  permute_192 = None
        mm_203: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_463, arg463_1);  view_463 = arg463_1 = None
        view_464: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_203, _shape_param_497);  mm_203 = _shape_param_497 = None
        convert_element_type_400: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_464, torch.float32);  view_464 = None
        add_152: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_398, convert_element_type_400);  convert_element_type_398 = convert_element_type_400 = None
        convert_element_type_401: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_202, torch.float32);  mm_202 = None
        permute_193: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
        clone_71: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_465: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_71, _shape_param_498);  clone_71 = _shape_param_498 = None
        view_466: "bf16[4096, 384]" = torch.ops.aten.view.default(view_465, _shape_param_499);  view_465 = _shape_param_499 = None
        permute_194: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_466, [1, 0])
        mm_204: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_194, arg461_1);  permute_194 = arg461_1 = None
        mm_205: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_466, arg464_1);  view_466 = arg464_1 = None
        view_467: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_205, _shape_param_500);  mm_205 = _shape_param_500 = None
        convert_element_type_402: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_467, torch.float32);  view_467 = None
        add_153: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_152, convert_element_type_402);  add_152 = convert_element_type_402 = None
        convert_element_type_403: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_204, torch.float32);  mm_204 = None
        mul_551: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_153, arg465_1);  arg465_1 = None
        mul_552: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg466_1, arg467_1)
        mul_553: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_153, mul_552);  add_153 = mul_552 = None
        sum_79: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 1], True, dtype = torch.float32);  mul_553 = None
        view_468: "f32[512]" = torch.ops.aten.view.default(sum_79, _shape_param_501);  sum_79 = _shape_param_501 = None
        mul_554: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_551, arg466_1)
        mul_555: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_551, arg467_1);  mul_551 = None
        sum_80: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True, dtype = torch.float32);  mul_554 = None
        add_154: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_150, mul_555);  add_150 = mul_555 = None
        pow_79: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg467_1, 3);  arg467_1 = None
        mul_556: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_80, -0.5);  sum_80 = None
        mul_557: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_556, pow_79);  mul_556 = pow_79 = None
        expand_31: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_557, _shape_param_502);  mul_557 = _shape_param_502 = None
        div_48: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_31, 512);  expand_31 = None
        pow_80: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg466_1, 1.0);  arg466_1 = None
        mul_558: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_80, 2.0);  pow_80 = None
        mul_559: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_48, mul_558);  div_48 = mul_558 = None
        add_155: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_154, mul_559);  add_154 = mul_559 = None
        convert_element_type_404: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16)
        convert_element_type_405: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg468_1, torch.bfloat16);  arg468_1 = None
        mul_560: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_405, 1.1111111111111112);  convert_element_type_405 = None
        mul_561: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_404, mul_560);  convert_element_type_404 = mul_560 = None
        view_469: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_561, _shape_param_503);  mul_561 = _shape_param_503 = None
        permute_195: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_469, [1, 0])
        mm_206: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_195, arg469_1);  permute_195 = arg469_1 = None
        mm_207: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_469, arg470_1);  view_469 = arg470_1 = None
        view_470: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_207, _shape_param_504);  mm_207 = _shape_param_504 = None
        convert_element_type_406: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_470, torch.float32);  view_470 = None
        convert_element_type_407: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_206, torch.float32);  mm_206 = None
        convert_element_type_408: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg471_1, torch.float32);  arg471_1 = None
        mul_562: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_408, 1.1111111111111112);  convert_element_type_408 = None
        mul_563: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_406, mul_562);  convert_element_type_406 = mul_562 = None
        view_471: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg472_1, _shape_param_505);  arg472_1 = _shape_param_505 = None
        mul_564: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_471, 0.5)
        convert_element_type_409: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_471, torch.float32)
        pow_81: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_409, 3.0)
        mul_565: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_81, 0.044715);  pow_81 = None
        add_156: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_471, mul_565);  view_471 = mul_565 = None
        mul_566: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_156, 0.7978845608028654);  add_156 = None
        tanh_10: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_566);  mul_566 = None
        add_157: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_567: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_564, add_157)
        mul_568: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_563, mul_567);  mul_567 = None
        view_472: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg473_1, _shape_param_506);  arg473_1 = _shape_param_506 = None
        mul_569: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_563, view_472);  mul_563 = view_472 = None
        convert_element_type_410: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_568, torch.bfloat16);  mul_568 = None
        view_473: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_410, _shape_param_507);  convert_element_type_410 = _shape_param_507 = None
        permute_196: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_473, [1, 0])
        mm_208: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_196, arg474_1);  permute_196 = None
        mm_209: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_473, arg475_1);  view_473 = arg475_1 = None
        view_474: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_209, _shape_param_508);  mm_209 = _shape_param_508 = None
        convert_element_type_411: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_474, torch.float32);  view_474 = None
        convert_element_type_412: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_208, torch.float32);  mm_208 = None
        mul_570: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_569, mul_564);  mul_564 = None
        mul_571: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_569, add_157);  mul_569 = add_157 = None
        convert_element_type_413: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_571, torch.bfloat16);  mul_571 = None
        mul_572: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_31: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_572);  mul_572 = None
        mul_573: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_570, sub_31);  mul_570 = sub_31 = None
        mul_574: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_573, 0.7978845608028654);  mul_573 = None
        convert_element_type_414: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_574, torch.bfloat16)
        mul_575: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_574, 0.044715);  mul_574 = None
        pow_82: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_409, 2.0);  convert_element_type_409 = None
        mul_576: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_82, 3.0);  pow_82 = None
        mul_577: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_575, mul_576);  mul_575 = mul_576 = None
        convert_element_type_415: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_577, torch.bfloat16);  mul_577 = None
        add_158: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_414, convert_element_type_415);  convert_element_type_414 = convert_element_type_415 = None
        mul_578: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_413, 0.5);  convert_element_type_413 = None
        add_159: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_158, mul_578);  add_158 = mul_578 = None
        view_475: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_159, _shape_param_509);  add_159 = _shape_param_509 = None
        permute_197: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_475, [1, 0])
        mm_210: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_197, arg474_1);  permute_197 = arg474_1 = None
        mm_211: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_475, arg476_1);  view_475 = arg476_1 = None
        view_476: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_211, _shape_param_510);  mm_211 = _shape_param_510 = None
        convert_element_type_416: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_476, torch.float32);  view_476 = None
        add_160: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_411, convert_element_type_416);  convert_element_type_411 = convert_element_type_416 = None
        convert_element_type_417: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_210, torch.float32);  mm_210 = None
        mul_579: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_160, arg477_1);  arg477_1 = None
        mul_580: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg478_1, arg479_1)
        mul_581: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_160, mul_580);  add_160 = mul_580 = None
        sum_81: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 1], True, dtype = torch.float32);  mul_581 = None
        view_477: "f32[512]" = torch.ops.aten.view.default(sum_81, _shape_param_511);  sum_81 = _shape_param_511 = None
        mul_582: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_579, arg478_1)
        mul_583: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_579, arg479_1);  mul_579 = None
        sum_82: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_582, [2], True, dtype = torch.float32);  mul_582 = None
        add_161: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_155, mul_583);  add_155 = mul_583 = None
        pow_83: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg479_1, 3);  arg479_1 = None
        mul_584: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_82, -0.5);  sum_82 = None
        mul_585: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_584, pow_83);  mul_584 = pow_83 = None
        expand_32: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_585, _shape_param_512);  mul_585 = _shape_param_512 = None
        div_49: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_32, 512);  expand_32 = None
        pow_84: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg478_1, 1.0);  arg478_1 = None
        mul_586: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_84, 2.0);  pow_84 = None
        mul_587: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_49, mul_586);  div_49 = mul_586 = None
        add_162: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_161, mul_587);  add_161 = mul_587 = None
        convert_element_type_418: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_162, torch.bfloat16)
        convert_element_type_419: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg480_1, torch.bfloat16);  arg480_1 = None
        mul_588: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_419, 1.1111111111111112);  convert_element_type_419 = None
        mul_589: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_418, mul_588);  convert_element_type_418 = mul_588 = None
        view_478: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_589, _shape_param_513);  mul_589 = _shape_param_513 = None
        permute_198: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_478, [1, 0])
        mm_212: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_198, arg481_1);  permute_198 = arg481_1 = None
        mm_213: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_478, arg482_1);  view_478 = arg482_1 = None
        view_479: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_213, _shape_param_514);  mm_213 = _shape_param_514 = None
        convert_element_type_420: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_212, torch.float32);  mm_212 = None
        view_480: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_479, _shape_param_515);  view_479 = _shape_param_515 = None
        permute_199: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None
        clone_72: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
        view_481: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_72, _shape_param_516);  clone_72 = _shape_param_516 = None
        bmm_72: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg483_1, view_481);  arg483_1 = None
        bmm_73: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_481, arg484_1);  view_481 = arg484_1 = None
        view_482: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_72, _shape_param_517);  bmm_72 = _shape_param_517 = None
        view_483: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_73, _shape_param_518);  bmm_73 = _shape_param_518 = None
        convert_element_type_421: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg485_1, torch.bfloat16);  arg485_1 = None
        mul_590: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_421, 1.1111111111111112);  convert_element_type_421 = None
        mul_591: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_483, mul_590);  view_483 = mul_590 = None
        convert_element_type_422: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_591, torch.float32);  mul_591 = None
        convert_element_type_423: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg486_1, torch.float32);  arg486_1 = None
        sub_32: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_423, arg487_1);  convert_element_type_423 = arg487_1 = None
        exp_19: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_32);  sub_32 = None
        div_50: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, arg488_1);  exp_19 = arg488_1 = None
        mul_592: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_422, div_50);  convert_element_type_422 = None
        sum_83: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_592, [-1], True)
        neg_18: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_50);  div_50 = None
        fma_18: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_18, sum_83, mul_592);  neg_18 = sum_83 = mul_592 = None
        convert_element_type_424: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None
        view_484: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_424, _shape_param_519);  convert_element_type_424 = _shape_param_519 = None
        view_485: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_484, _shape_param_520);  view_484 = _shape_param_520 = None
        view_486: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_485, _shape_param_521);  _shape_param_521 = None
        convert_element_type_425: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_485, torch.float32);  view_485 = None
        add_163: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_151, convert_element_type_425);  add_151 = convert_element_type_425 = None
        bmm_74: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg489_1, view_486);  arg489_1 = None
        bmm_75: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_486, arg490_1);  view_486 = arg490_1 = None
        view_487: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_74, _shape_param_522);  bmm_74 = _shape_param_522 = None
        view_488: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_75, _shape_param_523);  bmm_75 = _shape_param_523 = None
        permute_200: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_487, [0, 1, 3, 2]);  view_487 = None
        permute_201: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
        clone_73: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
        view_489: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_73, _shape_param_524);  clone_73 = _shape_param_524 = None
        view_490: "bf16[4096, 384]" = torch.ops.aten.view.default(view_489, _shape_param_525);  view_489 = _shape_param_525 = None
        permute_202: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_214: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_202, arg491_1);  permute_202 = None
        mm_215: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_490, arg492_1);  view_490 = arg492_1 = None
        view_491: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_215, _shape_param_526);  mm_215 = _shape_param_526 = None
        convert_element_type_426: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_491, torch.float32);  view_491 = None
        convert_element_type_427: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_214, torch.float32);  mm_214 = None
        permute_203: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_200, [0, 2, 1, 3]);  permute_200 = None
        view_492: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_203, _shape_param_527);  permute_203 = _shape_param_527 = None
        clone_74: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_492, memory_format = torch.contiguous_format);  view_492 = None
        view_493: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_74, _shape_param_528);  clone_74 = _shape_param_528 = None
        permute_204: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_493, [1, 0])
        mm_216: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_204, arg491_1);  permute_204 = None
        mm_217: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_493, arg493_1);  view_493 = arg493_1 = None
        view_494: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_217, _shape_param_529);  mm_217 = _shape_param_529 = None
        convert_element_type_428: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_494, torch.float32);  view_494 = None
        add_164: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_426, convert_element_type_428);  convert_element_type_426 = convert_element_type_428 = None
        convert_element_type_429: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_216, torch.float32);  mm_216 = None
        permute_205: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
        clone_75: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
        view_495: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_75, _shape_param_530);  clone_75 = _shape_param_530 = None
        view_496: "bf16[4096, 384]" = torch.ops.aten.view.default(view_495, _shape_param_531);  view_495 = _shape_param_531 = None
        permute_206: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_218: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_206, arg491_1);  permute_206 = arg491_1 = None
        mm_219: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_496, arg494_1);  view_496 = arg494_1 = None
        view_497: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_219, _shape_param_532);  mm_219 = _shape_param_532 = None
        convert_element_type_430: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_497, torch.float32);  view_497 = None
        add_165: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_164, convert_element_type_430);  add_164 = convert_element_type_430 = None
        convert_element_type_431: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_218, torch.float32);  mm_218 = None
        mul_593: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_165, arg495_1);  arg495_1 = None
        mul_594: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg496_1, arg497_1)
        mul_595: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_165, mul_594);  add_165 = mul_594 = None
        sum_84: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 1], True, dtype = torch.float32);  mul_595 = None
        view_498: "f32[512]" = torch.ops.aten.view.default(sum_84, _shape_param_533);  sum_84 = _shape_param_533 = None
        mul_596: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_593, arg496_1)
        mul_597: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_593, arg497_1);  mul_593 = None
        sum_85: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [2], True, dtype = torch.float32);  mul_596 = None
        add_166: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_162, mul_597);  add_162 = mul_597 = None
        pow_85: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg497_1, 3);  arg497_1 = None
        mul_598: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_85, -0.5);  sum_85 = None
        mul_599: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_598, pow_85);  mul_598 = pow_85 = None
        expand_33: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_599, _shape_param_534);  mul_599 = _shape_param_534 = None
        div_51: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_33, 512);  expand_33 = None
        pow_86: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg496_1, 1.0);  arg496_1 = None
        mul_600: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_86, 2.0);  pow_86 = None
        mul_601: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_51, mul_600);  div_51 = mul_600 = None
        add_167: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_166, mul_601);  add_166 = mul_601 = None
        convert_element_type_432: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_167, torch.bfloat16)
        convert_element_type_433: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg498_1, torch.bfloat16);  arg498_1 = None
        mul_602: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_433, 1.1111111111111112);  convert_element_type_433 = None
        mul_603: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_432, mul_602);  convert_element_type_432 = mul_602 = None
        view_499: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_603, _shape_param_535);  mul_603 = _shape_param_535 = None
        permute_207: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_499, [1, 0])
        mm_220: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_207, arg499_1);  permute_207 = arg499_1 = None
        mm_221: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_499, arg500_1);  view_499 = arg500_1 = None
        view_500: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_221, _shape_param_536);  mm_221 = _shape_param_536 = None
        convert_element_type_434: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_500, torch.float32);  view_500 = None
        convert_element_type_435: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_220, torch.float32);  mm_220 = None
        convert_element_type_436: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg501_1, torch.float32);  arg501_1 = None
        mul_604: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_436, 1.1111111111111112);  convert_element_type_436 = None
        mul_605: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_434, mul_604);  convert_element_type_434 = mul_604 = None
        view_501: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg502_1, _shape_param_537);  arg502_1 = _shape_param_537 = None
        mul_606: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        convert_element_type_437: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_501, torch.float32)
        pow_87: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_437, 3.0)
        mul_607: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_87, 0.044715);  pow_87 = None
        add_168: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_501, mul_607);  view_501 = mul_607 = None
        mul_608: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_168, 0.7978845608028654);  add_168 = None
        tanh_11: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_608);  mul_608 = None
        add_169: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_609: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_606, add_169)
        mul_610: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_605, mul_609);  mul_609 = None
        view_502: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg503_1, _shape_param_538);  arg503_1 = _shape_param_538 = None
        mul_611: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_605, view_502);  mul_605 = view_502 = None
        convert_element_type_438: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_610, torch.bfloat16);  mul_610 = None
        view_503: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_438, _shape_param_539);  convert_element_type_438 = _shape_param_539 = None
        permute_208: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_503, [1, 0])
        mm_222: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_208, arg504_1);  permute_208 = None
        mm_223: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_503, arg505_1);  view_503 = arg505_1 = None
        view_504: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_223, _shape_param_540);  mm_223 = _shape_param_540 = None
        convert_element_type_439: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_504, torch.float32);  view_504 = None
        convert_element_type_440: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_222, torch.float32);  mm_222 = None
        mul_612: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_611, mul_606);  mul_606 = None
        mul_613: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_611, add_169);  mul_611 = add_169 = None
        convert_element_type_441: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_613, torch.bfloat16);  mul_613 = None
        mul_614: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_33: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_614);  mul_614 = None
        mul_615: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_612, sub_33);  mul_612 = sub_33 = None
        mul_616: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_615, 0.7978845608028654);  mul_615 = None
        convert_element_type_442: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_616, torch.bfloat16)
        mul_617: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_616, 0.044715);  mul_616 = None
        pow_88: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_437, 2.0);  convert_element_type_437 = None
        mul_618: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_88, 3.0);  pow_88 = None
        mul_619: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
        convert_element_type_443: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_619, torch.bfloat16);  mul_619 = None
        add_170: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_442, convert_element_type_443);  convert_element_type_442 = convert_element_type_443 = None
        mul_620: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_441, 0.5);  convert_element_type_441 = None
        add_171: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_170, mul_620);  add_170 = mul_620 = None
        view_505: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_171, _shape_param_541);  add_171 = _shape_param_541 = None
        permute_209: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_505, [1, 0])
        mm_224: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_209, arg504_1);  permute_209 = arg504_1 = None
        mm_225: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_505, arg506_1);  view_505 = arg506_1 = None
        view_506: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_225, _shape_param_542);  mm_225 = _shape_param_542 = None
        convert_element_type_444: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_506, torch.float32);  view_506 = None
        add_172: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_439, convert_element_type_444);  convert_element_type_439 = convert_element_type_444 = None
        convert_element_type_445: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_224, torch.float32);  mm_224 = None
        mul_621: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_172, arg507_1);  arg507_1 = None
        mul_622: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg508_1, arg509_1)
        mul_623: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_172, mul_622);  add_172 = mul_622 = None
        sum_86: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1], True, dtype = torch.float32);  mul_623 = None
        view_507: "f32[512]" = torch.ops.aten.view.default(sum_86, _shape_param_543);  sum_86 = _shape_param_543 = None
        mul_624: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_621, arg508_1)
        mul_625: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_621, arg509_1);  mul_621 = None
        sum_87: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_624, [2], True, dtype = torch.float32);  mul_624 = None
        add_173: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_167, mul_625);  add_167 = mul_625 = None
        pow_89: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg509_1, 3);  arg509_1 = None
        mul_626: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_87, -0.5);  sum_87 = None
        mul_627: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_626, pow_89);  mul_626 = pow_89 = None
        expand_34: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_627, _shape_param_544);  mul_627 = _shape_param_544 = None
        div_52: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_34, 512);  expand_34 = None
        pow_90: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg508_1, 1.0);  arg508_1 = None
        mul_628: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_90, 2.0);  pow_90 = None
        mul_629: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_52, mul_628);  div_52 = mul_628 = None
        add_174: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_173, mul_629);  add_173 = mul_629 = None
        convert_element_type_446: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_174, torch.bfloat16)
        convert_element_type_447: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg510_1, torch.bfloat16);  arg510_1 = None
        mul_630: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_447, 1.1111111111111112);  convert_element_type_447 = None
        mul_631: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_446, mul_630);  convert_element_type_446 = mul_630 = None
        view_508: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_631, _shape_param_545);  mul_631 = _shape_param_545 = None
        permute_210: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_508, [1, 0])
        mm_226: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_210, arg511_1);  permute_210 = arg511_1 = None
        mm_227: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_508, arg512_1);  view_508 = arg512_1 = None
        view_509: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_227, _shape_param_546);  mm_227 = _shape_param_546 = None
        convert_element_type_448: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_226, torch.float32);  mm_226 = None
        view_510: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_509, _shape_param_547);  view_509 = _shape_param_547 = None
        permute_211: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
        clone_76: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
        view_511: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_76, _shape_param_548);  clone_76 = _shape_param_548 = None
        bmm_76: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg513_1, view_511);  arg513_1 = None
        bmm_77: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_511, arg514_1);  view_511 = arg514_1 = None
        view_512: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_76, _shape_param_549);  bmm_76 = _shape_param_549 = None
        view_513: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_77, _shape_param_550);  bmm_77 = _shape_param_550 = None
        convert_element_type_449: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg515_1, torch.bfloat16);  arg515_1 = None
        mul_632: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_449, 1.1111111111111112);  convert_element_type_449 = None
        mul_633: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_513, mul_632);  view_513 = mul_632 = None
        convert_element_type_450: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_633, torch.float32);  mul_633 = None
        convert_element_type_451: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg516_1, torch.float32);  arg516_1 = None
        sub_34: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_451, arg517_1);  convert_element_type_451 = arg517_1 = None
        exp_20: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
        div_53: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, arg518_1);  exp_20 = arg518_1 = None
        mul_634: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_450, div_53);  convert_element_type_450 = None
        sum_88: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_634, [-1], True)
        neg_19: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_53);  div_53 = None
        fma_19: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_19, sum_88, mul_634);  neg_19 = sum_88 = mul_634 = None
        convert_element_type_452: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None
        view_514: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_452, _shape_param_551);  convert_element_type_452 = _shape_param_551 = None
        view_515: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_514, _shape_param_552);  view_514 = _shape_param_552 = None
        view_516: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_515, _shape_param_553);  _shape_param_553 = None
        convert_element_type_453: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_515, torch.float32);  view_515 = None
        add_175: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_163, convert_element_type_453);  add_163 = convert_element_type_453 = None
        bmm_78: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg519_1, view_516);  arg519_1 = None
        bmm_79: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_516, arg520_1);  view_516 = arg520_1 = None
        view_517: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_78, _shape_param_554);  bmm_78 = _shape_param_554 = None
        view_518: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_79, _shape_param_555);  bmm_79 = _shape_param_555 = None
        permute_212: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_517, [0, 1, 3, 2]);  view_517 = None
        permute_213: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_77: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_519: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_77, _shape_param_556);  clone_77 = _shape_param_556 = None
        view_520: "bf16[4096, 384]" = torch.ops.aten.view.default(view_519, _shape_param_557);  view_519 = _shape_param_557 = None
        permute_214: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_228: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_214, arg521_1);  permute_214 = None
        mm_229: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_520, arg522_1);  view_520 = arg522_1 = None
        view_521: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_229, _shape_param_558);  mm_229 = _shape_param_558 = None
        convert_element_type_454: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_521, torch.float32);  view_521 = None
        convert_element_type_455: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_228, torch.float32);  mm_228 = None
        permute_215: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_212, [0, 2, 1, 3]);  permute_212 = None
        view_522: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_215, _shape_param_559);  permute_215 = _shape_param_559 = None
        clone_78: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_522, memory_format = torch.contiguous_format);  view_522 = None
        view_523: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_78, _shape_param_560);  clone_78 = _shape_param_560 = None
        permute_216: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_523, [1, 0])
        mm_230: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_216, arg521_1);  permute_216 = None
        mm_231: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_523, arg523_1);  view_523 = arg523_1 = None
        view_524: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_231, _shape_param_561);  mm_231 = _shape_param_561 = None
        convert_element_type_456: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_524, torch.float32);  view_524 = None
        add_176: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_454, convert_element_type_456);  convert_element_type_454 = convert_element_type_456 = None
        convert_element_type_457: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_230, torch.float32);  mm_230 = None
        permute_217: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None
        clone_79: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
        view_525: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_79, _shape_param_562);  clone_79 = _shape_param_562 = None
        view_526: "bf16[4096, 384]" = torch.ops.aten.view.default(view_525, _shape_param_563);  view_525 = _shape_param_563 = None
        permute_218: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_526, [1, 0])
        mm_232: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_218, arg521_1);  permute_218 = arg521_1 = None
        mm_233: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_526, arg524_1);  view_526 = arg524_1 = None
        view_527: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_233, _shape_param_564);  mm_233 = _shape_param_564 = None
        convert_element_type_458: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_527, torch.float32);  view_527 = None
        add_177: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_176, convert_element_type_458);  add_176 = convert_element_type_458 = None
        convert_element_type_459: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_232, torch.float32);  mm_232 = None
        mul_635: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_177, arg525_1);  arg525_1 = None
        mul_636: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg526_1, arg527_1)
        mul_637: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_177, mul_636);  add_177 = mul_636 = None
        sum_89: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_637, [0, 1], True, dtype = torch.float32);  mul_637 = None
        view_528: "f32[512]" = torch.ops.aten.view.default(sum_89, _shape_param_565);  sum_89 = _shape_param_565 = None
        mul_638: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_635, arg526_1)
        mul_639: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_635, arg527_1);  mul_635 = None
        sum_90: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [2], True, dtype = torch.float32);  mul_638 = None
        add_178: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_174, mul_639);  add_174 = mul_639 = None
        pow_91: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg527_1, 3);  arg527_1 = None
        mul_640: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_90, -0.5);  sum_90 = None
        mul_641: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_640, pow_91);  mul_640 = pow_91 = None
        expand_35: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_641, _shape_param_566);  mul_641 = _shape_param_566 = None
        div_54: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_35, 512);  expand_35 = None
        pow_92: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg526_1, 1.0);  arg526_1 = None
        mul_642: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_92, 2.0);  pow_92 = None
        mul_643: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_54, mul_642);  div_54 = mul_642 = None
        add_179: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_178, mul_643);  add_178 = mul_643 = None
        convert_element_type_460: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_179, torch.bfloat16)
        convert_element_type_461: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg528_1, torch.bfloat16);  arg528_1 = None
        mul_644: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_461, 1.1111111111111112);  convert_element_type_461 = None
        mul_645: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_460, mul_644);  convert_element_type_460 = mul_644 = None
        view_529: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_645, _shape_param_567);  mul_645 = _shape_param_567 = None
        permute_219: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_234: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_219, arg529_1);  permute_219 = arg529_1 = None
        mm_235: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_529, arg530_1);  view_529 = arg530_1 = None
        view_530: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_235, _shape_param_568);  mm_235 = _shape_param_568 = None
        convert_element_type_462: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_530, torch.float32);  view_530 = None
        convert_element_type_463: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_234, torch.float32);  mm_234 = None
        convert_element_type_464: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg531_1, torch.float32);  arg531_1 = None
        mul_646: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_464, 1.1111111111111112);  convert_element_type_464 = None
        mul_647: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_462, mul_646);  convert_element_type_462 = mul_646 = None
        view_531: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg532_1, _shape_param_569);  arg532_1 = _shape_param_569 = None
        mul_648: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_531, 0.5)
        convert_element_type_465: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_531, torch.float32)
        pow_93: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_465, 3.0)
        mul_649: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_93, 0.044715);  pow_93 = None
        add_180: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_531, mul_649);  view_531 = mul_649 = None
        mul_650: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_180, 0.7978845608028654);  add_180 = None
        tanh_12: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_650);  mul_650 = None
        add_181: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_651: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_648, add_181)
        mul_652: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_647, mul_651);  mul_651 = None
        view_532: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg533_1, _shape_param_570);  arg533_1 = _shape_param_570 = None
        mul_653: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_647, view_532);  mul_647 = view_532 = None
        convert_element_type_466: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_652, torch.bfloat16);  mul_652 = None
        view_533: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_466, _shape_param_571);  convert_element_type_466 = _shape_param_571 = None
        permute_220: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_533, [1, 0])
        mm_236: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_220, arg534_1);  permute_220 = None
        mm_237: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_533, arg535_1);  view_533 = arg535_1 = None
        view_534: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_237, _shape_param_572);  mm_237 = _shape_param_572 = None
        convert_element_type_467: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_534, torch.float32);  view_534 = None
        convert_element_type_468: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_236, torch.float32);  mm_236 = None
        mul_654: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_653, mul_648);  mul_648 = None
        mul_655: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_653, add_181);  mul_653 = add_181 = None
        convert_element_type_469: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_655, torch.bfloat16);  mul_655 = None
        mul_656: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_35: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_656);  mul_656 = None
        mul_657: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_654, sub_35);  mul_654 = sub_35 = None
        mul_658: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_657, 0.7978845608028654);  mul_657 = None
        convert_element_type_470: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_658, torch.bfloat16)
        mul_659: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_658, 0.044715);  mul_658 = None
        pow_94: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_465, 2.0);  convert_element_type_465 = None
        mul_660: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_94, 3.0);  pow_94 = None
        mul_661: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_659, mul_660);  mul_659 = mul_660 = None
        convert_element_type_471: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_661, torch.bfloat16);  mul_661 = None
        add_182: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_470, convert_element_type_471);  convert_element_type_470 = convert_element_type_471 = None
        mul_662: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_469, 0.5);  convert_element_type_469 = None
        add_183: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_182, mul_662);  add_182 = mul_662 = None
        view_535: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_183, _shape_param_573);  add_183 = _shape_param_573 = None
        permute_221: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_238: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_221, arg534_1);  permute_221 = arg534_1 = None
        mm_239: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_535, arg536_1);  view_535 = arg536_1 = None
        view_536: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_239, _shape_param_574);  mm_239 = _shape_param_574 = None
        convert_element_type_472: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_536, torch.float32);  view_536 = None
        add_184: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_467, convert_element_type_472);  convert_element_type_467 = convert_element_type_472 = None
        convert_element_type_473: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_238, torch.float32);  mm_238 = None
        mul_663: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_184, arg537_1);  arg537_1 = None
        mul_664: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg538_1, arg539_1)
        mul_665: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_184, mul_664);  add_184 = mul_664 = None
        sum_91: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_665, [0, 1], True, dtype = torch.float32);  mul_665 = None
        view_537: "f32[512]" = torch.ops.aten.view.default(sum_91, _shape_param_575);  sum_91 = _shape_param_575 = None
        mul_666: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_663, arg538_1)
        mul_667: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_663, arg539_1);  mul_663 = None
        sum_92: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_666, [2], True, dtype = torch.float32);  mul_666 = None
        add_185: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_179, mul_667);  add_179 = mul_667 = None
        pow_95: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg539_1, 3);  arg539_1 = None
        mul_668: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_92, -0.5);  sum_92 = None
        mul_669: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_668, pow_95);  mul_668 = pow_95 = None
        expand_36: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_669, _shape_param_576);  mul_669 = _shape_param_576 = None
        div_55: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_36, 512);  expand_36 = None
        pow_96: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg538_1, 1.0);  arg538_1 = None
        mul_670: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_96, 2.0);  pow_96 = None
        mul_671: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_55, mul_670);  div_55 = mul_670 = None
        add_186: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_185, mul_671);  add_185 = mul_671 = None
        convert_element_type_474: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16)
        convert_element_type_475: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg540_1, torch.bfloat16);  arg540_1 = None
        mul_672: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_475, 1.1111111111111112);  convert_element_type_475 = None
        mul_673: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_474, mul_672);  convert_element_type_474 = mul_672 = None
        view_538: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_673, _shape_param_577);  mul_673 = _shape_param_577 = None
        permute_222: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_538, [1, 0])
        mm_240: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_222, arg541_1);  permute_222 = arg541_1 = None
        mm_241: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_538, arg542_1);  view_538 = arg542_1 = None
        view_539: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_241, _shape_param_578);  mm_241 = _shape_param_578 = None
        convert_element_type_476: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_240, torch.float32);  mm_240 = None
        view_540: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_539, _shape_param_579);  view_539 = _shape_param_579 = None
        permute_223: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
        clone_80: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_223, memory_format = torch.contiguous_format);  permute_223 = None
        view_541: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_80, _shape_param_580);  clone_80 = _shape_param_580 = None
        bmm_80: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg543_1, view_541);  arg543_1 = None
        bmm_81: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_541, arg544_1);  view_541 = arg544_1 = None
        view_542: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_80, _shape_param_581);  bmm_80 = _shape_param_581 = None
        view_543: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_81, _shape_param_582);  bmm_81 = _shape_param_582 = None
        convert_element_type_477: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg545_1, torch.bfloat16);  arg545_1 = None
        mul_674: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_477, 1.1111111111111112);  convert_element_type_477 = None
        mul_675: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_543, mul_674);  view_543 = mul_674 = None
        convert_element_type_478: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_675, torch.float32);  mul_675 = None
        convert_element_type_479: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg546_1, torch.float32);  arg546_1 = None
        sub_36: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_479, arg547_1);  convert_element_type_479 = arg547_1 = None
        exp_21: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
        div_56: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, arg548_1);  exp_21 = arg548_1 = None
        mul_676: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_478, div_56);  convert_element_type_478 = None
        sum_93: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_676, [-1], True)
        neg_20: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_56);  div_56 = None
        fma_20: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_20, sum_93, mul_676);  neg_20 = sum_93 = mul_676 = None
        convert_element_type_480: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None
        view_544: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_480, _shape_param_583);  convert_element_type_480 = _shape_param_583 = None
        view_545: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_544, _shape_param_584);  view_544 = _shape_param_584 = None
        view_546: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_545, _shape_param_585);  _shape_param_585 = None
        convert_element_type_481: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_545, torch.float32);  view_545 = None
        add_187: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_175, convert_element_type_481);  add_175 = convert_element_type_481 = None
        bmm_82: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg549_1, view_546);  arg549_1 = None
        bmm_83: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_546, arg550_1);  view_546 = arg550_1 = None
        view_547: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_82, _shape_param_586);  bmm_82 = _shape_param_586 = None
        view_548: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_83, _shape_param_587);  bmm_83 = _shape_param_587 = None
        permute_224: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_547, [0, 1, 3, 2]);  view_547 = None
        permute_225: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_542, [0, 2, 1, 3]);  view_542 = None
        clone_81: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
        view_549: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_81, _shape_param_588);  clone_81 = _shape_param_588 = None
        view_550: "bf16[4096, 384]" = torch.ops.aten.view.default(view_549, _shape_param_589);  view_549 = _shape_param_589 = None
        permute_226: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_550, [1, 0])
        mm_242: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_226, arg551_1);  permute_226 = None
        mm_243: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_550, arg552_1);  view_550 = arg552_1 = None
        view_551: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_243, _shape_param_590);  mm_243 = _shape_param_590 = None
        convert_element_type_482: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_551, torch.float32);  view_551 = None
        convert_element_type_483: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_242, torch.float32);  mm_242 = None
        permute_227: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_224, [0, 2, 1, 3]);  permute_224 = None
        view_552: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_227, _shape_param_591);  permute_227 = _shape_param_591 = None
        clone_82: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_552, memory_format = torch.contiguous_format);  view_552 = None
        view_553: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_82, _shape_param_592);  clone_82 = _shape_param_592 = None
        permute_228: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_553, [1, 0])
        mm_244: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_228, arg551_1);  permute_228 = None
        mm_245: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_553, arg553_1);  view_553 = arg553_1 = None
        view_554: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_245, _shape_param_593);  mm_245 = _shape_param_593 = None
        convert_element_type_484: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_554, torch.float32);  view_554 = None
        add_188: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_482, convert_element_type_484);  convert_element_type_482 = convert_element_type_484 = None
        convert_element_type_485: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_244, torch.float32);  mm_244 = None
        permute_229: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None
        clone_83: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
        view_555: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_83, _shape_param_594);  clone_83 = _shape_param_594 = None
        view_556: "bf16[4096, 384]" = torch.ops.aten.view.default(view_555, _shape_param_595);  view_555 = _shape_param_595 = None
        permute_230: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_246: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_230, arg551_1);  permute_230 = arg551_1 = None
        mm_247: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_556, arg554_1);  view_556 = arg554_1 = None
        view_557: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_247, _shape_param_596);  mm_247 = _shape_param_596 = None
        convert_element_type_486: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_557, torch.float32);  view_557 = None
        add_189: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_188, convert_element_type_486);  add_188 = convert_element_type_486 = None
        convert_element_type_487: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_246, torch.float32);  mm_246 = None
        mul_677: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_189, arg555_1);  arg555_1 = None
        mul_678: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg556_1, arg557_1)
        mul_679: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_189, mul_678);  add_189 = mul_678 = None
        sum_94: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 1], True, dtype = torch.float32);  mul_679 = None
        view_558: "f32[512]" = torch.ops.aten.view.default(sum_94, _shape_param_597);  sum_94 = _shape_param_597 = None
        mul_680: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_677, arg556_1)
        mul_681: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_677, arg557_1);  mul_677 = None
        sum_95: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_680, [2], True, dtype = torch.float32);  mul_680 = None
        add_190: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_186, mul_681);  add_186 = mul_681 = None
        pow_97: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg557_1, 3);  arg557_1 = None
        mul_682: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_95, -0.5);  sum_95 = None
        mul_683: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_682, pow_97);  mul_682 = pow_97 = None
        expand_37: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_683, _shape_param_598);  mul_683 = _shape_param_598 = None
        div_57: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_37, 512);  expand_37 = None
        pow_98: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg556_1, 1.0);  arg556_1 = None
        mul_684: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_98, 2.0);  pow_98 = None
        mul_685: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_57, mul_684);  div_57 = mul_684 = None
        add_191: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_190, mul_685);  add_190 = mul_685 = None
        convert_element_type_488: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_191, torch.bfloat16)
        convert_element_type_489: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg558_1, torch.bfloat16);  arg558_1 = None
        mul_686: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_489, 1.1111111111111112);  convert_element_type_489 = None
        mul_687: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_488, mul_686);  convert_element_type_488 = mul_686 = None
        view_559: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_687, _shape_param_599);  mul_687 = _shape_param_599 = None
        permute_231: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_559, [1, 0])
        mm_248: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_231, arg559_1);  permute_231 = arg559_1 = None
        mm_249: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_559, arg560_1);  view_559 = arg560_1 = None
        view_560: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_249, _shape_param_600);  mm_249 = _shape_param_600 = None
        convert_element_type_490: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_560, torch.float32);  view_560 = None
        convert_element_type_491: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_248, torch.float32);  mm_248 = None
        convert_element_type_492: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg561_1, torch.float32);  arg561_1 = None
        mul_688: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_492, 1.1111111111111112);  convert_element_type_492 = None
        mul_689: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_490, mul_688);  convert_element_type_490 = mul_688 = None
        view_561: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg562_1, _shape_param_601);  arg562_1 = _shape_param_601 = None
        mul_690: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_561, 0.5)
        convert_element_type_493: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_561, torch.float32)
        pow_99: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_493, 3.0)
        mul_691: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_99, 0.044715);  pow_99 = None
        add_192: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_561, mul_691);  view_561 = mul_691 = None
        mul_692: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_192, 0.7978845608028654);  add_192 = None
        tanh_13: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_692);  mul_692 = None
        add_193: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_693: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_690, add_193)
        mul_694: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_689, mul_693);  mul_693 = None
        view_562: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg563_1, _shape_param_602);  arg563_1 = _shape_param_602 = None
        mul_695: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_689, view_562);  mul_689 = view_562 = None
        convert_element_type_494: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_694, torch.bfloat16);  mul_694 = None
        view_563: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_494, _shape_param_603);  convert_element_type_494 = _shape_param_603 = None
        permute_232: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_563, [1, 0])
        mm_250: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_232, arg564_1);  permute_232 = None
        mm_251: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_563, arg565_1);  view_563 = arg565_1 = None
        view_564: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_251, _shape_param_604);  mm_251 = _shape_param_604 = None
        convert_element_type_495: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_564, torch.float32);  view_564 = None
        convert_element_type_496: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_250, torch.float32);  mm_250 = None
        mul_696: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_695, mul_690);  mul_690 = None
        mul_697: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_695, add_193);  mul_695 = add_193 = None
        convert_element_type_497: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_697, torch.bfloat16);  mul_697 = None
        mul_698: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_37: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_698);  mul_698 = None
        mul_699: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_696, sub_37);  mul_696 = sub_37 = None
        mul_700: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_699, 0.7978845608028654);  mul_699 = None
        convert_element_type_498: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_700, torch.bfloat16)
        mul_701: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_700, 0.044715);  mul_700 = None
        pow_100: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_493, 2.0);  convert_element_type_493 = None
        mul_702: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_100, 3.0);  pow_100 = None
        mul_703: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_701, mul_702);  mul_701 = mul_702 = None
        convert_element_type_499: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_703, torch.bfloat16);  mul_703 = None
        add_194: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_498, convert_element_type_499);  convert_element_type_498 = convert_element_type_499 = None
        mul_704: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_497, 0.5);  convert_element_type_497 = None
        add_195: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_194, mul_704);  add_194 = mul_704 = None
        view_565: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_195, _shape_param_605);  add_195 = _shape_param_605 = None
        permute_233: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_252: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_233, arg564_1);  permute_233 = arg564_1 = None
        mm_253: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_565, arg566_1);  view_565 = arg566_1 = None
        view_566: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_253, _shape_param_606);  mm_253 = _shape_param_606 = None
        convert_element_type_500: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_566, torch.float32);  view_566 = None
        add_196: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_495, convert_element_type_500);  convert_element_type_495 = convert_element_type_500 = None
        convert_element_type_501: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_252, torch.float32);  mm_252 = None
        mul_705: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_196, arg567_1);  arg567_1 = None
        mul_706: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg568_1, arg569_1)
        mul_707: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_196, mul_706);  add_196 = mul_706 = None
        sum_96: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 1], True, dtype = torch.float32);  mul_707 = None
        view_567: "f32[512]" = torch.ops.aten.view.default(sum_96, _shape_param_607);  sum_96 = _shape_param_607 = None
        mul_708: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_705, arg568_1)
        mul_709: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_705, arg569_1);  mul_705 = None
        sum_97: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_708, [2], True, dtype = torch.float32);  mul_708 = None
        add_197: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_191, mul_709);  add_191 = mul_709 = None
        pow_101: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg569_1, 3);  arg569_1 = None
        mul_710: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_97, -0.5);  sum_97 = None
        mul_711: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_710, pow_101);  mul_710 = pow_101 = None
        expand_38: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_711, _shape_param_608);  mul_711 = _shape_param_608 = None
        div_58: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_38, 512);  expand_38 = None
        pow_102: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg568_1, 1.0);  arg568_1 = None
        mul_712: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_102, 2.0);  pow_102 = None
        mul_713: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_58, mul_712);  div_58 = mul_712 = None
        add_198: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_197, mul_713);  add_197 = mul_713 = None
        convert_element_type_502: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_198, torch.bfloat16)
        convert_element_type_503: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg570_1, torch.bfloat16);  arg570_1 = None
        mul_714: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_503, 1.1111111111111112);  convert_element_type_503 = None
        mul_715: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_502, mul_714);  convert_element_type_502 = mul_714 = None
        view_568: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_715, _shape_param_609);  mul_715 = _shape_param_609 = None
        permute_234: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_254: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_234, arg571_1);  permute_234 = arg571_1 = None
        mm_255: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_568, arg572_1);  view_568 = arg572_1 = None
        view_569: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_255, _shape_param_610);  mm_255 = _shape_param_610 = None
        convert_element_type_504: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_254, torch.float32);  mm_254 = None
        view_570: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_569, _shape_param_611);  view_569 = _shape_param_611 = None
        permute_235: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
        clone_84: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
        view_571: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_84, _shape_param_612);  clone_84 = _shape_param_612 = None
        bmm_84: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg573_1, view_571);  arg573_1 = None
        bmm_85: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_571, arg574_1);  view_571 = arg574_1 = None
        view_572: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_84, _shape_param_613);  bmm_84 = _shape_param_613 = None
        view_573: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_85, _shape_param_614);  bmm_85 = _shape_param_614 = None
        convert_element_type_505: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg575_1, torch.bfloat16);  arg575_1 = None
        mul_716: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_505, 1.1111111111111112);  convert_element_type_505 = None
        mul_717: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_573, mul_716);  view_573 = mul_716 = None
        convert_element_type_506: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_717, torch.float32);  mul_717 = None
        convert_element_type_507: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg576_1, torch.float32);  arg576_1 = None
        sub_38: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_507, arg577_1);  convert_element_type_507 = arg577_1 = None
        exp_22: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
        div_59: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, arg578_1);  exp_22 = arg578_1 = None
        mul_718: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_506, div_59);  convert_element_type_506 = None
        sum_98: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_718, [-1], True)
        neg_21: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_59);  div_59 = None
        fma_21: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_21, sum_98, mul_718);  neg_21 = sum_98 = mul_718 = None
        convert_element_type_508: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None
        view_574: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_508, _shape_param_615);  convert_element_type_508 = _shape_param_615 = None
        view_575: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_574, _shape_param_616);  view_574 = _shape_param_616 = None
        view_576: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_575, _shape_param_617);  _shape_param_617 = None
        convert_element_type_509: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_575, torch.float32);  view_575 = None
        add_199: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_187, convert_element_type_509);  add_187 = convert_element_type_509 = None
        bmm_86: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg579_1, view_576);  arg579_1 = None
        bmm_87: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_576, arg580_1);  view_576 = arg580_1 = None
        view_577: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_86, _shape_param_618);  bmm_86 = _shape_param_618 = None
        view_578: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_87, _shape_param_619);  bmm_87 = _shape_param_619 = None
        permute_236: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_577, [0, 1, 3, 2]);  view_577 = None
        permute_237: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        clone_85: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_579: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_85, _shape_param_620);  clone_85 = _shape_param_620 = None
        view_580: "bf16[4096, 384]" = torch.ops.aten.view.default(view_579, _shape_param_621);  view_579 = _shape_param_621 = None
        permute_238: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_580, [1, 0])
        mm_256: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_238, arg581_1);  permute_238 = None
        mm_257: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_580, arg582_1);  view_580 = arg582_1 = None
        view_581: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_257, _shape_param_622);  mm_257 = _shape_param_622 = None
        convert_element_type_510: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_581, torch.float32);  view_581 = None
        convert_element_type_511: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_256, torch.float32);  mm_256 = None
        permute_239: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_236, [0, 2, 1, 3]);  permute_236 = None
        view_582: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_239, _shape_param_623);  permute_239 = _shape_param_623 = None
        clone_86: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_582, memory_format = torch.contiguous_format);  view_582 = None
        view_583: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_86, _shape_param_624);  clone_86 = _shape_param_624 = None
        permute_240: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_583, [1, 0])
        mm_258: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_240, arg581_1);  permute_240 = None
        mm_259: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_583, arg583_1);  view_583 = arg583_1 = None
        view_584: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_259, _shape_param_625);  mm_259 = _shape_param_625 = None
        convert_element_type_512: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_584, torch.float32);  view_584 = None
        add_200: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_510, convert_element_type_512);  convert_element_type_510 = convert_element_type_512 = None
        convert_element_type_513: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_258, torch.float32);  mm_258 = None
        permute_241: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
        clone_87: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
        view_585: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_87, _shape_param_626);  clone_87 = _shape_param_626 = None
        view_586: "bf16[4096, 384]" = torch.ops.aten.view.default(view_585, _shape_param_627);  view_585 = _shape_param_627 = None
        permute_242: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_586, [1, 0])
        mm_260: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_242, arg581_1);  permute_242 = arg581_1 = None
        mm_261: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_586, arg584_1);  view_586 = arg584_1 = None
        view_587: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_261, _shape_param_628);  mm_261 = _shape_param_628 = None
        convert_element_type_514: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_587, torch.float32);  view_587 = None
        add_201: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_200, convert_element_type_514);  add_200 = convert_element_type_514 = None
        convert_element_type_515: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_260, torch.float32);  mm_260 = None
        mul_719: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_201, arg585_1);  arg585_1 = None
        mul_720: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg586_1, arg587_1)
        mul_721: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_201, mul_720);  add_201 = mul_720 = None
        sum_99: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_721, [0, 1], True, dtype = torch.float32);  mul_721 = None
        view_588: "f32[512]" = torch.ops.aten.view.default(sum_99, _shape_param_629);  sum_99 = _shape_param_629 = None
        mul_722: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_719, arg586_1)
        mul_723: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_719, arg587_1);  mul_719 = None
        sum_100: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_722, [2], True, dtype = torch.float32);  mul_722 = None
        add_202: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_198, mul_723);  add_198 = mul_723 = None
        pow_103: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg587_1, 3);  arg587_1 = None
        mul_724: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_100, -0.5);  sum_100 = None
        mul_725: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_724, pow_103);  mul_724 = pow_103 = None
        expand_39: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_725, _shape_param_630);  mul_725 = _shape_param_630 = None
        div_60: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_39, 512);  expand_39 = None
        pow_104: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg586_1, 1.0);  arg586_1 = None
        mul_726: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_104, 2.0);  pow_104 = None
        mul_727: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_60, mul_726);  div_60 = mul_726 = None
        add_203: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_202, mul_727);  add_202 = mul_727 = None
        convert_element_type_516: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_203, torch.bfloat16)
        convert_element_type_517: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg588_1, torch.bfloat16);  arg588_1 = None
        mul_728: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_517, 1.1111111111111112);  convert_element_type_517 = None
        mul_729: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_516, mul_728);  convert_element_type_516 = mul_728 = None
        view_589: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_729, _shape_param_631);  mul_729 = _shape_param_631 = None
        permute_243: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_589, [1, 0])
        mm_262: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_243, arg589_1);  permute_243 = arg589_1 = None
        mm_263: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_589, arg590_1);  view_589 = arg590_1 = None
        view_590: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_263, _shape_param_632);  mm_263 = _shape_param_632 = None
        convert_element_type_518: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_590, torch.float32);  view_590 = None
        convert_element_type_519: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_262, torch.float32);  mm_262 = None
        convert_element_type_520: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg591_1, torch.float32);  arg591_1 = None
        mul_730: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_520, 1.1111111111111112);  convert_element_type_520 = None
        mul_731: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_518, mul_730);  convert_element_type_518 = mul_730 = None
        view_591: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg592_1, _shape_param_633);  arg592_1 = _shape_param_633 = None
        mul_732: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_591, 0.5)
        convert_element_type_521: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_591, torch.float32)
        pow_105: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_521, 3.0)
        mul_733: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_105, 0.044715);  pow_105 = None
        add_204: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_591, mul_733);  view_591 = mul_733 = None
        mul_734: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_204, 0.7978845608028654);  add_204 = None
        tanh_14: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_734);  mul_734 = None
        add_205: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_735: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_732, add_205)
        mul_736: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_731, mul_735);  mul_735 = None
        view_592: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg593_1, _shape_param_634);  arg593_1 = _shape_param_634 = None
        mul_737: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_731, view_592);  mul_731 = view_592 = None
        convert_element_type_522: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_736, torch.bfloat16);  mul_736 = None
        view_593: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_522, _shape_param_635);  convert_element_type_522 = _shape_param_635 = None
        permute_244: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_264: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_244, arg594_1);  permute_244 = None
        mm_265: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_593, arg595_1);  view_593 = arg595_1 = None
        view_594: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_265, _shape_param_636);  mm_265 = _shape_param_636 = None
        convert_element_type_523: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_594, torch.float32);  view_594 = None
        convert_element_type_524: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_264, torch.float32);  mm_264 = None
        mul_738: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_737, mul_732);  mul_732 = None
        mul_739: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_737, add_205);  mul_737 = add_205 = None
        convert_element_type_525: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_739, torch.bfloat16);  mul_739 = None
        mul_740: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_39: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_740);  mul_740 = None
        mul_741: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_738, sub_39);  mul_738 = sub_39 = None
        mul_742: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_741, 0.7978845608028654);  mul_741 = None
        convert_element_type_526: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_742, torch.bfloat16)
        mul_743: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_742, 0.044715);  mul_742 = None
        pow_106: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_521, 2.0);  convert_element_type_521 = None
        mul_744: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_106, 3.0);  pow_106 = None
        mul_745: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_743, mul_744);  mul_743 = mul_744 = None
        convert_element_type_527: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_745, torch.bfloat16);  mul_745 = None
        add_206: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_526, convert_element_type_527);  convert_element_type_526 = convert_element_type_527 = None
        mul_746: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_525, 0.5);  convert_element_type_525 = None
        add_207: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_206, mul_746);  add_206 = mul_746 = None
        view_595: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_207, _shape_param_637);  add_207 = _shape_param_637 = None
        permute_245: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_595, [1, 0])
        mm_266: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_245, arg594_1);  permute_245 = arg594_1 = None
        mm_267: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_595, arg596_1);  view_595 = arg596_1 = None
        view_596: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_267, _shape_param_638);  mm_267 = _shape_param_638 = None
        convert_element_type_528: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_596, torch.float32);  view_596 = None
        add_208: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_523, convert_element_type_528);  convert_element_type_523 = convert_element_type_528 = None
        convert_element_type_529: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_266, torch.float32);  mm_266 = None
        mul_747: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_208, arg597_1);  arg597_1 = None
        mul_748: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg598_1, arg599_1)
        mul_749: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_208, mul_748);  add_208 = mul_748 = None
        sum_101: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_749, [0, 1], True, dtype = torch.float32);  mul_749 = None
        view_597: "f32[512]" = torch.ops.aten.view.default(sum_101, _shape_param_639);  sum_101 = _shape_param_639 = None
        mul_750: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_747, arg598_1)
        mul_751: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_747, arg599_1);  mul_747 = None
        sum_102: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_750, [2], True, dtype = torch.float32);  mul_750 = None
        add_209: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_203, mul_751);  add_203 = mul_751 = None
        pow_107: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg599_1, 3);  arg599_1 = None
        mul_752: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_102, -0.5);  sum_102 = None
        mul_753: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_752, pow_107);  mul_752 = pow_107 = None
        expand_40: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_753, _shape_param_640);  mul_753 = _shape_param_640 = None
        div_61: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_40, 512);  expand_40 = None
        pow_108: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg598_1, 1.0);  arg598_1 = None
        mul_754: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_108, 2.0);  pow_108 = None
        mul_755: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_61, mul_754);  div_61 = mul_754 = None
        add_210: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_209, mul_755);  add_209 = mul_755 = None
        convert_element_type_530: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16)
        convert_element_type_531: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg600_1, torch.bfloat16);  arg600_1 = None
        mul_756: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_531, 1.1111111111111112);  convert_element_type_531 = None
        mul_757: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_530, mul_756);  convert_element_type_530 = mul_756 = None
        view_598: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_757, _shape_param_641);  mul_757 = _shape_param_641 = None
        permute_246: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_268: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_246, arg601_1);  permute_246 = arg601_1 = None
        mm_269: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_598, arg602_1);  view_598 = arg602_1 = None
        view_599: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_269, _shape_param_642);  mm_269 = _shape_param_642 = None
        convert_element_type_532: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_268, torch.float32);  mm_268 = None
        view_600: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_599, _shape_param_643);  view_599 = _shape_param_643 = None
        permute_247: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None
        clone_88: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_601: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_88, _shape_param_644);  clone_88 = _shape_param_644 = None
        bmm_88: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg603_1, view_601);  arg603_1 = None
        bmm_89: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_601, arg604_1);  view_601 = arg604_1 = None
        view_602: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_88, _shape_param_645);  bmm_88 = _shape_param_645 = None
        view_603: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_89, _shape_param_646);  bmm_89 = _shape_param_646 = None
        convert_element_type_533: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg605_1, torch.bfloat16);  arg605_1 = None
        mul_758: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_533, 1.1111111111111112);  convert_element_type_533 = None
        mul_759: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_603, mul_758);  view_603 = mul_758 = None
        convert_element_type_534: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_759, torch.float32);  mul_759 = None
        convert_element_type_535: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg606_1, torch.float32);  arg606_1 = None
        sub_40: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_535, arg607_1);  convert_element_type_535 = arg607_1 = None
        exp_23: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
        div_62: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, arg608_1);  exp_23 = arg608_1 = None
        mul_760: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_534, div_62);  convert_element_type_534 = None
        sum_103: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_760, [-1], True)
        neg_22: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_62);  div_62 = None
        fma_22: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_22, sum_103, mul_760);  neg_22 = sum_103 = mul_760 = None
        convert_element_type_536: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None
        view_604: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_536, _shape_param_647);  convert_element_type_536 = _shape_param_647 = None
        view_605: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_604, _shape_param_648);  view_604 = _shape_param_648 = None
        view_606: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_605, _shape_param_649);  _shape_param_649 = None
        convert_element_type_537: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_605, torch.float32);  view_605 = None
        add_211: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_199, convert_element_type_537);  add_199 = convert_element_type_537 = None
        bmm_90: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg609_1, view_606);  arg609_1 = None
        bmm_91: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_606, arg610_1);  view_606 = arg610_1 = None
        view_607: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_90, _shape_param_650);  bmm_90 = _shape_param_650 = None
        view_608: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_91, _shape_param_651);  bmm_91 = _shape_param_651 = None
        permute_248: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_607, [0, 1, 3, 2]);  view_607 = None
        permute_249: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_602, [0, 2, 1, 3]);  view_602 = None
        clone_89: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_609: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_89, _shape_param_652);  clone_89 = _shape_param_652 = None
        view_610: "bf16[4096, 384]" = torch.ops.aten.view.default(view_609, _shape_param_653);  view_609 = _shape_param_653 = None
        permute_250: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_610, [1, 0])
        mm_270: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_250, arg611_1);  permute_250 = None
        mm_271: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_610, arg612_1);  view_610 = arg612_1 = None
        view_611: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_271, _shape_param_654);  mm_271 = _shape_param_654 = None
        convert_element_type_538: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_611, torch.float32);  view_611 = None
        convert_element_type_539: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_270, torch.float32);  mm_270 = None
        permute_251: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_248, [0, 2, 1, 3]);  permute_248 = None
        view_612: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_251, _shape_param_655);  permute_251 = _shape_param_655 = None
        clone_90: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_612, memory_format = torch.contiguous_format);  view_612 = None
        view_613: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_90, _shape_param_656);  clone_90 = _shape_param_656 = None
        permute_252: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_613, [1, 0])
        mm_272: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_252, arg611_1);  permute_252 = None
        mm_273: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_613, arg613_1);  view_613 = arg613_1 = None
        view_614: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_273, _shape_param_657);  mm_273 = _shape_param_657 = None
        convert_element_type_540: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_614, torch.float32);  view_614 = None
        add_212: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_538, convert_element_type_540);  convert_element_type_538 = convert_element_type_540 = None
        convert_element_type_541: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_272, torch.float32);  mm_272 = None
        permute_253: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None
        clone_91: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
        view_615: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_91, _shape_param_658);  clone_91 = _shape_param_658 = None
        view_616: "bf16[4096, 384]" = torch.ops.aten.view.default(view_615, _shape_param_659);  view_615 = _shape_param_659 = None
        permute_254: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_616, [1, 0])
        mm_274: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_254, arg611_1);  permute_254 = arg611_1 = None
        mm_275: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_616, arg614_1);  view_616 = arg614_1 = None
        view_617: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_275, _shape_param_660);  mm_275 = _shape_param_660 = None
        convert_element_type_542: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_617, torch.float32);  view_617 = None
        add_213: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_212, convert_element_type_542);  add_212 = convert_element_type_542 = None
        convert_element_type_543: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_274, torch.float32);  mm_274 = None
        mul_761: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_213, arg615_1);  arg615_1 = None
        mul_762: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg616_1, arg617_1)
        mul_763: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_213, mul_762);  add_213 = mul_762 = None
        sum_104: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_763, [0, 1], True, dtype = torch.float32);  mul_763 = None
        view_618: "f32[512]" = torch.ops.aten.view.default(sum_104, _shape_param_661);  sum_104 = _shape_param_661 = None
        mul_764: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_761, arg616_1)
        mul_765: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_761, arg617_1);  mul_761 = None
        sum_105: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_764, [2], True, dtype = torch.float32);  mul_764 = None
        add_214: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_210, mul_765);  add_210 = mul_765 = None
        pow_109: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg617_1, 3);  arg617_1 = None
        mul_766: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_105, -0.5);  sum_105 = None
        mul_767: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_766, pow_109);  mul_766 = pow_109 = None
        expand_41: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_767, _shape_param_662);  mul_767 = _shape_param_662 = None
        div_63: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_41, 512);  expand_41 = None
        pow_110: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg616_1, 1.0);  arg616_1 = None
        mul_768: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_110, 2.0);  pow_110 = None
        mul_769: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_63, mul_768);  div_63 = mul_768 = None
        add_215: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_214, mul_769);  add_214 = mul_769 = None
        convert_element_type_544: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16)
        convert_element_type_545: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg618_1, torch.bfloat16);  arg618_1 = None
        mul_770: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_545, 1.1111111111111112);  convert_element_type_545 = None
        mul_771: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_544, mul_770);  convert_element_type_544 = mul_770 = None
        view_619: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_771, _shape_param_663);  mul_771 = _shape_param_663 = None
        permute_255: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_619, [1, 0])
        mm_276: "bf16[512, 1024]" = torch.ops.aten.mm.default(permute_255, arg619_1);  permute_255 = arg619_1 = None
        mm_277: "bf16[4096, 1024]" = torch.ops.aten.mm.default(view_619, arg620_1);  view_619 = arg620_1 = None
        view_620: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(mm_277, _shape_param_664);  mm_277 = _shape_param_664 = None
        convert_element_type_546: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_620, torch.float32);  view_620 = None
        convert_element_type_547: "f32[512, 1024]" = torch.ops.prims.convert_element_type.default(mm_276, torch.float32);  mm_276 = None
        convert_element_type_548: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(arg621_1, torch.float32);  arg621_1 = None
        mul_772: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_548, 1.1111111111111112);  convert_element_type_548 = None
        mul_773: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_546, mul_772);  convert_element_type_546 = mul_772 = None
        view_621: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg622_1, _shape_param_665);  arg622_1 = _shape_param_665 = None
        mul_774: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_621, 0.5)
        convert_element_type_549: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(view_621, torch.float32)
        pow_111: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_549, 3.0)
        mul_775: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_111, 0.044715);  pow_111 = None
        add_216: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_621, mul_775);  view_621 = mul_775 = None
        mul_776: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_216, 0.7978845608028654);  add_216 = None
        tanh_15: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_776);  mul_776 = None
        add_217: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_777: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_774, add_217)
        mul_778: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_773, mul_777);  mul_777 = None
        view_622: "bf16[32, 128, 1024]" = torch.ops.aten.view.default(arg623_1, _shape_param_666);  arg623_1 = _shape_param_666 = None
        mul_779: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_773, view_622);  mul_773 = view_622 = None
        convert_element_type_550: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_778, torch.bfloat16);  mul_778 = None
        view_623: "bf16[4096, 1024]" = torch.ops.aten.view.default(convert_element_type_550, _shape_param_667);  convert_element_type_550 = _shape_param_667 = None
        permute_256: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_623, [1, 0])
        mm_278: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_256, arg624_1);  permute_256 = None
        mm_279: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_623, arg625_1);  view_623 = arg625_1 = None
        view_624: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_279, _shape_param_668);  mm_279 = _shape_param_668 = None
        convert_element_type_551: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_624, torch.float32);  view_624 = None
        convert_element_type_552: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_278, torch.float32);  mm_278 = None
        mul_780: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_779, mul_774);  mul_774 = None
        mul_781: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_779, add_217);  mul_779 = add_217 = None
        convert_element_type_553: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_781, torch.bfloat16);  mul_781 = None
        mul_782: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_41: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_782);  mul_782 = None
        mul_783: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_780, sub_41);  mul_780 = sub_41 = None
        mul_784: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_783, 0.7978845608028654);  mul_783 = None
        convert_element_type_554: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_784, torch.bfloat16)
        mul_785: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_784, 0.044715);  mul_784 = None
        pow_112: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_549, 2.0);  convert_element_type_549 = None
        mul_786: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_112, 3.0);  pow_112 = None
        mul_787: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_785, mul_786);  mul_785 = mul_786 = None
        convert_element_type_555: "bf16[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(mul_787, torch.bfloat16);  mul_787 = None
        add_218: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_554, convert_element_type_555);  convert_element_type_554 = convert_element_type_555 = None
        mul_788: "bf16[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_553, 0.5);  convert_element_type_553 = None
        add_219: "bf16[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_218, mul_788);  add_218 = mul_788 = None
        view_625: "bf16[4096, 1024]" = torch.ops.aten.view.default(add_219, _shape_param_669);  add_219 = _shape_param_669 = None
        permute_257: "bf16[1024, 4096]" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_280: "bf16[1024, 512]" = torch.ops.aten.mm.default(permute_257, arg624_1);  permute_257 = arg624_1 = None
        mm_281: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_625, arg626_1);  view_625 = arg626_1 = None
        view_626: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_281, _shape_param_670);  mm_281 = _shape_param_670 = None
        convert_element_type_556: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_626, torch.float32);  view_626 = None
        add_220: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_551, convert_element_type_556);  convert_element_type_551 = convert_element_type_556 = None
        convert_element_type_557: "f32[1024, 512]" = torch.ops.prims.convert_element_type.default(mm_280, torch.float32);  mm_280 = None
        mul_789: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_220, arg627_1);  arg627_1 = None
        mul_790: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg628_1, arg629_1)
        mul_791: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_220, mul_790);  add_220 = mul_790 = None
        sum_106: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_791, [0, 1], True, dtype = torch.float32);  mul_791 = None
        view_627: "f32[512]" = torch.ops.aten.view.default(sum_106, _shape_param_671);  sum_106 = _shape_param_671 = None
        mul_792: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_789, arg628_1)
        mul_793: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_789, arg629_1);  mul_789 = None
        sum_107: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_792, [2], True, dtype = torch.float32);  mul_792 = None
        add_221: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_215, mul_793);  add_215 = mul_793 = None
        pow_113: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg629_1, 3);  arg629_1 = None
        mul_794: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_107, -0.5);  sum_107 = None
        mul_795: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_794, pow_113);  mul_794 = pow_113 = None
        expand_42: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_795, _shape_param_672);  mul_795 = _shape_param_672 = None
        div_64: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_42, 512);  expand_42 = None
        pow_114: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg628_1, 1.0);  arg628_1 = None
        mul_796: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_114, 2.0);  pow_114 = None
        mul_797: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_64, mul_796);  div_64 = mul_796 = None
        add_222: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_221, mul_797);  add_221 = mul_797 = None
        convert_element_type_558: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(add_222, torch.bfloat16)
        convert_element_type_559: "bf16[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg630_1, torch.bfloat16);  arg630_1 = None
        mul_798: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_559, 1.1111111111111112);  convert_element_type_559 = None
        mul_799: "bf16[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_558, mul_798);  convert_element_type_558 = mul_798 = None
        view_628: "bf16[4096, 512]" = torch.ops.aten.view.default(mul_799, _shape_param_673);  mul_799 = _shape_param_673 = None
        permute_258: "bf16[512, 4096]" = torch.ops.aten.permute.default(view_628, [1, 0])
        mm_282: "bf16[512, 384]" = torch.ops.aten.mm.default(permute_258, arg631_1);  permute_258 = arg631_1 = None
        mm_283: "bf16[4096, 384]" = torch.ops.aten.mm.default(view_628, arg632_1);  view_628 = arg632_1 = None
        view_629: "bf16[32, 128, 384]" = torch.ops.aten.view.default(mm_283, _shape_param_674);  mm_283 = _shape_param_674 = None
        convert_element_type_560: "f32[512, 384]" = torch.ops.prims.convert_element_type.default(mm_282, torch.float32);  mm_282 = None
        view_630: "bf16[32, 128, 6, 64]" = torch.ops.aten.view.default(view_629, _shape_param_675);  view_629 = _shape_param_675 = None
        permute_259: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
        clone_92: "bf16[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_631: "bf16[192, 128, 64]" = torch.ops.aten.view.default(clone_92, _shape_param_676);  clone_92 = _shape_param_676 = None
        bmm_92: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(arg633_1, view_631);  arg633_1 = None
        bmm_93: "bf16[192, 128, 128]" = torch.ops.aten.bmm.default(view_631, arg634_1);  view_631 = arg634_1 = None
        view_632: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_92, _shape_param_677);  bmm_92 = _shape_param_677 = None
        view_633: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(bmm_93, _shape_param_678);  bmm_93 = _shape_param_678 = None
        convert_element_type_561: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(arg635_1, torch.bfloat16);  arg635_1 = None
        mul_800: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_561, 1.1111111111111112);  convert_element_type_561 = None
        mul_801: "bf16[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_633, mul_800);  view_633 = mul_800 = None
        convert_element_type_562: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(mul_801, torch.float32);  mul_801 = None
        full_6: "f32[32, 1, 128, 128]" = torch.ops.aten.full.default(_shape_param_679, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_679 = None
        view_634: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(arg636_1, _shape_param_680);  arg636_1 = _shape_param_680 = None
        permute_260: "f32[6, 128, 128]" = torch.ops.aten.permute.default(arg637_1, [2, 0, 1]);  arg637_1 = None
        unsqueeze_5: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_260, 0);  permute_260 = None
        add_223: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_5, full_6);  unsqueeze_5 = full_6 = None
        add_224: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_634, add_223);  view_634 = add_223 = None
        convert_element_type_563: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(add_224, torch.bfloat16);  add_224 = None
        convert_element_type_564: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_563, torch.float32);  convert_element_type_563 = None
        sub_42: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_564, arg638_1);  convert_element_type_564 = arg638_1 = None
        exp_24: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
        div_65: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_24, arg639_1);  exp_24 = arg639_1 = None
        mul_802: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_562, div_65);  convert_element_type_562 = None
        sum_108: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_802, [-1], True)
        neg_23: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_65);  div_65 = None
        fma_23: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_23, sum_108, mul_802);  neg_23 = sum_108 = mul_802 = None
        convert_element_type_565: "bf16[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None
        view_635: "bf16[192, 128, 128]" = torch.ops.aten.view.default(convert_element_type_565, _shape_param_681);  convert_element_type_565 = _shape_param_681 = None
        view_636: "bf16[32, 6, 128, 128]" = torch.ops.aten.view.default(view_635, _shape_param_682);  view_635 = _shape_param_682 = None
        view_637: "bf16[192, 128, 128]" = torch.ops.aten.view.default(view_636, _shape_param_683);  _shape_param_683 = None
        convert_element_type_566: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(view_636, torch.float32);  view_636 = None
        add_225: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_211, convert_element_type_566);  add_211 = convert_element_type_566 = None
        sum_109: "f32[1, 6, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_225, [0], True, dtype = torch.float32);  add_225 = None
        squeeze_1: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sum_109, 0);  sum_109 = None
        permute_261: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
        _unsafe_masked_index_put_accumulate_2: "f32[32, 6]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, full_3, [arg640_1], permute_261);  full_4 = full_3 = arg640_1 = permute_261 = None
        bmm_94: "bf16[192, 64, 128]" = torch.ops.aten.bmm.default(arg641_1, view_637);  arg641_1 = None
        bmm_95: "bf16[192, 128, 64]" = torch.ops.aten.bmm.default(view_637, arg642_1);  view_637 = arg642_1 = None
        view_638: "bf16[32, 6, 64, 128]" = torch.ops.aten.view.default(bmm_94, _shape_param_684);  bmm_94 = _shape_param_684 = None
        view_639: "bf16[32, 6, 128, 64]" = torch.ops.aten.view.default(bmm_95, _shape_param_685);  bmm_95 = _shape_param_685 = None
        permute_262: "bf16[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_638, [0, 1, 3, 2]);  view_638 = None
        permute_263: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_632, [0, 2, 1, 3]);  view_632 = None
        clone_93: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_263, memory_format = torch.contiguous_format);  permute_263 = None
        view_640: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_93, _shape_param_686);  clone_93 = _shape_param_686 = None
        view_641: "bf16[4096, 384]" = torch.ops.aten.view.default(view_640, _shape_param_687);  view_640 = _shape_param_687 = None
        permute_264: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_641, [1, 0])
        mm_284: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_264, arg643_1);  permute_264 = None
        mm_285: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_641, arg644_1);  view_641 = arg644_1 = None
        view_642: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_285, _shape_param_688);  mm_285 = _shape_param_688 = None
        convert_element_type_567: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_642, torch.float32);  view_642 = None
        convert_element_type_568: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_284, torch.float32);  mm_284 = None
        permute_265: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_262, [0, 2, 1, 3]);  permute_262 = None
        view_643: "bf16[32, 128, 384]" = torch.ops.aten.view.default(permute_265, _shape_param_689);  permute_265 = _shape_param_689 = None
        clone_94: "bf16[32, 128, 384]" = torch.ops.aten.clone.default(view_643, memory_format = torch.contiguous_format);  view_643 = None
        view_644: "bf16[4096, 384]" = torch.ops.aten.view.default(clone_94, _shape_param_690);  clone_94 = _shape_param_690 = None
        permute_266: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_644, [1, 0])
        mm_286: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_266, arg643_1);  permute_266 = None
        mm_287: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_644, arg645_1);  view_644 = arg645_1 = None
        view_645: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_287, _shape_param_691);  mm_287 = _shape_param_691 = None
        convert_element_type_569: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_645, torch.float32);  view_645 = None
        add_226: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_567, convert_element_type_569);  convert_element_type_567 = convert_element_type_569 = None
        convert_element_type_570: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_286, torch.float32);  mm_286 = None
        permute_267: "bf16[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_639, [0, 2, 1, 3]);  view_639 = None
        clone_95: "bf16[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_646: "bf16[32, 128, 384]" = torch.ops.aten.view.default(clone_95, _shape_param_692);  clone_95 = _shape_param_692 = None
        view_647: "bf16[4096, 384]" = torch.ops.aten.view.default(view_646, _shape_param_693);  view_646 = _shape_param_693 = None
        permute_268: "bf16[384, 4096]" = torch.ops.aten.permute.default(view_647, [1, 0])
        mm_288: "bf16[384, 512]" = torch.ops.aten.mm.default(permute_268, arg643_1);  permute_268 = arg643_1 = None
        mm_289: "bf16[4096, 512]" = torch.ops.aten.mm.default(view_647, arg646_1);  view_647 = arg646_1 = None
        view_648: "bf16[32, 128, 512]" = torch.ops.aten.view.default(mm_289, _shape_param_694);  mm_289 = _shape_param_694 = None
        convert_element_type_571: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(view_648, torch.float32);  view_648 = None
        add_227: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_226, convert_element_type_571);  add_226 = convert_element_type_571 = None
        convert_element_type_572: "f32[384, 512]" = torch.ops.prims.convert_element_type.default(mm_288, torch.float32);  mm_288 = None
        mul_803: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_227, arg647_1);  arg647_1 = None
        mul_804: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(arg648_1, arg401_1);  arg401_1 = None
        mul_805: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_804, 1.1111111111111112);  mul_804 = None
        mul_806: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_805, arg649_1)
        mul_807: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_227, mul_806);  add_227 = mul_806 = None
        sum_110: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_807, [0, 1], True, dtype = torch.float32);  mul_807 = None
        view_649: "f32[512]" = torch.ops.aten.view.default(sum_110, _shape_param_695);  sum_110 = _shape_param_695 = None
        mul_808: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_803, mul_805)
        mul_809: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_803, arg649_1);  mul_803 = None
        sum_111: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_808, [2], True, dtype = torch.float32);  mul_808 = None
        add_228: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_222, mul_809);  add_222 = mul_809 = None
        pow_115: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg649_1, 3);  arg649_1 = None
        mul_810: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_111, -0.5);  sum_111 = None
        mul_811: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_810, pow_115);  mul_810 = pow_115 = None
        expand_43: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_811, _shape_param_696);  mul_811 = _shape_param_696 = None
        div_66: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_43, 512);  expand_43 = None
        pow_116: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_805, 1.0);  mul_805 = None
        mul_812: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_116, 2.0);  pow_116 = None
        mul_813: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_66, mul_812);  div_66 = mul_812 = None
        add_229: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_228, mul_813);  add_228 = mul_813 = None
        convert_element_type_573: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(arg648_1, torch.float32);  arg648_1 = None
        mul_814: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_573, 1.1111111111111112);  convert_element_type_573 = None
        mul_815: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_229, mul_814);  add_229 = mul_814 = None
        _unsafe_masked_index_put_accumulate_3: "f32[250112, 512]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_5, unsqueeze_4, [arg403_1], mul_815);  full_5 = unsqueeze_4 = arg403_1 = mul_815 = None
        add_230: "f32[250112, 512]" = torch.ops.aten.add.Tensor(add_131, _unsafe_masked_index_put_accumulate_3);  add_131 = _unsafe_masked_index_put_accumulate_3 = None
        return (view_6, convert_element_type_12, convert_element_type_17, convert_element_type_22, view_15, convert_element_type_25, convert_element_type_32, convert_element_type_34, convert_element_type_36, view_35, convert_element_type_39, convert_element_type_46, convert_element_type_48, convert_element_type_50, view_56, convert_element_type_54, convert_element_type_59, convert_element_type_64, view_65, convert_element_type_67, convert_element_type_74, convert_element_type_76, convert_element_type_78, view_85, convert_element_type_81, convert_element_type_88, convert_element_type_90, convert_element_type_92, view_106, convert_element_type_96, convert_element_type_101, convert_element_type_106, view_115, convert_element_type_109, convert_element_type_116, convert_element_type_118, convert_element_type_120, view_135, convert_element_type_123, convert_element_type_130, convert_element_type_132, convert_element_type_134, view_156, convert_element_type_138, convert_element_type_143, convert_element_type_148, view_165, convert_element_type_151, convert_element_type_158, convert_element_type_160, convert_element_type_162, view_185, convert_element_type_165, convert_element_type_172, convert_element_type_174, convert_element_type_176, view_206, convert_element_type_180, convert_element_type_185, convert_element_type_190, view_215, convert_element_type_193, convert_element_type_200, convert_element_type_202, convert_element_type_204, view_235, convert_element_type_207, convert_element_type_214, convert_element_type_216, convert_element_type_218, view_256, convert_element_type_222, convert_element_type_227, convert_element_type_232, view_265, convert_element_type_235, convert_element_type_242, convert_element_type_244, convert_element_type_246, view_285, convert_element_type_249, convert_element_type_256, convert_element_type_258, convert_element_type_260, view_306, convert_element_type_264, convert_element_type_269, convert_element_type_274, view_315, convert_element_type_277, convert_element_type_284, convert_element_type_286, convert_element_type_288, view_335, convert_element_type_291, convert_element_type_298, convert_element_type_300, convert_element_type_302, view_356, convert_element_type_306, convert_element_type_311, convert_element_type_316, view_365, convert_element_type_319, convert_element_type_326, convert_element_type_328, convert_element_type_330, view_385, convert_element_type_333, _unsafe_masked_index_put_accumulate, convert_element_type_341, convert_element_type_343, convert_element_type_345, view_407, view_408, convert_element_type_351, convert_element_type_356, convert_element_type_361, view_417, convert_element_type_364, convert_element_type_371, convert_element_type_373, convert_element_type_375, view_438, convert_element_type_379, convert_element_type_384, convert_element_type_389, view_447, convert_element_type_392, convert_element_type_399, convert_element_type_401, convert_element_type_403, view_468, convert_element_type_407, convert_element_type_412, convert_element_type_417, view_477, convert_element_type_420, convert_element_type_427, convert_element_type_429, convert_element_type_431, view_498, convert_element_type_435, convert_element_type_440, convert_element_type_445, view_507, convert_element_type_448, convert_element_type_455, convert_element_type_457, convert_element_type_459, view_528, convert_element_type_463, convert_element_type_468, convert_element_type_473, view_537, convert_element_type_476, convert_element_type_483, convert_element_type_485, convert_element_type_487, view_558, convert_element_type_491, convert_element_type_496, convert_element_type_501, view_567, convert_element_type_504, convert_element_type_511, convert_element_type_513, convert_element_type_515, view_588, convert_element_type_519, convert_element_type_524, convert_element_type_529, view_597, convert_element_type_532, convert_element_type_539, convert_element_type_541, convert_element_type_543, view_618, convert_element_type_547, convert_element_type_552, convert_element_type_557, view_627, convert_element_type_560, _unsafe_masked_index_put_accumulate_2, convert_element_type_568, convert_element_type_570, convert_element_type_572, view_649, add_230)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
