"""
Standalone repro captured via capture_hook.
Label: GPTNeoForSequenceClassification
Pattern hash: e732f9a597ca
Shape hash: f66205c6
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "b8[32, 2]", arg3_1: "b8[32, 1]", arg4_1: "bf16[32, 2]", arg5_1: "bf16[32, 2]", arg6_1: "i64[32]", arg7_1: "i64[32]", arg8_1: "bf16[4096, 2048]", arg9_1: "bf16[2, 2048]", arg10_1: "f32[2048]", arg11_1: "f32[32, 128, 2048]", arg12_1: "f32[32, 128, 1]", arg13_1: "bf16[2048, 8192]", arg14_1: "bf16[4096, 8192]", arg15_1: "bf16[4096, 8192]", arg16_1: "bf16[8192, 2048]", arg17_1: "bf16[4096, 2048]", arg18_1: "f32[2048]", arg19_1: "f32[32, 128, 2048]", arg20_1: "f32[32, 128, 1]", arg21_1: "bf16[2048, 2048]", arg22_1: "bf16[4096, 2048]", arg23_1: "bf16[512, 128, 128]", arg24_1: "bf16[512, 128, 128]", arg25_1: "f32[32, 16, 128, 128]", arg26_1: "b8[1, 1, 2048, 2048]", arg27_1: "bf16[512, 128, 128]", arg28_1: "bf16[512, 128, 128]", arg29_1: "bf16[4096, 2048]", arg30_1: "bf16[2048, 2048]", arg31_1: "bf16[2048, 2048]", arg32_1: "bf16[2048, 2048]", arg33_1: "f32[2048]", arg34_1: "f32[32, 128, 2048]", arg35_1: "f32[32, 128, 1]", arg36_1: "bf16[2048, 8192]", arg37_1: "bf16[4096, 8192]", arg38_1: "bf16[4096, 8192]", arg39_1: "bf16[8192, 2048]", arg40_1: "bf16[4096, 2048]", arg41_1: "f32[2048]", arg42_1: "f32[32, 128, 2048]", arg43_1: "f32[32, 128, 1]", arg44_1: "bf16[2048, 2048]", arg45_1: "bf16[4096, 2048]", arg46_1: "bf16[512, 128, 128]", arg47_1: "bf16[512, 128, 128]", arg48_1: "f32[32, 16, 128, 128]", arg49_1: "b8[1, 1, 2048, 2048]", arg50_1: "bf16[512, 128, 128]", arg51_1: "bf16[512, 128, 128]", arg52_1: "bf16[4096, 2048]", arg53_1: "bf16[2048, 2048]", arg54_1: "bf16[2048, 2048]", arg55_1: "bf16[2048, 2048]", arg56_1: "f32[2048]", arg57_1: "f32[32, 128, 2048]", arg58_1: "f32[32, 128, 1]", arg59_1: "bf16[2048, 8192]", arg60_1: "bf16[4096, 8192]", arg61_1: "bf16[4096, 8192]", arg62_1: "bf16[8192, 2048]", arg63_1: "bf16[4096, 2048]", arg64_1: "f32[2048]", arg65_1: "f32[32, 128, 2048]", arg66_1: "f32[32, 128, 1]", arg67_1: "bf16[2048, 2048]", arg68_1: "bf16[4096, 2048]", arg69_1: "bf16[512, 128, 128]", arg70_1: "bf16[512, 128, 128]", arg71_1: "f32[32, 16, 128, 128]", arg72_1: "b8[1, 1, 2048, 2048]", arg73_1: "bf16[512, 128, 128]", arg74_1: "bf16[512, 128, 128]", arg75_1: "bf16[4096, 2048]", arg76_1: "bf16[2048, 2048]", arg77_1: "bf16[2048, 2048]", arg78_1: "bf16[2048, 2048]", arg79_1: "f32[2048]", arg80_1: "f32[32, 128, 2048]", arg81_1: "f32[32, 128, 1]", arg82_1: "bf16[2048, 8192]", arg83_1: "bf16[4096, 8192]", arg84_1: "bf16[4096, 8192]", arg85_1: "bf16[8192, 2048]", arg86_1: "bf16[4096, 2048]", arg87_1: "f32[2048]", arg88_1: "f32[32, 128, 2048]", arg89_1: "f32[32, 128, 1]", arg90_1: "bf16[2048, 2048]", arg91_1: "bf16[4096, 2048]", arg92_1: "bf16[512, 128, 128]", arg93_1: "bf16[512, 128, 128]", arg94_1: "f32[32, 16, 128, 128]", arg95_1: "b8[1, 1, 2048, 2048]", arg96_1: "bf16[512, 128, 128]", arg97_1: "bf16[512, 128, 128]", arg98_1: "bf16[4096, 2048]", arg99_1: "bf16[2048, 2048]", arg100_1: "bf16[2048, 2048]", arg101_1: "bf16[2048, 2048]", arg102_1: "f32[2048]", arg103_1: "f32[32, 128, 2048]", arg104_1: "f32[32, 128, 1]", arg105_1: "bf16[2048, 8192]", arg106_1: "bf16[4096, 8192]", arg107_1: "bf16[4096, 8192]", arg108_1: "bf16[8192, 2048]", arg109_1: "bf16[4096, 2048]", arg110_1: "f32[2048]", arg111_1: "f32[32, 128, 2048]", arg112_1: "f32[32, 128, 1]", arg113_1: "bf16[2048, 2048]", arg114_1: "bf16[4096, 2048]", arg115_1: "bf16[512, 128, 128]", arg116_1: "bf16[512, 128, 128]", arg117_1: "f32[32, 16, 128, 128]", arg118_1: "b8[1, 1, 2048, 2048]", arg119_1: "bf16[512, 128, 128]", arg120_1: "bf16[512, 128, 128]", arg121_1: "bf16[4096, 2048]", arg122_1: "bf16[2048, 2048]", arg123_1: "bf16[2048, 2048]", arg124_1: "bf16[2048, 2048]", arg125_1: "f32[2048]", arg126_1: "f32[32, 128, 2048]", arg127_1: "f32[32, 128, 1]", arg128_1: "bf16[2048, 8192]", arg129_1: "bf16[4096, 8192]", arg130_1: "bf16[4096, 8192]", arg131_1: "bf16[8192, 2048]", arg132_1: "bf16[4096, 2048]", arg133_1: "f32[2048]", arg134_1: "f32[32, 128, 2048]", arg135_1: "f32[32, 128, 1]", arg136_1: "bf16[2048, 2048]", arg137_1: "bf16[4096, 2048]", arg138_1: "bf16[512, 128, 128]", arg139_1: "bf16[512, 128, 128]", arg140_1: "f32[32, 16, 128, 128]", arg141_1: "b8[1, 1, 2048, 2048]", arg142_1: "bf16[512, 128, 128]", arg143_1: "bf16[512, 128, 128]", arg144_1: "bf16[4096, 2048]", arg145_1: "bf16[2048, 2048]", arg146_1: "bf16[2048, 2048]", arg147_1: "bf16[2048, 2048]", arg148_1: "f32[2048]", arg149_1: "f32[32, 128, 2048]", arg150_1: "f32[32, 128, 1]", arg151_1: "bf16[2048, 8192]", arg152_1: "bf16[4096, 8192]", arg153_1: "bf16[4096, 8192]", arg154_1: "bf16[8192, 2048]", arg155_1: "bf16[4096, 2048]", arg156_1: "f32[2048]", arg157_1: "f32[32, 128, 2048]", arg158_1: "f32[32, 128, 1]", arg159_1: "bf16[2048, 2048]", arg160_1: "bf16[4096, 2048]", arg161_1: "bf16[512, 128, 128]", arg162_1: "bf16[512, 128, 128]", arg163_1: "f32[32, 16, 128, 128]", arg164_1: "b8[1, 1, 2048, 2048]", arg165_1: "bf16[512, 128, 128]", arg166_1: "bf16[512, 128, 128]", arg167_1: "bf16[4096, 2048]", arg168_1: "bf16[2048, 2048]", arg169_1: "bf16[2048, 2048]", arg170_1: "bf16[2048, 2048]", arg171_1: "f32[2048]", arg172_1: "f32[32, 128, 2048]", arg173_1: "f32[32, 128, 1]", arg174_1: "bf16[2048, 8192]", arg175_1: "bf16[4096, 8192]", arg176_1: "bf16[4096, 8192]", arg177_1: "bf16[8192, 2048]", arg178_1: "bf16[4096, 2048]", arg179_1: "f32[2048]", arg180_1: "f32[32, 128, 2048]", arg181_1: "f32[32, 128, 1]", arg182_1: "bf16[2048, 2048]", arg183_1: "bf16[4096, 2048]", arg184_1: "bf16[512, 128, 128]", arg185_1: "bf16[512, 128, 128]", arg186_1: "f32[32, 16, 128, 128]", arg187_1: "b8[1, 1, 2048, 2048]", arg188_1: "bf16[512, 128, 128]", arg189_1: "bf16[512, 128, 128]", arg190_1: "bf16[4096, 2048]", arg191_1: "bf16[2048, 2048]", arg192_1: "bf16[2048, 2048]", arg193_1: "bf16[2048, 2048]", arg194_1: "f32[2048]", arg195_1: "f32[32, 128, 2048]", arg196_1: "f32[32, 128, 1]", arg197_1: "bf16[2048, 8192]", arg198_1: "bf16[4096, 8192]", arg199_1: "bf16[4096, 8192]", arg200_1: "bf16[8192, 2048]", arg201_1: "bf16[4096, 2048]", arg202_1: "f32[2048]", arg203_1: "f32[32, 128, 2048]", arg204_1: "f32[32, 128, 1]", arg205_1: "bf16[2048, 2048]", arg206_1: "bf16[4096, 2048]", arg207_1: "bf16[512, 128, 128]", arg208_1: "bf16[512, 128, 128]", arg209_1: "f32[32, 16, 128, 128]", arg210_1: "b8[1, 1, 2048, 2048]", arg211_1: "bf16[512, 128, 128]", arg212_1: "bf16[512, 128, 128]", arg213_1: "bf16[4096, 2048]", arg214_1: "bf16[2048, 2048]", arg215_1: "bf16[2048, 2048]", arg216_1: "bf16[2048, 2048]", arg217_1: "f32[2048]", arg218_1: "f32[32, 128, 2048]", arg219_1: "f32[32, 128, 1]", arg220_1: "bf16[2048, 8192]", arg221_1: "bf16[4096, 8192]", arg222_1: "bf16[4096, 8192]", arg223_1: "bf16[8192, 2048]", arg224_1: "bf16[4096, 2048]", arg225_1: "f32[2048]", arg226_1: "f32[32, 128, 2048]", arg227_1: "f32[32, 128, 1]", arg228_1: "bf16[2048, 2048]", arg229_1: "bf16[4096, 2048]", arg230_1: "bf16[512, 128, 128]", arg231_1: "bf16[512, 128, 128]", arg232_1: "f32[32, 16, 128, 128]", arg233_1: "b8[1, 1, 2048, 2048]", arg234_1: "bf16[512, 128, 128]", arg235_1: "bf16[512, 128, 128]", arg236_1: "bf16[4096, 2048]", arg237_1: "bf16[2048, 2048]", arg238_1: "bf16[2048, 2048]", arg239_1: "bf16[2048, 2048]", arg240_1: "f32[2048]", arg241_1: "f32[32, 128, 2048]", arg242_1: "f32[32, 128, 1]", arg243_1: "bf16[2048, 8192]", arg244_1: "bf16[4096, 8192]", arg245_1: "bf16[4096, 8192]", arg246_1: "bf16[8192, 2048]", arg247_1: "bf16[4096, 2048]", arg248_1: "f32[2048]", arg249_1: "f32[32, 128, 2048]", arg250_1: "f32[32, 128, 1]", arg251_1: "bf16[2048, 2048]", arg252_1: "bf16[4096, 2048]", arg253_1: "bf16[512, 128, 128]", arg254_1: "bf16[512, 128, 128]", arg255_1: "f32[32, 16, 128, 128]", arg256_1: "b8[1, 1, 2048, 2048]", arg257_1: "bf16[512, 128, 128]", arg258_1: "bf16[512, 128, 128]", arg259_1: "bf16[4096, 2048]", arg260_1: "bf16[2048, 2048]", arg261_1: "bf16[2048, 2048]", arg262_1: "bf16[2048, 2048]", arg263_1: "f32[2048]", arg264_1: "f32[32, 128, 2048]", arg265_1: "f32[32, 128, 1]", arg266_1: "bf16[2048, 8192]", arg267_1: "bf16[4096, 8192]", arg268_1: "bf16[4096, 8192]", arg269_1: "bf16[8192, 2048]", arg270_1: "bf16[4096, 2048]", arg271_1: "f32[2048]", arg272_1: "f32[32, 128, 2048]", arg273_1: "f32[32, 128, 1]", arg274_1: "bf16[2048, 2048]", arg275_1: "bf16[4096, 2048]", arg276_1: "bf16[512, 128, 128]", arg277_1: "bf16[512, 128, 128]", arg278_1: "f32[32, 16, 128, 128]", arg279_1: "b8[1, 1, 2048, 2048]", arg280_1: "bf16[512, 128, 128]", arg281_1: "bf16[512, 128, 128]", arg282_1: "bf16[4096, 2048]", arg283_1: "bf16[2048, 2048]", arg284_1: "bf16[2048, 2048]", arg285_1: "bf16[2048, 2048]", arg286_1: "f32[2048]", arg287_1: "f32[32, 128, 2048]", arg288_1: "f32[32, 128, 1]", arg289_1: "bf16[2048, 8192]", arg290_1: "bf16[4096, 8192]", arg291_1: "bf16[4096, 8192]", arg292_1: "bf16[8192, 2048]", arg293_1: "bf16[4096, 2048]", arg294_1: "f32[2048]", arg295_1: "f32[32, 128, 2048]", arg296_1: "f32[32, 128, 1]", arg297_1: "bf16[2048, 2048]", arg298_1: "bf16[4096, 2048]", arg299_1: "bf16[512, 128, 128]", arg300_1: "bf16[512, 128, 128]", arg301_1: "f32[32, 16, 128, 128]", arg302_1: "b8[1, 1, 2048, 2048]", arg303_1: "bf16[512, 128, 128]", arg304_1: "bf16[512, 128, 128]", arg305_1: "bf16[4096, 2048]", arg306_1: "bf16[2048, 2048]", arg307_1: "bf16[2048, 2048]", arg308_1: "bf16[2048, 2048]", arg309_1: "f32[2048]", arg310_1: "f32[32, 128, 2048]", arg311_1: "f32[32, 128, 1]", arg312_1: "bf16[2048, 8192]", arg313_1: "bf16[4096, 8192]", arg314_1: "bf16[4096, 8192]", arg315_1: "bf16[8192, 2048]", arg316_1: "bf16[4096, 2048]", arg317_1: "f32[2048]", arg318_1: "f32[32, 128, 2048]", arg319_1: "f32[32, 128, 1]", arg320_1: "bf16[2048, 2048]", arg321_1: "bf16[4096, 2048]", arg322_1: "bf16[512, 128, 128]", arg323_1: "bf16[512, 128, 128]", arg324_1: "f32[32, 16, 128, 128]", arg325_1: "b8[1, 1, 2048, 2048]", arg326_1: "bf16[512, 128, 128]", arg327_1: "bf16[512, 128, 128]", arg328_1: "bf16[4096, 2048]", arg329_1: "bf16[2048, 2048]", arg330_1: "bf16[2048, 2048]", arg331_1: "bf16[2048, 2048]", arg332_1: "f32[2048]", arg333_1: "f32[32, 128, 2048]", arg334_1: "f32[32, 128, 1]", arg335_1: "bf16[2048, 8192]", arg336_1: "bf16[4096, 8192]", arg337_1: "bf16[4096, 8192]", arg338_1: "bf16[8192, 2048]", arg339_1: "bf16[4096, 2048]", arg340_1: "f32[2048]", arg341_1: "f32[32, 128, 2048]", arg342_1: "f32[32, 128, 1]", arg343_1: "bf16[2048, 2048]", arg344_1: "bf16[4096, 2048]", arg345_1: "bf16[512, 128, 128]", arg346_1: "bf16[512, 128, 128]", arg347_1: "f32[32, 16, 128, 128]", arg348_1: "b8[1, 1, 2048, 2048]", arg349_1: "bf16[512, 128, 128]", arg350_1: "bf16[512, 128, 128]", arg351_1: "bf16[4096, 2048]", arg352_1: "bf16[2048, 2048]", arg353_1: "bf16[2048, 2048]", arg354_1: "bf16[2048, 2048]", arg355_1: "f32[2048]", arg356_1: "f32[32, 128, 2048]", arg357_1: "f32[32, 128, 1]", arg358_1: "bf16[2048, 8192]", arg359_1: "bf16[4096, 8192]", arg360_1: "bf16[4096, 8192]", arg361_1: "bf16[8192, 2048]", arg362_1: "bf16[4096, 2048]", arg363_1: "f32[2048]", arg364_1: "f32[32, 128, 2048]", arg365_1: "f32[32, 128, 1]", arg366_1: "bf16[2048, 2048]", arg367_1: "bf16[4096, 2048]", arg368_1: "bf16[512, 128, 128]", arg369_1: "bf16[512, 128, 128]", arg370_1: "f32[32, 16, 128, 128]", arg371_1: "b8[1, 1, 2048, 2048]", arg372_1: "bf16[512, 128, 128]", arg373_1: "bf16[512, 128, 128]", arg374_1: "bf16[4096, 2048]", arg375_1: "bf16[2048, 2048]", arg376_1: "bf16[2048, 2048]", arg377_1: "bf16[2048, 2048]", arg378_1: "f32[2048]", arg379_1: "f32[32, 128, 2048]", arg380_1: "f32[32, 128, 1]", arg381_1: "bf16[2048, 8192]", arg382_1: "bf16[4096, 8192]", arg383_1: "bf16[4096, 8192]", arg384_1: "bf16[8192, 2048]", arg385_1: "bf16[4096, 2048]", arg386_1: "f32[2048]", arg387_1: "f32[32, 128, 2048]", arg388_1: "f32[32, 128, 1]", arg389_1: "bf16[2048, 2048]", arg390_1: "bf16[4096, 2048]", arg391_1: "bf16[512, 128, 128]", arg392_1: "bf16[512, 128, 128]", arg393_1: "f32[32, 16, 128, 128]", arg394_1: "b8[1, 1, 2048, 2048]", arg395_1: "bf16[512, 128, 128]", arg396_1: "bf16[512, 128, 128]", arg397_1: "bf16[4096, 2048]", arg398_1: "bf16[2048, 2048]", arg399_1: "bf16[2048, 2048]", arg400_1: "bf16[2048, 2048]", arg401_1: "f32[2048]", arg402_1: "f32[32, 128, 2048]", arg403_1: "f32[32, 128, 1]", arg404_1: "bf16[2048, 8192]", arg405_1: "bf16[4096, 8192]", arg406_1: "bf16[4096, 8192]", arg407_1: "bf16[8192, 2048]", arg408_1: "bf16[4096, 2048]", arg409_1: "f32[2048]", arg410_1: "f32[32, 128, 2048]", arg411_1: "f32[32, 128, 1]", arg412_1: "bf16[2048, 2048]", arg413_1: "bf16[4096, 2048]", arg414_1: "bf16[512, 128, 128]", arg415_1: "bf16[512, 128, 128]", arg416_1: "f32[32, 16, 128, 128]", arg417_1: "b8[1, 1, 2048, 2048]", arg418_1: "bf16[512, 128, 128]", arg419_1: "bf16[512, 128, 128]", arg420_1: "bf16[4096, 2048]", arg421_1: "bf16[2048, 2048]", arg422_1: "bf16[2048, 2048]", arg423_1: "bf16[2048, 2048]", arg424_1: "f32[2048]", arg425_1: "f32[32, 128, 2048]", arg426_1: "f32[32, 128, 1]", arg427_1: "bf16[2048, 8192]", arg428_1: "bf16[4096, 8192]", arg429_1: "bf16[4096, 8192]", arg430_1: "bf16[8192, 2048]", arg431_1: "bf16[4096, 2048]", arg432_1: "f32[2048]", arg433_1: "f32[32, 128, 2048]", arg434_1: "f32[32, 128, 1]", arg435_1: "bf16[2048, 2048]", arg436_1: "bf16[4096, 2048]", arg437_1: "bf16[512, 128, 128]", arg438_1: "bf16[512, 128, 128]", arg439_1: "f32[32, 16, 128, 128]", arg440_1: "b8[1, 1, 2048, 2048]", arg441_1: "bf16[512, 128, 128]", arg442_1: "bf16[512, 128, 128]", arg443_1: "bf16[4096, 2048]", arg444_1: "bf16[2048, 2048]", arg445_1: "bf16[2048, 2048]", arg446_1: "bf16[2048, 2048]", arg447_1: "f32[2048]", arg448_1: "f32[32, 128, 2048]", arg449_1: "f32[32, 128, 1]", arg450_1: "bf16[2048, 8192]", arg451_1: "bf16[4096, 8192]", arg452_1: "bf16[4096, 8192]", arg453_1: "bf16[8192, 2048]", arg454_1: "bf16[4096, 2048]", arg455_1: "f32[2048]", arg456_1: "f32[32, 128, 2048]", arg457_1: "f32[32, 128, 1]", arg458_1: "bf16[2048, 2048]", arg459_1: "bf16[4096, 2048]", arg460_1: "bf16[512, 128, 128]", arg461_1: "bf16[512, 128, 128]", arg462_1: "f32[32, 16, 128, 128]", arg463_1: "b8[1, 1, 2048, 2048]", arg464_1: "bf16[512, 128, 128]", arg465_1: "bf16[512, 128, 128]", arg466_1: "bf16[4096, 2048]", arg467_1: "bf16[2048, 2048]", arg468_1: "bf16[2048, 2048]", arg469_1: "bf16[2048, 2048]", arg470_1: "f32[2048]", arg471_1: "f32[32, 128, 2048]", arg472_1: "f32[32, 128, 1]", arg473_1: "bf16[2048, 8192]", arg474_1: "bf16[4096, 8192]", arg475_1: "bf16[4096, 8192]", arg476_1: "bf16[8192, 2048]", arg477_1: "bf16[4096, 2048]", arg478_1: "f32[2048]", arg479_1: "f32[32, 128, 2048]", arg480_1: "f32[32, 128, 1]", arg481_1: "bf16[2048, 2048]", arg482_1: "bf16[4096, 2048]", arg483_1: "bf16[512, 128, 128]", arg484_1: "bf16[512, 128, 128]", arg485_1: "f32[32, 16, 128, 128]", arg486_1: "b8[1, 1, 2048, 2048]", arg487_1: "bf16[512, 128, 128]", arg488_1: "bf16[512, 128, 128]", arg489_1: "bf16[4096, 2048]", arg490_1: "bf16[2048, 2048]", arg491_1: "bf16[2048, 2048]", arg492_1: "bf16[2048, 2048]", arg493_1: "f32[2048]", arg494_1: "f32[32, 128, 2048]", arg495_1: "f32[32, 128, 1]", arg496_1: "bf16[2048, 8192]", arg497_1: "bf16[4096, 8192]", arg498_1: "bf16[4096, 8192]", arg499_1: "bf16[8192, 2048]", arg500_1: "bf16[4096, 2048]", arg501_1: "f32[2048]", arg502_1: "f32[32, 128, 2048]", arg503_1: "f32[32, 128, 1]", arg504_1: "bf16[2048, 2048]", arg505_1: "bf16[4096, 2048]", arg506_1: "bf16[512, 128, 128]", arg507_1: "bf16[512, 128, 128]", arg508_1: "f32[32, 16, 128, 128]", arg509_1: "b8[1, 1, 2048, 2048]", arg510_1: "bf16[512, 128, 128]", arg511_1: "bf16[512, 128, 128]", arg512_1: "bf16[4096, 2048]", arg513_1: "bf16[2048, 2048]", arg514_1: "bf16[2048, 2048]", arg515_1: "bf16[2048, 2048]", arg516_1: "f32[2048]", arg517_1: "f32[32, 128, 2048]", arg518_1: "f32[32, 128, 1]", arg519_1: "bf16[2048, 8192]", arg520_1: "bf16[4096, 8192]", arg521_1: "bf16[4096, 8192]", arg522_1: "bf16[8192, 2048]", arg523_1: "bf16[4096, 2048]", arg524_1: "f32[2048]", arg525_1: "f32[32, 128, 2048]", arg526_1: "f32[32, 128, 1]", arg527_1: "bf16[2048, 2048]", arg528_1: "bf16[4096, 2048]", arg529_1: "bf16[512, 128, 128]", arg530_1: "bf16[512, 128, 128]", arg531_1: "f32[32, 16, 128, 128]", arg532_1: "b8[1, 1, 2048, 2048]", arg533_1: "bf16[512, 128, 128]", arg534_1: "bf16[512, 128, 128]", arg535_1: "bf16[4096, 2048]", arg536_1: "bf16[2048, 2048]", arg537_1: "bf16[2048, 2048]", arg538_1: "bf16[2048, 2048]", arg539_1: "f32[2048]", arg540_1: "f32[32, 128, 2048]", arg541_1: "f32[32, 128, 1]", arg542_1: "bf16[2048, 8192]", arg543_1: "bf16[4096, 8192]", arg544_1: "bf16[4096, 8192]", arg545_1: "bf16[8192, 2048]", arg546_1: "bf16[4096, 2048]", arg547_1: "f32[2048]", arg548_1: "f32[32, 128, 2048]", arg549_1: "f32[32, 128, 1]", arg550_1: "bf16[2048, 2048]", arg551_1: "bf16[4096, 2048]", arg552_1: "bf16[512, 128, 128]", arg553_1: "bf16[512, 128, 128]", arg554_1: "i64[1, 128]", arg555_1: "i64[32, 128]", arg556_1: "bf16[512, 128, 128]", arg557_1: "b8[1, 1, 2048, 2048]", arg558_1: "f32[32, 16, 128, 1]", arg559_1: "f32[32, 16, 128, 1]", arg560_1: "bf16[512, 128, 128]", arg561_1: "bf16[512, 128, 128]", arg562_1: "bf16[4096, 2048]", arg563_1: "bf16[2048, 2048]", arg564_1: "bf16[2048, 2048]", arg565_1: "bf16[2048, 2048]", arg566_1: "f32[2048]", arg567_1: "f32[32, 128, 2048]", arg568_1: "f32[1, 128, 2048]", arg569_1: "f32[32, 128, 1]", arg570_1: "f32[32, 128, 1]", arg571_1: "i64[32, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where: "f32[32, 2]" = torch.ops.aten.where.self(arg2_1, scalar_tensor_1, scalar_tensor);  arg2_1 = scalar_tensor_1 = scalar_tensor = None
        full: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[32, 1]" = torch.ops.aten.where.self(arg3_1, div, full);  arg3_1 = div = None
        mul: "f32[32, 2]" = torch.ops.aten.mul.Tensor(where, where_1);  where = where_1 = None
        convert_element_type: "bf16[32, 2]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        convert_element_type_2: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.float32);  arg4_1 = None
        amax: "f32[32, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [1], True)
        sub: "f32[32, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax);  convert_element_type_2 = amax = None
        exp: "f32[32, 2]" = torch.ops.aten.exp.default(sub)
        sum_1: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(exp, [1], True);  exp = None
        log: "f32[32, 1]" = torch.ops.aten.log.default(sum_1);  sum_1 = None
        sub_1: "f32[32, 2]" = torch.ops.aten.sub.Tensor(sub, log);  sub = log = None
        convert_element_type_3: "bf16[32, 2]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[32, 2]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp_1: "f32[32, 2]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_2: "f32[32, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[32, 2]" = torch.ops.aten.mul.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        sub_2: "f32[32, 2]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[32, 2]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        add: "bf16[32, 2]" = torch.ops.aten.add.Tensor(arg5_1, convert_element_type_5);  arg5_1 = convert_element_type_5 = None
        full_1: "bf16[32, 128, 2]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        index_put: "bf16[32, 128, 2]" = torch.ops.aten.index_put.default(full_1, [arg6_1, arg7_1], add, True);  full_1 = arg7_1 = add = None
        view: "bf16[4096, 2]" = torch.ops.aten.view.default(index_put, _shape_param_1);  index_put = _shape_param_1 = None
        permute: "bf16[2, 4096]" = torch.ops.aten.permute.default(view, [1, 0])
        mm: "bf16[2, 2048]" = torch.ops.aten.mm.default(permute, arg8_1);  permute = arg8_1 = None
        mm_1: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view, arg9_1);  view = arg9_1 = None
        view_1: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_1, _shape_param_2);  mm_1 = _shape_param_2 = None
        convert_element_type_6: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_1, torch.float32);  view_1 = None
        convert_element_type_7: "f32[2, 2048]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mul_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_6, arg10_1);  arg10_1 = None
        mul_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, 2048)
        sum_3: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, arg11_1);  mul_2 = None
        sum_4: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg11_1, sum_4);  sum_4 = None
        sub_3: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_4: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_3, mul_5);  sub_3 = mul_5 = None
        mul_6: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg12_1, sub_4);  arg12_1 = sub_4 = None
        mul_7: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_6, arg11_1);  arg11_1 = None
        sum_5: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_6: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_6, [0, 1]);  convert_element_type_6 = None
        convert_element_type_8: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16)
        view_2: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_8, _shape_param_3);  convert_element_type_8 = _shape_param_3 = None
        mm_2: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_2, arg13_1);  arg13_1 = None
        permute_1: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_2, [1, 0])
        mm_3: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_1, arg14_1);  permute_1 = arg14_1 = None
        sum_7: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_2, [0], True, dtype = torch.float32);  view_2 = None
        view_3: "f32[2048]" = torch.ops.aten.view.default(sum_7, _shape_param_4);  sum_7 = _shape_param_4 = None
        convert_element_type_9: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        view_4: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_2, _shape_param_5);  mm_2 = _shape_param_5 = None
        convert_element_type_10: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_4, torch.float32);  view_4 = None
        convert_element_type_11: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_12: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_9, torch.float32);  convert_element_type_9 = None
        view_5: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg15_1, _shape_param_6);  arg15_1 = _shape_param_6 = None
        mul_8: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_5, 0.5)
        mul_9: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_10, mul_8);  mul_8 = None
        convert_element_type_13: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32)
        pow_1: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_13, 3.0)
        mul_10: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_1: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_5, mul_10);  view_5 = mul_10 = None
        mul_11: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_1, 0.7978845608028654);  add_1 = None
        tanh: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_2: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_12: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_10, add_2);  convert_element_type_10 = add_2 = None
        convert_element_type_14: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_13: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_5: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_13);  mul_13 = None
        mul_14: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_9, sub_5);  mul_9 = sub_5 = None
        mul_15: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_14, 0.7978845608028654);  mul_14 = None
        convert_element_type_15: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16)
        mul_16: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_15, 0.044715);  mul_15 = None
        pow_2: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_13, 2.0);  convert_element_type_13 = None
        mul_17: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_18: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        convert_element_type_16: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        add_3: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_15, convert_element_type_16);  convert_element_type_15 = convert_element_type_16 = None
        mul_19: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 0.5);  convert_element_type_14 = None
        add_4: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_3, mul_19);  add_3 = mul_19 = None
        view_6: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_4, _shape_param_7);  add_4 = _shape_param_7 = None
        mm_4: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_6, arg16_1);  arg16_1 = None
        permute_2: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_6, [1, 0])
        mm_5: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_2, arg17_1);  permute_2 = arg17_1 = None
        sum_8: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_6, [0], True, dtype = torch.float32);  view_6 = None
        view_7: "f32[8192]" = torch.ops.aten.view.default(sum_8, _shape_param_8);  sum_8 = _shape_param_8 = None
        convert_element_type_17: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        view_8: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_4, _shape_param_9);  mm_4 = _shape_param_9 = None
        convert_element_type_18: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        convert_element_type_19: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_20: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_17, torch.float32);  convert_element_type_17 = None
        mul_20: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_18, arg18_1);  arg18_1 = None
        mul_21: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_20, 2048)
        sum_9: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True)
        mul_22: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_20, arg19_1);  mul_20 = None
        sum_10: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True);  mul_22 = None
        mul_23: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg19_1, sum_10);  sum_10 = None
        sub_6: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_21, sum_9);  mul_21 = sum_9 = None
        sub_7: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_6, mul_23);  sub_6 = mul_23 = None
        mul_24: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg20_1, sub_7);  arg20_1 = sub_7 = None
        mul_25: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_18, arg19_1);  arg19_1 = None
        sum_11: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_25, [0, 1]);  mul_25 = None
        sum_12: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_18, [0, 1]);  convert_element_type_18 = None
        add_5: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_6, mul_24);  mul_6 = mul_24 = None
        convert_element_type_21: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16)
        view_9: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_21, _shape_param_10);  convert_element_type_21 = _shape_param_10 = None
        mm_6: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_9, arg21_1);  arg21_1 = None
        permute_3: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_9, [1, 0])
        mm_7: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_3, arg22_1);  permute_3 = arg22_1 = None
        sum_13: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_9, [0], True, dtype = torch.float32);  view_9 = None
        view_10: "f32[2048]" = torch.ops.aten.view.default(sum_13, _shape_param_11);  sum_13 = _shape_param_11 = None
        convert_element_type_22: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_10, torch.bfloat16);  view_10 = None
        view_11: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_6, _shape_param_12);  mm_6 = _shape_param_12 = None
        convert_element_type_23: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_24: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_22, torch.float32);  convert_element_type_22 = None
        view_12: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_11, _shape_param_13);  view_11 = _shape_param_13 = None
        permute_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        clone: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_13: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone, _shape_param_14);  clone = _shape_param_14 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg23_1, view_13);  arg23_1 = None
        bmm_1: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_13, arg24_1);  view_13 = arg24_1 = None
        view_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_15);  bmm = _shape_param_15 = None
        view_15: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_16);  bmm_1 = _shape_param_16 = None
        convert_element_type_25: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None
        mul_26: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_25, arg25_1);  convert_element_type_25 = None
        sum_14: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_26, [-1], True)
        neg: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg25_1);  arg25_1 = None
        fma: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_14, mul_26);  neg = sum_14 = mul_26 = None
        convert_element_type_26: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_1: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg26_1, 2, 0, 128);  arg26_1 = None
        slice_2: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 128);  slice_1 = None
        where_2: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_2, convert_element_type_26, full_2);  slice_2 = convert_element_type_26 = None
        view_16: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_2, _shape_param_17);  where_2 = _shape_param_17 = None
        bmm_2: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg27_1, view_16);  arg27_1 = None
        bmm_3: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_16, arg28_1);  view_16 = arg28_1 = None
        view_17: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_18);  bmm_2 = _shape_param_18 = None
        view_18: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_3, _shape_param_19);  bmm_3 = _shape_param_19 = None
        convert_element_type_27: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        permute_5: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_27, [0, 1, 3, 2]);  convert_element_type_27 = None
        convert_element_type_28: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_5, torch.bfloat16);  permute_5 = None
        convert_element_type_29: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_18, torch.bfloat16);  view_18 = None
        permute_6: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_1: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_19: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_1, _shape_param_20);  clone_1 = _shape_param_20 = None
        permute_7: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_28, [0, 2, 1, 3]);  convert_element_type_28 = None
        view_20: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_7, _shape_param_21);  permute_7 = _shape_param_21 = None
        permute_8: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_29, [0, 2, 1, 3]);  convert_element_type_29 = None
        clone_2: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_21: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_2, _shape_param_22);  clone_2 = _shape_param_22 = None
        view_22: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_19, _shape_param_23);  view_19 = _shape_param_23 = None
        permute_9: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_22, [1, 0])
        mm_8: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_9, arg29_1);  permute_9 = None
        mm_9: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_22, arg30_1);  view_22 = arg30_1 = None
        view_23: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_9, _shape_param_24);  mm_9 = _shape_param_24 = None
        convert_element_type_30: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_23, torch.float32);  view_23 = None
        convert_element_type_31: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        clone_3: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_20, memory_format = torch.contiguous_format);  view_20 = None
        view_24: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_3, _shape_param_25);  clone_3 = _shape_param_25 = None
        permute_10: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_24, [1, 0])
        mm_10: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_10, arg29_1);  permute_10 = None
        mm_11: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_24, arg31_1);  view_24 = arg31_1 = None
        view_25: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_11, _shape_param_26);  mm_11 = _shape_param_26 = None
        convert_element_type_32: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_25, torch.float32);  view_25 = None
        add_6: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_30, convert_element_type_32);  convert_element_type_30 = convert_element_type_32 = None
        convert_element_type_33: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        view_26: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_21, _shape_param_27);  view_21 = _shape_param_27 = None
        permute_11: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_26, [1, 0])
        mm_12: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_11, arg29_1);  permute_11 = arg29_1 = None
        mm_13: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_26, arg32_1);  view_26 = arg32_1 = None
        view_27: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_13, _shape_param_28);  mm_13 = _shape_param_28 = None
        convert_element_type_34: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        add_7: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_6, convert_element_type_34);  add_6 = convert_element_type_34 = None
        convert_element_type_35: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        mul_27: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_7, arg33_1);  arg33_1 = None
        mul_28: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_27, 2048)
        sum_15: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True)
        mul_29: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_27, arg34_1);  mul_27 = None
        sum_16: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_29, [2], True);  mul_29 = None
        mul_30: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg34_1, sum_16);  sum_16 = None
        sub_8: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_28, sum_15);  mul_28 = sum_15 = None
        sub_9: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_8, mul_30);  sub_8 = mul_30 = None
        mul_31: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg35_1, sub_9);  arg35_1 = sub_9 = None
        mul_32: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_7, arg34_1);  arg34_1 = None
        sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_32, [0, 1]);  mul_32 = None
        sum_18: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_7, [0, 1]);  add_7 = None
        add_8: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_5, mul_31);  add_5 = mul_31 = None
        convert_element_type_36: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16)
        view_28: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_29);  convert_element_type_36 = _shape_param_29 = None
        mm_14: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_28, arg36_1);  arg36_1 = None
        permute_12: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_15: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_12, arg37_1);  permute_12 = arg37_1 = None
        sum_19: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_28, [0], True, dtype = torch.float32);  view_28 = None
        view_29: "f32[2048]" = torch.ops.aten.view.default(sum_19, _shape_param_30);  sum_19 = _shape_param_30 = None
        convert_element_type_37: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_29, torch.bfloat16);  view_29 = None
        view_30: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_14, _shape_param_31);  mm_14 = _shape_param_31 = None
        convert_element_type_38: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        convert_element_type_39: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_40: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_37, torch.float32);  convert_element_type_37 = None
        view_31: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg38_1, _shape_param_32);  arg38_1 = _shape_param_32 = None
        mul_33: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_31, 0.5)
        mul_34: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_38, mul_33);  mul_33 = None
        convert_element_type_41: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_31, torch.float32)
        pow_3: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_41, 3.0)
        mul_35: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_9: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_31, mul_35);  view_31 = mul_35 = None
        mul_36: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh_1: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_36);  mul_36 = None
        add_10: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_37: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_38, add_10);  convert_element_type_38 = add_10 = None
        convert_element_type_42: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        mul_38: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_10: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_38);  mul_38 = None
        mul_39: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_34, sub_10);  mul_34 = sub_10 = None
        mul_40: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_39, 0.7978845608028654);  mul_39 = None
        convert_element_type_43: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16)
        mul_41: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_40, 0.044715);  mul_40 = None
        pow_4: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_41, 2.0);  convert_element_type_41 = None
        mul_42: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_4, 3.0);  pow_4 = None
        mul_43: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
        convert_element_type_44: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None
        add_11: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_43, convert_element_type_44);  convert_element_type_43 = convert_element_type_44 = None
        mul_44: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 0.5);  convert_element_type_42 = None
        add_12: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_11, mul_44);  add_11 = mul_44 = None
        view_32: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_12, _shape_param_33);  add_12 = _shape_param_33 = None
        mm_16: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_32, arg39_1);  arg39_1 = None
        permute_13: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_17: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_13, arg40_1);  permute_13 = arg40_1 = None
        sum_20: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_32, [0], True, dtype = torch.float32);  view_32 = None
        view_33: "f32[8192]" = torch.ops.aten.view.default(sum_20, _shape_param_34);  sum_20 = _shape_param_34 = None
        convert_element_type_45: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_33, torch.bfloat16);  view_33 = None
        view_34: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_16, _shape_param_35);  mm_16 = _shape_param_35 = None
        convert_element_type_46: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        convert_element_type_47: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_48: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_45, torch.float32);  convert_element_type_45 = None
        mul_45: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_46, arg41_1);  arg41_1 = None
        mul_46: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_45, 2048)
        sum_21: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_45, [2], True)
        mul_47: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_45, arg42_1);  mul_45 = None
        sum_22: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_47, [2], True);  mul_47 = None
        mul_48: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg42_1, sum_22);  sum_22 = None
        sub_11: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_46, sum_21);  mul_46 = sum_21 = None
        sub_12: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_11, mul_48);  sub_11 = mul_48 = None
        mul_49: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg43_1, sub_12);  arg43_1 = sub_12 = None
        mul_50: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_46, arg42_1);  arg42_1 = None
        sum_23: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_50, [0, 1]);  mul_50 = None
        sum_24: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_46, [0, 1]);  convert_element_type_46 = None
        add_13: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_8, mul_49);  add_8 = mul_49 = None
        convert_element_type_49: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16)
        view_35: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_36);  convert_element_type_49 = _shape_param_36 = None
        mm_18: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_35, arg44_1);  arg44_1 = None
        permute_14: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_35, [1, 0])
        mm_19: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_14, arg45_1);  permute_14 = arg45_1 = None
        sum_25: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_35, [0], True, dtype = torch.float32);  view_35 = None
        view_36: "f32[2048]" = torch.ops.aten.view.default(sum_25, _shape_param_37);  sum_25 = _shape_param_37 = None
        convert_element_type_50: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_36, torch.bfloat16);  view_36 = None
        view_37: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_18, _shape_param_38);  mm_18 = _shape_param_38 = None
        convert_element_type_51: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_52: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_50, torch.float32);  convert_element_type_50 = None
        view_38: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_37, _shape_param_39);  view_37 = _shape_param_39 = None
        permute_15: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
        clone_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_39: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_4, _shape_param_40);  clone_4 = _shape_param_40 = None
        bmm_4: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg46_1, view_39);  arg46_1 = None
        bmm_5: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_39, arg47_1);  view_39 = arg47_1 = None
        view_40: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_41);  bmm_4 = _shape_param_41 = None
        view_41: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_5, _shape_param_42);  bmm_5 = _shape_param_42 = None
        convert_element_type_53: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_51: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_53, arg48_1);  convert_element_type_53 = None
        sum_26: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_51, [-1], True)
        neg_1: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg48_1);  arg48_1 = None
        fma_1: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_1, sum_26, mul_51);  neg_1 = sum_26 = mul_51 = None
        convert_element_type_54: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        slice_3: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg49_1, 2, 0, 128);  arg49_1 = None
        slice_4: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 128);  slice_3 = None
        where_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_4, convert_element_type_54, full_2);  slice_4 = convert_element_type_54 = None
        view_42: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_3, _shape_param_43);  where_3 = _shape_param_43 = None
        bmm_6: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg50_1, view_42);  arg50_1 = None
        bmm_7: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_42, arg51_1);  view_42 = arg51_1 = None
        view_43: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_44);  bmm_6 = _shape_param_44 = None
        view_44: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_7, _shape_param_45);  bmm_7 = _shape_param_45 = None
        convert_element_type_55: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        permute_16: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_55, [0, 1, 3, 2]);  convert_element_type_55 = None
        convert_element_type_56: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_16, torch.bfloat16);  permute_16 = None
        convert_element_type_57: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_44, torch.bfloat16);  view_44 = None
        permute_17: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        clone_5: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_45: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_5, _shape_param_46);  clone_5 = _shape_param_46 = None
        permute_18: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_56, [0, 2, 1, 3]);  convert_element_type_56 = None
        view_46: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_18, _shape_param_47);  permute_18 = _shape_param_47 = None
        permute_19: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_57, [0, 2, 1, 3]);  convert_element_type_57 = None
        clone_6: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        view_47: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_6, _shape_param_48);  clone_6 = _shape_param_48 = None
        view_48: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_45, _shape_param_49);  view_45 = _shape_param_49 = None
        permute_20: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_48, [1, 0])
        mm_20: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_20, arg52_1);  permute_20 = None
        mm_21: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_48, arg53_1);  view_48 = arg53_1 = None
        view_49: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_21, _shape_param_50);  mm_21 = _shape_param_50 = None
        convert_element_type_58: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_49, torch.float32);  view_49 = None
        convert_element_type_59: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        clone_7: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_46, memory_format = torch.contiguous_format);  view_46 = None
        view_50: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_7, _shape_param_51);  clone_7 = _shape_param_51 = None
        permute_21: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_50, [1, 0])
        mm_22: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_21, arg52_1);  permute_21 = None
        mm_23: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_50, arg54_1);  view_50 = arg54_1 = None
        view_51: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_23, _shape_param_52);  mm_23 = _shape_param_52 = None
        convert_element_type_60: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_51, torch.float32);  view_51 = None
        add_14: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_58, convert_element_type_60);  convert_element_type_58 = convert_element_type_60 = None
        convert_element_type_61: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        view_52: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_47, _shape_param_53);  view_47 = _shape_param_53 = None
        permute_22: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_52, [1, 0])
        mm_24: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_22, arg52_1);  permute_22 = arg52_1 = None
        mm_25: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_52, arg55_1);  view_52 = arg55_1 = None
        view_53: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_25, _shape_param_54);  mm_25 = _shape_param_54 = None
        convert_element_type_62: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_53, torch.float32);  view_53 = None
        add_15: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_14, convert_element_type_62);  add_14 = convert_element_type_62 = None
        convert_element_type_63: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        mul_52: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_15, arg56_1);  arg56_1 = None
        mul_53: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_52, 2048)
        sum_27: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_52, [2], True)
        mul_54: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_52, arg57_1);  mul_52 = None
        sum_28: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_54, [2], True);  mul_54 = None
        mul_55: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg57_1, sum_28);  sum_28 = None
        sub_13: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_53, sum_27);  mul_53 = sum_27 = None
        sub_14: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_13, mul_55);  sub_13 = mul_55 = None
        mul_56: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg58_1, sub_14);  arg58_1 = sub_14 = None
        mul_57: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_15, arg57_1);  arg57_1 = None
        sum_29: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_57, [0, 1]);  mul_57 = None
        sum_30: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_15, [0, 1]);  add_15 = None
        add_16: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_13, mul_56);  add_13 = mul_56 = None
        convert_element_type_64: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16)
        view_54: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_64, _shape_param_55);  convert_element_type_64 = _shape_param_55 = None
        mm_26: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_54, arg59_1);  arg59_1 = None
        permute_23: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_54, [1, 0])
        mm_27: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_23, arg60_1);  permute_23 = arg60_1 = None
        sum_31: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_54, [0], True, dtype = torch.float32);  view_54 = None
        view_55: "f32[2048]" = torch.ops.aten.view.default(sum_31, _shape_param_56);  sum_31 = _shape_param_56 = None
        convert_element_type_65: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_55, torch.bfloat16);  view_55 = None
        view_56: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_26, _shape_param_57);  mm_26 = _shape_param_57 = None
        convert_element_type_66: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_56, torch.float32);  view_56 = None
        convert_element_type_67: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_68: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_65, torch.float32);  convert_element_type_65 = None
        view_57: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg61_1, _shape_param_58);  arg61_1 = _shape_param_58 = None
        mul_58: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_57, 0.5)
        mul_59: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_66, mul_58);  mul_58 = None
        convert_element_type_69: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32)
        pow_5: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 3.0)
        mul_60: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_17: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_57, mul_60);  view_57 = mul_60 = None
        mul_61: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_2: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_61);  mul_61 = None
        add_18: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_62: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_66, add_18);  convert_element_type_66 = add_18 = None
        convert_element_type_70: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        mul_63: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_15: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_63);  mul_63 = None
        mul_64: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_59, sub_15);  mul_59 = sub_15 = None
        mul_65: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_64, 0.7978845608028654);  mul_64 = None
        convert_element_type_71: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16)
        mul_66: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_65, 0.044715);  mul_65 = None
        pow_6: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_69, 2.0);  convert_element_type_69 = None
        mul_67: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_6, 3.0);  pow_6 = None
        mul_68: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
        convert_element_type_72: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_68, torch.bfloat16);  mul_68 = None
        add_19: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_71, convert_element_type_72);  convert_element_type_71 = convert_element_type_72 = None
        mul_69: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 0.5);  convert_element_type_70 = None
        add_20: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_19, mul_69);  add_19 = mul_69 = None
        view_58: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_20, _shape_param_59);  add_20 = _shape_param_59 = None
        mm_28: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_58, arg62_1);  arg62_1 = None
        permute_24: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_58, [1, 0])
        mm_29: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_24, arg63_1);  permute_24 = arg63_1 = None
        sum_32: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_58, [0], True, dtype = torch.float32);  view_58 = None
        view_59: "f32[8192]" = torch.ops.aten.view.default(sum_32, _shape_param_60);  sum_32 = _shape_param_60 = None
        convert_element_type_73: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_59, torch.bfloat16);  view_59 = None
        view_60: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_28, _shape_param_61);  mm_28 = _shape_param_61 = None
        convert_element_type_74: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_60, torch.float32);  view_60 = None
        convert_element_type_75: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_76: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_73, torch.float32);  convert_element_type_73 = None
        mul_70: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_74, arg64_1);  arg64_1 = None
        mul_71: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_70, 2048)
        sum_33: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_70, [2], True)
        mul_72: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_70, arg65_1);  mul_70 = None
        sum_34: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_72, [2], True);  mul_72 = None
        mul_73: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg65_1, sum_34);  sum_34 = None
        sub_16: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_71, sum_33);  mul_71 = sum_33 = None
        sub_17: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_16, mul_73);  sub_16 = mul_73 = None
        mul_74: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg66_1, sub_17);  arg66_1 = sub_17 = None
        mul_75: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_74, arg65_1);  arg65_1 = None
        sum_35: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_75, [0, 1]);  mul_75 = None
        sum_36: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_74, [0, 1]);  convert_element_type_74 = None
        add_21: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_16, mul_74);  add_16 = mul_74 = None
        convert_element_type_77: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16)
        view_61: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_77, _shape_param_62);  convert_element_type_77 = _shape_param_62 = None
        mm_30: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_61, arg67_1);  arg67_1 = None
        permute_25: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_61, [1, 0])
        mm_31: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_25, arg68_1);  permute_25 = arg68_1 = None
        sum_37: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_61, [0], True, dtype = torch.float32);  view_61 = None
        view_62: "f32[2048]" = torch.ops.aten.view.default(sum_37, _shape_param_63);  sum_37 = _shape_param_63 = None
        convert_element_type_78: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_62, torch.bfloat16);  view_62 = None
        view_63: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_30, _shape_param_64);  mm_30 = _shape_param_64 = None
        convert_element_type_79: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_80: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_78, torch.float32);  convert_element_type_78 = None
        view_64: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_63, _shape_param_65);  view_63 = _shape_param_65 = None
        permute_26: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
        clone_8: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_65: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_8, _shape_param_66);  clone_8 = _shape_param_66 = None
        bmm_8: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg69_1, view_65);  arg69_1 = None
        bmm_9: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_65, arg70_1);  view_65 = arg70_1 = None
        view_66: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_67);  bmm_8 = _shape_param_67 = None
        view_67: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_9, _shape_param_68);  bmm_9 = _shape_param_68 = None
        convert_element_type_81: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_67, torch.float32);  view_67 = None
        mul_76: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_81, arg71_1);  convert_element_type_81 = None
        sum_38: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_76, [-1], True)
        neg_2: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg71_1);  arg71_1 = None
        fma_2: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_2, sum_38, mul_76);  neg_2 = sum_38 = mul_76 = None
        convert_element_type_82: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        slice_5: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg72_1, 2, 0, 128);  arg72_1 = None
        slice_6: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 128);  slice_5 = None
        where_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_6, convert_element_type_82, full_2);  slice_6 = convert_element_type_82 = None
        view_68: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_4, _shape_param_69);  where_4 = _shape_param_69 = None
        bmm_10: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg73_1, view_68);  arg73_1 = None
        bmm_11: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_68, arg74_1);  view_68 = arg74_1 = None
        view_69: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_70);  bmm_10 = _shape_param_70 = None
        view_70: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_11, _shape_param_71);  bmm_11 = _shape_param_71 = None
        convert_element_type_83: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        permute_27: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_83, [0, 1, 3, 2]);  convert_element_type_83 = None
        convert_element_type_84: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_27, torch.bfloat16);  permute_27 = None
        convert_element_type_85: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_70, torch.bfloat16);  view_70 = None
        permute_28: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
        clone_9: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_71: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_9, _shape_param_72);  clone_9 = _shape_param_72 = None
        permute_29: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_84, [0, 2, 1, 3]);  convert_element_type_84 = None
        view_72: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_29, _shape_param_73);  permute_29 = _shape_param_73 = None
        permute_30: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_85, [0, 2, 1, 3]);  convert_element_type_85 = None
        clone_10: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_73: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_10, _shape_param_74);  clone_10 = _shape_param_74 = None
        view_74: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_71, _shape_param_75);  view_71 = _shape_param_75 = None
        permute_31: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_74, [1, 0])
        mm_32: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_31, arg75_1);  permute_31 = None
        mm_33: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_74, arg76_1);  view_74 = arg76_1 = None
        view_75: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_33, _shape_param_76);  mm_33 = _shape_param_76 = None
        convert_element_type_86: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_75, torch.float32);  view_75 = None
        convert_element_type_87: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_32, torch.float32);  mm_32 = None
        clone_11: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_72, memory_format = torch.contiguous_format);  view_72 = None
        view_76: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_11, _shape_param_77);  clone_11 = _shape_param_77 = None
        permute_32: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_76, [1, 0])
        mm_34: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_32, arg75_1);  permute_32 = None
        mm_35: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_76, arg77_1);  view_76 = arg77_1 = None
        view_77: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_35, _shape_param_78);  mm_35 = _shape_param_78 = None
        convert_element_type_88: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_77, torch.float32);  view_77 = None
        add_22: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_86, convert_element_type_88);  convert_element_type_86 = convert_element_type_88 = None
        convert_element_type_89: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        view_78: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_73, _shape_param_79);  view_73 = _shape_param_79 = None
        permute_33: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_78, [1, 0])
        mm_36: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_33, arg75_1);  permute_33 = arg75_1 = None
        mm_37: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_78, arg78_1);  view_78 = arg78_1 = None
        view_79: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_37, _shape_param_80);  mm_37 = _shape_param_80 = None
        convert_element_type_90: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        add_23: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_22, convert_element_type_90);  add_22 = convert_element_type_90 = None
        convert_element_type_91: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        mul_77: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_23, arg79_1);  arg79_1 = None
        mul_78: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_77, 2048)
        sum_39: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_77, [2], True)
        mul_79: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_77, arg80_1);  mul_77 = None
        sum_40: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_79, [2], True);  mul_79 = None
        mul_80: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg80_1, sum_40);  sum_40 = None
        sub_18: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_78, sum_39);  mul_78 = sum_39 = None
        sub_19: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_18, mul_80);  sub_18 = mul_80 = None
        mul_81: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg81_1, sub_19);  arg81_1 = sub_19 = None
        mul_82: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_23, arg80_1);  arg80_1 = None
        sum_41: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_82, [0, 1]);  mul_82 = None
        sum_42: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_23, [0, 1]);  add_23 = None
        add_24: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_21, mul_81);  add_21 = mul_81 = None
        convert_element_type_92: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16)
        view_80: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_92, _shape_param_81);  convert_element_type_92 = _shape_param_81 = None
        mm_38: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_80, arg82_1);  arg82_1 = None
        permute_34: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_80, [1, 0])
        mm_39: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_34, arg83_1);  permute_34 = arg83_1 = None
        sum_43: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_80, [0], True, dtype = torch.float32);  view_80 = None
        view_81: "f32[2048]" = torch.ops.aten.view.default(sum_43, _shape_param_82);  sum_43 = _shape_param_82 = None
        convert_element_type_93: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_81, torch.bfloat16);  view_81 = None
        view_82: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_38, _shape_param_83);  mm_38 = _shape_param_83 = None
        convert_element_type_94: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_82, torch.float32);  view_82 = None
        convert_element_type_95: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_96: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_93, torch.float32);  convert_element_type_93 = None
        view_83: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg84_1, _shape_param_84);  arg84_1 = _shape_param_84 = None
        mul_83: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_83, 0.5)
        mul_84: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_94, mul_83);  mul_83 = None
        convert_element_type_97: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_83, torch.float32)
        pow_7: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_97, 3.0)
        mul_85: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_25: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_83, mul_85);  view_83 = mul_85 = None
        mul_86: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_3: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_26: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_87: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_94, add_26);  convert_element_type_94 = add_26 = None
        convert_element_type_98: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None
        mul_88: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_20: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_88);  mul_88 = None
        mul_89: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_84, sub_20);  mul_84 = sub_20 = None
        mul_90: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_89, 0.7978845608028654);  mul_89 = None
        convert_element_type_99: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16)
        mul_91: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_90, 0.044715);  mul_90 = None
        pow_8: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_97, 2.0);  convert_element_type_97 = None
        mul_92: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_8, 3.0);  pow_8 = None
        mul_93: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None
        convert_element_type_100: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        add_27: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_99, convert_element_type_100);  convert_element_type_99 = convert_element_type_100 = None
        mul_94: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_98, 0.5);  convert_element_type_98 = None
        add_28: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_27, mul_94);  add_27 = mul_94 = None
        view_84: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_28, _shape_param_85);  add_28 = _shape_param_85 = None
        mm_40: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_84, arg85_1);  arg85_1 = None
        permute_35: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_84, [1, 0])
        mm_41: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_35, arg86_1);  permute_35 = arg86_1 = None
        sum_44: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_84, [0], True, dtype = torch.float32);  view_84 = None
        view_85: "f32[8192]" = torch.ops.aten.view.default(sum_44, _shape_param_86);  sum_44 = _shape_param_86 = None
        convert_element_type_101: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_85, torch.bfloat16);  view_85 = None
        view_86: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_40, _shape_param_87);  mm_40 = _shape_param_87 = None
        convert_element_type_102: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_86, torch.float32);  view_86 = None
        convert_element_type_103: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_104: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_101, torch.float32);  convert_element_type_101 = None
        mul_95: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_102, arg87_1);  arg87_1 = None
        mul_96: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_95, 2048)
        sum_45: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True)
        mul_97: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_95, arg88_1);  mul_95 = None
        sum_46: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_97, [2], True);  mul_97 = None
        mul_98: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg88_1, sum_46);  sum_46 = None
        sub_21: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_96, sum_45);  mul_96 = sum_45 = None
        sub_22: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_21, mul_98);  sub_21 = mul_98 = None
        mul_99: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg89_1, sub_22);  arg89_1 = sub_22 = None
        mul_100: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_102, arg88_1);  arg88_1 = None
        sum_47: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_100, [0, 1]);  mul_100 = None
        sum_48: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_102, [0, 1]);  convert_element_type_102 = None
        add_29: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_24, mul_99);  add_24 = mul_99 = None
        convert_element_type_105: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16)
        view_87: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_105, _shape_param_88);  convert_element_type_105 = _shape_param_88 = None
        mm_42: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_87, arg90_1);  arg90_1 = None
        permute_36: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_87, [1, 0])
        mm_43: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_36, arg91_1);  permute_36 = arg91_1 = None
        sum_49: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_87, [0], True, dtype = torch.float32);  view_87 = None
        view_88: "f32[2048]" = torch.ops.aten.view.default(sum_49, _shape_param_89);  sum_49 = _shape_param_89 = None
        convert_element_type_106: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_88, torch.bfloat16);  view_88 = None
        view_89: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_42, _shape_param_90);  mm_42 = _shape_param_90 = None
        convert_element_type_107: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_108: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_106, torch.float32);  convert_element_type_106 = None
        view_90: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_89, _shape_param_91);  view_89 = _shape_param_91 = None
        permute_37: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        clone_12: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_91: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_12, _shape_param_92);  clone_12 = _shape_param_92 = None
        bmm_12: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg92_1, view_91);  arg92_1 = None
        bmm_13: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_91, arg93_1);  view_91 = arg93_1 = None
        view_92: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_93);  bmm_12 = _shape_param_93 = None
        view_93: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_13, _shape_param_94);  bmm_13 = _shape_param_94 = None
        convert_element_type_109: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        mul_101: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_109, arg94_1);  convert_element_type_109 = None
        sum_50: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [-1], True)
        neg_3: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg94_1);  arg94_1 = None
        fma_3: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_3, sum_50, mul_101);  neg_3 = sum_50 = mul_101 = None
        convert_element_type_110: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        slice_7: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg95_1, 2, 0, 128);  arg95_1 = None
        slice_8: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 128);  slice_7 = None
        where_5: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_8, convert_element_type_110, full_2);  slice_8 = convert_element_type_110 = None
        view_94: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_5, _shape_param_95);  where_5 = _shape_param_95 = None
        bmm_14: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg96_1, view_94);  arg96_1 = None
        bmm_15: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_94, arg97_1);  view_94 = arg97_1 = None
        view_95: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_96);  bmm_14 = _shape_param_96 = None
        view_96: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_15, _shape_param_97);  bmm_15 = _shape_param_97 = None
        convert_element_type_111: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_95, torch.float32);  view_95 = None
        permute_38: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_111, [0, 1, 3, 2]);  convert_element_type_111 = None
        convert_element_type_112: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_38, torch.bfloat16);  permute_38 = None
        convert_element_type_113: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_96, torch.bfloat16);  view_96 = None
        permute_39: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None
        clone_13: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_97: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_13, _shape_param_98);  clone_13 = _shape_param_98 = None
        permute_40: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_112, [0, 2, 1, 3]);  convert_element_type_112 = None
        view_98: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_40, _shape_param_99);  permute_40 = _shape_param_99 = None
        permute_41: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_113, [0, 2, 1, 3]);  convert_element_type_113 = None
        clone_14: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_99: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_14, _shape_param_100);  clone_14 = _shape_param_100 = None
        view_100: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_97, _shape_param_101);  view_97 = _shape_param_101 = None
        permute_42: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_100, [1, 0])
        mm_44: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_42, arg98_1);  permute_42 = None
        mm_45: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_100, arg99_1);  view_100 = arg99_1 = None
        view_101: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_45, _shape_param_102);  mm_45 = _shape_param_102 = None
        convert_element_type_114: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None
        convert_element_type_115: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        clone_15: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_98, memory_format = torch.contiguous_format);  view_98 = None
        view_102: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_15, _shape_param_103);  clone_15 = _shape_param_103 = None
        permute_43: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_102, [1, 0])
        mm_46: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_43, arg98_1);  permute_43 = None
        mm_47: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_102, arg100_1);  view_102 = arg100_1 = None
        view_103: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_47, _shape_param_104);  mm_47 = _shape_param_104 = None
        convert_element_type_116: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_103, torch.float32);  view_103 = None
        add_30: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_114, convert_element_type_116);  convert_element_type_114 = convert_element_type_116 = None
        convert_element_type_117: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        view_104: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_99, _shape_param_105);  view_99 = _shape_param_105 = None
        permute_44: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_104, [1, 0])
        mm_48: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_44, arg98_1);  permute_44 = arg98_1 = None
        mm_49: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_104, arg101_1);  view_104 = arg101_1 = None
        view_105: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_49, _shape_param_106);  mm_49 = _shape_param_106 = None
        convert_element_type_118: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        add_31: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_30, convert_element_type_118);  add_30 = convert_element_type_118 = None
        convert_element_type_119: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_48, torch.float32);  mm_48 = None
        mul_102: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_31, arg102_1);  arg102_1 = None
        mul_103: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_102, 2048)
        sum_51: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_102, [2], True)
        mul_104: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_102, arg103_1);  mul_102 = None
        sum_52: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [2], True);  mul_104 = None
        mul_105: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg103_1, sum_52);  sum_52 = None
        sub_23: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_103, sum_51);  mul_103 = sum_51 = None
        sub_24: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_23, mul_105);  sub_23 = mul_105 = None
        mul_106: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg104_1, sub_24);  arg104_1 = sub_24 = None
        mul_107: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_31, arg103_1);  arg103_1 = None
        sum_53: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_107, [0, 1]);  mul_107 = None
        sum_54: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_31, [0, 1]);  add_31 = None
        add_32: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_29, mul_106);  add_29 = mul_106 = None
        convert_element_type_120: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16)
        view_106: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_120, _shape_param_107);  convert_element_type_120 = _shape_param_107 = None
        mm_50: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_106, arg105_1);  arg105_1 = None
        permute_45: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_106, [1, 0])
        mm_51: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_45, arg106_1);  permute_45 = arg106_1 = None
        sum_55: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_106, [0], True, dtype = torch.float32);  view_106 = None
        view_107: "f32[2048]" = torch.ops.aten.view.default(sum_55, _shape_param_108);  sum_55 = _shape_param_108 = None
        convert_element_type_121: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_107, torch.bfloat16);  view_107 = None
        view_108: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_50, _shape_param_109);  mm_50 = _shape_param_109 = None
        convert_element_type_122: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_108, torch.float32);  view_108 = None
        convert_element_type_123: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_124: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_121, torch.float32);  convert_element_type_121 = None
        view_109: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg107_1, _shape_param_110);  arg107_1 = _shape_param_110 = None
        mul_108: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_109: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_122, mul_108);  mul_108 = None
        convert_element_type_125: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_109, torch.float32)
        pow_9: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_125, 3.0)
        mul_110: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_33: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_109, mul_110);  view_109 = mul_110 = None
        mul_111: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_4: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_111);  mul_111 = None
        add_34: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_112: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_122, add_34);  convert_element_type_122 = add_34 = None
        convert_element_type_126: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        mul_113: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_25: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_113);  mul_113 = None
        mul_114: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_109, sub_25);  mul_109 = sub_25 = None
        mul_115: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_114, 0.7978845608028654);  mul_114 = None
        convert_element_type_127: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_115, torch.bfloat16)
        mul_116: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_115, 0.044715);  mul_115 = None
        pow_10: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_125, 2.0);  convert_element_type_125 = None
        mul_117: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_10, 3.0);  pow_10 = None
        mul_118: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        convert_element_type_128: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        add_35: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_127, convert_element_type_128);  convert_element_type_127 = convert_element_type_128 = None
        mul_119: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.5);  convert_element_type_126 = None
        add_36: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_35, mul_119);  add_35 = mul_119 = None
        view_110: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_36, _shape_param_111);  add_36 = _shape_param_111 = None
        mm_52: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_110, arg108_1);  arg108_1 = None
        permute_46: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_110, [1, 0])
        mm_53: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_46, arg109_1);  permute_46 = arg109_1 = None
        sum_56: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_110, [0], True, dtype = torch.float32);  view_110 = None
        view_111: "f32[8192]" = torch.ops.aten.view.default(sum_56, _shape_param_112);  sum_56 = _shape_param_112 = None
        convert_element_type_129: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_111, torch.bfloat16);  view_111 = None
        view_112: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_52, _shape_param_113);  mm_52 = _shape_param_113 = None
        convert_element_type_130: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_112, torch.float32);  view_112 = None
        convert_element_type_131: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_132: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_129, torch.float32);  convert_element_type_129 = None
        mul_120: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_130, arg110_1);  arg110_1 = None
        mul_121: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, 2048)
        sum_57: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_120, [2], True)
        mul_122: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, arg111_1);  mul_120 = None
        sum_58: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True);  mul_122 = None
        mul_123: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg111_1, sum_58);  sum_58 = None
        sub_26: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_121, sum_57);  mul_121 = sum_57 = None
        sub_27: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_26, mul_123);  sub_26 = mul_123 = None
        mul_124: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg112_1, sub_27);  arg112_1 = sub_27 = None
        mul_125: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_130, arg111_1);  arg111_1 = None
        sum_59: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_125, [0, 1]);  mul_125 = None
        sum_60: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_130, [0, 1]);  convert_element_type_130 = None
        add_37: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_32, mul_124);  add_32 = mul_124 = None
        convert_element_type_133: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16)
        view_113: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_133, _shape_param_114);  convert_element_type_133 = _shape_param_114 = None
        mm_54: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_113, arg113_1);  arg113_1 = None
        permute_47: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_113, [1, 0])
        mm_55: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_47, arg114_1);  permute_47 = arg114_1 = None
        sum_61: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_113, [0], True, dtype = torch.float32);  view_113 = None
        view_114: "f32[2048]" = torch.ops.aten.view.default(sum_61, _shape_param_115);  sum_61 = _shape_param_115 = None
        convert_element_type_134: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_114, torch.bfloat16);  view_114 = None
        view_115: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_54, _shape_param_116);  mm_54 = _shape_param_116 = None
        convert_element_type_135: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_136: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_134, torch.float32);  convert_element_type_134 = None
        view_116: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_115, _shape_param_117);  view_115 = _shape_param_117 = None
        permute_48: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_116, [0, 2, 1, 3]);  view_116 = None
        clone_16: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_117: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_16, _shape_param_118);  clone_16 = _shape_param_118 = None
        bmm_16: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg115_1, view_117);  arg115_1 = None
        bmm_17: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_117, arg116_1);  view_117 = arg116_1 = None
        view_118: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_119);  bmm_16 = _shape_param_119 = None
        view_119: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_17, _shape_param_120);  bmm_17 = _shape_param_120 = None
        convert_element_type_137: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        mul_126: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_137, arg117_1);  convert_element_type_137 = None
        sum_62: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_126, [-1], True)
        neg_4: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg117_1);  arg117_1 = None
        fma_4: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_4, sum_62, mul_126);  neg_4 = sum_62 = mul_126 = None
        convert_element_type_138: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        slice_9: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg118_1, 2, 0, 128);  arg118_1 = None
        slice_10: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_9, 3, 0, 128);  slice_9 = None
        where_6: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_10, convert_element_type_138, full_2);  slice_10 = convert_element_type_138 = None
        view_120: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_6, _shape_param_121);  where_6 = _shape_param_121 = None
        bmm_18: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg119_1, view_120);  arg119_1 = None
        bmm_19: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_120, arg120_1);  view_120 = arg120_1 = None
        view_121: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_122);  bmm_18 = _shape_param_122 = None
        view_122: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_19, _shape_param_123);  bmm_19 = _shape_param_123 = None
        convert_element_type_139: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_121, torch.float32);  view_121 = None
        permute_49: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_139, [0, 1, 3, 2]);  convert_element_type_139 = None
        convert_element_type_140: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_49, torch.bfloat16);  permute_49 = None
        convert_element_type_141: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_122, torch.bfloat16);  view_122 = None
        permute_50: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        clone_17: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_123: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_17, _shape_param_124);  clone_17 = _shape_param_124 = None
        permute_51: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_140, [0, 2, 1, 3]);  convert_element_type_140 = None
        view_124: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_51, _shape_param_125);  permute_51 = _shape_param_125 = None
        permute_52: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_141, [0, 2, 1, 3]);  convert_element_type_141 = None
        clone_18: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_125: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_18, _shape_param_126);  clone_18 = _shape_param_126 = None
        view_126: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_123, _shape_param_127);  view_123 = _shape_param_127 = None
        permute_53: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_126, [1, 0])
        mm_56: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_53, arg121_1);  permute_53 = None
        mm_57: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_126, arg122_1);  view_126 = arg122_1 = None
        view_127: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_57, _shape_param_128);  mm_57 = _shape_param_128 = None
        convert_element_type_142: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_127, torch.float32);  view_127 = None
        convert_element_type_143: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        clone_19: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_124, memory_format = torch.contiguous_format);  view_124 = None
        view_128: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_19, _shape_param_129);  clone_19 = _shape_param_129 = None
        permute_54: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_128, [1, 0])
        mm_58: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_54, arg121_1);  permute_54 = None
        mm_59: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_128, arg123_1);  view_128 = arg123_1 = None
        view_129: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_59, _shape_param_130);  mm_59 = _shape_param_130 = None
        convert_element_type_144: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        add_38: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_142, convert_element_type_144);  convert_element_type_142 = convert_element_type_144 = None
        convert_element_type_145: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        view_130: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_125, _shape_param_131);  view_125 = _shape_param_131 = None
        permute_55: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_130, [1, 0])
        mm_60: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_55, arg121_1);  permute_55 = arg121_1 = None
        mm_61: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_130, arg124_1);  view_130 = arg124_1 = None
        view_131: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_61, _shape_param_132);  mm_61 = _shape_param_132 = None
        convert_element_type_146: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None
        add_39: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_38, convert_element_type_146);  add_38 = convert_element_type_146 = None
        convert_element_type_147: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        mul_127: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_39, arg125_1);  arg125_1 = None
        mul_128: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_127, 2048)
        sum_63: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [2], True)
        mul_129: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_127, arg126_1);  mul_127 = None
        sum_64: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True);  mul_129 = None
        mul_130: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg126_1, sum_64);  sum_64 = None
        sub_28: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_128, sum_63);  mul_128 = sum_63 = None
        sub_29: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_28, mul_130);  sub_28 = mul_130 = None
        mul_131: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg127_1, sub_29);  arg127_1 = sub_29 = None
        mul_132: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_39, arg126_1);  arg126_1 = None
        sum_65: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_132, [0, 1]);  mul_132 = None
        sum_66: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_39, [0, 1]);  add_39 = None
        add_40: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_37, mul_131);  add_37 = mul_131 = None
        convert_element_type_148: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16)
        view_132: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_148, _shape_param_133);  convert_element_type_148 = _shape_param_133 = None
        mm_62: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_132, arg128_1);  arg128_1 = None
        permute_56: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_132, [1, 0])
        mm_63: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_56, arg129_1);  permute_56 = arg129_1 = None
        sum_67: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_132, [0], True, dtype = torch.float32);  view_132 = None
        view_133: "f32[2048]" = torch.ops.aten.view.default(sum_67, _shape_param_134);  sum_67 = _shape_param_134 = None
        convert_element_type_149: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_133, torch.bfloat16);  view_133 = None
        view_134: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_62, _shape_param_135);  mm_62 = _shape_param_135 = None
        convert_element_type_150: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_134, torch.float32);  view_134 = None
        convert_element_type_151: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_152: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_149, torch.float32);  convert_element_type_149 = None
        view_135: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg130_1, _shape_param_136);  arg130_1 = _shape_param_136 = None
        mul_133: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_135, 0.5)
        mul_134: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_150, mul_133);  mul_133 = None
        convert_element_type_153: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_135, torch.float32)
        pow_11: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 3.0)
        mul_135: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_41: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_135, mul_135);  view_135 = mul_135 = None
        mul_136: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_5: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_136);  mul_136 = None
        add_42: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_137: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_150, add_42);  convert_element_type_150 = add_42 = None
        convert_element_type_154: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None
        mul_138: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_30: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_138);  mul_138 = None
        mul_139: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_134, sub_30);  mul_134 = sub_30 = None
        mul_140: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_139, 0.7978845608028654);  mul_139 = None
        convert_element_type_155: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_140, torch.bfloat16)
        mul_141: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_140, 0.044715);  mul_140 = None
        pow_12: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_153, 2.0);  convert_element_type_153 = None
        mul_142: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_12, 3.0);  pow_12 = None
        mul_143: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        convert_element_type_156: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None
        add_43: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_155, convert_element_type_156);  convert_element_type_155 = convert_element_type_156 = None
        mul_144: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_154, 0.5);  convert_element_type_154 = None
        add_44: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_43, mul_144);  add_43 = mul_144 = None
        view_136: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_44, _shape_param_137);  add_44 = _shape_param_137 = None
        mm_64: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_136, arg131_1);  arg131_1 = None
        permute_57: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_136, [1, 0])
        mm_65: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_57, arg132_1);  permute_57 = arg132_1 = None
        sum_68: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_136, [0], True, dtype = torch.float32);  view_136 = None
        view_137: "f32[8192]" = torch.ops.aten.view.default(sum_68, _shape_param_138);  sum_68 = _shape_param_138 = None
        convert_element_type_157: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_137, torch.bfloat16);  view_137 = None
        view_138: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_64, _shape_param_139);  mm_64 = _shape_param_139 = None
        convert_element_type_158: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_138, torch.float32);  view_138 = None
        convert_element_type_159: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_160: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_157, torch.float32);  convert_element_type_157 = None
        mul_145: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_158, arg133_1);  arg133_1 = None
        mul_146: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_145, 2048)
        sum_69: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True)
        mul_147: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_145, arg134_1);  mul_145 = None
        sum_70: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
        mul_148: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg134_1, sum_70);  sum_70 = None
        sub_31: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_146, sum_69);  mul_146 = sum_69 = None
        sub_32: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_31, mul_148);  sub_31 = mul_148 = None
        mul_149: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg135_1, sub_32);  arg135_1 = sub_32 = None
        mul_150: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_158, arg134_1);  arg134_1 = None
        sum_71: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1]);  mul_150 = None
        sum_72: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_158, [0, 1]);  convert_element_type_158 = None
        add_45: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_40, mul_149);  add_40 = mul_149 = None
        convert_element_type_161: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16)
        view_139: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_161, _shape_param_140);  convert_element_type_161 = _shape_param_140 = None
        mm_66: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_139, arg136_1);  arg136_1 = None
        permute_58: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_139, [1, 0])
        mm_67: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_58, arg137_1);  permute_58 = arg137_1 = None
        sum_73: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_139, [0], True, dtype = torch.float32);  view_139 = None
        view_140: "f32[2048]" = torch.ops.aten.view.default(sum_73, _shape_param_141);  sum_73 = _shape_param_141 = None
        convert_element_type_162: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_140, torch.bfloat16);  view_140 = None
        view_141: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_66, _shape_param_142);  mm_66 = _shape_param_142 = None
        convert_element_type_163: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_164: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_162, torch.float32);  convert_element_type_162 = None
        view_142: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_141, _shape_param_143);  view_141 = _shape_param_143 = None
        permute_59: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
        clone_20: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_143: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_20, _shape_param_144);  clone_20 = _shape_param_144 = None
        bmm_20: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg138_1, view_143);  arg138_1 = None
        bmm_21: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_143, arg139_1);  view_143 = arg139_1 = None
        view_144: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_145);  bmm_20 = _shape_param_145 = None
        view_145: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_21, _shape_param_146);  bmm_21 = _shape_param_146 = None
        convert_element_type_165: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_145, torch.float32);  view_145 = None
        mul_151: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_165, arg140_1);  convert_element_type_165 = None
        sum_74: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [-1], True)
        neg_5: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg140_1);  arg140_1 = None
        fma_5: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_5, sum_74, mul_151);  neg_5 = sum_74 = mul_151 = None
        convert_element_type_166: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        slice_11: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg141_1, 2, 0, 128);  arg141_1 = None
        slice_12: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 128);  slice_11 = None
        where_7: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_12, convert_element_type_166, full_2);  slice_12 = convert_element_type_166 = None
        view_146: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_7, _shape_param_147);  where_7 = _shape_param_147 = None
        bmm_22: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg142_1, view_146);  arg142_1 = None
        bmm_23: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_146, arg143_1);  view_146 = arg143_1 = None
        view_147: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_148);  bmm_22 = _shape_param_148 = None
        view_148: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_23, _shape_param_149);  bmm_23 = _shape_param_149 = None
        convert_element_type_167: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_147, torch.float32);  view_147 = None
        permute_60: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_167, [0, 1, 3, 2]);  convert_element_type_167 = None
        convert_element_type_168: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_60, torch.bfloat16);  permute_60 = None
        convert_element_type_169: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_148, torch.bfloat16);  view_148 = None
        permute_61: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1, 3]);  view_144 = None
        clone_21: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
        view_149: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_21, _shape_param_150);  clone_21 = _shape_param_150 = None
        permute_62: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_168, [0, 2, 1, 3]);  convert_element_type_168 = None
        view_150: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_62, _shape_param_151);  permute_62 = _shape_param_151 = None
        permute_63: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_169, [0, 2, 1, 3]);  convert_element_type_169 = None
        clone_22: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        view_151: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_22, _shape_param_152);  clone_22 = _shape_param_152 = None
        view_152: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_149, _shape_param_153);  view_149 = _shape_param_153 = None
        permute_64: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_152, [1, 0])
        mm_68: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_64, arg144_1);  permute_64 = None
        mm_69: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_152, arg145_1);  view_152 = arg145_1 = None
        view_153: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_69, _shape_param_154);  mm_69 = _shape_param_154 = None
        convert_element_type_170: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_153, torch.float32);  view_153 = None
        convert_element_type_171: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        clone_23: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_150, memory_format = torch.contiguous_format);  view_150 = None
        view_154: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_23, _shape_param_155);  clone_23 = _shape_param_155 = None
        permute_65: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_154, [1, 0])
        mm_70: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_65, arg144_1);  permute_65 = None
        mm_71: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_154, arg146_1);  view_154 = arg146_1 = None
        view_155: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_71, _shape_param_156);  mm_71 = _shape_param_156 = None
        convert_element_type_172: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_155, torch.float32);  view_155 = None
        add_46: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_170, convert_element_type_172);  convert_element_type_170 = convert_element_type_172 = None
        convert_element_type_173: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        view_156: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_151, _shape_param_157);  view_151 = _shape_param_157 = None
        permute_66: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_72: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_66, arg144_1);  permute_66 = arg144_1 = None
        mm_73: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_156, arg147_1);  view_156 = arg147_1 = None
        view_157: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_73, _shape_param_158);  mm_73 = _shape_param_158 = None
        convert_element_type_174: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None
        add_47: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_46, convert_element_type_174);  add_46 = convert_element_type_174 = None
        convert_element_type_175: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        mul_152: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_47, arg148_1);  arg148_1 = None
        mul_153: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, 2048)
        sum_75: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True)
        mul_154: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, arg149_1);  mul_152 = None
        sum_76: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True);  mul_154 = None
        mul_155: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg149_1, sum_76);  sum_76 = None
        sub_33: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_153, sum_75);  mul_153 = sum_75 = None
        sub_34: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_33, mul_155);  sub_33 = mul_155 = None
        mul_156: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg150_1, sub_34);  arg150_1 = sub_34 = None
        mul_157: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_47, arg149_1);  arg149_1 = None
        sum_77: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_157, [0, 1]);  mul_157 = None
        sum_78: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_47, [0, 1]);  add_47 = None
        add_48: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_45, mul_156);  add_45 = mul_156 = None
        convert_element_type_176: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16)
        view_158: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_176, _shape_param_159);  convert_element_type_176 = _shape_param_159 = None
        mm_74: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_158, arg151_1);  arg151_1 = None
        permute_67: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_158, [1, 0])
        mm_75: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_67, arg152_1);  permute_67 = arg152_1 = None
        sum_79: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_158, [0], True, dtype = torch.float32);  view_158 = None
        view_159: "f32[2048]" = torch.ops.aten.view.default(sum_79, _shape_param_160);  sum_79 = _shape_param_160 = None
        convert_element_type_177: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_159, torch.bfloat16);  view_159 = None
        view_160: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_74, _shape_param_161);  mm_74 = _shape_param_161 = None
        convert_element_type_178: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_160, torch.float32);  view_160 = None
        convert_element_type_179: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_180: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_177, torch.float32);  convert_element_type_177 = None
        view_161: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg153_1, _shape_param_162);  arg153_1 = _shape_param_162 = None
        mul_158: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_161, 0.5)
        mul_159: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_178, mul_158);  mul_158 = None
        convert_element_type_181: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_161, torch.float32)
        pow_13: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_181, 3.0)
        mul_160: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_49: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_161, mul_160);  view_161 = mul_160 = None
        mul_161: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_6: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_161);  mul_161 = None
        add_50: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_162: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_178, add_50);  convert_element_type_178 = add_50 = None
        convert_element_type_182: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None
        mul_163: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_35: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_163);  mul_163 = None
        mul_164: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_159, sub_35);  mul_159 = sub_35 = None
        mul_165: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_164, 0.7978845608028654);  mul_164 = None
        convert_element_type_183: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16)
        mul_166: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_165, 0.044715);  mul_165 = None
        pow_14: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_181, 2.0);  convert_element_type_181 = None
        mul_167: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_168: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
        convert_element_type_184: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        add_51: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_183, convert_element_type_184);  convert_element_type_183 = convert_element_type_184 = None
        mul_169: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_182, 0.5);  convert_element_type_182 = None
        add_52: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_51, mul_169);  add_51 = mul_169 = None
        view_162: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_52, _shape_param_163);  add_52 = _shape_param_163 = None
        mm_76: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_162, arg154_1);  arg154_1 = None
        permute_68: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_162, [1, 0])
        mm_77: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_68, arg155_1);  permute_68 = arg155_1 = None
        sum_80: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_162, [0], True, dtype = torch.float32);  view_162 = None
        view_163: "f32[8192]" = torch.ops.aten.view.default(sum_80, _shape_param_164);  sum_80 = _shape_param_164 = None
        convert_element_type_185: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_163, torch.bfloat16);  view_163 = None
        view_164: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_76, _shape_param_165);  mm_76 = _shape_param_165 = None
        convert_element_type_186: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_164, torch.float32);  view_164 = None
        convert_element_type_187: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_188: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_185, torch.float32);  convert_element_type_185 = None
        mul_170: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_186, arg156_1);  arg156_1 = None
        mul_171: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, 2048)
        sum_81: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True)
        mul_172: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, arg157_1);  mul_170 = None
        sum_82: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
        mul_173: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg157_1, sum_82);  sum_82 = None
        sub_36: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_171, sum_81);  mul_171 = sum_81 = None
        sub_37: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_36, mul_173);  sub_36 = mul_173 = None
        mul_174: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg158_1, sub_37);  arg158_1 = sub_37 = None
        mul_175: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_186, arg157_1);  arg157_1 = None
        sum_83: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
        sum_84: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_186, [0, 1]);  convert_element_type_186 = None
        add_53: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_48, mul_174);  add_48 = mul_174 = None
        convert_element_type_189: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16)
        view_165: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_189, _shape_param_166);  convert_element_type_189 = _shape_param_166 = None
        mm_78: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_165, arg159_1);  arg159_1 = None
        permute_69: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_165, [1, 0])
        mm_79: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_69, arg160_1);  permute_69 = arg160_1 = None
        sum_85: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True, dtype = torch.float32);  view_165 = None
        view_166: "f32[2048]" = torch.ops.aten.view.default(sum_85, _shape_param_167);  sum_85 = _shape_param_167 = None
        convert_element_type_190: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_166, torch.bfloat16);  view_166 = None
        view_167: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_78, _shape_param_168);  mm_78 = _shape_param_168 = None
        convert_element_type_191: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_192: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_190, torch.float32);  convert_element_type_190 = None
        view_168: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_167, _shape_param_169);  view_167 = _shape_param_169 = None
        permute_70: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
        clone_24: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_169: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_24, _shape_param_170);  clone_24 = _shape_param_170 = None
        bmm_24: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg161_1, view_169);  arg161_1 = None
        bmm_25: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_169, arg162_1);  view_169 = arg162_1 = None
        view_170: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_171);  bmm_24 = _shape_param_171 = None
        view_171: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_25, _shape_param_172);  bmm_25 = _shape_param_172 = None
        convert_element_type_193: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_171, torch.float32);  view_171 = None
        mul_176: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_193, arg163_1);  convert_element_type_193 = None
        sum_86: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [-1], True)
        neg_6: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg163_1);  arg163_1 = None
        fma_6: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_6, sum_86, mul_176);  neg_6 = sum_86 = mul_176 = None
        convert_element_type_194: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        slice_13: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg164_1, 2, 0, 128);  arg164_1 = None
        slice_14: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_13, 3, 0, 128);  slice_13 = None
        where_8: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_14, convert_element_type_194, full_2);  slice_14 = convert_element_type_194 = None
        view_172: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_8, _shape_param_173);  where_8 = _shape_param_173 = None
        bmm_26: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg165_1, view_172);  arg165_1 = None
        bmm_27: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_172, arg166_1);  view_172 = arg166_1 = None
        view_173: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_174);  bmm_26 = _shape_param_174 = None
        view_174: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_27, _shape_param_175);  bmm_27 = _shape_param_175 = None
        convert_element_type_195: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        permute_71: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_195, [0, 1, 3, 2]);  convert_element_type_195 = None
        convert_element_type_196: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_71, torch.bfloat16);  permute_71 = None
        convert_element_type_197: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_174, torch.bfloat16);  view_174 = None
        permute_72: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        clone_25: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_175: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_25, _shape_param_176);  clone_25 = _shape_param_176 = None
        permute_73: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_196, [0, 2, 1, 3]);  convert_element_type_196 = None
        view_176: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_73, _shape_param_177);  permute_73 = _shape_param_177 = None
        permute_74: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_197, [0, 2, 1, 3]);  convert_element_type_197 = None
        clone_26: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
        view_177: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_26, _shape_param_178);  clone_26 = _shape_param_178 = None
        view_178: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_175, _shape_param_179);  view_175 = _shape_param_179 = None
        permute_75: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_178, [1, 0])
        mm_80: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_75, arg167_1);  permute_75 = None
        mm_81: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_178, arg168_1);  view_178 = arg168_1 = None
        view_179: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_81, _shape_param_180);  mm_81 = _shape_param_180 = None
        convert_element_type_198: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_179, torch.float32);  view_179 = None
        convert_element_type_199: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        clone_27: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_176, memory_format = torch.contiguous_format);  view_176 = None
        view_180: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_27, _shape_param_181);  clone_27 = _shape_param_181 = None
        permute_76: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_180, [1, 0])
        mm_82: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_76, arg167_1);  permute_76 = None
        mm_83: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_180, arg169_1);  view_180 = arg169_1 = None
        view_181: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_83, _shape_param_182);  mm_83 = _shape_param_182 = None
        convert_element_type_200: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_181, torch.float32);  view_181 = None
        add_54: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_198, convert_element_type_200);  convert_element_type_198 = convert_element_type_200 = None
        convert_element_type_201: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        view_182: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_177, _shape_param_183);  view_177 = _shape_param_183 = None
        permute_77: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_182, [1, 0])
        mm_84: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_77, arg167_1);  permute_77 = arg167_1 = None
        mm_85: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_182, arg170_1);  view_182 = arg170_1 = None
        view_183: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_85, _shape_param_184);  mm_85 = _shape_param_184 = None
        convert_element_type_202: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_183, torch.float32);  view_183 = None
        add_55: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_54, convert_element_type_202);  add_54 = convert_element_type_202 = None
        convert_element_type_203: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        mul_177: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_55, arg171_1);  arg171_1 = None
        mul_178: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_177, 2048)
        sum_87: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True)
        mul_179: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_177, arg172_1);  mul_177 = None
        sum_88: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True);  mul_179 = None
        mul_180: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg172_1, sum_88);  sum_88 = None
        sub_38: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_178, sum_87);  mul_178 = sum_87 = None
        sub_39: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_38, mul_180);  sub_38 = mul_180 = None
        mul_181: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg173_1, sub_39);  arg173_1 = sub_39 = None
        mul_182: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_55, arg172_1);  arg172_1 = None
        sum_89: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 1]);  mul_182 = None
        sum_90: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_55, [0, 1]);  add_55 = None
        add_56: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_53, mul_181);  add_53 = mul_181 = None
        convert_element_type_204: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16)
        view_184: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_204, _shape_param_185);  convert_element_type_204 = _shape_param_185 = None
        mm_86: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_184, arg174_1);  arg174_1 = None
        permute_78: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_184, [1, 0])
        mm_87: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_78, arg175_1);  permute_78 = arg175_1 = None
        sum_91: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_184, [0], True, dtype = torch.float32);  view_184 = None
        view_185: "f32[2048]" = torch.ops.aten.view.default(sum_91, _shape_param_186);  sum_91 = _shape_param_186 = None
        convert_element_type_205: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_185, torch.bfloat16);  view_185 = None
        view_186: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_86, _shape_param_187);  mm_86 = _shape_param_187 = None
        convert_element_type_206: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_186, torch.float32);  view_186 = None
        convert_element_type_207: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_208: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_205, torch.float32);  convert_element_type_205 = None
        view_187: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg176_1, _shape_param_188);  arg176_1 = _shape_param_188 = None
        mul_183: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_187, 0.5)
        mul_184: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_206, mul_183);  mul_183 = None
        convert_element_type_209: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_187, torch.float32)
        pow_15: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_209, 3.0)
        mul_185: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_57: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_187, mul_185);  view_187 = mul_185 = None
        mul_186: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_7: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_186);  mul_186 = None
        add_58: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_187: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_206, add_58);  convert_element_type_206 = add_58 = None
        convert_element_type_210: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_187, torch.bfloat16);  mul_187 = None
        mul_188: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_40: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_188);  mul_188 = None
        mul_189: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_184, sub_40);  mul_184 = sub_40 = None
        mul_190: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_189, 0.7978845608028654);  mul_189 = None
        convert_element_type_211: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16)
        mul_191: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_190, 0.044715);  mul_190 = None
        pow_16: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_209, 2.0);  convert_element_type_209 = None
        mul_192: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_193: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None
        convert_element_type_212: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16);  mul_193 = None
        add_59: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_211, convert_element_type_212);  convert_element_type_211 = convert_element_type_212 = None
        mul_194: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_210, 0.5);  convert_element_type_210 = None
        add_60: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_59, mul_194);  add_59 = mul_194 = None
        view_188: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_60, _shape_param_189);  add_60 = _shape_param_189 = None
        mm_88: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_188, arg177_1);  arg177_1 = None
        permute_79: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_188, [1, 0])
        mm_89: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_79, arg178_1);  permute_79 = arg178_1 = None
        sum_92: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True, dtype = torch.float32);  view_188 = None
        view_189: "f32[8192]" = torch.ops.aten.view.default(sum_92, _shape_param_190);  sum_92 = _shape_param_190 = None
        convert_element_type_213: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_189, torch.bfloat16);  view_189 = None
        view_190: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_88, _shape_param_191);  mm_88 = _shape_param_191 = None
        convert_element_type_214: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_190, torch.float32);  view_190 = None
        convert_element_type_215: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_216: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_213, torch.float32);  convert_element_type_213 = None
        mul_195: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_214, arg179_1);  arg179_1 = None
        mul_196: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_195, 2048)
        sum_93: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
        mul_197: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_195, arg180_1);  mul_195 = None
        sum_94: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
        mul_198: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg180_1, sum_94);  sum_94 = None
        sub_41: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_196, sum_93);  mul_196 = sum_93 = None
        sub_42: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_41, mul_198);  sub_41 = mul_198 = None
        mul_199: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg181_1, sub_42);  arg181_1 = sub_42 = None
        mul_200: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_214, arg180_1);  arg180_1 = None
        sum_95: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
        sum_96: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_214, [0, 1]);  convert_element_type_214 = None
        add_61: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_56, mul_199);  add_56 = mul_199 = None
        convert_element_type_217: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16)
        view_191: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_217, _shape_param_192);  convert_element_type_217 = _shape_param_192 = None
        mm_90: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_191, arg182_1);  arg182_1 = None
        permute_80: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_191, [1, 0])
        mm_91: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_80, arg183_1);  permute_80 = arg183_1 = None
        sum_97: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_191, [0], True, dtype = torch.float32);  view_191 = None
        view_192: "f32[2048]" = torch.ops.aten.view.default(sum_97, _shape_param_193);  sum_97 = _shape_param_193 = None
        convert_element_type_218: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_192, torch.bfloat16);  view_192 = None
        view_193: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_90, _shape_param_194);  mm_90 = _shape_param_194 = None
        convert_element_type_219: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_220: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_218, torch.float32);  convert_element_type_218 = None
        view_194: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_193, _shape_param_195);  view_193 = _shape_param_195 = None
        permute_81: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        clone_28: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_195: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_28, _shape_param_196);  clone_28 = _shape_param_196 = None
        bmm_28: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg184_1, view_195);  arg184_1 = None
        bmm_29: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_195, arg185_1);  view_195 = arg185_1 = None
        view_196: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_197);  bmm_28 = _shape_param_197 = None
        view_197: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_29, _shape_param_198);  bmm_29 = _shape_param_198 = None
        convert_element_type_221: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        mul_201: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_221, arg186_1);  convert_element_type_221 = None
        sum_98: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [-1], True)
        neg_7: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg186_1);  arg186_1 = None
        fma_7: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_7, sum_98, mul_201);  neg_7 = sum_98 = mul_201 = None
        convert_element_type_222: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        slice_15: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg187_1, 2, 0, 128);  arg187_1 = None
        slice_16: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 128);  slice_15 = None
        where_9: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_16, convert_element_type_222, full_2);  slice_16 = convert_element_type_222 = None
        view_198: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_9, _shape_param_199);  where_9 = _shape_param_199 = None
        bmm_30: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg188_1, view_198);  arg188_1 = None
        bmm_31: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_198, arg189_1);  view_198 = arg189_1 = None
        view_199: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_200);  bmm_30 = _shape_param_200 = None
        view_200: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_31, _shape_param_201);  bmm_31 = _shape_param_201 = None
        convert_element_type_223: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        permute_82: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_223, [0, 1, 3, 2]);  convert_element_type_223 = None
        convert_element_type_224: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_82, torch.bfloat16);  permute_82 = None
        convert_element_type_225: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_200, torch.bfloat16);  view_200 = None
        permute_83: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
        clone_29: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_201: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_29, _shape_param_202);  clone_29 = _shape_param_202 = None
        permute_84: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_224, [0, 2, 1, 3]);  convert_element_type_224 = None
        view_202: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_84, _shape_param_203);  permute_84 = _shape_param_203 = None
        permute_85: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_225, [0, 2, 1, 3]);  convert_element_type_225 = None
        clone_30: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        view_203: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_30, _shape_param_204);  clone_30 = _shape_param_204 = None
        view_204: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_201, _shape_param_205);  view_201 = _shape_param_205 = None
        permute_86: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_204, [1, 0])
        mm_92: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_86, arg190_1);  permute_86 = None
        mm_93: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_204, arg191_1);  view_204 = arg191_1 = None
        view_205: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_93, _shape_param_206);  mm_93 = _shape_param_206 = None
        convert_element_type_226: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        convert_element_type_227: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        clone_31: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_202, memory_format = torch.contiguous_format);  view_202 = None
        view_206: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_31, _shape_param_207);  clone_31 = _shape_param_207 = None
        permute_87: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_206, [1, 0])
        mm_94: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_87, arg190_1);  permute_87 = None
        mm_95: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_206, arg192_1);  view_206 = arg192_1 = None
        view_207: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_95, _shape_param_208);  mm_95 = _shape_param_208 = None
        convert_element_type_228: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_207, torch.float32);  view_207 = None
        add_62: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_226, convert_element_type_228);  convert_element_type_226 = convert_element_type_228 = None
        convert_element_type_229: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        view_208: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_203, _shape_param_209);  view_203 = _shape_param_209 = None
        permute_88: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_208, [1, 0])
        mm_96: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_88, arg190_1);  permute_88 = arg190_1 = None
        mm_97: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_208, arg193_1);  view_208 = arg193_1 = None
        view_209: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_97, _shape_param_210);  mm_97 = _shape_param_210 = None
        convert_element_type_230: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        add_63: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_62, convert_element_type_230);  add_62 = convert_element_type_230 = None
        convert_element_type_231: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        mul_202: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_63, arg194_1);  arg194_1 = None
        mul_203: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_202, 2048)
        sum_99: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True)
        mul_204: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_202, arg195_1);  mul_202 = None
        sum_100: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
        mul_205: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg195_1, sum_100);  sum_100 = None
        sub_43: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_203, sum_99);  mul_203 = sum_99 = None
        sub_44: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_43, mul_205);  sub_43 = mul_205 = None
        mul_206: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg196_1, sub_44);  arg196_1 = sub_44 = None
        mul_207: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_63, arg195_1);  arg195_1 = None
        sum_101: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
        sum_102: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_63, [0, 1]);  add_63 = None
        add_64: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_61, mul_206);  add_61 = mul_206 = None
        convert_element_type_232: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16)
        view_210: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_232, _shape_param_211);  convert_element_type_232 = _shape_param_211 = None
        mm_98: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_210, arg197_1);  arg197_1 = None
        permute_89: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_99: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_89, arg198_1);  permute_89 = arg198_1 = None
        sum_103: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True, dtype = torch.float32);  view_210 = None
        view_211: "f32[2048]" = torch.ops.aten.view.default(sum_103, _shape_param_212);  sum_103 = _shape_param_212 = None
        convert_element_type_233: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_211, torch.bfloat16);  view_211 = None
        view_212: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_98, _shape_param_213);  mm_98 = _shape_param_213 = None
        convert_element_type_234: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        convert_element_type_235: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_236: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_233, torch.float32);  convert_element_type_233 = None
        view_213: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg199_1, _shape_param_214);  arg199_1 = _shape_param_214 = None
        mul_208: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_213, 0.5)
        mul_209: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_234, mul_208);  mul_208 = None
        convert_element_type_237: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_213, torch.float32)
        pow_17: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_237, 3.0)
        mul_210: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_65: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_213, mul_210);  view_213 = mul_210 = None
        mul_211: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_8: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_211);  mul_211 = None
        add_66: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_212: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_234, add_66);  convert_element_type_234 = add_66 = None
        convert_element_type_238: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_212, torch.bfloat16);  mul_212 = None
        mul_213: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_45: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_213);  mul_213 = None
        mul_214: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_209, sub_45);  mul_209 = sub_45 = None
        mul_215: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_214, 0.7978845608028654);  mul_214 = None
        convert_element_type_239: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_215, torch.bfloat16)
        mul_216: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_215, 0.044715);  mul_215 = None
        pow_18: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_237, 2.0);  convert_element_type_237 = None
        mul_217: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_218: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
        convert_element_type_240: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_218, torch.bfloat16);  mul_218 = None
        add_67: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_239, convert_element_type_240);  convert_element_type_239 = convert_element_type_240 = None
        mul_219: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_238, 0.5);  convert_element_type_238 = None
        add_68: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_67, mul_219);  add_67 = mul_219 = None
        view_214: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_68, _shape_param_215);  add_68 = _shape_param_215 = None
        mm_100: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_214, arg200_1);  arg200_1 = None
        permute_90: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_214, [1, 0])
        mm_101: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_90, arg201_1);  permute_90 = arg201_1 = None
        sum_104: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True, dtype = torch.float32);  view_214 = None
        view_215: "f32[8192]" = torch.ops.aten.view.default(sum_104, _shape_param_216);  sum_104 = _shape_param_216 = None
        convert_element_type_241: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_215, torch.bfloat16);  view_215 = None
        view_216: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_100, _shape_param_217);  mm_100 = _shape_param_217 = None
        convert_element_type_242: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_216, torch.float32);  view_216 = None
        convert_element_type_243: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_244: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_241, torch.float32);  convert_element_type_241 = None
        mul_220: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_242, arg202_1);  arg202_1 = None
        mul_221: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_220, 2048)
        sum_105: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [2], True)
        mul_222: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_220, arg203_1);  mul_220 = None
        sum_106: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
        mul_223: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg203_1, sum_106);  sum_106 = None
        sub_46: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_221, sum_105);  mul_221 = sum_105 = None
        sub_47: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_46, mul_223);  sub_46 = mul_223 = None
        mul_224: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg204_1, sub_47);  arg204_1 = sub_47 = None
        mul_225: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_242, arg203_1);  arg203_1 = None
        sum_107: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1]);  mul_225 = None
        sum_108: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_242, [0, 1]);  convert_element_type_242 = None
        add_69: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_64, mul_224);  add_64 = mul_224 = None
        convert_element_type_245: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16)
        view_217: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_245, _shape_param_218);  convert_element_type_245 = _shape_param_218 = None
        mm_102: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_217, arg205_1);  arg205_1 = None
        permute_91: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_217, [1, 0])
        mm_103: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_91, arg206_1);  permute_91 = arg206_1 = None
        sum_109: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_217, [0], True, dtype = torch.float32);  view_217 = None
        view_218: "f32[2048]" = torch.ops.aten.view.default(sum_109, _shape_param_219);  sum_109 = _shape_param_219 = None
        convert_element_type_246: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_218, torch.bfloat16);  view_218 = None
        view_219: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_102, _shape_param_220);  mm_102 = _shape_param_220 = None
        convert_element_type_247: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_248: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_246, torch.float32);  convert_element_type_246 = None
        view_220: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_219, _shape_param_221);  view_219 = _shape_param_221 = None
        permute_92: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
        clone_32: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_221: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_32, _shape_param_222);  clone_32 = _shape_param_222 = None
        bmm_32: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg207_1, view_221);  arg207_1 = None
        bmm_33: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_221, arg208_1);  view_221 = arg208_1 = None
        view_222: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_223);  bmm_32 = _shape_param_223 = None
        view_223: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_33, _shape_param_224);  bmm_33 = _shape_param_224 = None
        convert_element_type_249: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_223, torch.float32);  view_223 = None
        mul_226: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_249, arg209_1);  convert_element_type_249 = None
        sum_110: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [-1], True)
        neg_8: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg209_1);  arg209_1 = None
        fma_8: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_8, sum_110, mul_226);  neg_8 = sum_110 = mul_226 = None
        convert_element_type_250: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        slice_17: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg210_1, 2, 0, 128);  arg210_1 = None
        slice_18: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_17, 3, 0, 128);  slice_17 = None
        where_10: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_18, convert_element_type_250, full_2);  slice_18 = convert_element_type_250 = None
        view_224: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_10, _shape_param_225);  where_10 = _shape_param_225 = None
        bmm_34: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg211_1, view_224);  arg211_1 = None
        bmm_35: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_224, arg212_1);  view_224 = arg212_1 = None
        view_225: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_226);  bmm_34 = _shape_param_226 = None
        view_226: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_35, _shape_param_227);  bmm_35 = _shape_param_227 = None
        convert_element_type_251: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_225, torch.float32);  view_225 = None
        permute_93: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_251, [0, 1, 3, 2]);  convert_element_type_251 = None
        convert_element_type_252: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_93, torch.bfloat16);  permute_93 = None
        convert_element_type_253: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_226, torch.bfloat16);  view_226 = None
        permute_94: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
        clone_33: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_227: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_33, _shape_param_228);  clone_33 = _shape_param_228 = None
        permute_95: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_252, [0, 2, 1, 3]);  convert_element_type_252 = None
        view_228: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_95, _shape_param_229);  permute_95 = _shape_param_229 = None
        permute_96: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_253, [0, 2, 1, 3]);  convert_element_type_253 = None
        clone_34: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
        view_229: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_34, _shape_param_230);  clone_34 = _shape_param_230 = None
        view_230: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_227, _shape_param_231);  view_227 = _shape_param_231 = None
        permute_97: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_230, [1, 0])
        mm_104: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_97, arg213_1);  permute_97 = None
        mm_105: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_230, arg214_1);  view_230 = arg214_1 = None
        view_231: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_105, _shape_param_232);  mm_105 = _shape_param_232 = None
        convert_element_type_254: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_231, torch.float32);  view_231 = None
        convert_element_type_255: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        clone_35: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_228, memory_format = torch.contiguous_format);  view_228 = None
        view_232: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_35, _shape_param_233);  clone_35 = _shape_param_233 = None
        permute_98: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_232, [1, 0])
        mm_106: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_98, arg213_1);  permute_98 = None
        mm_107: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_232, arg215_1);  view_232 = arg215_1 = None
        view_233: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_107, _shape_param_234);  mm_107 = _shape_param_234 = None
        convert_element_type_256: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_233, torch.float32);  view_233 = None
        add_70: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_254, convert_element_type_256);  convert_element_type_254 = convert_element_type_256 = None
        convert_element_type_257: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        view_234: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_229, _shape_param_235);  view_229 = _shape_param_235 = None
        permute_99: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_234, [1, 0])
        mm_108: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_99, arg213_1);  permute_99 = arg213_1 = None
        mm_109: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_234, arg216_1);  view_234 = arg216_1 = None
        view_235: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_109, _shape_param_236);  mm_109 = _shape_param_236 = None
        convert_element_type_258: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_235, torch.float32);  view_235 = None
        add_71: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_70, convert_element_type_258);  add_70 = convert_element_type_258 = None
        convert_element_type_259: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        mul_227: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_71, arg217_1);  arg217_1 = None
        mul_228: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_227, 2048)
        sum_111: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
        mul_229: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_227, arg218_1);  mul_227 = None
        sum_112: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
        mul_230: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg218_1, sum_112);  sum_112 = None
        sub_48: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_228, sum_111);  mul_228 = sum_111 = None
        sub_49: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_48, mul_230);  sub_48 = mul_230 = None
        mul_231: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg219_1, sub_49);  arg219_1 = sub_49 = None
        mul_232: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_71, arg218_1);  arg218_1 = None
        sum_113: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
        sum_114: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_71, [0, 1]);  add_71 = None
        add_72: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_69, mul_231);  add_69 = mul_231 = None
        convert_element_type_260: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16)
        view_236: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_260, _shape_param_237);  convert_element_type_260 = _shape_param_237 = None
        mm_110: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_236, arg220_1);  arg220_1 = None
        permute_100: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_236, [1, 0])
        mm_111: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_100, arg221_1);  permute_100 = arg221_1 = None
        sum_115: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_236, [0], True, dtype = torch.float32);  view_236 = None
        view_237: "f32[2048]" = torch.ops.aten.view.default(sum_115, _shape_param_238);  sum_115 = _shape_param_238 = None
        convert_element_type_261: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_237, torch.bfloat16);  view_237 = None
        view_238: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_110, _shape_param_239);  mm_110 = _shape_param_239 = None
        convert_element_type_262: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_238, torch.float32);  view_238 = None
        convert_element_type_263: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_264: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_261, torch.float32);  convert_element_type_261 = None
        view_239: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg222_1, _shape_param_240);  arg222_1 = _shape_param_240 = None
        mul_233: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
        mul_234: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_262, mul_233);  mul_233 = None
        convert_element_type_265: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32)
        pow_19: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_265, 3.0)
        mul_235: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_73: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_239, mul_235);  view_239 = mul_235 = None
        mul_236: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_9: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_236);  mul_236 = None
        add_74: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_237: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_262, add_74);  convert_element_type_262 = add_74 = None
        convert_element_type_266: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_237, torch.bfloat16);  mul_237 = None
        mul_238: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_50: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_238);  mul_238 = None
        mul_239: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_234, sub_50);  mul_234 = sub_50 = None
        mul_240: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_239, 0.7978845608028654);  mul_239 = None
        convert_element_type_267: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_240, torch.bfloat16)
        mul_241: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_240, 0.044715);  mul_240 = None
        pow_20: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_265, 2.0);  convert_element_type_265 = None
        mul_242: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_243: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
        convert_element_type_268: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_243, torch.bfloat16);  mul_243 = None
        add_75: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_267, convert_element_type_268);  convert_element_type_267 = convert_element_type_268 = None
        mul_244: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_266, 0.5);  convert_element_type_266 = None
        add_76: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_75, mul_244);  add_75 = mul_244 = None
        view_240: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_76, _shape_param_241);  add_76 = _shape_param_241 = None
        mm_112: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_240, arg223_1);  arg223_1 = None
        permute_101: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_240, [1, 0])
        mm_113: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_101, arg224_1);  permute_101 = arg224_1 = None
        sum_116: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True, dtype = torch.float32);  view_240 = None
        view_241: "f32[8192]" = torch.ops.aten.view.default(sum_116, _shape_param_242);  sum_116 = _shape_param_242 = None
        convert_element_type_269: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_241, torch.bfloat16);  view_241 = None
        view_242: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_112, _shape_param_243);  mm_112 = _shape_param_243 = None
        convert_element_type_270: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        convert_element_type_271: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_272: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_269, torch.float32);  convert_element_type_269 = None
        mul_245: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_270, arg225_1);  arg225_1 = None
        mul_246: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_245, 2048)
        sum_117: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True)
        mul_247: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_245, arg226_1);  mul_245 = None
        sum_118: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True);  mul_247 = None
        mul_248: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg226_1, sum_118);  sum_118 = None
        sub_51: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_246, sum_117);  mul_246 = sum_117 = None
        sub_52: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_51, mul_248);  sub_51 = mul_248 = None
        mul_249: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg227_1, sub_52);  arg227_1 = sub_52 = None
        mul_250: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_270, arg226_1);  arg226_1 = None
        sum_119: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1]);  mul_250 = None
        sum_120: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_270, [0, 1]);  convert_element_type_270 = None
        add_77: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_72, mul_249);  add_72 = mul_249 = None
        convert_element_type_273: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16)
        view_243: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_273, _shape_param_244);  convert_element_type_273 = _shape_param_244 = None
        mm_114: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_243, arg228_1);  arg228_1 = None
        permute_102: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_243, [1, 0])
        mm_115: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_102, arg229_1);  permute_102 = arg229_1 = None
        sum_121: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True, dtype = torch.float32);  view_243 = None
        view_244: "f32[2048]" = torch.ops.aten.view.default(sum_121, _shape_param_245);  sum_121 = _shape_param_245 = None
        convert_element_type_274: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_244, torch.bfloat16);  view_244 = None
        view_245: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_114, _shape_param_246);  mm_114 = _shape_param_246 = None
        convert_element_type_275: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_276: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_274, torch.float32);  convert_element_type_274 = None
        view_246: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_245, _shape_param_247);  view_245 = _shape_param_247 = None
        permute_103: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        clone_36: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        view_247: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_36, _shape_param_248);  clone_36 = _shape_param_248 = None
        bmm_36: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg230_1, view_247);  arg230_1 = None
        bmm_37: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_247, arg231_1);  view_247 = arg231_1 = None
        view_248: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_249);  bmm_36 = _shape_param_249 = None
        view_249: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_37, _shape_param_250);  bmm_37 = _shape_param_250 = None
        convert_element_type_277: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_249, torch.float32);  view_249 = None
        mul_251: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_277, arg232_1);  convert_element_type_277 = None
        sum_122: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
        neg_9: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg232_1);  arg232_1 = None
        fma_9: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_9, sum_122, mul_251);  neg_9 = sum_122 = mul_251 = None
        convert_element_type_278: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        slice_19: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg233_1, 2, 0, 128);  arg233_1 = None
        slice_20: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 128);  slice_19 = None
        where_11: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_20, convert_element_type_278, full_2);  slice_20 = convert_element_type_278 = None
        view_250: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_11, _shape_param_251);  where_11 = _shape_param_251 = None
        bmm_38: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg234_1, view_250);  arg234_1 = None
        bmm_39: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_250, arg235_1);  view_250 = arg235_1 = None
        view_251: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_252);  bmm_38 = _shape_param_252 = None
        view_252: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_39, _shape_param_253);  bmm_39 = _shape_param_253 = None
        convert_element_type_279: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_251, torch.float32);  view_251 = None
        permute_104: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_279, [0, 1, 3, 2]);  convert_element_type_279 = None
        convert_element_type_280: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_104, torch.bfloat16);  permute_104 = None
        convert_element_type_281: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_252, torch.bfloat16);  view_252 = None
        permute_105: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_248, [0, 2, 1, 3]);  view_248 = None
        clone_37: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_253: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_37, _shape_param_254);  clone_37 = _shape_param_254 = None
        permute_106: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_280, [0, 2, 1, 3]);  convert_element_type_280 = None
        view_254: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_106, _shape_param_255);  permute_106 = _shape_param_255 = None
        permute_107: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_281, [0, 2, 1, 3]);  convert_element_type_281 = None
        clone_38: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
        view_255: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_38, _shape_param_256);  clone_38 = _shape_param_256 = None
        view_256: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_253, _shape_param_257);  view_253 = _shape_param_257 = None
        permute_108: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_256, [1, 0])
        mm_116: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_108, arg236_1);  permute_108 = None
        mm_117: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_256, arg237_1);  view_256 = arg237_1 = None
        view_257: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_117, _shape_param_258);  mm_117 = _shape_param_258 = None
        convert_element_type_282: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_257, torch.float32);  view_257 = None
        convert_element_type_283: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        clone_39: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_254, memory_format = torch.contiguous_format);  view_254 = None
        view_258: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_39, _shape_param_259);  clone_39 = _shape_param_259 = None
        permute_109: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_258, [1, 0])
        mm_118: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_109, arg236_1);  permute_109 = None
        mm_119: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_258, arg238_1);  view_258 = arg238_1 = None
        view_259: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_119, _shape_param_260);  mm_119 = _shape_param_260 = None
        convert_element_type_284: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_259, torch.float32);  view_259 = None
        add_78: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_282, convert_element_type_284);  convert_element_type_282 = convert_element_type_284 = None
        convert_element_type_285: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        view_260: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_255, _shape_param_261);  view_255 = _shape_param_261 = None
        permute_110: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_260, [1, 0])
        mm_120: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_110, arg236_1);  permute_110 = arg236_1 = None
        mm_121: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_260, arg239_1);  view_260 = arg239_1 = None
        view_261: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_121, _shape_param_262);  mm_121 = _shape_param_262 = None
        convert_element_type_286: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        add_79: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_78, convert_element_type_286);  add_78 = convert_element_type_286 = None
        convert_element_type_287: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        mul_252: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_79, arg240_1);  arg240_1 = None
        mul_253: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_252, 2048)
        sum_123: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
        mul_254: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_252, arg241_1);  mul_252 = None
        sum_124: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
        mul_255: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg241_1, sum_124);  sum_124 = None
        sub_53: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_253, sum_123);  mul_253 = sum_123 = None
        sub_54: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_53, mul_255);  sub_53 = mul_255 = None
        mul_256: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg242_1, sub_54);  arg242_1 = sub_54 = None
        mul_257: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_79, arg241_1);  arg241_1 = None
        sum_125: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
        sum_126: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_79, [0, 1]);  add_79 = None
        add_80: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_77, mul_256);  add_77 = mul_256 = None
        convert_element_type_288: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16)
        view_262: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_288, _shape_param_263);  convert_element_type_288 = _shape_param_263 = None
        mm_122: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_262, arg243_1);  arg243_1 = None
        permute_111: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_262, [1, 0])
        mm_123: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_111, arg244_1);  permute_111 = arg244_1 = None
        sum_127: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_262, [0], True, dtype = torch.float32);  view_262 = None
        view_263: "f32[2048]" = torch.ops.aten.view.default(sum_127, _shape_param_264);  sum_127 = _shape_param_264 = None
        convert_element_type_289: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_263, torch.bfloat16);  view_263 = None
        view_264: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_122, _shape_param_265);  mm_122 = _shape_param_265 = None
        convert_element_type_290: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_264, torch.float32);  view_264 = None
        convert_element_type_291: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_292: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_289, torch.float32);  convert_element_type_289 = None
        view_265: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg245_1, _shape_param_266);  arg245_1 = _shape_param_266 = None
        mul_258: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_265, 0.5)
        mul_259: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_290, mul_258);  mul_258 = None
        convert_element_type_293: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_265, torch.float32)
        pow_21: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_293, 3.0)
        mul_260: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_81: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_265, mul_260);  view_265 = mul_260 = None
        mul_261: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_10: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_261);  mul_261 = None
        add_82: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_262: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_290, add_82);  convert_element_type_290 = add_82 = None
        convert_element_type_294: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_262, torch.bfloat16);  mul_262 = None
        mul_263: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_55: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_263);  mul_263 = None
        mul_264: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_259, sub_55);  mul_259 = sub_55 = None
        mul_265: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_264, 0.7978845608028654);  mul_264 = None
        convert_element_type_295: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_265, torch.bfloat16)
        mul_266: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_265, 0.044715);  mul_265 = None
        pow_22: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_293, 2.0);  convert_element_type_293 = None
        mul_267: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_268: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
        convert_element_type_296: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_268, torch.bfloat16);  mul_268 = None
        add_83: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_295, convert_element_type_296);  convert_element_type_295 = convert_element_type_296 = None
        mul_269: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_294, 0.5);  convert_element_type_294 = None
        add_84: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_83, mul_269);  add_83 = mul_269 = None
        view_266: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_84, _shape_param_267);  add_84 = _shape_param_267 = None
        mm_124: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_266, arg246_1);  arg246_1 = None
        permute_112: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_266, [1, 0])
        mm_125: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_112, arg247_1);  permute_112 = arg247_1 = None
        sum_128: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True, dtype = torch.float32);  view_266 = None
        view_267: "f32[8192]" = torch.ops.aten.view.default(sum_128, _shape_param_268);  sum_128 = _shape_param_268 = None
        convert_element_type_297: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_267, torch.bfloat16);  view_267 = None
        view_268: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_124, _shape_param_269);  mm_124 = _shape_param_269 = None
        convert_element_type_298: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_268, torch.float32);  view_268 = None
        convert_element_type_299: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_300: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_297, torch.float32);  convert_element_type_297 = None
        mul_270: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_298, arg248_1);  arg248_1 = None
        mul_271: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_270, 2048)
        sum_129: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
        mul_272: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_270, arg249_1);  mul_270 = None
        sum_130: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
        mul_273: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg249_1, sum_130);  sum_130 = None
        sub_56: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_271, sum_129);  mul_271 = sum_129 = None
        sub_57: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_56, mul_273);  sub_56 = mul_273 = None
        mul_274: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg250_1, sub_57);  arg250_1 = sub_57 = None
        mul_275: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_298, arg249_1);  arg249_1 = None
        sum_131: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
        sum_132: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_298, [0, 1]);  convert_element_type_298 = None
        add_85: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_80, mul_274);  add_80 = mul_274 = None
        convert_element_type_301: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16)
        view_269: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_301, _shape_param_270);  convert_element_type_301 = _shape_param_270 = None
        mm_126: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_269, arg251_1);  arg251_1 = None
        permute_113: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_269, [1, 0])
        mm_127: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_113, arg252_1);  permute_113 = arg252_1 = None
        sum_133: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True, dtype = torch.float32);  view_269 = None
        view_270: "f32[2048]" = torch.ops.aten.view.default(sum_133, _shape_param_271);  sum_133 = _shape_param_271 = None
        convert_element_type_302: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_270, torch.bfloat16);  view_270 = None
        view_271: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_126, _shape_param_272);  mm_126 = _shape_param_272 = None
        convert_element_type_303: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_304: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_302, torch.float32);  convert_element_type_302 = None
        view_272: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_271, _shape_param_273);  view_271 = _shape_param_273 = None
        permute_114: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
        clone_40: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_273: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_40, _shape_param_274);  clone_40 = _shape_param_274 = None
        bmm_40: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg253_1, view_273);  arg253_1 = None
        bmm_41: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_273, arg254_1);  view_273 = arg254_1 = None
        view_274: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_275);  bmm_40 = _shape_param_275 = None
        view_275: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_41, _shape_param_276);  bmm_41 = _shape_param_276 = None
        convert_element_type_305: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_275, torch.float32);  view_275 = None
        mul_276: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_305, arg255_1);  convert_element_type_305 = None
        sum_134: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
        neg_10: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg255_1);  arg255_1 = None
        fma_10: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_10, sum_134, mul_276);  neg_10 = sum_134 = mul_276 = None
        convert_element_type_306: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        slice_21: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg256_1, 2, 0, 128);  arg256_1 = None
        slice_22: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_21, 3, 0, 128);  slice_21 = None
        where_12: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_22, convert_element_type_306, full_2);  slice_22 = convert_element_type_306 = None
        view_276: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_12, _shape_param_277);  where_12 = _shape_param_277 = None
        bmm_42: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg257_1, view_276);  arg257_1 = None
        bmm_43: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_276, arg258_1);  view_276 = arg258_1 = None
        view_277: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_278);  bmm_42 = _shape_param_278 = None
        view_278: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_43, _shape_param_279);  bmm_43 = _shape_param_279 = None
        convert_element_type_307: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_277, torch.float32);  view_277 = None
        permute_115: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_307, [0, 1, 3, 2]);  convert_element_type_307 = None
        convert_element_type_308: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_115, torch.bfloat16);  permute_115 = None
        convert_element_type_309: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_278, torch.bfloat16);  view_278 = None
        permute_116: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        clone_41: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_279: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_41, _shape_param_280);  clone_41 = _shape_param_280 = None
        permute_117: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_308, [0, 2, 1, 3]);  convert_element_type_308 = None
        view_280: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_117, _shape_param_281);  permute_117 = _shape_param_281 = None
        permute_118: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_309, [0, 2, 1, 3]);  convert_element_type_309 = None
        clone_42: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
        view_281: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_42, _shape_param_282);  clone_42 = _shape_param_282 = None
        view_282: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_279, _shape_param_283);  view_279 = _shape_param_283 = None
        permute_119: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_282, [1, 0])
        mm_128: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_119, arg259_1);  permute_119 = None
        mm_129: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_282, arg260_1);  view_282 = arg260_1 = None
        view_283: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_129, _shape_param_284);  mm_129 = _shape_param_284 = None
        convert_element_type_310: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        convert_element_type_311: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        clone_43: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_280, memory_format = torch.contiguous_format);  view_280 = None
        view_284: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_43, _shape_param_285);  clone_43 = _shape_param_285 = None
        permute_120: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_284, [1, 0])
        mm_130: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_120, arg259_1);  permute_120 = None
        mm_131: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_284, arg261_1);  view_284 = arg261_1 = None
        view_285: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_131, _shape_param_286);  mm_131 = _shape_param_286 = None
        convert_element_type_312: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_285, torch.float32);  view_285 = None
        add_86: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_310, convert_element_type_312);  convert_element_type_310 = convert_element_type_312 = None
        convert_element_type_313: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        view_286: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_281, _shape_param_287);  view_281 = _shape_param_287 = None
        permute_121: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_286, [1, 0])
        mm_132: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_121, arg259_1);  permute_121 = arg259_1 = None
        mm_133: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_286, arg262_1);  view_286 = arg262_1 = None
        view_287: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_133, _shape_param_288);  mm_133 = _shape_param_288 = None
        convert_element_type_314: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        add_87: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_86, convert_element_type_314);  add_86 = convert_element_type_314 = None
        convert_element_type_315: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None
        mul_277: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_87, arg263_1);  arg263_1 = None
        mul_278: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_277, 2048)
        sum_135: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True)
        mul_279: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_277, arg264_1);  mul_277 = None
        sum_136: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
        mul_280: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg264_1, sum_136);  sum_136 = None
        sub_58: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_278, sum_135);  mul_278 = sum_135 = None
        sub_59: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_58, mul_280);  sub_58 = mul_280 = None
        mul_281: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg265_1, sub_59);  arg265_1 = sub_59 = None
        mul_282: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_87, arg264_1);  arg264_1 = None
        sum_137: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1]);  mul_282 = None
        sum_138: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_87, [0, 1]);  add_87 = None
        add_88: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_85, mul_281);  add_85 = mul_281 = None
        convert_element_type_316: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16)
        view_288: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_316, _shape_param_289);  convert_element_type_316 = _shape_param_289 = None
        mm_134: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_288, arg266_1);  arg266_1 = None
        permute_122: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_135: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_122, arg267_1);  permute_122 = arg267_1 = None
        sum_139: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        view_289: "f32[2048]" = torch.ops.aten.view.default(sum_139, _shape_param_290);  sum_139 = _shape_param_290 = None
        convert_element_type_317: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        view_290: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_134, _shape_param_291);  mm_134 = _shape_param_291 = None
        convert_element_type_318: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_290, torch.float32);  view_290 = None
        convert_element_type_319: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_320: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_317, torch.float32);  convert_element_type_317 = None
        view_291: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg268_1, _shape_param_292);  arg268_1 = _shape_param_292 = None
        mul_283: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_291, 0.5)
        mul_284: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_318, mul_283);  mul_283 = None
        convert_element_type_321: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_291, torch.float32)
        pow_23: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_321, 3.0)
        mul_285: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_89: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_291, mul_285);  view_291 = mul_285 = None
        mul_286: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_11: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_286);  mul_286 = None
        add_90: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_287: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_318, add_90);  convert_element_type_318 = add_90 = None
        convert_element_type_322: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_287, torch.bfloat16);  mul_287 = None
        mul_288: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_60: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_288);  mul_288 = None
        mul_289: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_284, sub_60);  mul_284 = sub_60 = None
        mul_290: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_289, 0.7978845608028654);  mul_289 = None
        convert_element_type_323: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16)
        mul_291: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_290, 0.044715);  mul_290 = None
        pow_24: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_321, 2.0);  convert_element_type_321 = None
        mul_292: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_293: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        convert_element_type_324: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_293, torch.bfloat16);  mul_293 = None
        add_91: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_323, convert_element_type_324);  convert_element_type_323 = convert_element_type_324 = None
        mul_294: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_322, 0.5);  convert_element_type_322 = None
        add_92: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_91, mul_294);  add_91 = mul_294 = None
        view_292: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_92, _shape_param_293);  add_92 = _shape_param_293 = None
        mm_136: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_292, arg269_1);  arg269_1 = None
        permute_123: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_292, [1, 0])
        mm_137: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_123, arg270_1);  permute_123 = arg270_1 = None
        sum_140: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_292, [0], True, dtype = torch.float32);  view_292 = None
        view_293: "f32[8192]" = torch.ops.aten.view.default(sum_140, _shape_param_294);  sum_140 = _shape_param_294 = None
        convert_element_type_325: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_293, torch.bfloat16);  view_293 = None
        view_294: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_136, _shape_param_295);  mm_136 = _shape_param_295 = None
        convert_element_type_326: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_294, torch.float32);  view_294 = None
        convert_element_type_327: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_328: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_325, torch.float32);  convert_element_type_325 = None
        mul_295: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_326, arg271_1);  arg271_1 = None
        mul_296: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_295, 2048)
        sum_141: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True)
        mul_297: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_295, arg272_1);  mul_295 = None
        sum_142: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
        mul_298: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg272_1, sum_142);  sum_142 = None
        sub_61: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_296, sum_141);  mul_296 = sum_141 = None
        sub_62: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_61, mul_298);  sub_61 = mul_298 = None
        mul_299: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg273_1, sub_62);  arg273_1 = sub_62 = None
        mul_300: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_326, arg272_1);  arg272_1 = None
        sum_143: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1]);  mul_300 = None
        sum_144: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_326, [0, 1]);  convert_element_type_326 = None
        add_93: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_88, mul_299);  add_88 = mul_299 = None
        convert_element_type_329: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16)
        view_295: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_329, _shape_param_296);  convert_element_type_329 = _shape_param_296 = None
        mm_138: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_295, arg274_1);  arg274_1 = None
        permute_124: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_295, [1, 0])
        mm_139: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_124, arg275_1);  permute_124 = arg275_1 = None
        sum_145: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_295, [0], True, dtype = torch.float32);  view_295 = None
        view_296: "f32[2048]" = torch.ops.aten.view.default(sum_145, _shape_param_297);  sum_145 = _shape_param_297 = None
        convert_element_type_330: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_296, torch.bfloat16);  view_296 = None
        view_297: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_138, _shape_param_298);  mm_138 = _shape_param_298 = None
        convert_element_type_331: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_332: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_330, torch.float32);  convert_element_type_330 = None
        view_298: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_297, _shape_param_299);  view_297 = _shape_param_299 = None
        permute_125: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
        clone_44: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_299: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_44, _shape_param_300);  clone_44 = _shape_param_300 = None
        bmm_44: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg276_1, view_299);  arg276_1 = None
        bmm_45: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_299, arg277_1);  view_299 = arg277_1 = None
        view_300: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_301);  bmm_44 = _shape_param_301 = None
        view_301: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_45, _shape_param_302);  bmm_45 = _shape_param_302 = None
        convert_element_type_333: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_301, torch.float32);  view_301 = None
        mul_301: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_333, arg278_1);  convert_element_type_333 = None
        sum_146: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
        neg_11: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg278_1);  arg278_1 = None
        fma_11: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_11, sum_146, mul_301);  neg_11 = sum_146 = mul_301 = None
        convert_element_type_334: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        slice_23: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg279_1, 2, 0, 128);  arg279_1 = None
        slice_24: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 128);  slice_23 = None
        where_13: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_24, convert_element_type_334, full_2);  slice_24 = convert_element_type_334 = None
        view_302: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_13, _shape_param_303);  where_13 = _shape_param_303 = None
        bmm_46: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg280_1, view_302);  arg280_1 = None
        bmm_47: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_302, arg281_1);  view_302 = arg281_1 = None
        view_303: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_304);  bmm_46 = _shape_param_304 = None
        view_304: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_47, _shape_param_305);  bmm_47 = _shape_param_305 = None
        convert_element_type_335: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_303, torch.float32);  view_303 = None
        permute_126: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_335, [0, 1, 3, 2]);  convert_element_type_335 = None
        convert_element_type_336: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_126, torch.bfloat16);  permute_126 = None
        convert_element_type_337: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_304, torch.bfloat16);  view_304 = None
        permute_127: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_45: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_305: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_45, _shape_param_306);  clone_45 = _shape_param_306 = None
        permute_128: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_336, [0, 2, 1, 3]);  convert_element_type_336 = None
        view_306: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_128, _shape_param_307);  permute_128 = _shape_param_307 = None
        permute_129: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_337, [0, 2, 1, 3]);  convert_element_type_337 = None
        clone_46: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_307: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_46, _shape_param_308);  clone_46 = _shape_param_308 = None
        view_308: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_305, _shape_param_309);  view_305 = _shape_param_309 = None
        permute_130: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_140: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_130, arg282_1);  permute_130 = None
        mm_141: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_308, arg283_1);  view_308 = arg283_1 = None
        view_309: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_141, _shape_param_310);  mm_141 = _shape_param_310 = None
        convert_element_type_338: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_309, torch.float32);  view_309 = None
        convert_element_type_339: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        clone_47: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_306, memory_format = torch.contiguous_format);  view_306 = None
        view_310: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_47, _shape_param_311);  clone_47 = _shape_param_311 = None
        permute_131: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_142: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_131, arg282_1);  permute_131 = None
        mm_143: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_310, arg284_1);  view_310 = arg284_1 = None
        view_311: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_143, _shape_param_312);  mm_143 = _shape_param_312 = None
        convert_element_type_340: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_311, torch.float32);  view_311 = None
        add_94: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_338, convert_element_type_340);  convert_element_type_338 = convert_element_type_340 = None
        convert_element_type_341: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        view_312: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_307, _shape_param_313);  view_307 = _shape_param_313 = None
        permute_132: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_312, [1, 0])
        mm_144: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_132, arg282_1);  permute_132 = arg282_1 = None
        mm_145: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_312, arg285_1);  view_312 = arg285_1 = None
        view_313: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_145, _shape_param_314);  mm_145 = _shape_param_314 = None
        convert_element_type_342: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_313, torch.float32);  view_313 = None
        add_95: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_94, convert_element_type_342);  add_94 = convert_element_type_342 = None
        convert_element_type_343: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        mul_302: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_95, arg286_1);  arg286_1 = None
        mul_303: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_302, 2048)
        sum_147: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True)
        mul_304: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_302, arg287_1);  mul_302 = None
        sum_148: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
        mul_305: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg287_1, sum_148);  sum_148 = None
        sub_63: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_303, sum_147);  mul_303 = sum_147 = None
        sub_64: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_63, mul_305);  sub_63 = mul_305 = None
        mul_306: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg288_1, sub_64);  arg288_1 = sub_64 = None
        mul_307: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_95, arg287_1);  arg287_1 = None
        sum_149: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
        sum_150: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_95, [0, 1]);  add_95 = None
        add_96: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_93, mul_306);  add_93 = mul_306 = None
        convert_element_type_344: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16)
        view_314: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_344, _shape_param_315);  convert_element_type_344 = _shape_param_315 = None
        mm_146: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_314, arg289_1);  arg289_1 = None
        permute_133: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_314, [1, 0])
        mm_147: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_133, arg290_1);  permute_133 = arg290_1 = None
        sum_151: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True, dtype = torch.float32);  view_314 = None
        view_315: "f32[2048]" = torch.ops.aten.view.default(sum_151, _shape_param_316);  sum_151 = _shape_param_316 = None
        convert_element_type_345: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_315, torch.bfloat16);  view_315 = None
        view_316: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_146, _shape_param_317);  mm_146 = _shape_param_317 = None
        convert_element_type_346: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_316, torch.float32);  view_316 = None
        convert_element_type_347: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_348: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_345, torch.float32);  convert_element_type_345 = None
        view_317: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg291_1, _shape_param_318);  arg291_1 = _shape_param_318 = None
        mul_308: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_317, 0.5)
        mul_309: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_346, mul_308);  mul_308 = None
        convert_element_type_349: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_317, torch.float32)
        pow_25: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_349, 3.0)
        mul_310: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_97: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_317, mul_310);  view_317 = mul_310 = None
        mul_311: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_12: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_311);  mul_311 = None
        add_98: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_312: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_346, add_98);  convert_element_type_346 = add_98 = None
        convert_element_type_350: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None
        mul_313: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_65: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_313);  mul_313 = None
        mul_314: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_309, sub_65);  mul_309 = sub_65 = None
        mul_315: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_314, 0.7978845608028654);  mul_314 = None
        convert_element_type_351: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_315, torch.bfloat16)
        mul_316: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_315, 0.044715);  mul_315 = None
        pow_26: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_349, 2.0);  convert_element_type_349 = None
        mul_317: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
        mul_318: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        convert_element_type_352: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_318, torch.bfloat16);  mul_318 = None
        add_99: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_351, convert_element_type_352);  convert_element_type_351 = convert_element_type_352 = None
        mul_319: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_350, 0.5);  convert_element_type_350 = None
        add_100: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_99, mul_319);  add_99 = mul_319 = None
        view_318: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_100, _shape_param_319);  add_100 = _shape_param_319 = None
        mm_148: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_318, arg292_1);  arg292_1 = None
        permute_134: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_149: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_134, arg293_1);  permute_134 = arg293_1 = None
        sum_152: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True, dtype = torch.float32);  view_318 = None
        view_319: "f32[8192]" = torch.ops.aten.view.default(sum_152, _shape_param_320);  sum_152 = _shape_param_320 = None
        convert_element_type_353: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_319, torch.bfloat16);  view_319 = None
        view_320: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_148, _shape_param_321);  mm_148 = _shape_param_321 = None
        convert_element_type_354: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_320, torch.float32);  view_320 = None
        convert_element_type_355: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_356: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_353, torch.float32);  convert_element_type_353 = None
        mul_320: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_354, arg294_1);  arg294_1 = None
        mul_321: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_320, 2048)
        sum_153: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True)
        mul_322: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_320, arg295_1);  mul_320 = None
        sum_154: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True);  mul_322 = None
        mul_323: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg295_1, sum_154);  sum_154 = None
        sub_66: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_321, sum_153);  mul_321 = sum_153 = None
        sub_67: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_66, mul_323);  sub_66 = mul_323 = None
        mul_324: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg296_1, sub_67);  arg296_1 = sub_67 = None
        mul_325: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_354, arg295_1);  arg295_1 = None
        sum_155: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1]);  mul_325 = None
        sum_156: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_354, [0, 1]);  convert_element_type_354 = None
        add_101: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_96, mul_324);  add_96 = mul_324 = None
        convert_element_type_357: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16)
        view_321: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_357, _shape_param_322);  convert_element_type_357 = _shape_param_322 = None
        mm_150: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_321, arg297_1);  arg297_1 = None
        permute_135: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_321, [1, 0])
        mm_151: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_135, arg298_1);  permute_135 = arg298_1 = None
        sum_157: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_321, [0], True, dtype = torch.float32);  view_321 = None
        view_322: "f32[2048]" = torch.ops.aten.view.default(sum_157, _shape_param_323);  sum_157 = _shape_param_323 = None
        convert_element_type_358: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_322, torch.bfloat16);  view_322 = None
        view_323: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_150, _shape_param_324);  mm_150 = _shape_param_324 = None
        convert_element_type_359: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        convert_element_type_360: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_358, torch.float32);  convert_element_type_358 = None
        view_324: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_323, _shape_param_325);  view_323 = _shape_param_325 = None
        permute_136: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
        clone_48: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
        view_325: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_48, _shape_param_326);  clone_48 = _shape_param_326 = None
        bmm_48: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg299_1, view_325);  arg299_1 = None
        bmm_49: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_325, arg300_1);  view_325 = arg300_1 = None
        view_326: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_48, _shape_param_327);  bmm_48 = _shape_param_327 = None
        view_327: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_49, _shape_param_328);  bmm_49 = _shape_param_328 = None
        convert_element_type_361: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_327, torch.float32);  view_327 = None
        mul_326: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_361, arg301_1);  convert_element_type_361 = None
        sum_158: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [-1], True)
        neg_12: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg301_1);  arg301_1 = None
        fma_12: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_12, sum_158, mul_326);  neg_12 = sum_158 = mul_326 = None
        convert_element_type_362: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None
        slice_25: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg302_1, 2, 0, 128);  arg302_1 = None
        slice_26: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_25, 3, 0, 128);  slice_25 = None
        where_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_26, convert_element_type_362, full_2);  slice_26 = convert_element_type_362 = None
        view_328: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_14, _shape_param_329);  where_14 = _shape_param_329 = None
        bmm_50: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg303_1, view_328);  arg303_1 = None
        bmm_51: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_328, arg304_1);  view_328 = arg304_1 = None
        view_329: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_50, _shape_param_330);  bmm_50 = _shape_param_330 = None
        view_330: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_51, _shape_param_331);  bmm_51 = _shape_param_331 = None
        convert_element_type_363: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_329, torch.float32);  view_329 = None
        permute_137: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_363, [0, 1, 3, 2]);  convert_element_type_363 = None
        convert_element_type_364: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_137, torch.bfloat16);  permute_137 = None
        convert_element_type_365: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_330, torch.bfloat16);  view_330 = None
        permute_138: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        clone_49: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_138, memory_format = torch.contiguous_format);  permute_138 = None
        view_331: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_49, _shape_param_332);  clone_49 = _shape_param_332 = None
        permute_139: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_364, [0, 2, 1, 3]);  convert_element_type_364 = None
        view_332: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_139, _shape_param_333);  permute_139 = _shape_param_333 = None
        permute_140: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_365, [0, 2, 1, 3]);  convert_element_type_365 = None
        clone_50: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_333: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_50, _shape_param_334);  clone_50 = _shape_param_334 = None
        view_334: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_331, _shape_param_335);  view_331 = _shape_param_335 = None
        permute_141: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_152: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_141, arg305_1);  permute_141 = None
        mm_153: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_334, arg306_1);  view_334 = arg306_1 = None
        view_335: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_153, _shape_param_336);  mm_153 = _shape_param_336 = None
        convert_element_type_366: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_335, torch.float32);  view_335 = None
        convert_element_type_367: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        clone_51: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_332, memory_format = torch.contiguous_format);  view_332 = None
        view_336: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_51, _shape_param_337);  clone_51 = _shape_param_337 = None
        permute_142: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_154: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_142, arg305_1);  permute_142 = None
        mm_155: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_336, arg307_1);  view_336 = arg307_1 = None
        view_337: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_155, _shape_param_338);  mm_155 = _shape_param_338 = None
        convert_element_type_368: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_337, torch.float32);  view_337 = None
        add_102: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_366, convert_element_type_368);  convert_element_type_366 = convert_element_type_368 = None
        convert_element_type_369: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        view_338: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_333, _shape_param_339);  view_333 = _shape_param_339 = None
        permute_143: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_156: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_143, arg305_1);  permute_143 = arg305_1 = None
        mm_157: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_338, arg308_1);  view_338 = arg308_1 = None
        view_339: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_157, _shape_param_340);  mm_157 = _shape_param_340 = None
        convert_element_type_370: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        add_103: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_102, convert_element_type_370);  add_102 = convert_element_type_370 = None
        convert_element_type_371: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        mul_327: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_103, arg309_1);  arg309_1 = None
        mul_328: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_327, 2048)
        sum_159: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
        mul_329: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_327, arg310_1);  mul_327 = None
        sum_160: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
        mul_330: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg310_1, sum_160);  sum_160 = None
        sub_68: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_328, sum_159);  mul_328 = sum_159 = None
        sub_69: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_68, mul_330);  sub_68 = mul_330 = None
        mul_331: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg311_1, sub_69);  arg311_1 = sub_69 = None
        mul_332: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_103, arg310_1);  arg310_1 = None
        sum_161: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
        sum_162: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_103, [0, 1]);  add_103 = None
        add_104: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_101, mul_331);  add_101 = mul_331 = None
        convert_element_type_372: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16)
        view_340: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_372, _shape_param_341);  convert_element_type_372 = _shape_param_341 = None
        mm_158: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_340, arg312_1);  arg312_1 = None
        permute_144: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_340, [1, 0])
        mm_159: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_144, arg313_1);  permute_144 = arg313_1 = None
        sum_163: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True, dtype = torch.float32);  view_340 = None
        view_341: "f32[2048]" = torch.ops.aten.view.default(sum_163, _shape_param_342);  sum_163 = _shape_param_342 = None
        convert_element_type_373: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_341, torch.bfloat16);  view_341 = None
        view_342: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_158, _shape_param_343);  mm_158 = _shape_param_343 = None
        convert_element_type_374: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_342, torch.float32);  view_342 = None
        convert_element_type_375: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_376: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_373, torch.float32);  convert_element_type_373 = None
        view_343: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg314_1, _shape_param_344);  arg314_1 = _shape_param_344 = None
        mul_333: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
        mul_334: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_374, mul_333);  mul_333 = None
        convert_element_type_377: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_343, torch.float32)
        pow_27: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_377, 3.0)
        mul_335: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_105: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_343, mul_335);  view_343 = mul_335 = None
        mul_336: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_105, 0.7978845608028654);  add_105 = None
        tanh_13: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_336);  mul_336 = None
        add_106: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_337: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_374, add_106);  convert_element_type_374 = add_106 = None
        convert_element_type_378: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_337, torch.bfloat16);  mul_337 = None
        mul_338: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_70: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_338);  mul_338 = None
        mul_339: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_334, sub_70);  mul_334 = sub_70 = None
        mul_340: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_339, 0.7978845608028654);  mul_339 = None
        convert_element_type_379: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_340, torch.bfloat16)
        mul_341: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_340, 0.044715);  mul_340 = None
        pow_28: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_377, 2.0);  convert_element_type_377 = None
        mul_342: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_28, 3.0);  pow_28 = None
        mul_343: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
        convert_element_type_380: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_343, torch.bfloat16);  mul_343 = None
        add_107: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_379, convert_element_type_380);  convert_element_type_379 = convert_element_type_380 = None
        mul_344: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_378, 0.5);  convert_element_type_378 = None
        add_108: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_107, mul_344);  add_107 = mul_344 = None
        view_344: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_108, _shape_param_345);  add_108 = _shape_param_345 = None
        mm_160: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_344, arg315_1);  arg315_1 = None
        permute_145: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_344, [1, 0])
        mm_161: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_145, arg316_1);  permute_145 = arg316_1 = None
        sum_164: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True, dtype = torch.float32);  view_344 = None
        view_345: "f32[8192]" = torch.ops.aten.view.default(sum_164, _shape_param_346);  sum_164 = _shape_param_346 = None
        convert_element_type_381: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_345, torch.bfloat16);  view_345 = None
        view_346: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_160, _shape_param_347);  mm_160 = _shape_param_347 = None
        convert_element_type_382: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_346, torch.float32);  view_346 = None
        convert_element_type_383: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None
        convert_element_type_384: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_381, torch.float32);  convert_element_type_381 = None
        mul_345: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_382, arg317_1);  arg317_1 = None
        mul_346: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_345, 2048)
        sum_165: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [2], True)
        mul_347: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_345, arg318_1);  mul_345 = None
        sum_166: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True);  mul_347 = None
        mul_348: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg318_1, sum_166);  sum_166 = None
        sub_71: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_346, sum_165);  mul_346 = sum_165 = None
        sub_72: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_71, mul_348);  sub_71 = mul_348 = None
        mul_349: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg319_1, sub_72);  arg319_1 = sub_72 = None
        mul_350: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_382, arg318_1);  arg318_1 = None
        sum_167: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1]);  mul_350 = None
        sum_168: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_382, [0, 1]);  convert_element_type_382 = None
        add_109: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_104, mul_349);  add_104 = mul_349 = None
        convert_element_type_385: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16)
        view_347: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_385, _shape_param_348);  convert_element_type_385 = _shape_param_348 = None
        mm_162: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_347, arg320_1);  arg320_1 = None
        permute_146: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_347, [1, 0])
        mm_163: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_146, arg321_1);  permute_146 = arg321_1 = None
        sum_169: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True, dtype = torch.float32);  view_347 = None
        view_348: "f32[2048]" = torch.ops.aten.view.default(sum_169, _shape_param_349);  sum_169 = _shape_param_349 = None
        convert_element_type_386: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_348, torch.bfloat16);  view_348 = None
        view_349: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_162, _shape_param_350);  mm_162 = _shape_param_350 = None
        convert_element_type_387: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        convert_element_type_388: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_386, torch.float32);  convert_element_type_386 = None
        view_350: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_349, _shape_param_351);  view_349 = _shape_param_351 = None
        permute_147: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
        clone_52: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_351: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_52, _shape_param_352);  clone_52 = _shape_param_352 = None
        bmm_52: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg322_1, view_351);  arg322_1 = None
        bmm_53: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_351, arg323_1);  view_351 = arg323_1 = None
        view_352: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_52, _shape_param_353);  bmm_52 = _shape_param_353 = None
        view_353: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_53, _shape_param_354);  bmm_53 = _shape_param_354 = None
        convert_element_type_389: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_353, torch.float32);  view_353 = None
        mul_351: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_389, arg324_1);  convert_element_type_389 = None
        sum_170: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [-1], True)
        neg_13: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg324_1);  arg324_1 = None
        fma_13: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_13, sum_170, mul_351);  neg_13 = sum_170 = mul_351 = None
        convert_element_type_390: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None
        slice_27: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg325_1, 2, 0, 128);  arg325_1 = None
        slice_28: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_27, 3, 0, 128);  slice_27 = None
        where_15: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_28, convert_element_type_390, full_2);  slice_28 = convert_element_type_390 = None
        view_354: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_15, _shape_param_355);  where_15 = _shape_param_355 = None
        bmm_54: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg326_1, view_354);  arg326_1 = None
        bmm_55: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_354, arg327_1);  view_354 = arg327_1 = None
        view_355: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_54, _shape_param_356);  bmm_54 = _shape_param_356 = None
        view_356: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_55, _shape_param_357);  bmm_55 = _shape_param_357 = None
        convert_element_type_391: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_355, torch.float32);  view_355 = None
        permute_148: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_391, [0, 1, 3, 2]);  convert_element_type_391 = None
        convert_element_type_392: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_148, torch.bfloat16);  permute_148 = None
        convert_element_type_393: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_356, torch.bfloat16);  view_356 = None
        permute_149: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        clone_53: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        view_357: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_53, _shape_param_358);  clone_53 = _shape_param_358 = None
        permute_150: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_392, [0, 2, 1, 3]);  convert_element_type_392 = None
        view_358: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_150, _shape_param_359);  permute_150 = _shape_param_359 = None
        permute_151: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_393, [0, 2, 1, 3]);  convert_element_type_393 = None
        clone_54: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None
        view_359: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_54, _shape_param_360);  clone_54 = _shape_param_360 = None
        view_360: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_357, _shape_param_361);  view_357 = _shape_param_361 = None
        permute_152: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_360, [1, 0])
        mm_164: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_152, arg328_1);  permute_152 = None
        mm_165: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_360, arg329_1);  view_360 = arg329_1 = None
        view_361: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_165, _shape_param_362);  mm_165 = _shape_param_362 = None
        convert_element_type_394: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_361, torch.float32);  view_361 = None
        convert_element_type_395: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        clone_55: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_358, memory_format = torch.contiguous_format);  view_358 = None
        view_362: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_55, _shape_param_363);  clone_55 = _shape_param_363 = None
        permute_153: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_362, [1, 0])
        mm_166: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_153, arg328_1);  permute_153 = None
        mm_167: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_362, arg330_1);  view_362 = arg330_1 = None
        view_363: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_167, _shape_param_364);  mm_167 = _shape_param_364 = None
        convert_element_type_396: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_363, torch.float32);  view_363 = None
        add_110: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_394, convert_element_type_396);  convert_element_type_394 = convert_element_type_396 = None
        convert_element_type_397: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        view_364: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_359, _shape_param_365);  view_359 = _shape_param_365 = None
        permute_154: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_168: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_154, arg328_1);  permute_154 = arg328_1 = None
        mm_169: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_364, arg331_1);  view_364 = arg331_1 = None
        view_365: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_169, _shape_param_366);  mm_169 = _shape_param_366 = None
        convert_element_type_398: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_365, torch.float32);  view_365 = None
        add_111: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_110, convert_element_type_398);  add_110 = convert_element_type_398 = None
        convert_element_type_399: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        mul_352: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_111, arg332_1);  arg332_1 = None
        mul_353: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_352, 2048)
        sum_171: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True)
        mul_354: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_352, arg333_1);  mul_352 = None
        sum_172: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True);  mul_354 = None
        mul_355: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg333_1, sum_172);  sum_172 = None
        sub_73: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_353, sum_171);  mul_353 = sum_171 = None
        sub_74: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_73, mul_355);  sub_73 = mul_355 = None
        mul_356: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg334_1, sub_74);  arg334_1 = sub_74 = None
        mul_357: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_111, arg333_1);  arg333_1 = None
        sum_173: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 1]);  mul_357 = None
        sum_174: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_111, [0, 1]);  add_111 = None
        add_112: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_109, mul_356);  add_109 = mul_356 = None
        convert_element_type_400: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16)
        view_366: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_400, _shape_param_367);  convert_element_type_400 = _shape_param_367 = None
        mm_170: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_366, arg335_1);  arg335_1 = None
        permute_155: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_171: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_155, arg336_1);  permute_155 = arg336_1 = None
        sum_175: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True, dtype = torch.float32);  view_366 = None
        view_367: "f32[2048]" = torch.ops.aten.view.default(sum_175, _shape_param_368);  sum_175 = _shape_param_368 = None
        convert_element_type_401: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_367, torch.bfloat16);  view_367 = None
        view_368: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_170, _shape_param_369);  mm_170 = _shape_param_369 = None
        convert_element_type_402: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_368, torch.float32);  view_368 = None
        convert_element_type_403: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        convert_element_type_404: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_401, torch.float32);  convert_element_type_401 = None
        view_369: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg337_1, _shape_param_370);  arg337_1 = _shape_param_370 = None
        mul_358: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_369, 0.5)
        mul_359: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_402, mul_358);  mul_358 = None
        convert_element_type_405: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_369, torch.float32)
        pow_29: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_405, 3.0)
        mul_360: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_113: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_369, mul_360);  view_369 = mul_360 = None
        mul_361: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        tanh_14: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_361);  mul_361 = None
        add_114: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_362: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_402, add_114);  convert_element_type_402 = add_114 = None
        convert_element_type_406: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None
        mul_363: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_75: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_363);  mul_363 = None
        mul_364: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_359, sub_75);  mul_359 = sub_75 = None
        mul_365: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_364, 0.7978845608028654);  mul_364 = None
        convert_element_type_407: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_365, torch.bfloat16)
        mul_366: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_365, 0.044715);  mul_365 = None
        pow_30: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_405, 2.0);  convert_element_type_405 = None
        mul_367: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
        mul_368: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
        convert_element_type_408: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_368, torch.bfloat16);  mul_368 = None
        add_115: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_407, convert_element_type_408);  convert_element_type_407 = convert_element_type_408 = None
        mul_369: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_406, 0.5);  convert_element_type_406 = None
        add_116: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_115, mul_369);  add_115 = mul_369 = None
        view_370: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_116, _shape_param_371);  add_116 = _shape_param_371 = None
        mm_172: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_370, arg338_1);  arg338_1 = None
        permute_156: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_370, [1, 0])
        mm_173: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_156, arg339_1);  permute_156 = arg339_1 = None
        sum_176: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_370, [0], True, dtype = torch.float32);  view_370 = None
        view_371: "f32[8192]" = torch.ops.aten.view.default(sum_176, _shape_param_372);  sum_176 = _shape_param_372 = None
        convert_element_type_409: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_371, torch.bfloat16);  view_371 = None
        view_372: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_172, _shape_param_373);  mm_172 = _shape_param_373 = None
        convert_element_type_410: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_372, torch.float32);  view_372 = None
        convert_element_type_411: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        convert_element_type_412: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_409, torch.float32);  convert_element_type_409 = None
        mul_370: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_410, arg340_1);  arg340_1 = None
        mul_371: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_370, 2048)
        sum_177: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True)
        mul_372: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_370, arg341_1);  mul_370 = None
        sum_178: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
        mul_373: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg341_1, sum_178);  sum_178 = None
        sub_76: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_371, sum_177);  mul_371 = sum_177 = None
        sub_77: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_76, mul_373);  sub_76 = mul_373 = None
        mul_374: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg342_1, sub_77);  arg342_1 = sub_77 = None
        mul_375: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_410, arg341_1);  arg341_1 = None
        sum_179: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
        sum_180: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_410, [0, 1]);  convert_element_type_410 = None
        add_117: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_112, mul_374);  add_112 = mul_374 = None
        convert_element_type_413: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16)
        view_373: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_413, _shape_param_374);  convert_element_type_413 = _shape_param_374 = None
        mm_174: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_373, arg343_1);  arg343_1 = None
        permute_157: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_373, [1, 0])
        mm_175: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_157, arg344_1);  permute_157 = arg344_1 = None
        sum_181: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_373, [0], True, dtype = torch.float32);  view_373 = None
        view_374: "f32[2048]" = torch.ops.aten.view.default(sum_181, _shape_param_375);  sum_181 = _shape_param_375 = None
        convert_element_type_414: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_374, torch.bfloat16);  view_374 = None
        view_375: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_174, _shape_param_376);  mm_174 = _shape_param_376 = None
        convert_element_type_415: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        convert_element_type_416: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_414, torch.float32);  convert_element_type_414 = None
        view_376: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_375, _shape_param_377);  view_375 = _shape_param_377 = None
        permute_158: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        clone_56: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_377: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_56, _shape_param_378);  clone_56 = _shape_param_378 = None
        bmm_56: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg345_1, view_377);  arg345_1 = None
        bmm_57: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_377, arg346_1);  view_377 = arg346_1 = None
        view_378: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_56, _shape_param_379);  bmm_56 = _shape_param_379 = None
        view_379: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_57, _shape_param_380);  bmm_57 = _shape_param_380 = None
        convert_element_type_417: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_379, torch.float32);  view_379 = None
        mul_376: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_417, arg347_1);  convert_element_type_417 = None
        sum_182: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [-1], True)
        neg_14: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg347_1);  arg347_1 = None
        fma_14: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_14, sum_182, mul_376);  neg_14 = sum_182 = mul_376 = None
        convert_element_type_418: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None
        slice_29: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg348_1, 2, 0, 128);  arg348_1 = None
        slice_30: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_29, 3, 0, 128);  slice_29 = None
        where_16: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_30, convert_element_type_418, full_2);  slice_30 = convert_element_type_418 = None
        view_380: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_16, _shape_param_381);  where_16 = _shape_param_381 = None
        bmm_58: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg349_1, view_380);  arg349_1 = None
        bmm_59: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_380, arg350_1);  view_380 = arg350_1 = None
        view_381: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_58, _shape_param_382);  bmm_58 = _shape_param_382 = None
        view_382: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_59, _shape_param_383);  bmm_59 = _shape_param_383 = None
        convert_element_type_419: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_381, torch.float32);  view_381 = None
        permute_159: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_419, [0, 1, 3, 2]);  convert_element_type_419 = None
        convert_element_type_420: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_159, torch.bfloat16);  permute_159 = None
        convert_element_type_421: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_382, torch.bfloat16);  view_382 = None
        permute_160: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        clone_57: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_383: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_57, _shape_param_384);  clone_57 = _shape_param_384 = None
        permute_161: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_420, [0, 2, 1, 3]);  convert_element_type_420 = None
        view_384: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_161, _shape_param_385);  permute_161 = _shape_param_385 = None
        permute_162: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_421, [0, 2, 1, 3]);  convert_element_type_421 = None
        clone_58: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
        view_385: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_58, _shape_param_386);  clone_58 = _shape_param_386 = None
        view_386: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_383, _shape_param_387);  view_383 = _shape_param_387 = None
        permute_163: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_386, [1, 0])
        mm_176: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_163, arg351_1);  permute_163 = None
        mm_177: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_386, arg352_1);  view_386 = arg352_1 = None
        view_387: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_177, _shape_param_388);  mm_177 = _shape_param_388 = None
        convert_element_type_422: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_387, torch.float32);  view_387 = None
        convert_element_type_423: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        clone_59: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_384, memory_format = torch.contiguous_format);  view_384 = None
        view_388: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_59, _shape_param_389);  clone_59 = _shape_param_389 = None
        permute_164: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_388, [1, 0])
        mm_178: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_164, arg351_1);  permute_164 = None
        mm_179: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_388, arg353_1);  view_388 = arg353_1 = None
        view_389: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_179, _shape_param_390);  mm_179 = _shape_param_390 = None
        convert_element_type_424: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_389, torch.float32);  view_389 = None
        add_118: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_422, convert_element_type_424);  convert_element_type_422 = convert_element_type_424 = None
        convert_element_type_425: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        view_390: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_385, _shape_param_391);  view_385 = _shape_param_391 = None
        permute_165: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_390, [1, 0])
        mm_180: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_165, arg351_1);  permute_165 = arg351_1 = None
        mm_181: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_390, arg354_1);  view_390 = arg354_1 = None
        view_391: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_181, _shape_param_392);  mm_181 = _shape_param_392 = None
        convert_element_type_426: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_391, torch.float32);  view_391 = None
        add_119: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_118, convert_element_type_426);  add_118 = convert_element_type_426 = None
        convert_element_type_427: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None
        mul_377: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_119, arg355_1);  arg355_1 = None
        mul_378: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_377, 2048)
        sum_183: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True)
        mul_379: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_377, arg356_1);  mul_377 = None
        sum_184: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [2], True);  mul_379 = None
        mul_380: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg356_1, sum_184);  sum_184 = None
        sub_78: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_378, sum_183);  mul_378 = sum_183 = None
        sub_79: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_78, mul_380);  sub_78 = mul_380 = None
        mul_381: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg357_1, sub_79);  arg357_1 = sub_79 = None
        mul_382: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_119, arg356_1);  arg356_1 = None
        sum_185: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1]);  mul_382 = None
        sum_186: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
        add_120: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_117, mul_381);  add_117 = mul_381 = None
        convert_element_type_428: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16)
        view_392: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_428, _shape_param_393);  convert_element_type_428 = _shape_param_393 = None
        mm_182: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_392, arg358_1);  arg358_1 = None
        permute_166: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_183: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_166, arg359_1);  permute_166 = arg359_1 = None
        sum_187: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True, dtype = torch.float32);  view_392 = None
        view_393: "f32[2048]" = torch.ops.aten.view.default(sum_187, _shape_param_394);  sum_187 = _shape_param_394 = None
        convert_element_type_429: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_393, torch.bfloat16);  view_393 = None
        view_394: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_182, _shape_param_395);  mm_182 = _shape_param_395 = None
        convert_element_type_430: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_394, torch.float32);  view_394 = None
        convert_element_type_431: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_432: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_429, torch.float32);  convert_element_type_429 = None
        view_395: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg360_1, _shape_param_396);  arg360_1 = _shape_param_396 = None
        mul_383: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_395, 0.5)
        mul_384: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_430, mul_383);  mul_383 = None
        convert_element_type_433: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_395, torch.float32)
        pow_31: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_433, 3.0)
        mul_385: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_31, 0.044715);  pow_31 = None
        add_121: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_395, mul_385);  view_395 = mul_385 = None
        mul_386: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_121, 0.7978845608028654);  add_121 = None
        tanh_15: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_386);  mul_386 = None
        add_122: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_387: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_430, add_122);  convert_element_type_430 = add_122 = None
        convert_element_type_434: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_387, torch.bfloat16);  mul_387 = None
        mul_388: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_80: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_388);  mul_388 = None
        mul_389: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_384, sub_80);  mul_384 = sub_80 = None
        mul_390: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_389, 0.7978845608028654);  mul_389 = None
        convert_element_type_435: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_390, torch.bfloat16)
        mul_391: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_390, 0.044715);  mul_390 = None
        pow_32: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_433, 2.0);  convert_element_type_433 = None
        mul_392: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
        mul_393: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
        convert_element_type_436: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_393, torch.bfloat16);  mul_393 = None
        add_123: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_435, convert_element_type_436);  convert_element_type_435 = convert_element_type_436 = None
        mul_394: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_434, 0.5);  convert_element_type_434 = None
        add_124: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_123, mul_394);  add_123 = mul_394 = None
        view_396: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_124, _shape_param_397);  add_124 = _shape_param_397 = None
        mm_184: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_396, arg361_1);  arg361_1 = None
        permute_167: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_396, [1, 0])
        mm_185: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_167, arg362_1);  permute_167 = arg362_1 = None
        sum_188: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_396, [0], True, dtype = torch.float32);  view_396 = None
        view_397: "f32[8192]" = torch.ops.aten.view.default(sum_188, _shape_param_398);  sum_188 = _shape_param_398 = None
        convert_element_type_437: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_397, torch.bfloat16);  view_397 = None
        view_398: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_184, _shape_param_399);  mm_184 = _shape_param_399 = None
        convert_element_type_438: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        convert_element_type_439: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None
        convert_element_type_440: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_437, torch.float32);  convert_element_type_437 = None
        mul_395: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_438, arg363_1);  arg363_1 = None
        mul_396: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_395, 2048)
        sum_189: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
        mul_397: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_395, arg364_1);  mul_395 = None
        sum_190: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
        mul_398: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg364_1, sum_190);  sum_190 = None
        sub_81: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_396, sum_189);  mul_396 = sum_189 = None
        sub_82: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_81, mul_398);  sub_81 = mul_398 = None
        mul_399: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg365_1, sub_82);  arg365_1 = sub_82 = None
        mul_400: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_438, arg364_1);  arg364_1 = None
        sum_191: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
        sum_192: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_438, [0, 1]);  convert_element_type_438 = None
        add_125: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_120, mul_399);  add_120 = mul_399 = None
        convert_element_type_441: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16)
        view_399: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_441, _shape_param_400);  convert_element_type_441 = _shape_param_400 = None
        mm_186: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_399, arg366_1);  arg366_1 = None
        permute_168: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_399, [1, 0])
        mm_187: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_168, arg367_1);  permute_168 = arg367_1 = None
        sum_193: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_399, [0], True, dtype = torch.float32);  view_399 = None
        view_400: "f32[2048]" = torch.ops.aten.view.default(sum_193, _shape_param_401);  sum_193 = _shape_param_401 = None
        convert_element_type_442: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_400, torch.bfloat16);  view_400 = None
        view_401: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_186, _shape_param_402);  mm_186 = _shape_param_402 = None
        convert_element_type_443: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None
        convert_element_type_444: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_442, torch.float32);  convert_element_type_442 = None
        view_402: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_401, _shape_param_403);  view_401 = _shape_param_403 = None
        permute_169: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
        clone_60: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_403: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_60, _shape_param_404);  clone_60 = _shape_param_404 = None
        bmm_60: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg368_1, view_403);  arg368_1 = None
        bmm_61: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_403, arg369_1);  view_403 = arg369_1 = None
        view_404: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_60, _shape_param_405);  bmm_60 = _shape_param_405 = None
        view_405: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_61, _shape_param_406);  bmm_61 = _shape_param_406 = None
        convert_element_type_445: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_405, torch.float32);  view_405 = None
        mul_401: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_445, arg370_1);  convert_element_type_445 = None
        sum_194: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [-1], True)
        neg_15: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg370_1);  arg370_1 = None
        fma_15: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_15, sum_194, mul_401);  neg_15 = sum_194 = mul_401 = None
        convert_element_type_446: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None
        slice_31: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg371_1, 2, 0, 128);  arg371_1 = None
        slice_32: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_31, 3, 0, 128);  slice_31 = None
        where_17: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_32, convert_element_type_446, full_2);  slice_32 = convert_element_type_446 = None
        view_406: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_17, _shape_param_407);  where_17 = _shape_param_407 = None
        bmm_62: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg372_1, view_406);  arg372_1 = None
        bmm_63: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_406, arg373_1);  view_406 = arg373_1 = None
        view_407: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_62, _shape_param_408);  bmm_62 = _shape_param_408 = None
        view_408: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_63, _shape_param_409);  bmm_63 = _shape_param_409 = None
        convert_element_type_447: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_407, torch.float32);  view_407 = None
        permute_170: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_447, [0, 1, 3, 2]);  convert_element_type_447 = None
        convert_element_type_448: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_170, torch.bfloat16);  permute_170 = None
        convert_element_type_449: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_408, torch.bfloat16);  view_408 = None
        permute_171: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        clone_61: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_409: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_61, _shape_param_410);  clone_61 = _shape_param_410 = None
        permute_172: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_448, [0, 2, 1, 3]);  convert_element_type_448 = None
        view_410: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_172, _shape_param_411);  permute_172 = _shape_param_411 = None
        permute_173: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_449, [0, 2, 1, 3]);  convert_element_type_449 = None
        clone_62: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
        view_411: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_62, _shape_param_412);  clone_62 = _shape_param_412 = None
        view_412: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_409, _shape_param_413);  view_409 = _shape_param_413 = None
        permute_174: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_412, [1, 0])
        mm_188: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_174, arg374_1);  permute_174 = None
        mm_189: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_412, arg375_1);  view_412 = arg375_1 = None
        view_413: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_189, _shape_param_414);  mm_189 = _shape_param_414 = None
        convert_element_type_450: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_413, torch.float32);  view_413 = None
        convert_element_type_451: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        clone_63: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_410, memory_format = torch.contiguous_format);  view_410 = None
        view_414: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_63, _shape_param_415);  clone_63 = _shape_param_415 = None
        permute_175: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_414, [1, 0])
        mm_190: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_175, arg374_1);  permute_175 = None
        mm_191: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_414, arg376_1);  view_414 = arg376_1 = None
        view_415: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_191, _shape_param_416);  mm_191 = _shape_param_416 = None
        convert_element_type_452: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        add_126: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_450, convert_element_type_452);  convert_element_type_450 = convert_element_type_452 = None
        convert_element_type_453: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        view_416: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_411, _shape_param_417);  view_411 = _shape_param_417 = None
        permute_176: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_416, [1, 0])
        mm_192: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_176, arg374_1);  permute_176 = arg374_1 = None
        mm_193: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_416, arg377_1);  view_416 = arg377_1 = None
        view_417: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_193, _shape_param_418);  mm_193 = _shape_param_418 = None
        convert_element_type_454: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_417, torch.float32);  view_417 = None
        add_127: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_126, convert_element_type_454);  add_126 = convert_element_type_454 = None
        convert_element_type_455: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        mul_402: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_127, arg378_1);  arg378_1 = None
        mul_403: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_402, 2048)
        sum_195: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
        mul_404: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_402, arg379_1);  mul_402 = None
        sum_196: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        mul_405: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg379_1, sum_196);  sum_196 = None
        sub_83: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_403, sum_195);  mul_403 = sum_195 = None
        sub_84: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_83, mul_405);  sub_83 = mul_405 = None
        mul_406: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg380_1, sub_84);  arg380_1 = sub_84 = None
        mul_407: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_127, arg379_1);  arg379_1 = None
        sum_197: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        sum_198: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_127, [0, 1]);  add_127 = None
        add_128: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_125, mul_406);  add_125 = mul_406 = None
        convert_element_type_456: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16)
        view_418: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_456, _shape_param_419);  convert_element_type_456 = _shape_param_419 = None
        mm_194: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_418, arg381_1);  arg381_1 = None
        permute_177: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_418, [1, 0])
        mm_195: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_177, arg382_1);  permute_177 = arg382_1 = None
        sum_199: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_418, [0], True, dtype = torch.float32);  view_418 = None
        view_419: "f32[2048]" = torch.ops.aten.view.default(sum_199, _shape_param_420);  sum_199 = _shape_param_420 = None
        convert_element_type_457: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_419, torch.bfloat16);  view_419 = None
        view_420: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_194, _shape_param_421);  mm_194 = _shape_param_421 = None
        convert_element_type_458: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_420, torch.float32);  view_420 = None
        convert_element_type_459: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        convert_element_type_460: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_457, torch.float32);  convert_element_type_457 = None
        view_421: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg383_1, _shape_param_422);  arg383_1 = _shape_param_422 = None
        mul_408: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_421, 0.5)
        mul_409: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_458, mul_408);  mul_408 = None
        convert_element_type_461: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_421, torch.float32)
        pow_33: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_461, 3.0)
        mul_410: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_129: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_421, mul_410);  view_421 = mul_410 = None
        mul_411: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_129, 0.7978845608028654);  add_129 = None
        tanh_16: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_411);  mul_411 = None
        add_130: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_412: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_458, add_130);  convert_element_type_458 = add_130 = None
        convert_element_type_462: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_412, torch.bfloat16);  mul_412 = None
        mul_413: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_85: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_413);  mul_413 = None
        mul_414: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_409, sub_85);  mul_409 = sub_85 = None
        mul_415: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_414, 0.7978845608028654);  mul_414 = None
        convert_element_type_463: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_415, torch.bfloat16)
        mul_416: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_415, 0.044715);  mul_415 = None
        pow_34: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_461, 2.0);  convert_element_type_461 = None
        mul_417: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
        mul_418: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_416, mul_417);  mul_416 = mul_417 = None
        convert_element_type_464: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_418, torch.bfloat16);  mul_418 = None
        add_131: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_463, convert_element_type_464);  convert_element_type_463 = convert_element_type_464 = None
        mul_419: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_462, 0.5);  convert_element_type_462 = None
        add_132: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_131, mul_419);  add_131 = mul_419 = None
        view_422: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_132, _shape_param_423);  add_132 = _shape_param_423 = None
        mm_196: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_422, arg384_1);  arg384_1 = None
        permute_178: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_422, [1, 0])
        mm_197: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_178, arg385_1);  permute_178 = arg385_1 = None
        sum_200: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True, dtype = torch.float32);  view_422 = None
        view_423: "f32[8192]" = torch.ops.aten.view.default(sum_200, _shape_param_424);  sum_200 = _shape_param_424 = None
        convert_element_type_465: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_423, torch.bfloat16);  view_423 = None
        view_424: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_196, _shape_param_425);  mm_196 = _shape_param_425 = None
        convert_element_type_466: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_424, torch.float32);  view_424 = None
        convert_element_type_467: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None
        convert_element_type_468: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_465, torch.float32);  convert_element_type_465 = None
        mul_420: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_466, arg386_1);  arg386_1 = None
        mul_421: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_420, 2048)
        sum_201: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_420, [2], True)
        mul_422: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_420, arg387_1);  mul_420 = None
        sum_202: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True);  mul_422 = None
        mul_423: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg387_1, sum_202);  sum_202 = None
        sub_86: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_421, sum_201);  mul_421 = sum_201 = None
        sub_87: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_86, mul_423);  sub_86 = mul_423 = None
        mul_424: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg388_1, sub_87);  arg388_1 = sub_87 = None
        mul_425: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_466, arg387_1);  arg387_1 = None
        sum_203: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_425, [0, 1]);  mul_425 = None
        sum_204: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_466, [0, 1]);  convert_element_type_466 = None
        add_133: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_128, mul_424);  add_128 = mul_424 = None
        convert_element_type_469: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16)
        view_425: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_469, _shape_param_426);  convert_element_type_469 = _shape_param_426 = None
        mm_198: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_425, arg389_1);  arg389_1 = None
        permute_179: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_425, [1, 0])
        mm_199: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_179, arg390_1);  permute_179 = arg390_1 = None
        sum_205: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_425, [0], True, dtype = torch.float32);  view_425 = None
        view_426: "f32[2048]" = torch.ops.aten.view.default(sum_205, _shape_param_427);  sum_205 = _shape_param_427 = None
        convert_element_type_470: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_426, torch.bfloat16);  view_426 = None
        view_427: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_198, _shape_param_428);  mm_198 = _shape_param_428 = None
        convert_element_type_471: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None
        convert_element_type_472: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_470, torch.float32);  convert_element_type_470 = None
        view_428: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_427, _shape_param_429);  view_427 = _shape_param_429 = None
        permute_180: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
        clone_64: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_429: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_64, _shape_param_430);  clone_64 = _shape_param_430 = None
        bmm_64: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg391_1, view_429);  arg391_1 = None
        bmm_65: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_429, arg392_1);  view_429 = arg392_1 = None
        view_430: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_64, _shape_param_431);  bmm_64 = _shape_param_431 = None
        view_431: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_65, _shape_param_432);  bmm_65 = _shape_param_432 = None
        convert_element_type_473: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_431, torch.float32);  view_431 = None
        mul_426: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_473, arg393_1);  convert_element_type_473 = None
        sum_206: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [-1], True)
        neg_16: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg393_1);  arg393_1 = None
        fma_16: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_16, sum_206, mul_426);  neg_16 = sum_206 = mul_426 = None
        convert_element_type_474: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None
        slice_33: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg394_1, 2, 0, 128);  arg394_1 = None
        slice_34: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_33, 3, 0, 128);  slice_33 = None
        where_18: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_34, convert_element_type_474, full_2);  slice_34 = convert_element_type_474 = None
        view_432: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_18, _shape_param_433);  where_18 = _shape_param_433 = None
        bmm_66: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg395_1, view_432);  arg395_1 = None
        bmm_67: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_432, arg396_1);  view_432 = arg396_1 = None
        view_433: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_66, _shape_param_434);  bmm_66 = _shape_param_434 = None
        view_434: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_67, _shape_param_435);  bmm_67 = _shape_param_435 = None
        convert_element_type_475: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_433, torch.float32);  view_433 = None
        permute_181: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_475, [0, 1, 3, 2]);  convert_element_type_475 = None
        convert_element_type_476: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_181, torch.bfloat16);  permute_181 = None
        convert_element_type_477: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_434, torch.bfloat16);  view_434 = None
        permute_182: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
        clone_65: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_435: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_65, _shape_param_436);  clone_65 = _shape_param_436 = None
        permute_183: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_476, [0, 2, 1, 3]);  convert_element_type_476 = None
        view_436: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_183, _shape_param_437);  permute_183 = _shape_param_437 = None
        permute_184: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_477, [0, 2, 1, 3]);  convert_element_type_477 = None
        clone_66: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_437: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_66, _shape_param_438);  clone_66 = _shape_param_438 = None
        view_438: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_435, _shape_param_439);  view_435 = _shape_param_439 = None
        permute_185: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_438, [1, 0])
        mm_200: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_185, arg397_1);  permute_185 = None
        mm_201: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_438, arg398_1);  view_438 = arg398_1 = None
        view_439: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_201, _shape_param_440);  mm_201 = _shape_param_440 = None
        convert_element_type_478: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_439, torch.float32);  view_439 = None
        convert_element_type_479: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        clone_67: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_436, memory_format = torch.contiguous_format);  view_436 = None
        view_440: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_67, _shape_param_441);  clone_67 = _shape_param_441 = None
        permute_186: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_440, [1, 0])
        mm_202: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_186, arg397_1);  permute_186 = None
        mm_203: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_440, arg399_1);  view_440 = arg399_1 = None
        view_441: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_203, _shape_param_442);  mm_203 = _shape_param_442 = None
        convert_element_type_480: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_441, torch.float32);  view_441 = None
        add_134: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_478, convert_element_type_480);  convert_element_type_478 = convert_element_type_480 = None
        convert_element_type_481: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_202, torch.float32);  mm_202 = None
        view_442: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_437, _shape_param_443);  view_437 = _shape_param_443 = None
        permute_187: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_442, [1, 0])
        mm_204: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_187, arg397_1);  permute_187 = arg397_1 = None
        mm_205: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_442, arg400_1);  view_442 = arg400_1 = None
        view_443: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_205, _shape_param_444);  mm_205 = _shape_param_444 = None
        convert_element_type_482: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_443, torch.float32);  view_443 = None
        add_135: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_134, convert_element_type_482);  add_134 = convert_element_type_482 = None
        convert_element_type_483: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_204, torch.float32);  mm_204 = None
        mul_427: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_135, arg401_1);  arg401_1 = None
        mul_428: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_427, 2048)
        sum_207: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
        mul_429: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_427, arg402_1);  mul_427 = None
        sum_208: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
        mul_430: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg402_1, sum_208);  sum_208 = None
        sub_88: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_428, sum_207);  mul_428 = sum_207 = None
        sub_89: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_88, mul_430);  sub_88 = mul_430 = None
        mul_431: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg403_1, sub_89);  arg403_1 = sub_89 = None
        mul_432: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_135, arg402_1);  arg402_1 = None
        sum_209: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
        sum_210: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
        add_136: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_133, mul_431);  add_133 = mul_431 = None
        convert_element_type_484: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16)
        view_444: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_484, _shape_param_445);  convert_element_type_484 = _shape_param_445 = None
        mm_206: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_444, arg404_1);  arg404_1 = None
        permute_188: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_444, [1, 0])
        mm_207: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_188, arg405_1);  permute_188 = arg405_1 = None
        sum_211: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_444, [0], True, dtype = torch.float32);  view_444 = None
        view_445: "f32[2048]" = torch.ops.aten.view.default(sum_211, _shape_param_446);  sum_211 = _shape_param_446 = None
        convert_element_type_485: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_445, torch.bfloat16);  view_445 = None
        view_446: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_206, _shape_param_447);  mm_206 = _shape_param_447 = None
        convert_element_type_486: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_446, torch.float32);  view_446 = None
        convert_element_type_487: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        convert_element_type_488: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_485, torch.float32);  convert_element_type_485 = None
        view_447: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg406_1, _shape_param_448);  arg406_1 = _shape_param_448 = None
        mul_433: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_447, 0.5)
        mul_434: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_486, mul_433);  mul_433 = None
        convert_element_type_489: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_447, torch.float32)
        pow_35: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_489, 3.0)
        mul_435: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_35, 0.044715);  pow_35 = None
        add_137: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_447, mul_435);  view_447 = mul_435 = None
        mul_436: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_137, 0.7978845608028654);  add_137 = None
        tanh_17: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_436);  mul_436 = None
        add_138: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_437: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_486, add_138);  convert_element_type_486 = add_138 = None
        convert_element_type_490: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_437, torch.bfloat16);  mul_437 = None
        mul_438: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_90: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_438);  mul_438 = None
        mul_439: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_434, sub_90);  mul_434 = sub_90 = None
        mul_440: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_439, 0.7978845608028654);  mul_439 = None
        convert_element_type_491: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_440, torch.bfloat16)
        mul_441: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_440, 0.044715);  mul_440 = None
        pow_36: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_489, 2.0);  convert_element_type_489 = None
        mul_442: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_443: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
        convert_element_type_492: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_443, torch.bfloat16);  mul_443 = None
        add_139: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_491, convert_element_type_492);  convert_element_type_491 = convert_element_type_492 = None
        mul_444: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_490, 0.5);  convert_element_type_490 = None
        add_140: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_139, mul_444);  add_139 = mul_444 = None
        view_448: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_140, _shape_param_449);  add_140 = _shape_param_449 = None
        mm_208: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_448, arg407_1);  arg407_1 = None
        permute_189: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_209: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_189, arg408_1);  permute_189 = arg408_1 = None
        sum_212: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True, dtype = torch.float32);  view_448 = None
        view_449: "f32[8192]" = torch.ops.aten.view.default(sum_212, _shape_param_450);  sum_212 = _shape_param_450 = None
        convert_element_type_493: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_449, torch.bfloat16);  view_449 = None
        view_450: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_208, _shape_param_451);  mm_208 = _shape_param_451 = None
        convert_element_type_494: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_450, torch.float32);  view_450 = None
        convert_element_type_495: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_209, torch.float32);  mm_209 = None
        convert_element_type_496: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_493, torch.float32);  convert_element_type_493 = None
        mul_445: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_494, arg409_1);  arg409_1 = None
        mul_446: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_445, 2048)
        sum_213: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True)
        mul_447: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_445, arg410_1);  mul_445 = None
        sum_214: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
        mul_448: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg410_1, sum_214);  sum_214 = None
        sub_91: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_446, sum_213);  mul_446 = sum_213 = None
        sub_92: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_91, mul_448);  sub_91 = mul_448 = None
        mul_449: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg411_1, sub_92);  arg411_1 = sub_92 = None
        mul_450: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_494, arg410_1);  arg410_1 = None
        sum_215: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1]);  mul_450 = None
        sum_216: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_494, [0, 1]);  convert_element_type_494 = None
        add_141: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_136, mul_449);  add_136 = mul_449 = None
        convert_element_type_497: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16)
        view_451: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_497, _shape_param_452);  convert_element_type_497 = _shape_param_452 = None
        mm_210: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_451, arg412_1);  arg412_1 = None
        permute_190: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_451, [1, 0])
        mm_211: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_190, arg413_1);  permute_190 = arg413_1 = None
        sum_217: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True, dtype = torch.float32);  view_451 = None
        view_452: "f32[2048]" = torch.ops.aten.view.default(sum_217, _shape_param_453);  sum_217 = _shape_param_453 = None
        convert_element_type_498: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_452, torch.bfloat16);  view_452 = None
        view_453: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_210, _shape_param_454);  mm_210 = _shape_param_454 = None
        convert_element_type_499: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_211, torch.float32);  mm_211 = None
        convert_element_type_500: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_498, torch.float32);  convert_element_type_498 = None
        view_454: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_453, _shape_param_455);  view_453 = _shape_param_455 = None
        permute_191: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
        clone_68: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
        view_455: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_68, _shape_param_456);  clone_68 = _shape_param_456 = None
        bmm_68: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg414_1, view_455);  arg414_1 = None
        bmm_69: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_455, arg415_1);  view_455 = arg415_1 = None
        view_456: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_68, _shape_param_457);  bmm_68 = _shape_param_457 = None
        view_457: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_69, _shape_param_458);  bmm_69 = _shape_param_458 = None
        convert_element_type_501: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_457, torch.float32);  view_457 = None
        mul_451: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_501, arg416_1);  convert_element_type_501 = None
        sum_218: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [-1], True)
        neg_17: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg416_1);  arg416_1 = None
        fma_17: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_17, sum_218, mul_451);  neg_17 = sum_218 = mul_451 = None
        convert_element_type_502: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None
        slice_35: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg417_1, 2, 0, 128);  arg417_1 = None
        slice_36: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 128);  slice_35 = None
        where_19: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_36, convert_element_type_502, full_2);  slice_36 = convert_element_type_502 = None
        view_458: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_19, _shape_param_459);  where_19 = _shape_param_459 = None
        bmm_70: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg418_1, view_458);  arg418_1 = None
        bmm_71: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_458, arg419_1);  view_458 = arg419_1 = None
        view_459: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_70, _shape_param_460);  bmm_70 = _shape_param_460 = None
        view_460: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_71, _shape_param_461);  bmm_71 = _shape_param_461 = None
        convert_element_type_503: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        permute_192: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_503, [0, 1, 3, 2]);  convert_element_type_503 = None
        convert_element_type_504: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_192, torch.bfloat16);  permute_192 = None
        convert_element_type_505: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_460, torch.bfloat16);  view_460 = None
        permute_193: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        clone_69: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_461: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_69, _shape_param_462);  clone_69 = _shape_param_462 = None
        permute_194: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_504, [0, 2, 1, 3]);  convert_element_type_504 = None
        view_462: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_194, _shape_param_463);  permute_194 = _shape_param_463 = None
        permute_195: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_505, [0, 2, 1, 3]);  convert_element_type_505 = None
        clone_70: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_463: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_70, _shape_param_464);  clone_70 = _shape_param_464 = None
        view_464: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_461, _shape_param_465);  view_461 = _shape_param_465 = None
        permute_196: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_464, [1, 0])
        mm_212: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_196, arg420_1);  permute_196 = None
        mm_213: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_464, arg421_1);  view_464 = arg421_1 = None
        view_465: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_213, _shape_param_466);  mm_213 = _shape_param_466 = None
        convert_element_type_506: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_465, torch.float32);  view_465 = None
        convert_element_type_507: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_212, torch.float32);  mm_212 = None
        clone_71: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_462, memory_format = torch.contiguous_format);  view_462 = None
        view_466: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_71, _shape_param_467);  clone_71 = _shape_param_467 = None
        permute_197: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_466, [1, 0])
        mm_214: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_197, arg420_1);  permute_197 = None
        mm_215: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_466, arg422_1);  view_466 = arg422_1 = None
        view_467: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_215, _shape_param_468);  mm_215 = _shape_param_468 = None
        convert_element_type_508: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_467, torch.float32);  view_467 = None
        add_142: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_506, convert_element_type_508);  convert_element_type_506 = convert_element_type_508 = None
        convert_element_type_509: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_214, torch.float32);  mm_214 = None
        view_468: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_463, _shape_param_469);  view_463 = _shape_param_469 = None
        permute_198: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_468, [1, 0])
        mm_216: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_198, arg420_1);  permute_198 = arg420_1 = None
        mm_217: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_468, arg423_1);  view_468 = arg423_1 = None
        view_469: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_217, _shape_param_470);  mm_217 = _shape_param_470 = None
        convert_element_type_510: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_469, torch.float32);  view_469 = None
        add_143: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_142, convert_element_type_510);  add_142 = convert_element_type_510 = None
        convert_element_type_511: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_216, torch.float32);  mm_216 = None
        mul_452: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_143, arg424_1);  arg424_1 = None
        mul_453: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_452, 2048)
        sum_219: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_452, [2], True)
        mul_454: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_452, arg425_1);  mul_452 = None
        sum_220: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True);  mul_454 = None
        mul_455: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg425_1, sum_220);  sum_220 = None
        sub_93: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_453, sum_219);  mul_453 = sum_219 = None
        sub_94: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_93, mul_455);  sub_93 = mul_455 = None
        mul_456: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg426_1, sub_94);  arg426_1 = sub_94 = None
        mul_457: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_143, arg425_1);  arg425_1 = None
        sum_221: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 1]);  mul_457 = None
        sum_222: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
        add_144: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_141, mul_456);  add_141 = mul_456 = None
        convert_element_type_512: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16)
        view_470: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_512, _shape_param_471);  convert_element_type_512 = _shape_param_471 = None
        mm_218: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_470, arg427_1);  arg427_1 = None
        permute_199: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_219: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_199, arg428_1);  permute_199 = arg428_1 = None
        sum_223: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_470, [0], True, dtype = torch.float32);  view_470 = None
        view_471: "f32[2048]" = torch.ops.aten.view.default(sum_223, _shape_param_472);  sum_223 = _shape_param_472 = None
        convert_element_type_513: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_471, torch.bfloat16);  view_471 = None
        view_472: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_218, _shape_param_473);  mm_218 = _shape_param_473 = None
        convert_element_type_514: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_472, torch.float32);  view_472 = None
        convert_element_type_515: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_219, torch.float32);  mm_219 = None
        convert_element_type_516: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_513, torch.float32);  convert_element_type_513 = None
        view_473: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg429_1, _shape_param_474);  arg429_1 = _shape_param_474 = None
        mul_458: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_473, 0.5)
        mul_459: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_514, mul_458);  mul_458 = None
        convert_element_type_517: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_473, torch.float32)
        pow_37: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_517, 3.0)
        mul_460: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_145: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_473, mul_460);  view_473 = mul_460 = None
        mul_461: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_145, 0.7978845608028654);  add_145 = None
        tanh_18: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_461);  mul_461 = None
        add_146: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_462: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_514, add_146);  convert_element_type_514 = add_146 = None
        convert_element_type_518: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_462, torch.bfloat16);  mul_462 = None
        mul_463: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_95: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_463);  mul_463 = None
        mul_464: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_459, sub_95);  mul_459 = sub_95 = None
        mul_465: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_464, 0.7978845608028654);  mul_464 = None
        convert_element_type_519: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_465, torch.bfloat16)
        mul_466: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_465, 0.044715);  mul_465 = None
        pow_38: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_517, 2.0);  convert_element_type_517 = None
        mul_467: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_468: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        convert_element_type_520: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_468, torch.bfloat16);  mul_468 = None
        add_147: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_519, convert_element_type_520);  convert_element_type_519 = convert_element_type_520 = None
        mul_469: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_518, 0.5);  convert_element_type_518 = None
        add_148: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_147, mul_469);  add_147 = mul_469 = None
        view_474: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_148, _shape_param_475);  add_148 = _shape_param_475 = None
        mm_220: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_474, arg430_1);  arg430_1 = None
        permute_200: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_474, [1, 0])
        mm_221: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_200, arg431_1);  permute_200 = arg431_1 = None
        sum_224: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_474, [0], True, dtype = torch.float32);  view_474 = None
        view_475: "f32[8192]" = torch.ops.aten.view.default(sum_224, _shape_param_476);  sum_224 = _shape_param_476 = None
        convert_element_type_521: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_475, torch.bfloat16);  view_475 = None
        view_476: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_220, _shape_param_477);  mm_220 = _shape_param_477 = None
        convert_element_type_522: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_476, torch.float32);  view_476 = None
        convert_element_type_523: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_221, torch.float32);  mm_221 = None
        convert_element_type_524: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_521, torch.float32);  convert_element_type_521 = None
        mul_470: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_522, arg432_1);  arg432_1 = None
        mul_471: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_470, 2048)
        sum_225: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [2], True)
        mul_472: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_470, arg433_1);  mul_470 = None
        sum_226: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
        mul_473: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg433_1, sum_226);  sum_226 = None
        sub_96: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_471, sum_225);  mul_471 = sum_225 = None
        sub_97: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_96, mul_473);  sub_96 = mul_473 = None
        mul_474: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg434_1, sub_97);  arg434_1 = sub_97 = None
        mul_475: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_522, arg433_1);  arg433_1 = None
        sum_227: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
        sum_228: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_522, [0, 1]);  convert_element_type_522 = None
        add_149: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_144, mul_474);  add_144 = mul_474 = None
        convert_element_type_525: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16)
        view_477: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_525, _shape_param_478);  convert_element_type_525 = _shape_param_478 = None
        mm_222: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_477, arg435_1);  arg435_1 = None
        permute_201: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_477, [1, 0])
        mm_223: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_201, arg436_1);  permute_201 = arg436_1 = None
        sum_229: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_477, [0], True, dtype = torch.float32);  view_477 = None
        view_478: "f32[2048]" = torch.ops.aten.view.default(sum_229, _shape_param_479);  sum_229 = _shape_param_479 = None
        convert_element_type_526: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_478, torch.bfloat16);  view_478 = None
        view_479: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_222, _shape_param_480);  mm_222 = _shape_param_480 = None
        convert_element_type_527: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_223, torch.float32);  mm_223 = None
        convert_element_type_528: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_526, torch.float32);  convert_element_type_526 = None
        view_480: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_479, _shape_param_481);  view_479 = _shape_param_481 = None
        permute_202: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None
        clone_72: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_481: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_72, _shape_param_482);  clone_72 = _shape_param_482 = None
        bmm_72: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg437_1, view_481);  arg437_1 = None
        bmm_73: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_481, arg438_1);  view_481 = arg438_1 = None
        view_482: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_72, _shape_param_483);  bmm_72 = _shape_param_483 = None
        view_483: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_73, _shape_param_484);  bmm_73 = _shape_param_484 = None
        convert_element_type_529: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_483, torch.float32);  view_483 = None
        mul_476: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_529, arg439_1);  convert_element_type_529 = None
        sum_230: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [-1], True)
        neg_18: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg439_1);  arg439_1 = None
        fma_18: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_18, sum_230, mul_476);  neg_18 = sum_230 = mul_476 = None
        convert_element_type_530: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None
        slice_37: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg440_1, 2, 0, 128);  arg440_1 = None
        slice_38: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_37, 3, 0, 128);  slice_37 = None
        where_20: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_38, convert_element_type_530, full_2);  slice_38 = convert_element_type_530 = None
        view_484: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_20, _shape_param_485);  where_20 = _shape_param_485 = None
        bmm_74: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg441_1, view_484);  arg441_1 = None
        bmm_75: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_484, arg442_1);  view_484 = arg442_1 = None
        view_485: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_74, _shape_param_486);  bmm_74 = _shape_param_486 = None
        view_486: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_75, _shape_param_487);  bmm_75 = _shape_param_487 = None
        convert_element_type_531: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_485, torch.float32);  view_485 = None
        permute_203: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_531, [0, 1, 3, 2]);  convert_element_type_531 = None
        convert_element_type_532: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_203, torch.bfloat16);  permute_203 = None
        convert_element_type_533: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_486, torch.bfloat16);  view_486 = None
        permute_204: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_482, [0, 2, 1, 3]);  view_482 = None
        clone_73: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_487: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_73, _shape_param_488);  clone_73 = _shape_param_488 = None
        permute_205: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_532, [0, 2, 1, 3]);  convert_element_type_532 = None
        view_488: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_205, _shape_param_489);  permute_205 = _shape_param_489 = None
        permute_206: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_533, [0, 2, 1, 3]);  convert_element_type_533 = None
        clone_74: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
        view_489: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_74, _shape_param_490);  clone_74 = _shape_param_490 = None
        view_490: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_487, _shape_param_491);  view_487 = _shape_param_491 = None
        permute_207: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_490, [1, 0])
        mm_224: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_207, arg443_1);  permute_207 = None
        mm_225: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_490, arg444_1);  view_490 = arg444_1 = None
        view_491: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_225, _shape_param_492);  mm_225 = _shape_param_492 = None
        convert_element_type_534: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_491, torch.float32);  view_491 = None
        convert_element_type_535: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_224, torch.float32);  mm_224 = None
        clone_75: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_488, memory_format = torch.contiguous_format);  view_488 = None
        view_492: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_75, _shape_param_493);  clone_75 = _shape_param_493 = None
        permute_208: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_492, [1, 0])
        mm_226: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_208, arg443_1);  permute_208 = None
        mm_227: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_492, arg445_1);  view_492 = arg445_1 = None
        view_493: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_227, _shape_param_494);  mm_227 = _shape_param_494 = None
        convert_element_type_536: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_493, torch.float32);  view_493 = None
        add_150: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_534, convert_element_type_536);  convert_element_type_534 = convert_element_type_536 = None
        convert_element_type_537: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_226, torch.float32);  mm_226 = None
        view_494: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_489, _shape_param_495);  view_489 = _shape_param_495 = None
        permute_209: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_228: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_209, arg443_1);  permute_209 = arg443_1 = None
        mm_229: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_494, arg446_1);  view_494 = arg446_1 = None
        view_495: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_229, _shape_param_496);  mm_229 = _shape_param_496 = None
        convert_element_type_538: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_495, torch.float32);  view_495 = None
        add_151: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_150, convert_element_type_538);  add_150 = convert_element_type_538 = None
        convert_element_type_539: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_228, torch.float32);  mm_228 = None
        mul_477: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_151, arg447_1);  arg447_1 = None
        mul_478: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_477, 2048)
        sum_231: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True)
        mul_479: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_477, arg448_1);  mul_477 = None
        sum_232: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
        mul_480: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg448_1, sum_232);  sum_232 = None
        sub_98: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_478, sum_231);  mul_478 = sum_231 = None
        sub_99: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_98, mul_480);  sub_98 = mul_480 = None
        mul_481: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg449_1, sub_99);  arg449_1 = sub_99 = None
        mul_482: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_151, arg448_1);  arg448_1 = None
        sum_233: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1]);  mul_482 = None
        sum_234: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_151, [0, 1]);  add_151 = None
        add_152: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_149, mul_481);  add_149 = mul_481 = None
        convert_element_type_540: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16)
        view_496: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_540, _shape_param_497);  convert_element_type_540 = _shape_param_497 = None
        mm_230: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_496, arg450_1);  arg450_1 = None
        permute_210: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_231: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_210, arg451_1);  permute_210 = arg451_1 = None
        sum_235: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True, dtype = torch.float32);  view_496 = None
        view_497: "f32[2048]" = torch.ops.aten.view.default(sum_235, _shape_param_498);  sum_235 = _shape_param_498 = None
        convert_element_type_541: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_497, torch.bfloat16);  view_497 = None
        view_498: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_230, _shape_param_499);  mm_230 = _shape_param_499 = None
        convert_element_type_542: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_498, torch.float32);  view_498 = None
        convert_element_type_543: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_231, torch.float32);  mm_231 = None
        convert_element_type_544: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_541, torch.float32);  convert_element_type_541 = None
        view_499: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg452_1, _shape_param_500);  arg452_1 = _shape_param_500 = None
        mul_483: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_499, 0.5)
        mul_484: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_542, mul_483);  mul_483 = None
        convert_element_type_545: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_499, torch.float32)
        pow_39: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_545, 3.0)
        mul_485: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_39, 0.044715);  pow_39 = None
        add_153: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_499, mul_485);  view_499 = mul_485 = None
        mul_486: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_153, 0.7978845608028654);  add_153 = None
        tanh_19: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_486);  mul_486 = None
        add_154: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_487: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_542, add_154);  convert_element_type_542 = add_154 = None
        convert_element_type_546: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_487, torch.bfloat16);  mul_487 = None
        mul_488: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_100: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_488);  mul_488 = None
        mul_489: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_484, sub_100);  mul_484 = sub_100 = None
        mul_490: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_489, 0.7978845608028654);  mul_489 = None
        convert_element_type_547: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_490, torch.bfloat16)
        mul_491: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_490, 0.044715);  mul_490 = None
        pow_40: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_545, 2.0);  convert_element_type_545 = None
        mul_492: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_493: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
        convert_element_type_548: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_493, torch.bfloat16);  mul_493 = None
        add_155: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_547, convert_element_type_548);  convert_element_type_547 = convert_element_type_548 = None
        mul_494: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_546, 0.5);  convert_element_type_546 = None
        add_156: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_155, mul_494);  add_155 = mul_494 = None
        view_500: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_156, _shape_param_501);  add_156 = _shape_param_501 = None
        mm_232: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_500, arg453_1);  arg453_1 = None
        permute_211: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_500, [1, 0])
        mm_233: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_211, arg454_1);  permute_211 = arg454_1 = None
        sum_236: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True, dtype = torch.float32);  view_500 = None
        view_501: "f32[8192]" = torch.ops.aten.view.default(sum_236, _shape_param_502);  sum_236 = _shape_param_502 = None
        convert_element_type_549: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_501, torch.bfloat16);  view_501 = None
        view_502: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_232, _shape_param_503);  mm_232 = _shape_param_503 = None
        convert_element_type_550: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_502, torch.float32);  view_502 = None
        convert_element_type_551: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_233, torch.float32);  mm_233 = None
        convert_element_type_552: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_549, torch.float32);  convert_element_type_549 = None
        mul_495: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_550, arg455_1);  arg455_1 = None
        mul_496: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_495, 2048)
        sum_237: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
        mul_497: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_495, arg456_1);  mul_495 = None
        sum_238: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
        mul_498: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg456_1, sum_238);  sum_238 = None
        sub_101: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_496, sum_237);  mul_496 = sum_237 = None
        sub_102: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_101, mul_498);  sub_101 = mul_498 = None
        mul_499: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg457_1, sub_102);  arg457_1 = sub_102 = None
        mul_500: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_550, arg456_1);  arg456_1 = None
        sum_239: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
        sum_240: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_550, [0, 1]);  convert_element_type_550 = None
        add_157: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_152, mul_499);  add_152 = mul_499 = None
        convert_element_type_553: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16)
        view_503: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_553, _shape_param_504);  convert_element_type_553 = _shape_param_504 = None
        mm_234: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_503, arg458_1);  arg458_1 = None
        permute_212: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_503, [1, 0])
        mm_235: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_212, arg459_1);  permute_212 = arg459_1 = None
        sum_241: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True, dtype = torch.float32);  view_503 = None
        view_504: "f32[2048]" = torch.ops.aten.view.default(sum_241, _shape_param_505);  sum_241 = _shape_param_505 = None
        convert_element_type_554: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_504, torch.bfloat16);  view_504 = None
        view_505: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_234, _shape_param_506);  mm_234 = _shape_param_506 = None
        convert_element_type_555: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None
        convert_element_type_556: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_554, torch.float32);  convert_element_type_554 = None
        view_506: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_505, _shape_param_507);  view_505 = _shape_param_507 = None
        permute_213: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None
        clone_76: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_507: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_76, _shape_param_508);  clone_76 = _shape_param_508 = None
        bmm_76: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg460_1, view_507);  arg460_1 = None
        bmm_77: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_507, arg461_1);  view_507 = arg461_1 = None
        view_508: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_76, _shape_param_509);  bmm_76 = _shape_param_509 = None
        view_509: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_77, _shape_param_510);  bmm_77 = _shape_param_510 = None
        convert_element_type_557: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_509, torch.float32);  view_509 = None
        mul_501: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_557, arg462_1);  convert_element_type_557 = None
        sum_242: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [-1], True)
        neg_19: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg462_1);  arg462_1 = None
        fma_19: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_19, sum_242, mul_501);  neg_19 = sum_242 = mul_501 = None
        convert_element_type_558: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None
        slice_39: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg463_1, 2, 0, 128);  arg463_1 = None
        slice_40: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_39, 3, 0, 128);  slice_39 = None
        where_21: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_40, convert_element_type_558, full_2);  slice_40 = convert_element_type_558 = None
        view_510: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_21, _shape_param_511);  where_21 = _shape_param_511 = None
        bmm_78: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg464_1, view_510);  arg464_1 = None
        bmm_79: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_510, arg465_1);  view_510 = arg465_1 = None
        view_511: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_78, _shape_param_512);  bmm_78 = _shape_param_512 = None
        view_512: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_79, _shape_param_513);  bmm_79 = _shape_param_513 = None
        convert_element_type_559: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_511, torch.float32);  view_511 = None
        permute_214: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_559, [0, 1, 3, 2]);  convert_element_type_559 = None
        convert_element_type_560: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_214, torch.bfloat16);  permute_214 = None
        convert_element_type_561: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_512, torch.bfloat16);  view_512 = None
        permute_215: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
        clone_77: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_513: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_77, _shape_param_514);  clone_77 = _shape_param_514 = None
        permute_216: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_560, [0, 2, 1, 3]);  convert_element_type_560 = None
        view_514: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_216, _shape_param_515);  permute_216 = _shape_param_515 = None
        permute_217: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_561, [0, 2, 1, 3]);  convert_element_type_561 = None
        clone_78: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
        view_515: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_78, _shape_param_516);  clone_78 = _shape_param_516 = None
        view_516: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_513, _shape_param_517);  view_513 = _shape_param_517 = None
        permute_218: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_516, [1, 0])
        mm_236: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_218, arg466_1);  permute_218 = None
        mm_237: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_516, arg467_1);  view_516 = arg467_1 = None
        view_517: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_237, _shape_param_518);  mm_237 = _shape_param_518 = None
        convert_element_type_562: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_517, torch.float32);  view_517 = None
        convert_element_type_563: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_236, torch.float32);  mm_236 = None
        clone_79: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_514, memory_format = torch.contiguous_format);  view_514 = None
        view_518: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_79, _shape_param_519);  clone_79 = _shape_param_519 = None
        permute_219: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_518, [1, 0])
        mm_238: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_219, arg466_1);  permute_219 = None
        mm_239: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_518, arg468_1);  view_518 = arg468_1 = None
        view_519: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_239, _shape_param_520);  mm_239 = _shape_param_520 = None
        convert_element_type_564: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_519, torch.float32);  view_519 = None
        add_158: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_562, convert_element_type_564);  convert_element_type_562 = convert_element_type_564 = None
        convert_element_type_565: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_238, torch.float32);  mm_238 = None
        view_520: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_515, _shape_param_521);  view_515 = _shape_param_521 = None
        permute_220: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_240: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_220, arg466_1);  permute_220 = arg466_1 = None
        mm_241: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_520, arg469_1);  view_520 = arg469_1 = None
        view_521: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_241, _shape_param_522);  mm_241 = _shape_param_522 = None
        convert_element_type_566: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_521, torch.float32);  view_521 = None
        add_159: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_158, convert_element_type_566);  add_158 = convert_element_type_566 = None
        convert_element_type_567: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_240, torch.float32);  mm_240 = None
        mul_502: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_159, arg470_1);  arg470_1 = None
        mul_503: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_502, 2048)
        sum_243: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [2], True)
        mul_504: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_502, arg471_1);  mul_502 = None
        sum_244: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_504, [2], True);  mul_504 = None
        mul_505: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg471_1, sum_244);  sum_244 = None
        sub_103: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_503, sum_243);  mul_503 = sum_243 = None
        sub_104: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_103, mul_505);  sub_103 = mul_505 = None
        mul_506: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg472_1, sub_104);  arg472_1 = sub_104 = None
        mul_507: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_159, arg471_1);  arg471_1 = None
        sum_245: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 1]);  mul_507 = None
        sum_246: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
        add_160: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_157, mul_506);  add_157 = mul_506 = None
        convert_element_type_568: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16)
        view_522: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_568, _shape_param_523);  convert_element_type_568 = _shape_param_523 = None
        mm_242: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_522, arg473_1);  arg473_1 = None
        permute_221: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_522, [1, 0])
        mm_243: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_221, arg474_1);  permute_221 = arg474_1 = None
        sum_247: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True, dtype = torch.float32);  view_522 = None
        view_523: "f32[2048]" = torch.ops.aten.view.default(sum_247, _shape_param_524);  sum_247 = _shape_param_524 = None
        convert_element_type_569: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_523, torch.bfloat16);  view_523 = None
        view_524: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_242, _shape_param_525);  mm_242 = _shape_param_525 = None
        convert_element_type_570: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_524, torch.float32);  view_524 = None
        convert_element_type_571: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_243, torch.float32);  mm_243 = None
        convert_element_type_572: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_569, torch.float32);  convert_element_type_569 = None
        view_525: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg475_1, _shape_param_526);  arg475_1 = _shape_param_526 = None
        mul_508: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_525, 0.5)
        mul_509: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_570, mul_508);  mul_508 = None
        convert_element_type_573: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_525, torch.float32)
        pow_41: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_573, 3.0)
        mul_510: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_161: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_525, mul_510);  view_525 = mul_510 = None
        mul_511: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_20: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_511);  mul_511 = None
        add_162: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_512: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_570, add_162);  convert_element_type_570 = add_162 = None
        convert_element_type_574: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_512, torch.bfloat16);  mul_512 = None
        mul_513: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_105: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_513);  mul_513 = None
        mul_514: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_509, sub_105);  mul_509 = sub_105 = None
        mul_515: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_514, 0.7978845608028654);  mul_514 = None
        convert_element_type_575: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_515, torch.bfloat16)
        mul_516: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_515, 0.044715);  mul_515 = None
        pow_42: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_573, 2.0);  convert_element_type_573 = None
        mul_517: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_518: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_516, mul_517);  mul_516 = mul_517 = None
        convert_element_type_576: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_518, torch.bfloat16);  mul_518 = None
        add_163: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_575, convert_element_type_576);  convert_element_type_575 = convert_element_type_576 = None
        mul_519: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_574, 0.5);  convert_element_type_574 = None
        add_164: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_163, mul_519);  add_163 = mul_519 = None
        view_526: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_164, _shape_param_527);  add_164 = _shape_param_527 = None
        mm_244: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_526, arg476_1);  arg476_1 = None
        permute_222: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_526, [1, 0])
        mm_245: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_222, arg477_1);  permute_222 = arg477_1 = None
        sum_248: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True, dtype = torch.float32);  view_526 = None
        view_527: "f32[8192]" = torch.ops.aten.view.default(sum_248, _shape_param_528);  sum_248 = _shape_param_528 = None
        convert_element_type_577: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_527, torch.bfloat16);  view_527 = None
        view_528: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_244, _shape_param_529);  mm_244 = _shape_param_529 = None
        convert_element_type_578: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_528, torch.float32);  view_528 = None
        convert_element_type_579: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_245, torch.float32);  mm_245 = None
        convert_element_type_580: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_577, torch.float32);  convert_element_type_577 = None
        mul_520: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_578, arg478_1);  arg478_1 = None
        mul_521: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_520, 2048)
        sum_249: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_520, [2], True)
        mul_522: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_520, arg479_1);  mul_520 = None
        sum_250: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True);  mul_522 = None
        mul_523: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg479_1, sum_250);  sum_250 = None
        sub_106: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_521, sum_249);  mul_521 = sum_249 = None
        sub_107: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_106, mul_523);  sub_106 = mul_523 = None
        mul_524: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg480_1, sub_107);  arg480_1 = sub_107 = None
        mul_525: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_578, arg479_1);  arg479_1 = None
        sum_251: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_525, [0, 1]);  mul_525 = None
        sum_252: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_578, [0, 1]);  convert_element_type_578 = None
        add_165: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_160, mul_524);  add_160 = mul_524 = None
        convert_element_type_581: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_165, torch.bfloat16)
        view_529: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_581, _shape_param_530);  convert_element_type_581 = _shape_param_530 = None
        mm_246: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_529, arg481_1);  arg481_1 = None
        permute_223: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_247: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_223, arg482_1);  permute_223 = arg482_1 = None
        sum_253: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_529, [0], True, dtype = torch.float32);  view_529 = None
        view_530: "f32[2048]" = torch.ops.aten.view.default(sum_253, _shape_param_531);  sum_253 = _shape_param_531 = None
        convert_element_type_582: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_530, torch.bfloat16);  view_530 = None
        view_531: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_246, _shape_param_532);  mm_246 = _shape_param_532 = None
        convert_element_type_583: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_247, torch.float32);  mm_247 = None
        convert_element_type_584: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_582, torch.float32);  convert_element_type_582 = None
        view_532: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_531, _shape_param_533);  view_531 = _shape_param_533 = None
        permute_224: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_532, [0, 2, 1, 3]);  view_532 = None
        clone_80: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        view_533: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_80, _shape_param_534);  clone_80 = _shape_param_534 = None
        bmm_80: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg483_1, view_533);  arg483_1 = None
        bmm_81: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_533, arg484_1);  view_533 = arg484_1 = None
        view_534: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_80, _shape_param_535);  bmm_80 = _shape_param_535 = None
        view_535: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_81, _shape_param_536);  bmm_81 = _shape_param_536 = None
        convert_element_type_585: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_535, torch.float32);  view_535 = None
        mul_526: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_585, arg485_1);  convert_element_type_585 = None
        sum_254: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_526, [-1], True)
        neg_20: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg485_1);  arg485_1 = None
        fma_20: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_20, sum_254, mul_526);  neg_20 = sum_254 = mul_526 = None
        convert_element_type_586: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None
        slice_41: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg486_1, 2, 0, 128);  arg486_1 = None
        slice_42: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_41, 3, 0, 128);  slice_41 = None
        where_22: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_42, convert_element_type_586, full_2);  slice_42 = convert_element_type_586 = None
        view_536: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_22, _shape_param_537);  where_22 = _shape_param_537 = None
        bmm_82: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg487_1, view_536);  arg487_1 = None
        bmm_83: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_536, arg488_1);  view_536 = arg488_1 = None
        view_537: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_82, _shape_param_538);  bmm_82 = _shape_param_538 = None
        view_538: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_83, _shape_param_539);  bmm_83 = _shape_param_539 = None
        convert_element_type_587: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_537, torch.float32);  view_537 = None
        permute_225: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_587, [0, 1, 3, 2]);  convert_element_type_587 = None
        convert_element_type_588: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_225, torch.bfloat16);  permute_225 = None
        convert_element_type_589: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_538, torch.bfloat16);  view_538 = None
        permute_226: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
        clone_81: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_539: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_81, _shape_param_540);  clone_81 = _shape_param_540 = None
        permute_227: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_588, [0, 2, 1, 3]);  convert_element_type_588 = None
        view_540: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_227, _shape_param_541);  permute_227 = _shape_param_541 = None
        permute_228: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_589, [0, 2, 1, 3]);  convert_element_type_589 = None
        clone_82: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
        view_541: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_82, _shape_param_542);  clone_82 = _shape_param_542 = None
        view_542: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_539, _shape_param_543);  view_539 = _shape_param_543 = None
        permute_229: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_542, [1, 0])
        mm_248: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_229, arg489_1);  permute_229 = None
        mm_249: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_542, arg490_1);  view_542 = arg490_1 = None
        view_543: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_249, _shape_param_544);  mm_249 = _shape_param_544 = None
        convert_element_type_590: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_543, torch.float32);  view_543 = None
        convert_element_type_591: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_248, torch.float32);  mm_248 = None
        clone_83: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_540, memory_format = torch.contiguous_format);  view_540 = None
        view_544: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_83, _shape_param_545);  clone_83 = _shape_param_545 = None
        permute_230: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_544, [1, 0])
        mm_250: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_230, arg489_1);  permute_230 = None
        mm_251: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_544, arg491_1);  view_544 = arg491_1 = None
        view_545: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_251, _shape_param_546);  mm_251 = _shape_param_546 = None
        convert_element_type_592: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_545, torch.float32);  view_545 = None
        add_166: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_590, convert_element_type_592);  convert_element_type_590 = convert_element_type_592 = None
        convert_element_type_593: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_250, torch.float32);  mm_250 = None
        view_546: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_541, _shape_param_547);  view_541 = _shape_param_547 = None
        permute_231: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_546, [1, 0])
        mm_252: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_231, arg489_1);  permute_231 = arg489_1 = None
        mm_253: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_546, arg492_1);  view_546 = arg492_1 = None
        view_547: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_253, _shape_param_548);  mm_253 = _shape_param_548 = None
        convert_element_type_594: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_547, torch.float32);  view_547 = None
        add_167: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_166, convert_element_type_594);  add_166 = convert_element_type_594 = None
        convert_element_type_595: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_252, torch.float32);  mm_252 = None
        mul_527: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_167, arg493_1);  arg493_1 = None
        mul_528: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_527, 2048)
        sum_255: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True)
        mul_529: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_527, arg494_1);  mul_527 = None
        sum_256: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [2], True);  mul_529 = None
        mul_530: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg494_1, sum_256);  sum_256 = None
        sub_108: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_528, sum_255);  mul_528 = sum_255 = None
        sub_109: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_108, mul_530);  sub_108 = mul_530 = None
        mul_531: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg495_1, sub_109);  arg495_1 = sub_109 = None
        mul_532: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_167, arg494_1);  arg494_1 = None
        sum_257: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 1]);  mul_532 = None
        sum_258: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
        add_168: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_165, mul_531);  add_165 = mul_531 = None
        convert_element_type_596: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_168, torch.bfloat16)
        view_548: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_596, _shape_param_549);  convert_element_type_596 = _shape_param_549 = None
        mm_254: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_548, arg496_1);  arg496_1 = None
        permute_232: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_255: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_232, arg497_1);  permute_232 = arg497_1 = None
        sum_259: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True, dtype = torch.float32);  view_548 = None
        view_549: "f32[2048]" = torch.ops.aten.view.default(sum_259, _shape_param_550);  sum_259 = _shape_param_550 = None
        convert_element_type_597: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_549, torch.bfloat16);  view_549 = None
        view_550: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_254, _shape_param_551);  mm_254 = _shape_param_551 = None
        convert_element_type_598: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_550, torch.float32);  view_550 = None
        convert_element_type_599: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_255, torch.float32);  mm_255 = None
        convert_element_type_600: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_597, torch.float32);  convert_element_type_597 = None
        view_551: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg498_1, _shape_param_552);  arg498_1 = _shape_param_552 = None
        mul_533: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_551, 0.5)
        mul_534: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_598, mul_533);  mul_533 = None
        convert_element_type_601: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_551, torch.float32)
        pow_43: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_601, 3.0)
        mul_535: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_43, 0.044715);  pow_43 = None
        add_169: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_551, mul_535);  view_551 = mul_535 = None
        mul_536: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_169, 0.7978845608028654);  add_169 = None
        tanh_21: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_536);  mul_536 = None
        add_170: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_537: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_598, add_170);  convert_element_type_598 = add_170 = None
        convert_element_type_602: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_537, torch.bfloat16);  mul_537 = None
        mul_538: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_110: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_538);  mul_538 = None
        mul_539: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_534, sub_110);  mul_534 = sub_110 = None
        mul_540: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_539, 0.7978845608028654);  mul_539 = None
        convert_element_type_603: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_540, torch.bfloat16)
        mul_541: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_540, 0.044715);  mul_540 = None
        pow_44: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_601, 2.0);  convert_element_type_601 = None
        mul_542: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_543: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_541, mul_542);  mul_541 = mul_542 = None
        convert_element_type_604: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_543, torch.bfloat16);  mul_543 = None
        add_171: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_603, convert_element_type_604);  convert_element_type_603 = convert_element_type_604 = None
        mul_544: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_602, 0.5);  convert_element_type_602 = None
        add_172: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_171, mul_544);  add_171 = mul_544 = None
        view_552: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_172, _shape_param_553);  add_172 = _shape_param_553 = None
        mm_256: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_552, arg499_1);  arg499_1 = None
        permute_233: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_552, [1, 0])
        mm_257: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_233, arg500_1);  permute_233 = arg500_1 = None
        sum_260: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True, dtype = torch.float32);  view_552 = None
        view_553: "f32[8192]" = torch.ops.aten.view.default(sum_260, _shape_param_554);  sum_260 = _shape_param_554 = None
        convert_element_type_605: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_553, torch.bfloat16);  view_553 = None
        view_554: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_256, _shape_param_555);  mm_256 = _shape_param_555 = None
        convert_element_type_606: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_554, torch.float32);  view_554 = None
        convert_element_type_607: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_257, torch.float32);  mm_257 = None
        convert_element_type_608: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_605, torch.float32);  convert_element_type_605 = None
        mul_545: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_606, arg501_1);  arg501_1 = None
        mul_546: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_545, 2048)
        sum_261: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_545, [2], True)
        mul_547: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_545, arg502_1);  mul_545 = None
        sum_262: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_547, [2], True);  mul_547 = None
        mul_548: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg502_1, sum_262);  sum_262 = None
        sub_111: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_546, sum_261);  mul_546 = sum_261 = None
        sub_112: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_111, mul_548);  sub_111 = mul_548 = None
        mul_549: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg503_1, sub_112);  arg503_1 = sub_112 = None
        mul_550: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_606, arg502_1);  arg502_1 = None
        sum_263: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 1]);  mul_550 = None
        sum_264: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_606, [0, 1]);  convert_element_type_606 = None
        add_173: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_168, mul_549);  add_168 = mul_549 = None
        convert_element_type_609: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16)
        view_555: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_609, _shape_param_556);  convert_element_type_609 = _shape_param_556 = None
        mm_258: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_555, arg504_1);  arg504_1 = None
        permute_234: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_555, [1, 0])
        mm_259: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_234, arg505_1);  permute_234 = arg505_1 = None
        sum_265: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True, dtype = torch.float32);  view_555 = None
        view_556: "f32[2048]" = torch.ops.aten.view.default(sum_265, _shape_param_557);  sum_265 = _shape_param_557 = None
        convert_element_type_610: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_556, torch.bfloat16);  view_556 = None
        view_557: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_258, _shape_param_558);  mm_258 = _shape_param_558 = None
        convert_element_type_611: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_259, torch.float32);  mm_259 = None
        convert_element_type_612: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_610, torch.float32);  convert_element_type_610 = None
        view_558: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_557, _shape_param_559);  view_557 = _shape_param_559 = None
        permute_235: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
        clone_84: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
        view_559: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_84, _shape_param_560);  clone_84 = _shape_param_560 = None
        bmm_84: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg506_1, view_559);  arg506_1 = None
        bmm_85: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_559, arg507_1);  view_559 = arg507_1 = None
        view_560: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_84, _shape_param_561);  bmm_84 = _shape_param_561 = None
        view_561: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_85, _shape_param_562);  bmm_85 = _shape_param_562 = None
        convert_element_type_613: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_561, torch.float32);  view_561 = None
        mul_551: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_613, arg508_1);  convert_element_type_613 = None
        sum_266: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [-1], True)
        neg_21: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg508_1);  arg508_1 = None
        fma_21: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_21, sum_266, mul_551);  neg_21 = sum_266 = mul_551 = None
        convert_element_type_614: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None
        slice_43: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg509_1, 2, 0, 128);  arg509_1 = None
        slice_44: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_43, 3, 0, 128);  slice_43 = None
        where_23: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_44, convert_element_type_614, full_2);  slice_44 = convert_element_type_614 = None
        view_562: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_23, _shape_param_563);  where_23 = _shape_param_563 = None
        bmm_86: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg510_1, view_562);  arg510_1 = None
        bmm_87: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_562, arg511_1);  view_562 = arg511_1 = None
        view_563: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_86, _shape_param_564);  bmm_86 = _shape_param_564 = None
        view_564: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_87, _shape_param_565);  bmm_87 = _shape_param_565 = None
        convert_element_type_615: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_563, torch.float32);  view_563 = None
        permute_236: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_615, [0, 1, 3, 2]);  convert_element_type_615 = None
        convert_element_type_616: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_236, torch.bfloat16);  permute_236 = None
        convert_element_type_617: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_564, torch.bfloat16);  view_564 = None
        permute_237: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
        clone_85: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_565: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_85, _shape_param_566);  clone_85 = _shape_param_566 = None
        permute_238: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_616, [0, 2, 1, 3]);  convert_element_type_616 = None
        view_566: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_238, _shape_param_567);  permute_238 = _shape_param_567 = None
        permute_239: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_617, [0, 2, 1, 3]);  convert_element_type_617 = None
        clone_86: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_567: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_86, _shape_param_568);  clone_86 = _shape_param_568 = None
        view_568: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_565, _shape_param_569);  view_565 = _shape_param_569 = None
        permute_240: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_260: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_240, arg512_1);  permute_240 = None
        mm_261: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_568, arg513_1);  view_568 = arg513_1 = None
        view_569: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_261, _shape_param_570);  mm_261 = _shape_param_570 = None
        convert_element_type_618: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_569, torch.float32);  view_569 = None
        convert_element_type_619: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_260, torch.float32);  mm_260 = None
        clone_87: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_566, memory_format = torch.contiguous_format);  view_566 = None
        view_570: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_87, _shape_param_571);  clone_87 = _shape_param_571 = None
        permute_241: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_570, [1, 0])
        mm_262: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_241, arg512_1);  permute_241 = None
        mm_263: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_570, arg514_1);  view_570 = arg514_1 = None
        view_571: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_263, _shape_param_572);  mm_263 = _shape_param_572 = None
        convert_element_type_620: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_571, torch.float32);  view_571 = None
        add_174: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_618, convert_element_type_620);  convert_element_type_618 = convert_element_type_620 = None
        convert_element_type_621: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_262, torch.float32);  mm_262 = None
        view_572: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_567, _shape_param_573);  view_567 = _shape_param_573 = None
        permute_242: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_572, [1, 0])
        mm_264: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_242, arg512_1);  permute_242 = arg512_1 = None
        mm_265: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_572, arg515_1);  view_572 = arg515_1 = None
        view_573: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_265, _shape_param_574);  mm_265 = _shape_param_574 = None
        convert_element_type_622: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_573, torch.float32);  view_573 = None
        add_175: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_174, convert_element_type_622);  add_174 = convert_element_type_622 = None
        convert_element_type_623: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_264, torch.float32);  mm_264 = None
        mul_552: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_175, arg516_1);  arg516_1 = None
        mul_553: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_552, 2048)
        sum_267: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_552, [2], True)
        mul_554: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_552, arg517_1);  mul_552 = None
        sum_268: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True);  mul_554 = None
        mul_555: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg517_1, sum_268);  sum_268 = None
        sub_113: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_553, sum_267);  mul_553 = sum_267 = None
        sub_114: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_113, mul_555);  sub_113 = mul_555 = None
        mul_556: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg518_1, sub_114);  arg518_1 = sub_114 = None
        mul_557: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_175, arg517_1);  arg517_1 = None
        sum_269: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 1]);  mul_557 = None
        sum_270: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_175, [0, 1]);  add_175 = None
        add_176: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_173, mul_556);  add_173 = mul_556 = None
        convert_element_type_624: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16)
        view_574: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_624, _shape_param_575);  convert_element_type_624 = _shape_param_575 = None
        mm_266: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_574, arg519_1);  arg519_1 = None
        permute_243: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_574, [1, 0])
        mm_267: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_243, arg520_1);  permute_243 = arg520_1 = None
        sum_271: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_574, [0], True, dtype = torch.float32);  view_574 = None
        view_575: "f32[2048]" = torch.ops.aten.view.default(sum_271, _shape_param_576);  sum_271 = _shape_param_576 = None
        convert_element_type_625: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_575, torch.bfloat16);  view_575 = None
        view_576: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_266, _shape_param_577);  mm_266 = _shape_param_577 = None
        convert_element_type_626: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_576, torch.float32);  view_576 = None
        convert_element_type_627: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_267, torch.float32);  mm_267 = None
        convert_element_type_628: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_625, torch.float32);  convert_element_type_625 = None
        view_577: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg521_1, _shape_param_578);  arg521_1 = _shape_param_578 = None
        mul_558: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_577, 0.5)
        mul_559: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_626, mul_558);  mul_558 = None
        convert_element_type_629: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_577, torch.float32)
        pow_45: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_629, 3.0)
        mul_560: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_177: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_577, mul_560);  view_577 = mul_560 = None
        mul_561: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_177, 0.7978845608028654);  add_177 = None
        tanh_22: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_561);  mul_561 = None
        add_178: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_562: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_626, add_178);  convert_element_type_626 = add_178 = None
        convert_element_type_630: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_562, torch.bfloat16);  mul_562 = None
        mul_563: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_115: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_563);  mul_563 = None
        mul_564: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_559, sub_115);  mul_559 = sub_115 = None
        mul_565: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_564, 0.7978845608028654);  mul_564 = None
        convert_element_type_631: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_565, torch.bfloat16)
        mul_566: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_565, 0.044715);  mul_565 = None
        pow_46: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_629, 2.0);  convert_element_type_629 = None
        mul_567: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_568: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
        convert_element_type_632: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_568, torch.bfloat16);  mul_568 = None
        add_179: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_631, convert_element_type_632);  convert_element_type_631 = convert_element_type_632 = None
        mul_569: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_630, 0.5);  convert_element_type_630 = None
        add_180: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_179, mul_569);  add_179 = mul_569 = None
        view_578: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_180, _shape_param_579);  add_180 = _shape_param_579 = None
        mm_268: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_578, arg522_1);  arg522_1 = None
        permute_244: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_578, [1, 0])
        mm_269: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_244, arg523_1);  permute_244 = arg523_1 = None
        sum_272: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_578, [0], True, dtype = torch.float32);  view_578 = None
        view_579: "f32[8192]" = torch.ops.aten.view.default(sum_272, _shape_param_580);  sum_272 = _shape_param_580 = None
        convert_element_type_633: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_579, torch.bfloat16);  view_579 = None
        view_580: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_268, _shape_param_581);  mm_268 = _shape_param_581 = None
        convert_element_type_634: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_580, torch.float32);  view_580 = None
        convert_element_type_635: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_269, torch.float32);  mm_269 = None
        convert_element_type_636: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_633, torch.float32);  convert_element_type_633 = None
        mul_570: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_634, arg524_1);  arg524_1 = None
        mul_571: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_570, 2048)
        sum_273: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [2], True)
        mul_572: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_570, arg525_1);  mul_570 = None
        sum_274: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [2], True);  mul_572 = None
        mul_573: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg525_1, sum_274);  sum_274 = None
        sub_116: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_571, sum_273);  mul_571 = sum_273 = None
        sub_117: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_116, mul_573);  sub_116 = mul_573 = None
        mul_574: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg526_1, sub_117);  arg526_1 = sub_117 = None
        mul_575: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_634, arg525_1);  arg525_1 = None
        sum_275: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 1]);  mul_575 = None
        sum_276: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_634, [0, 1]);  convert_element_type_634 = None
        add_181: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_176, mul_574);  add_176 = mul_574 = None
        convert_element_type_637: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_181, torch.bfloat16)
        view_581: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_637, _shape_param_582);  convert_element_type_637 = _shape_param_582 = None
        mm_270: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_581, arg527_1);  arg527_1 = None
        permute_245: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_581, [1, 0])
        mm_271: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_245, arg528_1);  permute_245 = arg528_1 = None
        sum_277: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_581, [0], True, dtype = torch.float32);  view_581 = None
        view_582: "f32[2048]" = torch.ops.aten.view.default(sum_277, _shape_param_583);  sum_277 = _shape_param_583 = None
        convert_element_type_638: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_582, torch.bfloat16);  view_582 = None
        view_583: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_270, _shape_param_584);  mm_270 = _shape_param_584 = None
        convert_element_type_639: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_271, torch.float32);  mm_271 = None
        convert_element_type_640: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_638, torch.float32);  convert_element_type_638 = None
        view_584: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_583, _shape_param_585);  view_583 = _shape_param_585 = None
        permute_246: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_584, [0, 2, 1, 3]);  view_584 = None
        clone_88: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        view_585: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_88, _shape_param_586);  clone_88 = _shape_param_586 = None
        bmm_88: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg529_1, view_585);  arg529_1 = None
        bmm_89: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_585, arg530_1);  view_585 = arg530_1 = None
        view_586: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_88, _shape_param_587);  bmm_88 = _shape_param_587 = None
        view_587: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_89, _shape_param_588);  bmm_89 = _shape_param_588 = None
        convert_element_type_641: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_587, torch.float32);  view_587 = None
        mul_576: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_641, arg531_1);  convert_element_type_641 = None
        sum_278: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_576, [-1], True)
        neg_22: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg531_1);  arg531_1 = None
        fma_22: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_22, sum_278, mul_576);  neg_22 = sum_278 = mul_576 = None
        convert_element_type_642: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None
        slice_45: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg532_1, 2, 0, 128);  arg532_1 = None
        slice_46: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_45, 3, 0, 128);  slice_45 = None
        where_24: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_46, convert_element_type_642, full_2);  slice_46 = convert_element_type_642 = None
        view_588: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_24, _shape_param_589);  where_24 = _shape_param_589 = None
        bmm_90: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg533_1, view_588);  arg533_1 = None
        bmm_91: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_588, arg534_1);  view_588 = arg534_1 = None
        view_589: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_90, _shape_param_590);  bmm_90 = _shape_param_590 = None
        view_590: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_91, _shape_param_591);  bmm_91 = _shape_param_591 = None
        convert_element_type_643: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_589, torch.float32);  view_589 = None
        permute_247: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_643, [0, 1, 3, 2]);  convert_element_type_643 = None
        convert_element_type_644: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_247, torch.bfloat16);  permute_247 = None
        convert_element_type_645: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_590, torch.bfloat16);  view_590 = None
        permute_248: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_586, [0, 2, 1, 3]);  view_586 = None
        clone_89: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_591: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_89, _shape_param_592);  clone_89 = _shape_param_592 = None
        permute_249: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_644, [0, 2, 1, 3]);  convert_element_type_644 = None
        view_592: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_249, _shape_param_593);  permute_249 = _shape_param_593 = None
        permute_250: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_645, [0, 2, 1, 3]);  convert_element_type_645 = None
        clone_90: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_593: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_90, _shape_param_594);  clone_90 = _shape_param_594 = None
        view_594: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_591, _shape_param_595);  view_591 = _shape_param_595 = None
        permute_251: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_594, [1, 0])
        mm_272: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_251, arg535_1);  permute_251 = None
        mm_273: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_594, arg536_1);  view_594 = arg536_1 = None
        view_595: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_273, _shape_param_596);  mm_273 = _shape_param_596 = None
        convert_element_type_646: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_595, torch.float32);  view_595 = None
        convert_element_type_647: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_272, torch.float32);  mm_272 = None
        clone_91: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_592, memory_format = torch.contiguous_format);  view_592 = None
        view_596: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_91, _shape_param_597);  clone_91 = _shape_param_597 = None
        permute_252: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_596, [1, 0])
        mm_274: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_252, arg535_1);  permute_252 = None
        mm_275: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_596, arg537_1);  view_596 = arg537_1 = None
        view_597: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_275, _shape_param_598);  mm_275 = _shape_param_598 = None
        convert_element_type_648: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_597, torch.float32);  view_597 = None
        add_182: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_646, convert_element_type_648);  convert_element_type_646 = convert_element_type_648 = None
        convert_element_type_649: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_274, torch.float32);  mm_274 = None
        view_598: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_593, _shape_param_599);  view_593 = _shape_param_599 = None
        permute_253: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_276: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_253, arg535_1);  permute_253 = arg535_1 = None
        mm_277: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_598, arg538_1);  view_598 = arg538_1 = None
        view_599: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_277, _shape_param_600);  mm_277 = _shape_param_600 = None
        convert_element_type_650: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_599, torch.float32);  view_599 = None
        add_183: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_182, convert_element_type_650);  add_182 = convert_element_type_650 = None
        convert_element_type_651: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_276, torch.float32);  mm_276 = None
        mul_577: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_183, arg539_1);  arg539_1 = None
        mul_578: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_577, 2048)
        sum_279: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_577, [2], True)
        mul_579: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_577, arg540_1);  mul_577 = None
        sum_280: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True);  mul_579 = None
        mul_580: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg540_1, sum_280);  sum_280 = None
        sub_118: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_578, sum_279);  mul_578 = sum_279 = None
        sub_119: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_118, mul_580);  sub_118 = mul_580 = None
        mul_581: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg541_1, sub_119);  arg541_1 = sub_119 = None
        mul_582: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_183, arg540_1);  arg540_1 = None
        sum_281: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_582, [0, 1]);  mul_582 = None
        sum_282: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_183, [0, 1]);  add_183 = None
        add_184: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_181, mul_581);  add_181 = mul_581 = None
        convert_element_type_652: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16)
        view_600: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_652, _shape_param_601);  convert_element_type_652 = _shape_param_601 = None
        mm_278: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_600, arg542_1);  arg542_1 = None
        permute_254: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_279: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_254, arg543_1);  permute_254 = arg543_1 = None
        sum_283: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True, dtype = torch.float32);  view_600 = None
        view_601: "f32[2048]" = torch.ops.aten.view.default(sum_283, _shape_param_602);  sum_283 = _shape_param_602 = None
        convert_element_type_653: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_601, torch.bfloat16);  view_601 = None
        view_602: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_278, _shape_param_603);  mm_278 = _shape_param_603 = None
        convert_element_type_654: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_602, torch.float32);  view_602 = None
        convert_element_type_655: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None
        convert_element_type_656: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_653, torch.float32);  convert_element_type_653 = None
        view_603: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg544_1, _shape_param_604);  arg544_1 = _shape_param_604 = None
        mul_583: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_603, 0.5)
        mul_584: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_654, mul_583);  mul_583 = None
        convert_element_type_657: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_603, torch.float32)
        pow_47: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_657, 3.0)
        mul_585: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_47, 0.044715);  pow_47 = None
        add_185: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_603, mul_585);  view_603 = mul_585 = None
        mul_586: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_185, 0.7978845608028654);  add_185 = None
        tanh_23: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_586);  mul_586 = None
        add_186: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_587: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_654, add_186);  convert_element_type_654 = add_186 = None
        convert_element_type_658: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_587, torch.bfloat16);  mul_587 = None
        mul_588: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_120: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_588);  mul_588 = None
        mul_589: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_584, sub_120);  mul_584 = sub_120 = None
        mul_590: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_589, 0.7978845608028654);  mul_589 = None
        convert_element_type_659: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_590, torch.bfloat16)
        mul_591: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_590, 0.044715);  mul_590 = None
        pow_48: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_657, 2.0);  convert_element_type_657 = None
        mul_592: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_593: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_591, mul_592);  mul_591 = mul_592 = None
        convert_element_type_660: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_593, torch.bfloat16);  mul_593 = None
        add_187: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_659, convert_element_type_660);  convert_element_type_659 = convert_element_type_660 = None
        mul_594: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_658, 0.5);  convert_element_type_658 = None
        add_188: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_187, mul_594);  add_187 = mul_594 = None
        view_604: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_188, _shape_param_605);  add_188 = _shape_param_605 = None
        mm_280: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_604, arg545_1);  arg545_1 = None
        permute_255: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_604, [1, 0])
        mm_281: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_255, arg546_1);  permute_255 = arg546_1 = None
        sum_284: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True, dtype = torch.float32);  view_604 = None
        view_605: "f32[8192]" = torch.ops.aten.view.default(sum_284, _shape_param_606);  sum_284 = _shape_param_606 = None
        convert_element_type_661: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_605, torch.bfloat16);  view_605 = None
        view_606: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_280, _shape_param_607);  mm_280 = _shape_param_607 = None
        convert_element_type_662: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_606, torch.float32);  view_606 = None
        convert_element_type_663: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_281, torch.float32);  mm_281 = None
        convert_element_type_664: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_661, torch.float32);  convert_element_type_661 = None
        mul_595: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_662, arg547_1);  arg547_1 = None
        mul_596: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_595, 2048)
        sum_285: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [2], True)
        mul_597: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_595, arg548_1);  mul_595 = None
        sum_286: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [2], True);  mul_597 = None
        mul_598: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg548_1, sum_286);  sum_286 = None
        sub_121: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_596, sum_285);  mul_596 = sum_285 = None
        sub_122: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_121, mul_598);  sub_121 = mul_598 = None
        mul_599: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg549_1, sub_122);  arg549_1 = sub_122 = None
        mul_600: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_662, arg548_1);  arg548_1 = None
        sum_287: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 1]);  mul_600 = None
        sum_288: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_662, [0, 1]);  convert_element_type_662 = None
        add_189: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_184, mul_599);  add_184 = mul_599 = None
        convert_element_type_665: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16)
        view_607: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_665, _shape_param_608);  convert_element_type_665 = _shape_param_608 = None
        mm_282: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_607, arg550_1);  arg550_1 = None
        permute_256: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_607, [1, 0])
        mm_283: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_256, arg551_1);  permute_256 = arg551_1 = None
        sum_289: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True, dtype = torch.float32);  view_607 = None
        view_608: "f32[2048]" = torch.ops.aten.view.default(sum_289, _shape_param_609);  sum_289 = _shape_param_609 = None
        convert_element_type_666: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_608, torch.bfloat16);  view_608 = None
        view_609: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_282, _shape_param_610);  mm_282 = _shape_param_610 = None
        convert_element_type_667: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_283, torch.float32);  mm_283 = None
        convert_element_type_668: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_666, torch.float32);  convert_element_type_666 = None
        view_610: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_609, _shape_param_611);  view_609 = _shape_param_611 = None
        permute_257: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None
        clone_92: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
        view_611: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_92, _shape_param_612);  clone_92 = _shape_param_612 = None
        bmm_92: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg552_1, view_611);  arg552_1 = None
        bmm_93: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_611, arg553_1);  view_611 = arg553_1 = None
        view_612: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_92, _shape_param_613);  bmm_92 = _shape_param_613 = None
        view_613: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_93, _shape_param_614);  bmm_93 = _shape_param_614 = None
        convert_element_type_669: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_613, torch.float32);  view_613 = None
        unsqueeze: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(arg6_1, 1);  arg6_1 = None
        unsqueeze_1: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, 2);  unsqueeze = None
        unsqueeze_2: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        unsqueeze_3: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(arg554_1, 1)
        unsqueeze_4: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 3)
        unsqueeze_5: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
        full_3: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_4)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full_3, le);  full_3 = le = None
        index: "i64[32, 1, 128, 1]" = torch.ops.aten.index.Tensor(arg555_1, [unsqueeze_2, unsqueeze_4]);  unsqueeze_4 = None
        index_1: "i64[32, 1, 1, 128]" = torch.ops.aten.index.Tensor(arg555_1, [unsqueeze_2, unsqueeze_5]);  arg555_1 = unsqueeze_2 = unsqueeze_5 = None
        eq: "b8[32, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[32, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq);  bitwise_and = eq = None
        expand: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_615);  bitwise_and_1 = _shape_param_615 = None
        full_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand, full, full_4);  expand = full = full_4 = None
        view_614: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg556_1, _shape_param_616);  arg556_1 = _shape_param_616 = None
        slice_47: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg557_1, 2, 0, 128);  arg557_1 = None
        slice_48: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 128);  slice_47 = None
        full_5: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, view_614, full_5);  view_614 = full_5 = None
        add_190: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_26, where_25);  where_26 = where_25 = None
        sub_123: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_190, arg558_1);  add_190 = arg558_1 = None
        exp_2: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_123);  sub_123 = None
        div_1: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, arg559_1);  exp_2 = arg559_1 = None
        mul_601: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_669, div_1);  convert_element_type_669 = None
        sum_290: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [-1], True)
        neg_23: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_23: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_23, sum_290, mul_601);  neg_23 = sum_290 = mul_601 = None
        convert_element_type_670: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None
        where_27: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, convert_element_type_670, full_2);  slice_48 = convert_element_type_670 = full_2 = None
        view_615: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_27, _shape_param_617);  where_27 = _shape_param_617 = None
        bmm_94: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg560_1, view_615);  arg560_1 = None
        bmm_95: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_615, arg561_1);  view_615 = arg561_1 = None
        view_616: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_94, _shape_param_618);  bmm_94 = _shape_param_618 = None
        view_617: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_95, _shape_param_619);  bmm_95 = _shape_param_619 = None
        convert_element_type_671: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_616, torch.float32);  view_616 = None
        permute_258: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_671, [0, 1, 3, 2]);  convert_element_type_671 = None
        convert_element_type_672: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_258, torch.bfloat16);  permute_258 = None
        convert_element_type_673: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_617, torch.bfloat16);  view_617 = None
        permute_259: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_612, [0, 2, 1, 3]);  view_612 = None
        clone_93: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_618: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_93, _shape_param_620);  clone_93 = _shape_param_620 = None
        permute_260: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_672, [0, 2, 1, 3]);  convert_element_type_672 = None
        view_619: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_260, _shape_param_621);  permute_260 = _shape_param_621 = None
        permute_261: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_673, [0, 2, 1, 3]);  convert_element_type_673 = None
        clone_94: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_620: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_94, _shape_param_622);  clone_94 = _shape_param_622 = None
        view_621: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_618, _shape_param_623);  view_618 = _shape_param_623 = None
        permute_262: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_621, [1, 0])
        mm_284: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_262, arg562_1);  permute_262 = None
        mm_285: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_621, arg563_1);  view_621 = arg563_1 = None
        view_622: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_285, _shape_param_624);  mm_285 = _shape_param_624 = None
        convert_element_type_674: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_622, torch.float32);  view_622 = None
        convert_element_type_675: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_284, torch.float32);  mm_284 = None
        clone_95: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_619, memory_format = torch.contiguous_format);  view_619 = None
        view_623: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_95, _shape_param_625);  clone_95 = _shape_param_625 = None
        permute_263: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_623, [1, 0])
        mm_286: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_263, arg562_1);  permute_263 = None
        mm_287: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_623, arg564_1);  view_623 = arg564_1 = None
        view_624: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_287, _shape_param_626);  mm_287 = _shape_param_626 = None
        convert_element_type_676: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_624, torch.float32);  view_624 = None
        add_191: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_674, convert_element_type_676);  convert_element_type_674 = convert_element_type_676 = None
        convert_element_type_677: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_286, torch.float32);  mm_286 = None
        view_625: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_620, _shape_param_627);  view_620 = _shape_param_627 = None
        permute_264: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_288: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_264, arg562_1);  permute_264 = arg562_1 = None
        mm_289: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_625, arg565_1);  view_625 = arg565_1 = None
        view_626: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_289, _shape_param_628);  mm_289 = _shape_param_628 = None
        convert_element_type_678: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_626, torch.float32);  view_626 = None
        add_192: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_191, convert_element_type_678);  add_191 = convert_element_type_678 = None
        convert_element_type_679: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_288, torch.float32);  mm_288 = None
        mul_602: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_192, arg566_1);  arg566_1 = None
        mul_603: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_602, 2048)
        sum_291: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True)
        add_193: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(arg567_1, arg568_1);  arg567_1 = arg568_1 = None
        sub_124: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_193, arg569_1);  add_193 = arg569_1 = None
        mul_604: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_124, arg570_1);  sub_124 = None
        mul_605: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_602, mul_604);  mul_602 = None
        sum_292: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_605, [2], True);  mul_605 = None
        mul_606: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_604, sum_292);  sum_292 = None
        sub_125: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_603, sum_291);  mul_603 = sum_291 = None
        sub_126: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_125, mul_606);  sub_125 = mul_606 = None
        div_2: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(arg570_1, 2048);  arg570_1 = None
        mul_607: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_2, sub_126);  div_2 = sub_126 = None
        mul_608: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_192, mul_604);  mul_604 = None
        sum_293: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 1]);  mul_608 = None
        sum_294: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_192, [0, 1]);  add_192 = None
        add_194: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_189, mul_607);  add_189 = mul_607 = None
        sum_295: "f32[1, 128, 2048]" = torch.ops.aten.sum.dim_IntList(add_194, [0], True, dtype = torch.float32)
        full_6: "b8[1, 128, 1]" = torch.ops.aten.full.default(_shape_param_629, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_629 = None
        full_7: "f32[2048, 2048]" = torch.ops.aten.full.default(_shape_param_630, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_630 = None
        _unsafe_masked_index_put_accumulate: "f32[2048, 2048]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_7, full_6, [arg554_1], sum_295);  full_7 = full_6 = arg554_1 = sum_295 = None
        ge: "b8[32, 128]" = torch.ops.aten.ge.Scalar(arg571_1, 0)
        lt: "b8[32, 128]" = torch.ops.aten.lt.Scalar(arg571_1, 50257)
        bitwise_and_2: "b8[32, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg571_1, -1)
        bitwise_and_3: "b8[32, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne);  bitwise_and_2 = ne = None
        unsqueeze_6: "b8[32, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_8: "f32[50257, 2048]" = torch.ops.aten.full.default(_shape_param_631, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_631 = None
        _unsafe_masked_index_put_accumulate_1: "f32[50257, 2048]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_8, unsqueeze_6, [arg571_1], add_194);  full_8 = unsqueeze_6 = arg571_1 = add_194 = None
        return (convert_element_type_7, sum_5, sum_6, convert_element_type_11, convert_element_type_12, convert_element_type_19, convert_element_type_20, sum_11, sum_12, convert_element_type_23, convert_element_type_24, convert_element_type_31, convert_element_type_33, convert_element_type_35, sum_17, sum_18, convert_element_type_39, convert_element_type_40, convert_element_type_47, convert_element_type_48, sum_23, sum_24, convert_element_type_51, convert_element_type_52, convert_element_type_59, convert_element_type_61, convert_element_type_63, sum_29, sum_30, convert_element_type_67, convert_element_type_68, convert_element_type_75, convert_element_type_76, sum_35, sum_36, convert_element_type_79, convert_element_type_80, convert_element_type_87, convert_element_type_89, convert_element_type_91, sum_41, sum_42, convert_element_type_95, convert_element_type_96, convert_element_type_103, convert_element_type_104, sum_47, sum_48, convert_element_type_107, convert_element_type_108, convert_element_type_115, convert_element_type_117, convert_element_type_119, sum_53, sum_54, convert_element_type_123, convert_element_type_124, convert_element_type_131, convert_element_type_132, sum_59, sum_60, convert_element_type_135, convert_element_type_136, convert_element_type_143, convert_element_type_145, convert_element_type_147, sum_65, sum_66, convert_element_type_151, convert_element_type_152, convert_element_type_159, convert_element_type_160, sum_71, sum_72, convert_element_type_163, convert_element_type_164, convert_element_type_171, convert_element_type_173, convert_element_type_175, sum_77, sum_78, convert_element_type_179, convert_element_type_180, convert_element_type_187, convert_element_type_188, sum_83, sum_84, convert_element_type_191, convert_element_type_192, convert_element_type_199, convert_element_type_201, convert_element_type_203, sum_89, sum_90, convert_element_type_207, convert_element_type_208, convert_element_type_215, convert_element_type_216, sum_95, sum_96, convert_element_type_219, convert_element_type_220, convert_element_type_227, convert_element_type_229, convert_element_type_231, sum_101, sum_102, convert_element_type_235, convert_element_type_236, convert_element_type_243, convert_element_type_244, sum_107, sum_108, convert_element_type_247, convert_element_type_248, convert_element_type_255, convert_element_type_257, convert_element_type_259, sum_113, sum_114, convert_element_type_263, convert_element_type_264, convert_element_type_271, convert_element_type_272, sum_119, sum_120, convert_element_type_275, convert_element_type_276, convert_element_type_283, convert_element_type_285, convert_element_type_287, sum_125, sum_126, convert_element_type_291, convert_element_type_292, convert_element_type_299, convert_element_type_300, sum_131, sum_132, convert_element_type_303, convert_element_type_304, convert_element_type_311, convert_element_type_313, convert_element_type_315, sum_137, sum_138, convert_element_type_319, convert_element_type_320, convert_element_type_327, convert_element_type_328, sum_143, sum_144, convert_element_type_331, convert_element_type_332, convert_element_type_339, convert_element_type_341, convert_element_type_343, sum_149, sum_150, convert_element_type_347, convert_element_type_348, convert_element_type_355, convert_element_type_356, sum_155, sum_156, convert_element_type_359, convert_element_type_360, convert_element_type_367, convert_element_type_369, convert_element_type_371, sum_161, sum_162, convert_element_type_375, convert_element_type_376, convert_element_type_383, convert_element_type_384, sum_167, sum_168, convert_element_type_387, convert_element_type_388, convert_element_type_395, convert_element_type_397, convert_element_type_399, sum_173, sum_174, convert_element_type_403, convert_element_type_404, convert_element_type_411, convert_element_type_412, sum_179, sum_180, convert_element_type_415, convert_element_type_416, convert_element_type_423, convert_element_type_425, convert_element_type_427, sum_185, sum_186, convert_element_type_431, convert_element_type_432, convert_element_type_439, convert_element_type_440, sum_191, sum_192, convert_element_type_443, convert_element_type_444, convert_element_type_451, convert_element_type_453, convert_element_type_455, sum_197, sum_198, convert_element_type_459, convert_element_type_460, convert_element_type_467, convert_element_type_468, sum_203, sum_204, convert_element_type_471, convert_element_type_472, convert_element_type_479, convert_element_type_481, convert_element_type_483, sum_209, sum_210, convert_element_type_487, convert_element_type_488, convert_element_type_495, convert_element_type_496, sum_215, sum_216, convert_element_type_499, convert_element_type_500, convert_element_type_507, convert_element_type_509, convert_element_type_511, sum_221, sum_222, convert_element_type_515, convert_element_type_516, convert_element_type_523, convert_element_type_524, sum_227, sum_228, convert_element_type_527, convert_element_type_528, convert_element_type_535, convert_element_type_537, convert_element_type_539, sum_233, sum_234, convert_element_type_543, convert_element_type_544, convert_element_type_551, convert_element_type_552, sum_239, sum_240, convert_element_type_555, convert_element_type_556, convert_element_type_563, convert_element_type_565, convert_element_type_567, sum_245, sum_246, convert_element_type_571, convert_element_type_572, convert_element_type_579, convert_element_type_580, sum_251, sum_252, convert_element_type_583, convert_element_type_584, convert_element_type_591, convert_element_type_593, convert_element_type_595, sum_257, sum_258, convert_element_type_599, convert_element_type_600, convert_element_type_607, convert_element_type_608, sum_263, sum_264, convert_element_type_611, convert_element_type_612, convert_element_type_619, convert_element_type_621, convert_element_type_623, sum_269, sum_270, convert_element_type_627, convert_element_type_628, convert_element_type_635, convert_element_type_636, sum_275, sum_276, convert_element_type_639, convert_element_type_640, convert_element_type_647, convert_element_type_649, convert_element_type_651, sum_281, sum_282, convert_element_type_655, convert_element_type_656, convert_element_type_663, convert_element_type_664, sum_287, sum_288, convert_element_type_667, convert_element_type_668, convert_element_type_675, convert_element_type_677, convert_element_type_679, sum_293, sum_294, _unsafe_masked_index_put_accumulate, _unsafe_masked_index_put_accumulate_1)



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
