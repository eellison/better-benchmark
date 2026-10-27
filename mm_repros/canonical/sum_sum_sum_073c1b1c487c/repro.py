"""
Standalone repro captured via capture_hook.
Label: GPTJForQuestionAnswering
Pattern hash: 073c1b1c487c
Shape hash: a456585d
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
    def forward(self, arg0_1: "f32[]", arg1_1: "b8[1]", arg2_1: "i64[1]", arg3_1: "bf16[1, 128]", arg4_1: "f32[1, 1]", arg5_1: "f32[1, 1]", arg6_1: "bf16[1, 128]", arg7_1: "b8[1]", arg8_1: "i64[1]", arg9_1: "bf16[1, 128]", arg10_1: "f32[1, 1]", arg11_1: "f32[1, 1]", arg12_1: "bf16[1, 128]", arg13_1: "bf16[2, 4096]", arg14_1: "bf16[128, 4096]", arg15_1: "f32[4096]", arg16_1: "f32[1, 128, 4096]", arg17_1: "f32[1, 128, 1]", arg18_1: "bf16[4096, 16384]", arg19_1: "bf16[128, 16384]", arg20_1: "bf16[128, 16384]", arg21_1: "bf16[16384, 4096]", arg22_1: "bf16[128, 4096]", arg23_1: "bf16[128, 4096]", arg24_1: "bf16[4096, 4096]", arg25_1: "bf16[16, 128, 128]", arg26_1: "bf16[16, 256, 128]", arg27_1: "f32[1, 16, 128, 128]", arg28_1: "bf16[16, 256, 128]", arg29_1: "bf16[16, 128, 256]", arg30_1: "bf16[1, 128, 1, 32, 1]", arg31_1: "bf16[1, 128, 1, 32, 1]", arg32_1: "bf16[4096, 4096]", arg33_1: "bf16[4096, 4096]", arg34_1: "bf16[4096, 4096]", arg35_1: "f32[4096]", arg36_1: "f32[1, 128, 4096]", arg37_1: "f32[1, 128, 1]", arg38_1: "bf16[4096, 16384]", arg39_1: "bf16[128, 16384]", arg40_1: "bf16[128, 16384]", arg41_1: "bf16[16384, 4096]", arg42_1: "bf16[128, 4096]", arg43_1: "bf16[128, 4096]", arg44_1: "bf16[4096, 4096]", arg45_1: "bf16[16, 128, 128]", arg46_1: "bf16[16, 256, 128]", arg47_1: "f32[1, 16, 128, 128]", arg48_1: "bf16[16, 256, 128]", arg49_1: "bf16[16, 128, 256]", arg50_1: "bf16[1, 128, 1, 32, 1]", arg51_1: "bf16[1, 128, 1, 32, 1]", arg52_1: "bf16[4096, 4096]", arg53_1: "bf16[4096, 4096]", arg54_1: "bf16[4096, 4096]", arg55_1: "f32[4096]", arg56_1: "f32[1, 128, 4096]", arg57_1: "f32[1, 128, 1]", arg58_1: "bf16[4096, 16384]", arg59_1: "bf16[128, 16384]", arg60_1: "bf16[128, 16384]", arg61_1: "bf16[16384, 4096]", arg62_1: "bf16[128, 4096]", arg63_1: "bf16[128, 4096]", arg64_1: "bf16[4096, 4096]", arg65_1: "bf16[16, 128, 128]", arg66_1: "bf16[16, 256, 128]", arg67_1: "f32[1, 16, 128, 128]", arg68_1: "bf16[16, 256, 128]", arg69_1: "bf16[16, 128, 256]", arg70_1: "bf16[1, 128, 1, 32, 1]", arg71_1: "bf16[1, 128, 1, 32, 1]", arg72_1: "bf16[4096, 4096]", arg73_1: "bf16[4096, 4096]", arg74_1: "bf16[4096, 4096]", arg75_1: "f32[4096]", arg76_1: "f32[1, 128, 4096]", arg77_1: "f32[1, 128, 1]", arg78_1: "bf16[4096, 16384]", arg79_1: "bf16[128, 16384]", arg80_1: "bf16[128, 16384]", arg81_1: "bf16[16384, 4096]", arg82_1: "bf16[128, 4096]", arg83_1: "bf16[128, 4096]", arg84_1: "bf16[4096, 4096]", arg85_1: "bf16[16, 128, 128]", arg86_1: "bf16[16, 256, 128]", arg87_1: "f32[1, 16, 128, 128]", arg88_1: "bf16[16, 256, 128]", arg89_1: "bf16[16, 128, 256]", arg90_1: "bf16[1, 128, 1, 32, 1]", arg91_1: "bf16[1, 128, 1, 32, 1]", arg92_1: "bf16[4096, 4096]", arg93_1: "bf16[4096, 4096]", arg94_1: "bf16[4096, 4096]", arg95_1: "f32[4096]", arg96_1: "f32[1, 128, 4096]", arg97_1: "f32[1, 128, 1]", arg98_1: "bf16[4096, 16384]", arg99_1: "bf16[128, 16384]", arg100_1: "bf16[128, 16384]", arg101_1: "bf16[16384, 4096]", arg102_1: "bf16[128, 4096]", arg103_1: "bf16[128, 4096]", arg104_1: "bf16[4096, 4096]", arg105_1: "bf16[16, 128, 128]", arg106_1: "bf16[16, 256, 128]", arg107_1: "f32[1, 16, 128, 128]", arg108_1: "bf16[16, 256, 128]", arg109_1: "bf16[16, 128, 256]", arg110_1: "bf16[1, 128, 1, 32, 1]", arg111_1: "bf16[1, 128, 1, 32, 1]", arg112_1: "bf16[4096, 4096]", arg113_1: "bf16[4096, 4096]", arg114_1: "bf16[4096, 4096]", arg115_1: "f32[4096]", arg116_1: "f32[1, 128, 4096]", arg117_1: "f32[1, 128, 1]", arg118_1: "bf16[4096, 16384]", arg119_1: "bf16[128, 16384]", arg120_1: "bf16[128, 16384]", arg121_1: "bf16[16384, 4096]", arg122_1: "bf16[128, 4096]", arg123_1: "bf16[128, 4096]", arg124_1: "bf16[4096, 4096]", arg125_1: "bf16[16, 128, 128]", arg126_1: "bf16[16, 256, 128]", arg127_1: "f32[1, 16, 128, 128]", arg128_1: "bf16[16, 256, 128]", arg129_1: "bf16[16, 128, 256]", arg130_1: "bf16[1, 128, 1, 32, 1]", arg131_1: "bf16[1, 128, 1, 32, 1]", arg132_1: "bf16[4096, 4096]", arg133_1: "bf16[4096, 4096]", arg134_1: "bf16[4096, 4096]", arg135_1: "f32[4096]", arg136_1: "f32[1, 128, 4096]", arg137_1: "f32[1, 128, 1]", arg138_1: "bf16[4096, 16384]", arg139_1: "bf16[128, 16384]", arg140_1: "bf16[128, 16384]", arg141_1: "bf16[16384, 4096]", arg142_1: "bf16[128, 4096]", arg143_1: "bf16[128, 4096]", arg144_1: "bf16[4096, 4096]", arg145_1: "bf16[16, 128, 128]", arg146_1: "bf16[16, 256, 128]", arg147_1: "f32[1, 16, 128, 128]", arg148_1: "bf16[16, 256, 128]", arg149_1: "bf16[16, 128, 256]", arg150_1: "bf16[1, 128, 1, 32, 1]", arg151_1: "bf16[1, 128, 1, 32, 1]", arg152_1: "bf16[4096, 4096]", arg153_1: "bf16[4096, 4096]", arg154_1: "bf16[4096, 4096]", arg155_1: "f32[4096]", arg156_1: "f32[1, 128, 4096]", arg157_1: "f32[1, 128, 1]", arg158_1: "bf16[4096, 16384]", arg159_1: "bf16[128, 16384]", arg160_1: "bf16[128, 16384]", arg161_1: "bf16[16384, 4096]", arg162_1: "bf16[128, 4096]", arg163_1: "bf16[128, 4096]", arg164_1: "bf16[4096, 4096]", arg165_1: "bf16[16, 128, 128]", arg166_1: "bf16[16, 256, 128]", arg167_1: "f32[1, 16, 128, 128]", arg168_1: "bf16[16, 256, 128]", arg169_1: "bf16[16, 128, 256]", arg170_1: "bf16[1, 128, 1, 32, 1]", arg171_1: "bf16[1, 128, 1, 32, 1]", arg172_1: "bf16[4096, 4096]", arg173_1: "bf16[4096, 4096]", arg174_1: "bf16[4096, 4096]", arg175_1: "f32[4096]", arg176_1: "f32[1, 128, 4096]", arg177_1: "f32[1, 128, 1]", arg178_1: "bf16[4096, 16384]", arg179_1: "bf16[128, 16384]", arg180_1: "bf16[128, 16384]", arg181_1: "bf16[16384, 4096]", arg182_1: "bf16[128, 4096]", arg183_1: "bf16[128, 4096]", arg184_1: "bf16[4096, 4096]", arg185_1: "bf16[16, 128, 128]", arg186_1: "bf16[16, 256, 128]", arg187_1: "f32[1, 16, 128, 128]", arg188_1: "bf16[16, 256, 128]", arg189_1: "bf16[16, 128, 256]", arg190_1: "bf16[1, 128, 1, 32, 1]", arg191_1: "bf16[1, 128, 1, 32, 1]", arg192_1: "bf16[4096, 4096]", arg193_1: "bf16[4096, 4096]", arg194_1: "bf16[4096, 4096]", arg195_1: "f32[4096]", arg196_1: "f32[1, 128, 4096]", arg197_1: "f32[1, 128, 1]", arg198_1: "bf16[4096, 16384]", arg199_1: "bf16[128, 16384]", arg200_1: "bf16[128, 16384]", arg201_1: "bf16[16384, 4096]", arg202_1: "bf16[128, 4096]", arg203_1: "bf16[128, 4096]", arg204_1: "bf16[4096, 4096]", arg205_1: "bf16[16, 128, 128]", arg206_1: "bf16[16, 256, 128]", arg207_1: "f32[1, 16, 128, 128]", arg208_1: "bf16[16, 256, 128]", arg209_1: "bf16[16, 128, 256]", arg210_1: "bf16[1, 128, 1, 32, 1]", arg211_1: "bf16[1, 128, 1, 32, 1]", arg212_1: "bf16[4096, 4096]", arg213_1: "bf16[4096, 4096]", arg214_1: "bf16[4096, 4096]", arg215_1: "f32[4096]", arg216_1: "f32[1, 128, 4096]", arg217_1: "f32[1, 128, 1]", arg218_1: "bf16[4096, 16384]", arg219_1: "bf16[128, 16384]", arg220_1: "bf16[128, 16384]", arg221_1: "bf16[16384, 4096]", arg222_1: "bf16[128, 4096]", arg223_1: "bf16[128, 4096]", arg224_1: "bf16[4096, 4096]", arg225_1: "bf16[16, 128, 128]", arg226_1: "bf16[16, 256, 128]", arg227_1: "f32[1, 16, 128, 128]", arg228_1: "bf16[16, 256, 128]", arg229_1: "bf16[16, 128, 256]", arg230_1: "bf16[1, 128, 1, 32, 1]", arg231_1: "bf16[1, 128, 1, 32, 1]", arg232_1: "bf16[4096, 4096]", arg233_1: "bf16[4096, 4096]", arg234_1: "bf16[4096, 4096]", arg235_1: "f32[4096]", arg236_1: "f32[1, 128, 4096]", arg237_1: "f32[1, 128, 1]", arg238_1: "bf16[4096, 16384]", arg239_1: "bf16[128, 16384]", arg240_1: "bf16[128, 16384]", arg241_1: "bf16[16384, 4096]", arg242_1: "bf16[128, 4096]", arg243_1: "bf16[128, 4096]", arg244_1: "bf16[4096, 4096]", arg245_1: "bf16[16, 128, 128]", arg246_1: "bf16[16, 256, 128]", arg247_1: "f32[1, 16, 128, 128]", arg248_1: "bf16[16, 256, 128]", arg249_1: "bf16[16, 128, 256]", arg250_1: "bf16[1, 128, 1, 32, 1]", arg251_1: "bf16[1, 128, 1, 32, 1]", arg252_1: "bf16[4096, 4096]", arg253_1: "bf16[4096, 4096]", arg254_1: "bf16[4096, 4096]", arg255_1: "f32[4096]", arg256_1: "f32[1, 128, 4096]", arg257_1: "f32[1, 128, 1]", arg258_1: "bf16[4096, 16384]", arg259_1: "bf16[128, 16384]", arg260_1: "bf16[128, 16384]", arg261_1: "bf16[16384, 4096]", arg262_1: "bf16[128, 4096]", arg263_1: "bf16[128, 4096]", arg264_1: "bf16[4096, 4096]", arg265_1: "bf16[16, 128, 128]", arg266_1: "bf16[16, 256, 128]", arg267_1: "f32[1, 16, 128, 128]", arg268_1: "bf16[16, 256, 128]", arg269_1: "bf16[16, 128, 256]", arg270_1: "bf16[1, 128, 1, 32, 1]", arg271_1: "bf16[1, 128, 1, 32, 1]", arg272_1: "bf16[4096, 4096]", arg273_1: "bf16[4096, 4096]", arg274_1: "bf16[4096, 4096]", arg275_1: "f32[4096]", arg276_1: "f32[1, 128, 4096]", arg277_1: "f32[1, 128, 1]", arg278_1: "bf16[4096, 16384]", arg279_1: "bf16[128, 16384]", arg280_1: "bf16[128, 16384]", arg281_1: "bf16[16384, 4096]", arg282_1: "bf16[128, 4096]", arg283_1: "bf16[128, 4096]", arg284_1: "bf16[4096, 4096]", arg285_1: "bf16[16, 128, 128]", arg286_1: "bf16[16, 256, 128]", arg287_1: "f32[1, 16, 128, 128]", arg288_1: "bf16[16, 256, 128]", arg289_1: "bf16[16, 128, 256]", arg290_1: "bf16[1, 128, 1, 32, 1]", arg291_1: "bf16[1, 128, 1, 32, 1]", arg292_1: "bf16[4096, 4096]", arg293_1: "bf16[4096, 4096]", arg294_1: "bf16[4096, 4096]", arg295_1: "f32[4096]", arg296_1: "f32[1, 128, 4096]", arg297_1: "f32[1, 128, 1]", arg298_1: "bf16[4096, 16384]", arg299_1: "bf16[128, 16384]", arg300_1: "bf16[128, 16384]", arg301_1: "bf16[16384, 4096]", arg302_1: "bf16[128, 4096]", arg303_1: "bf16[128, 4096]", arg304_1: "bf16[4096, 4096]", arg305_1: "bf16[16, 128, 128]", arg306_1: "bf16[16, 256, 128]", arg307_1: "f32[1, 16, 128, 128]", arg308_1: "bf16[16, 256, 128]", arg309_1: "bf16[16, 128, 256]", arg310_1: "bf16[1, 128, 1, 32, 1]", arg311_1: "bf16[1, 128, 1, 32, 1]", arg312_1: "bf16[4096, 4096]", arg313_1: "bf16[4096, 4096]", arg314_1: "bf16[4096, 4096]", arg315_1: "f32[4096]", arg316_1: "f32[1, 128, 4096]", arg317_1: "f32[1, 128, 1]", arg318_1: "bf16[4096, 16384]", arg319_1: "bf16[128, 16384]", arg320_1: "bf16[128, 16384]", arg321_1: "bf16[16384, 4096]", arg322_1: "bf16[128, 4096]", arg323_1: "bf16[128, 4096]", arg324_1: "bf16[4096, 4096]", arg325_1: "bf16[16, 128, 128]", arg326_1: "bf16[16, 256, 128]", arg327_1: "f32[1, 16, 128, 128]", arg328_1: "bf16[16, 256, 128]", arg329_1: "bf16[16, 128, 256]", arg330_1: "bf16[1, 128, 1, 32, 1]", arg331_1: "bf16[1, 128, 1, 32, 1]", arg332_1: "bf16[4096, 4096]", arg333_1: "bf16[4096, 4096]", arg334_1: "bf16[4096, 4096]", arg335_1: "f32[4096]", arg336_1: "f32[1, 128, 4096]", arg337_1: "f32[1, 128, 1]", arg338_1: "bf16[4096, 16384]", arg339_1: "bf16[128, 16384]", arg340_1: "bf16[128, 16384]", arg341_1: "bf16[16384, 4096]", arg342_1: "bf16[128, 4096]", arg343_1: "bf16[128, 4096]", arg344_1: "bf16[4096, 4096]", arg345_1: "bf16[16, 128, 128]", arg346_1: "bf16[16, 256, 128]", arg347_1: "f32[1, 16, 128, 128]", arg348_1: "bf16[16, 256, 128]", arg349_1: "bf16[16, 128, 256]", arg350_1: "bf16[1, 128, 1, 32, 1]", arg351_1: "bf16[1, 128, 1, 32, 1]", arg352_1: "bf16[4096, 4096]", arg353_1: "bf16[4096, 4096]", arg354_1: "bf16[4096, 4096]", arg355_1: "f32[4096]", arg356_1: "f32[1, 128, 4096]", arg357_1: "f32[1, 128, 1]", arg358_1: "bf16[4096, 16384]", arg359_1: "bf16[128, 16384]", arg360_1: "bf16[128, 16384]", arg361_1: "bf16[16384, 4096]", arg362_1: "bf16[128, 4096]", arg363_1: "bf16[128, 4096]", arg364_1: "bf16[4096, 4096]", arg365_1: "bf16[16, 128, 128]", arg366_1: "bf16[16, 256, 128]", arg367_1: "f32[1, 16, 128, 128]", arg368_1: "bf16[16, 256, 128]", arg369_1: "bf16[16, 128, 256]", arg370_1: "bf16[1, 128, 1, 32, 1]", arg371_1: "bf16[1, 128, 1, 32, 1]", arg372_1: "bf16[4096, 4096]", arg373_1: "bf16[4096, 4096]", arg374_1: "bf16[4096, 4096]", arg375_1: "f32[4096]", arg376_1: "f32[1, 128, 4096]", arg377_1: "f32[1, 128, 1]", arg378_1: "bf16[4096, 16384]", arg379_1: "bf16[128, 16384]", arg380_1: "bf16[128, 16384]", arg381_1: "bf16[16384, 4096]", arg382_1: "bf16[128, 4096]", arg383_1: "bf16[128, 4096]", arg384_1: "bf16[4096, 4096]", arg385_1: "bf16[16, 128, 128]", arg386_1: "bf16[16, 256, 128]", arg387_1: "f32[1, 16, 128, 128]", arg388_1: "bf16[16, 256, 128]", arg389_1: "bf16[16, 128, 256]", arg390_1: "bf16[1, 128, 1, 32, 1]", arg391_1: "bf16[1, 128, 1, 32, 1]", arg392_1: "bf16[4096, 4096]", arg393_1: "bf16[4096, 4096]", arg394_1: "bf16[4096, 4096]", arg395_1: "f32[4096]", arg396_1: "f32[1, 128, 4096]", arg397_1: "f32[1, 128, 1]", arg398_1: "bf16[4096, 16384]", arg399_1: "bf16[128, 16384]", arg400_1: "bf16[128, 16384]", arg401_1: "bf16[16384, 4096]", arg402_1: "bf16[128, 4096]", arg403_1: "bf16[128, 4096]", arg404_1: "bf16[4096, 4096]", arg405_1: "bf16[16, 128, 128]", arg406_1: "bf16[16, 256, 128]", arg407_1: "f32[1, 16, 128, 128]", arg408_1: "bf16[16, 256, 128]", arg409_1: "bf16[16, 128, 256]", arg410_1: "bf16[1, 128, 1, 32, 1]", arg411_1: "bf16[1, 128, 1, 32, 1]", arg412_1: "bf16[4096, 4096]", arg413_1: "bf16[4096, 4096]", arg414_1: "bf16[4096, 4096]", arg415_1: "f32[4096]", arg416_1: "f32[1, 128, 4096]", arg417_1: "f32[1, 128, 1]", arg418_1: "bf16[4096, 16384]", arg419_1: "bf16[128, 16384]", arg420_1: "bf16[128, 16384]", arg421_1: "bf16[16384, 4096]", arg422_1: "bf16[128, 4096]", arg423_1: "bf16[128, 4096]", arg424_1: "bf16[4096, 4096]", arg425_1: "bf16[16, 128, 128]", arg426_1: "bf16[16, 256, 128]", arg427_1: "f32[1, 16, 128, 128]", arg428_1: "bf16[16, 256, 128]", arg429_1: "bf16[16, 128, 256]", arg430_1: "bf16[1, 128, 1, 32, 1]", arg431_1: "bf16[1, 128, 1, 32, 1]", arg432_1: "bf16[4096, 4096]", arg433_1: "bf16[4096, 4096]", arg434_1: "bf16[4096, 4096]", arg435_1: "f32[4096]", arg436_1: "f32[1, 128, 4096]", arg437_1: "f32[1, 128, 1]", arg438_1: "bf16[4096, 16384]", arg439_1: "bf16[128, 16384]", arg440_1: "bf16[128, 16384]", arg441_1: "bf16[16384, 4096]", arg442_1: "bf16[128, 4096]", arg443_1: "bf16[128, 4096]", arg444_1: "bf16[4096, 4096]", arg445_1: "bf16[16, 128, 128]", arg446_1: "bf16[16, 256, 128]", arg447_1: "f32[1, 16, 128, 128]", arg448_1: "bf16[16, 256, 128]", arg449_1: "bf16[16, 128, 256]", arg450_1: "bf16[1, 128, 1, 32, 1]", arg451_1: "bf16[1, 128, 1, 32, 1]", arg452_1: "bf16[4096, 4096]", arg453_1: "bf16[4096, 4096]", arg454_1: "bf16[4096, 4096]", arg455_1: "f32[4096]", arg456_1: "f32[1, 128, 4096]", arg457_1: "f32[1, 128, 1]", arg458_1: "bf16[4096, 16384]", arg459_1: "bf16[128, 16384]", arg460_1: "bf16[128, 16384]", arg461_1: "bf16[16384, 4096]", arg462_1: "bf16[128, 4096]", arg463_1: "bf16[128, 4096]", arg464_1: "bf16[4096, 4096]", arg465_1: "bf16[16, 128, 128]", arg466_1: "bf16[16, 256, 128]", arg467_1: "f32[1, 16, 128, 128]", arg468_1: "bf16[16, 256, 128]", arg469_1: "bf16[16, 128, 256]", arg470_1: "bf16[1, 128, 1, 32, 1]", arg471_1: "bf16[1, 128, 1, 32, 1]", arg472_1: "bf16[4096, 4096]", arg473_1: "bf16[4096, 4096]", arg474_1: "bf16[4096, 4096]", arg475_1: "f32[4096]", arg476_1: "f32[1, 128, 4096]", arg477_1: "f32[1, 128, 1]", arg478_1: "bf16[4096, 16384]", arg479_1: "bf16[128, 16384]", arg480_1: "bf16[128, 16384]", arg481_1: "bf16[16384, 4096]", arg482_1: "bf16[128, 4096]", arg483_1: "bf16[128, 4096]", arg484_1: "bf16[4096, 4096]", arg485_1: "bf16[16, 128, 128]", arg486_1: "bf16[16, 256, 128]", arg487_1: "f32[1, 16, 128, 128]", arg488_1: "bf16[16, 256, 128]", arg489_1: "bf16[16, 128, 256]", arg490_1: "bf16[1, 128, 1, 32, 1]", arg491_1: "bf16[1, 128, 1, 32, 1]", arg492_1: "bf16[4096, 4096]", arg493_1: "bf16[4096, 4096]", arg494_1: "bf16[4096, 4096]", arg495_1: "f32[4096]", arg496_1: "f32[1, 128, 4096]", arg497_1: "f32[1, 128, 1]", arg498_1: "bf16[4096, 16384]", arg499_1: "bf16[128, 16384]", arg500_1: "bf16[128, 16384]", arg501_1: "bf16[16384, 4096]", arg502_1: "bf16[128, 4096]", arg503_1: "bf16[128, 4096]", arg504_1: "bf16[4096, 4096]", arg505_1: "bf16[16, 128, 128]", arg506_1: "bf16[16, 256, 128]", arg507_1: "f32[1, 16, 128, 128]", arg508_1: "bf16[16, 256, 128]", arg509_1: "bf16[16, 128, 256]", arg510_1: "bf16[1, 128, 1, 32, 1]", arg511_1: "bf16[1, 128, 1, 32, 1]", arg512_1: "bf16[4096, 4096]", arg513_1: "bf16[4096, 4096]", arg514_1: "bf16[4096, 4096]", arg515_1: "f32[4096]", arg516_1: "f32[1, 128, 4096]", arg517_1: "f32[1, 128, 1]", arg518_1: "bf16[4096, 16384]", arg519_1: "bf16[128, 16384]", arg520_1: "bf16[128, 16384]", arg521_1: "bf16[16384, 4096]", arg522_1: "bf16[128, 4096]", arg523_1: "bf16[128, 4096]", arg524_1: "bf16[4096, 4096]", arg525_1: "bf16[16, 128, 128]", arg526_1: "bf16[16, 256, 128]", arg527_1: "f32[1, 16, 128, 128]", arg528_1: "bf16[16, 256, 128]", arg529_1: "bf16[16, 128, 256]", arg530_1: "bf16[1, 128, 1, 32, 1]", arg531_1: "bf16[1, 128, 1, 32, 1]", arg532_1: "bf16[4096, 4096]", arg533_1: "bf16[4096, 4096]", arg534_1: "bf16[4096, 4096]", arg535_1: "f32[4096]", arg536_1: "f32[1, 128, 4096]", arg537_1: "f32[1, 128, 1]", arg538_1: "bf16[4096, 16384]", arg539_1: "bf16[128, 16384]", arg540_1: "bf16[128, 16384]", arg541_1: "bf16[16384, 4096]", arg542_1: "bf16[128, 4096]", arg543_1: "bf16[128, 4096]", arg544_1: "bf16[4096, 4096]", arg545_1: "bf16[16, 128, 128]", arg546_1: "bf16[16, 256, 128]", arg547_1: "f32[1, 16, 128, 128]", arg548_1: "bf16[16, 256, 128]", arg549_1: "bf16[16, 128, 256]", arg550_1: "bf16[1, 128, 1, 32, 1]", arg551_1: "bf16[1, 128, 1, 32, 1]", arg552_1: "bf16[4096, 4096]", arg553_1: "bf16[4096, 4096]", arg554_1: "bf16[4096, 4096]", arg555_1: "f32[4096]", arg556_1: "bf16[1, 128, 4096]", arg557_1: "f32[1, 128, 4096]", arg558_1: "f32[1, 128, 1]", arg559_1: "f32[1, 128, 1]", arg560_1: "bf16[4096, 16384]", arg561_1: "bf16[128, 16384]", arg562_1: "bf16[128, 16384]", arg563_1: "bf16[16384, 4096]", arg564_1: "bf16[128, 4096]", arg565_1: "bf16[128, 4096]", arg566_1: "bf16[4096, 4096]", arg567_1: "bf16[16, 128, 128]", arg568_1: "bf16[16, 256, 128]", arg569_1: "f32[1, 16, 128, 128]", arg570_1: "bf16[16, 256, 128]", arg571_1: "bf16[16, 128, 256]", arg572_1: "bf16[1, 128, 1, 32, 1]", arg573_1: "bf16[1, 128, 1, 32, 1]", arg574_1: "bf16[4096, 4096]", arg575_1: "bf16[4096, 4096]", arg576_1: "bf16[4096, 4096]", arg577_1: "f32[4096]", arg578_1: "f32[1, 128, 1]", arg579_1: "f32[1, 128, 1]", arg580_1: "i64[1, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825, _shape_param_826, _shape_param_827, _shape_param_828, _shape_param_829, _shape_param_830, _shape_param_831, _shape_param_832, _shape_param_833, _shape_param_834, _shape_param_835, _shape_param_836, _shape_param_837, _shape_param_838, _shape_param_839, _shape_param_840, _shape_param_841, _shape_param_842, _shape_param_843, _shape_param_844, _shape_param_845, _shape_param_846, _shape_param_847, _shape_param_848, _shape_param_849, _shape_param_850):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, 2);  arg0_1 = None
        sum_1: "i64[]" = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_1, torch.float32);  sum_1 = None
        div_1: "f32[]" = torch.ops.aten.div.Tensor(div, convert_element_type);  convert_element_type = None
        clamp_min: "i64[1]" = torch.ops.aten.clamp_min.default(arg2_1, 0);  arg2_1 = None
        clamp_max: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min, 128);  clamp_min = None
        unsqueeze: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max, 1);  clamp_max = None
        ne: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, 128)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[1, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = None
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view: "i64[1, 128]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[1, 128]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[1, 128]" = torch.ops.aten.eq.Tensor(expand, view);  expand = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[1, 128]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[1, 1]" = torch.ops.aten.where.self(ne, div_1, full_1);  ne = div_1 = None
        mul: "f32[1, 128]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type_1: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_2: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.float32);  convert_element_type_1 = None
        convert_element_type_3: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        sub: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_3, arg4_1);  convert_element_type_3 = arg4_1 = None
        sub_1: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_4: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_5: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_4, torch.float32);  convert_element_type_4 = None
        exp: "f32[1, 128]" = torch.ops.aten.exp.default(convert_element_type_5);  convert_element_type_5 = None
        sum_2: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [1], True)
        mul_1: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp, sum_2);  exp = sum_2 = None
        sub_2: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, mul_1);  convert_element_type_2 = mul_1 = None
        convert_element_type_6: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        add: "bf16[1, 128]" = torch.ops.aten.add.Tensor(arg6_1, convert_element_type_6);  arg6_1 = convert_element_type_6 = None
        sum_3: "i64[]" = torch.ops.aten.sum.default(arg7_1);  arg7_1 = None
        convert_element_type_7: "f32[]" = torch.ops.prims.convert_element_type.default(sum_3, torch.float32);  sum_3 = None
        div_2: "f32[]" = torch.ops.aten.div.Tensor(div, convert_element_type_7);  div = convert_element_type_7 = None
        clamp_min_1: "i64[1]" = torch.ops.aten.clamp_min.default(arg8_1, 0);  arg8_1 = None
        clamp_max_1: "i64[1]" = torch.ops.aten.clamp_max.default(clamp_min_1, 128);  clamp_min_1 = None
        unsqueeze_1: "i64[1, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_1, 1);  clamp_max_1 = None
        ne_1: "b8[1, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_1, 128)
        where_3: "i64[1, 1]" = torch.ops.aten.where.self(ne_1, unsqueeze_1, full);  unsqueeze_1 = full = None
        expand_1: "i64[1, 128]" = torch.ops.aten.expand.default(where_3, _shape_param_2);  where_3 = _shape_param_2 = None
        eq_1: "b8[1, 128]" = torch.ops.aten.eq.Tensor(expand_1, view);  expand_1 = view = None
        where_4: "f32[1, 128]" = torch.ops.aten.where.self(eq_1, scalar_tensor_1, scalar_tensor);  eq_1 = scalar_tensor_1 = scalar_tensor = None
        where_5: "f32[1, 1]" = torch.ops.aten.where.self(ne_1, div_2, full_1);  ne_1 = div_2 = full_1 = None
        mul_2: "f32[1, 128]" = torch.ops.aten.mul.Tensor(where_4, where_5);  where_4 = where_5 = None
        convert_element_type_8: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        convert_element_type_9: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_8, torch.float32);  convert_element_type_8 = None
        convert_element_type_10: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        sub_3: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_10, arg10_1);  convert_element_type_10 = arg10_1 = None
        sub_4: "f32[1, 128]" = torch.ops.aten.sub.Tensor(sub_3, arg11_1);  sub_3 = arg11_1 = None
        convert_element_type_11: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_4, torch.bfloat16);  sub_4 = None
        convert_element_type_12: "f32[1, 128]" = torch.ops.prims.convert_element_type.default(convert_element_type_11, torch.float32);  convert_element_type_11 = None
        exp_1: "f32[1, 128]" = torch.ops.aten.exp.default(convert_element_type_12);  convert_element_type_12 = None
        sum_4: "f32[1, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_9, [1], True)
        mul_3: "f32[1, 128]" = torch.ops.aten.mul.Tensor(exp_1, sum_4);  exp_1 = sum_4 = None
        sub_5: "f32[1, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_9, mul_3);  convert_element_type_9 = mul_3 = None
        convert_element_type_13: "bf16[1, 128]" = torch.ops.prims.convert_element_type.default(sub_5, torch.bfloat16);  sub_5 = None
        add_1: "bf16[1, 128]" = torch.ops.aten.add.Tensor(arg12_1, convert_element_type_13);  arg12_1 = convert_element_type_13 = None
        unsqueeze_2: "bf16[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add, 2);  add = None
        unsqueeze_3: "bf16[1, 128, 1]" = torch.ops.aten.unsqueeze.default(add_1, 2);  add_1 = None
        cat: "bf16[1, 128, 2]" = torch.ops.aten.cat.default([unsqueeze_3, unsqueeze_2], 2);  unsqueeze_3 = unsqueeze_2 = None
        view_1: "bf16[128, 2]" = torch.ops.aten.view.default(cat, _shape_param_3);  cat = _shape_param_3 = None
        constant_pad_nd: "bf16[128, 8]" = torch.ops.aten.constant_pad_nd.default(view_1, _shape_param_4);  _shape_param_4 = None
        constant_pad_nd_1: "bf16[8, 4096]" = torch.ops.aten.constant_pad_nd.default(arg13_1, _shape_param_5);  arg13_1 = _shape_param_5 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        permute: "bf16[2, 128]" = torch.ops.aten.permute.default(view_1, [1, 0])
        mm_1: "bf16[2, 4096]" = torch.ops.aten.mm.default(permute, arg14_1);  permute = arg14_1 = None
        sum_5: "f32[1, 2]" = torch.ops.aten.sum.dim_IntList(view_1, [0], True, dtype = torch.float32);  view_1 = None
        view_2: "f32[2]" = torch.ops.aten.view.default(sum_5, _shape_param_6);  sum_5 = _shape_param_6 = None
        convert_element_type_14: "bf16[2]" = torch.ops.prims.convert_element_type.default(view_2, torch.bfloat16);  view_2 = None
        view_3: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_7);  mm = _shape_param_7 = None
        convert_element_type_15: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_3, torch.float32);  view_3 = None
        convert_element_type_16: "f32[2, 4096]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_17: "f32[2]" = torch.ops.prims.convert_element_type.default(convert_element_type_14, torch.float32);  convert_element_type_14 = None
        mul_4: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_15, arg15_1);  arg15_1 = None
        mul_5: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_4, 4096)
        sum_6: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True)
        mul_6: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_4, arg16_1);  mul_4 = None
        sum_7: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_6, [2], True);  mul_6 = None
        mul_7: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg16_1, sum_7);  sum_7 = None
        sub_6: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_5, sum_6);  mul_5 = sum_6 = None
        sub_7: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_6, mul_7);  sub_6 = mul_7 = None
        mul_8: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg17_1, sub_7);  arg17_1 = sub_7 = None
        mul_9: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_15, arg16_1);  arg16_1 = None
        sum_8: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_9, [0, 1]);  mul_9 = None
        sum_9: "f32[4096]" = torch.ops.aten.sum.dim_IntList(convert_element_type_15, [0, 1]);  convert_element_type_15 = None
        convert_element_type_18: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16)
        view_4: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_18, _shape_param_8);  convert_element_type_18 = _shape_param_8 = None
        mm_2: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_4, arg18_1);  arg18_1 = None
        permute_1: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_3: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_1, arg19_1);  arg19_1 = None
        sum_10: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32)
        view_5: "f32[4096]" = torch.ops.aten.view.default(sum_10, _shape_param_9);  sum_10 = _shape_param_9 = None
        convert_element_type_19: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_2, _shape_param_10);  mm_2 = _shape_param_10 = None
        convert_element_type_20: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_21: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_22: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_19, torch.float32);  convert_element_type_19 = None
        view_7: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg20_1, _shape_param_11);  arg20_1 = _shape_param_11 = None
        mul_10: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_7, 0.5)
        mul_11: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_20, mul_10);  mul_10 = None
        convert_element_type_23: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_7, torch.float32)
        pow_1: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 3.0)
        mul_12: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_2: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_7, mul_12);  view_7 = mul_12 = None
        mul_13: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_2, 0.7978845608028654);  add_2 = None
        tanh: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_13);  mul_13 = None
        add_3: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_14: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_20, add_3);  convert_element_type_20 = add_3 = None
        convert_element_type_24: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_14, torch.bfloat16);  mul_14 = None
        mul_15: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_8: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_15);  mul_15 = None
        mul_16: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_11, sub_8);  mul_11 = sub_8 = None
        mul_17: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_16, 0.7978845608028654);  mul_16 = None
        convert_element_type_25: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_17, torch.bfloat16)
        mul_18: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_17, 0.044715);  mul_17 = None
        pow_2: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_23, 2.0);  convert_element_type_23 = None
        mul_19: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_20: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_18, mul_19);  mul_18 = mul_19 = None
        convert_element_type_26: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_20, torch.bfloat16);  mul_20 = None
        add_4: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_25, convert_element_type_26);  convert_element_type_25 = convert_element_type_26 = None
        mul_21: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 0.5);  convert_element_type_24 = None
        add_5: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_4, mul_21);  add_4 = mul_21 = None
        view_8: "bf16[128, 16384]" = torch.ops.aten.view.default(add_5, _shape_param_12);  add_5 = _shape_param_12 = None
        mm_4: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_8, arg21_1);  arg21_1 = None
        permute_2: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_8, [1, 0])
        mm_5: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_2, arg22_1);  permute_2 = None
        sum_11: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_8, [0], True, dtype = torch.float32);  view_8 = None
        view_9: "f32[16384]" = torch.ops.aten.view.default(sum_11, _shape_param_13);  sum_11 = _shape_param_13 = None
        convert_element_type_27: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_9, torch.bfloat16);  view_9 = None
        view_10: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_4, _shape_param_14);  mm_4 = _shape_param_14 = None
        convert_element_type_28: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_29: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_30: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_27, torch.float32);  convert_element_type_27 = None
        mm_6: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_1, arg23_1);  permute_1 = arg23_1 = None
        mm_7: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_4, arg24_1);  view_4 = arg24_1 = None
        view_11: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_7, _shape_param_15);  mm_7 = _shape_param_15 = None
        convert_element_type_31: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        view_12: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_11, _shape_param_16);  view_11 = _shape_param_16 = None
        permute_3: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_12, [0, 2, 1, 3]);  view_12 = None
        view_13: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_3, _shape_param_17);  permute_3 = _shape_param_17 = None
        bmm: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg25_1, view_13);  arg25_1 = None
        bmm_1: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_13, arg26_1);  view_13 = arg26_1 = None
        view_14: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm, _shape_param_18);  bmm = _shape_param_18 = None
        view_15: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_19);  bmm_1 = _shape_param_19 = None
        convert_element_type_32: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_15, torch.float32);  view_15 = None
        mul_22: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_32, arg27_1);  convert_element_type_32 = None
        sum_12: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [-1], True)
        neg: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg27_1);  arg27_1 = None
        fma: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_12, mul_22);  neg = sum_12 = mul_22 = None
        convert_element_type_33: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        div_3: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_33, 16.0);  convert_element_type_33 = None
        view_16: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_3, _shape_param_20);  div_3 = _shape_param_20 = None
        bmm_2: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg28_1, view_16);  arg28_1 = None
        bmm_3: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_16, arg29_1);  view_16 = arg29_1 = None
        view_17: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_21);  bmm_2 = _shape_param_21 = None
        view_18: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_3, _shape_param_22);  bmm_3 = _shape_param_22 = None
        convert_element_type_34: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_17, torch.float32);  view_17 = None
        permute_4: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_34, [0, 1, 3, 2]);  convert_element_type_34 = None
        convert_element_type_35: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_4, torch.bfloat16);  permute_4 = None
        convert_element_type_36: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_18, torch.bfloat16);  view_18 = None
        permute_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_36, [0, 2, 1, 3]);  convert_element_type_36 = None
        permute_6: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_35, [0, 2, 1, 3]);  convert_element_type_35 = None
        slice_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_5, 3, 0, 64)
        slice_2: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_5, 3, 64, 256);  permute_5 = None
        slice_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_6, 3, 0, 64)
        slice_4: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_6, 3, 64, 256);  permute_6 = None
        expand_2: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg30_1, _shape_param_23);  arg30_1 = _shape_param_23 = None
        clone: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_19: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone, _shape_param_24);  clone = _shape_param_24 = None
        mul_23: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, view_19)
        view_20: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_23, _shape_param_25);  mul_23 = _shape_param_25 = None
        select: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_20, -1, 0)
        select_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_20, -1, 1);  view_20 = None
        neg_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select);  select = None
        full_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.full.default(_shape_param_26, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_26 = None
        slice_scatter: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_1, 3, 1, 9223372036854775807, 2);  neg_1 = None
        slice_scatter_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_1, 3, 0, 9223372036854775807, 2);  select_1 = None
        add_6: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None
        expand_3: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg31_1, _shape_param_27);  arg31_1 = _shape_param_27 = None
        clone_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_21: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_28);  clone_1 = _shape_param_28 = None
        mul_24: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1, view_21);  slice_1 = None
        add_7: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_6, mul_24);  add_6 = mul_24 = None
        mul_25: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_3, view_19);  view_19 = None
        view_22: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_25, _shape_param_29);  mul_25 = _shape_param_29 = None
        select_2: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_22, -1, 0)
        select_3: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_22, -1, 1);  view_22 = None
        neg_2: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_2);  select_2 = None
        slice_scatter_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_2, 3, 1, 9223372036854775807, 2);  neg_2 = None
        slice_scatter_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_3, 3, 0, 9223372036854775807, 2);  select_3 = None
        add_8: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None
        mul_26: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_3, view_21);  slice_3 = view_21 = None
        add_9: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_8, mul_26);  add_8 = mul_26 = None
        full_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.full.default(_shape_param_30, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_30 = None
        slice_scatter_4: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_2, 3, 64, 9223372036854775807);  slice_2 = None
        slice_scatter_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_7, 3, 0, 64);  add_7 = None
        add_10: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_5);  slice_scatter_4 = slice_scatter_5 = None
        slice_scatter_6: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_4, 3, 64, 9223372036854775807);  slice_4 = None
        slice_scatter_7: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_9, 3, 0, 64);  add_9 = None
        add_11: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_6, slice_scatter_7);  slice_scatter_6 = slice_scatter_7 = None
        permute_7: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_23: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_2, _shape_param_31);  clone_2 = _shape_param_31 = None
        view_24: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_11, _shape_param_32);  add_11 = _shape_param_32 = None
        view_25: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_10, _shape_param_33);  add_10 = _shape_param_33 = None
        view_26: "bf16[128, 4096]" = torch.ops.aten.view.default(view_23, _shape_param_34);  view_23 = _shape_param_34 = None
        permute_8: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_26, [1, 0])
        mm_8: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_8, arg22_1);  permute_8 = None
        mm_9: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_26, arg32_1);  view_26 = arg32_1 = None
        view_27: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_9, _shape_param_35);  mm_9 = _shape_param_35 = None
        convert_element_type_37: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        add_12: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_28, convert_element_type_37);  convert_element_type_28 = convert_element_type_37 = None
        convert_element_type_38: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        view_28: "bf16[128, 4096]" = torch.ops.aten.view.default(view_24, _shape_param_36);  view_24 = _shape_param_36 = None
        permute_9: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_10: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_9, arg22_1);  permute_9 = None
        mm_11: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_28, arg33_1);  view_28 = arg33_1 = None
        view_29: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_11, _shape_param_37);  mm_11 = _shape_param_37 = None
        convert_element_type_39: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        add_13: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_12, convert_element_type_39);  add_12 = convert_element_type_39 = None
        convert_element_type_40: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        view_30: "bf16[128, 4096]" = torch.ops.aten.view.default(view_25, _shape_param_38);  view_25 = _shape_param_38 = None
        permute_10: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_30, [1, 0])
        mm_12: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_10, arg22_1);  permute_10 = arg22_1 = None
        mm_13: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_30, arg34_1);  view_30 = arg34_1 = None
        view_31: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_13, _shape_param_39);  mm_13 = _shape_param_39 = None
        convert_element_type_41: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        add_14: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_13, convert_element_type_41);  add_13 = convert_element_type_41 = None
        convert_element_type_42: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        mul_27: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_14, arg35_1);  arg35_1 = None
        mul_28: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_27, 4096)
        sum_13: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True)
        mul_29: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_27, arg36_1);  mul_27 = None
        sum_14: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_29, [2], True);  mul_29 = None
        mul_30: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg36_1, sum_14);  sum_14 = None
        sub_9: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_28, sum_13);  mul_28 = sum_13 = None
        sub_10: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_9, mul_30);  sub_9 = mul_30 = None
        mul_31: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg37_1, sub_10);  arg37_1 = sub_10 = None
        mul_32: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_14, arg36_1);  arg36_1 = None
        sum_15: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_32, [0, 1]);  mul_32 = None
        sum_16: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_14, [0, 1]);  add_14 = None
        add_15: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_8, mul_31);  mul_8 = mul_31 = None
        convert_element_type_43: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_15, torch.bfloat16)
        view_32: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_43, _shape_param_40);  convert_element_type_43 = _shape_param_40 = None
        mm_14: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_32, arg38_1);  arg38_1 = None
        permute_11: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_15: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_11, arg39_1);  arg39_1 = None
        sum_17: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_32, [0], True, dtype = torch.float32)
        view_33: "f32[4096]" = torch.ops.aten.view.default(sum_17, _shape_param_41);  sum_17 = _shape_param_41 = None
        convert_element_type_44: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_33, torch.bfloat16);  view_33 = None
        view_34: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_14, _shape_param_42);  mm_14 = _shape_param_42 = None
        convert_element_type_45: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        convert_element_type_46: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_47: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_44, torch.float32);  convert_element_type_44 = None
        view_35: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg40_1, _shape_param_43);  arg40_1 = _shape_param_43 = None
        mul_33: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
        mul_34: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_45, mul_33);  mul_33 = None
        convert_element_type_48: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_35, torch.float32)
        pow_3: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_48, 3.0)
        mul_35: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_16: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_35, mul_35);  view_35 = mul_35 = None
        mul_36: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_16, 0.7978845608028654);  add_16 = None
        tanh_1: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_36);  mul_36 = None
        add_17: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_37: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_45, add_17);  convert_element_type_45 = add_17 = None
        convert_element_type_49: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        mul_38: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_11: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_38);  mul_38 = None
        mul_39: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_34, sub_11);  mul_34 = sub_11 = None
        mul_40: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_39, 0.7978845608028654);  mul_39 = None
        convert_element_type_50: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16)
        mul_41: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_40, 0.044715);  mul_40 = None
        pow_4: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_48, 2.0);  convert_element_type_48 = None
        mul_42: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_4, 3.0);  pow_4 = None
        mul_43: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
        convert_element_type_51: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None
        add_18: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_50, convert_element_type_51);  convert_element_type_50 = convert_element_type_51 = None
        mul_44: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 0.5);  convert_element_type_49 = None
        add_19: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_18, mul_44);  add_18 = mul_44 = None
        view_36: "bf16[128, 16384]" = torch.ops.aten.view.default(add_19, _shape_param_44);  add_19 = _shape_param_44 = None
        mm_16: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_36, arg41_1);  arg41_1 = None
        permute_12: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_36, [1, 0])
        mm_17: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_12, arg42_1);  permute_12 = None
        sum_18: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_36, [0], True, dtype = torch.float32);  view_36 = None
        view_37: "f32[16384]" = torch.ops.aten.view.default(sum_18, _shape_param_45);  sum_18 = _shape_param_45 = None
        convert_element_type_52: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_37, torch.bfloat16);  view_37 = None
        view_38: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_16, _shape_param_46);  mm_16 = _shape_param_46 = None
        convert_element_type_53: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None
        convert_element_type_54: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_55: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_52, torch.float32);  convert_element_type_52 = None
        mm_18: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_11, arg43_1);  permute_11 = arg43_1 = None
        mm_19: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_32, arg44_1);  view_32 = arg44_1 = None
        view_39: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_19, _shape_param_47);  mm_19 = _shape_param_47 = None
        convert_element_type_56: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        view_40: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_39, _shape_param_48);  view_39 = _shape_param_48 = None
        permute_13: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_40, [0, 2, 1, 3]);  view_40 = None
        view_41: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_13, _shape_param_49);  permute_13 = _shape_param_49 = None
        bmm_4: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg45_1, view_41);  arg45_1 = None
        bmm_5: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_41, arg46_1);  view_41 = arg46_1 = None
        view_42: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_4, _shape_param_50);  bmm_4 = _shape_param_50 = None
        view_43: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_5, _shape_param_51);  bmm_5 = _shape_param_51 = None
        convert_element_type_57: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_43, torch.float32);  view_43 = None
        mul_45: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_57, arg47_1);  convert_element_type_57 = None
        sum_19: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_45, [-1], True)
        neg_3: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg47_1);  arg47_1 = None
        fma_1: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_3, sum_19, mul_45);  neg_3 = sum_19 = mul_45 = None
        convert_element_type_58: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        div_4: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_58, 16.0);  convert_element_type_58 = None
        view_44: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_4, _shape_param_52);  div_4 = _shape_param_52 = None
        bmm_6: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg48_1, view_44);  arg48_1 = None
        bmm_7: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_44, arg49_1);  view_44 = arg49_1 = None
        view_45: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_53);  bmm_6 = _shape_param_53 = None
        view_46: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_7, _shape_param_54);  bmm_7 = _shape_param_54 = None
        convert_element_type_59: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        permute_14: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_59, [0, 1, 3, 2]);  convert_element_type_59 = None
        convert_element_type_60: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_14, torch.bfloat16);  permute_14 = None
        convert_element_type_61: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_46, torch.bfloat16);  view_46 = None
        permute_15: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_61, [0, 2, 1, 3]);  convert_element_type_61 = None
        permute_16: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_60, [0, 2, 1, 3]);  convert_element_type_60 = None
        slice_5: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_15, 3, 0, 64)
        slice_6: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_15, 3, 64, 256);  permute_15 = None
        slice_7: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_16, 3, 0, 64)
        slice_8: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_16, 3, 64, 256);  permute_16 = None
        expand_4: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg50_1, _shape_param_55);  arg50_1 = _shape_param_55 = None
        clone_3: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_47: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_3, _shape_param_56);  clone_3 = _shape_param_56 = None
        mul_46: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_5, view_47)
        view_48: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_46, _shape_param_57);  mul_46 = _shape_param_57 = None
        select_4: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_48, -1, 0)
        select_5: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_48, -1, 1);  view_48 = None
        neg_4: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_4);  select_4 = None
        slice_scatter_8: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_4, 3, 1, 9223372036854775807, 2);  neg_4 = None
        slice_scatter_9: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_5, 3, 0, 9223372036854775807, 2);  select_5 = None
        add_20: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_8, slice_scatter_9);  slice_scatter_8 = slice_scatter_9 = None
        expand_5: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg51_1, _shape_param_58);  arg51_1 = _shape_param_58 = None
        clone_4: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_49: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_59);  clone_4 = _shape_param_59 = None
        mul_47: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_5, view_49);  slice_5 = None
        add_21: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_20, mul_47);  add_20 = mul_47 = None
        mul_48: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_7, view_47);  view_47 = None
        view_50: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_48, _shape_param_60);  mul_48 = _shape_param_60 = None
        select_6: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_50, -1, 0)
        select_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_50, -1, 1);  view_50 = None
        neg_5: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_6);  select_6 = None
        slice_scatter_10: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_5, 3, 1, 9223372036854775807, 2);  neg_5 = None
        slice_scatter_11: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_7, 3, 0, 9223372036854775807, 2);  select_7 = None
        add_22: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_11);  slice_scatter_10 = slice_scatter_11 = None
        mul_49: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_7, view_49);  slice_7 = view_49 = None
        add_23: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_22, mul_49);  add_22 = mul_49 = None
        slice_scatter_12: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_6, 3, 64, 9223372036854775807);  slice_6 = None
        slice_scatter_13: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_21, 3, 0, 64);  add_21 = None
        add_24: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_12, slice_scatter_13);  slice_scatter_12 = slice_scatter_13 = None
        slice_scatter_14: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_8, 3, 64, 9223372036854775807);  slice_8 = None
        slice_scatter_15: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_23, 3, 0, 64);  add_23 = None
        add_25: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_14, slice_scatter_15);  slice_scatter_14 = slice_scatter_15 = None
        permute_17: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        clone_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_51: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_5, _shape_param_61);  clone_5 = _shape_param_61 = None
        view_52: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_25, _shape_param_62);  add_25 = _shape_param_62 = None
        view_53: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_24, _shape_param_63);  add_24 = _shape_param_63 = None
        view_54: "bf16[128, 4096]" = torch.ops.aten.view.default(view_51, _shape_param_64);  view_51 = _shape_param_64 = None
        permute_18: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_54, [1, 0])
        mm_20: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_18, arg42_1);  permute_18 = None
        mm_21: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_54, arg52_1);  view_54 = arg52_1 = None
        view_55: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_21, _shape_param_65);  mm_21 = _shape_param_65 = None
        convert_element_type_62: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        add_26: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_53, convert_element_type_62);  convert_element_type_53 = convert_element_type_62 = None
        convert_element_type_63: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        view_56: "bf16[128, 4096]" = torch.ops.aten.view.default(view_52, _shape_param_66);  view_52 = _shape_param_66 = None
        permute_19: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_56, [1, 0])
        mm_22: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_19, arg42_1);  permute_19 = None
        mm_23: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_56, arg53_1);  view_56 = arg53_1 = None
        view_57: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_23, _shape_param_67);  mm_23 = _shape_param_67 = None
        convert_element_type_64: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        add_27: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_26, convert_element_type_64);  add_26 = convert_element_type_64 = None
        convert_element_type_65: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        view_58: "bf16[128, 4096]" = torch.ops.aten.view.default(view_53, _shape_param_68);  view_53 = _shape_param_68 = None
        permute_20: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_58, [1, 0])
        mm_24: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_20, arg42_1);  permute_20 = arg42_1 = None
        mm_25: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_58, arg54_1);  view_58 = arg54_1 = None
        view_59: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_25, _shape_param_69);  mm_25 = _shape_param_69 = None
        convert_element_type_66: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_59, torch.float32);  view_59 = None
        add_28: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_27, convert_element_type_66);  add_27 = convert_element_type_66 = None
        convert_element_type_67: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        mul_50: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_28, arg55_1);  arg55_1 = None
        mul_51: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_50, 4096)
        sum_20: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_50, [2], True)
        mul_52: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_50, arg56_1);  mul_50 = None
        sum_21: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_52, [2], True);  mul_52 = None
        mul_53: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg56_1, sum_21);  sum_21 = None
        sub_12: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_51, sum_20);  mul_51 = sum_20 = None
        sub_13: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_12, mul_53);  sub_12 = mul_53 = None
        mul_54: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg57_1, sub_13);  arg57_1 = sub_13 = None
        mul_55: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_28, arg56_1);  arg56_1 = None
        sum_22: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_55, [0, 1]);  mul_55 = None
        sum_23: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_28, [0, 1]);  add_28 = None
        add_29: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_15, mul_54);  add_15 = mul_54 = None
        convert_element_type_68: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16)
        view_60: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_68, _shape_param_70);  convert_element_type_68 = _shape_param_70 = None
        mm_26: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_60, arg58_1);  arg58_1 = None
        permute_21: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_60, [1, 0])
        mm_27: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_21, arg59_1);  arg59_1 = None
        sum_24: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_60, [0], True, dtype = torch.float32)
        view_61: "f32[4096]" = torch.ops.aten.view.default(sum_24, _shape_param_71);  sum_24 = _shape_param_71 = None
        convert_element_type_69: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_61, torch.bfloat16);  view_61 = None
        view_62: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_26, _shape_param_72);  mm_26 = _shape_param_72 = None
        convert_element_type_70: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_62, torch.float32);  view_62 = None
        convert_element_type_71: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_72: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_69, torch.float32);  convert_element_type_69 = None
        view_63: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg60_1, _shape_param_73);  arg60_1 = _shape_param_73 = None
        mul_56: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
        mul_57: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_70, mul_56);  mul_56 = None
        convert_element_type_73: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_63, torch.float32)
        pow_5: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 3.0)
        mul_58: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_30: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_63, mul_58);  view_63 = mul_58 = None
        mul_59: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
        tanh_2: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_59);  mul_59 = None
        add_31: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_60: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_70, add_31);  convert_element_type_70 = add_31 = None
        convert_element_type_74: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_60, torch.bfloat16);  mul_60 = None
        mul_61: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_14: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_61);  mul_61 = None
        mul_62: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_57, sub_14);  mul_57 = sub_14 = None
        mul_63: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_62, 0.7978845608028654);  mul_62 = None
        convert_element_type_75: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16)
        mul_64: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_63, 0.044715);  mul_63 = None
        pow_6: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_73, 2.0);  convert_element_type_73 = None
        mul_65: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_6, 3.0);  pow_6 = None
        mul_66: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
        convert_element_type_76: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_66, torch.bfloat16);  mul_66 = None
        add_32: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_75, convert_element_type_76);  convert_element_type_75 = convert_element_type_76 = None
        mul_67: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 0.5);  convert_element_type_74 = None
        add_33: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_32, mul_67);  add_32 = mul_67 = None
        view_64: "bf16[128, 16384]" = torch.ops.aten.view.default(add_33, _shape_param_74);  add_33 = _shape_param_74 = None
        mm_28: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_64, arg61_1);  arg61_1 = None
        permute_22: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_64, [1, 0])
        mm_29: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_22, arg62_1);  permute_22 = None
        sum_25: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_64, [0], True, dtype = torch.float32);  view_64 = None
        view_65: "f32[16384]" = torch.ops.aten.view.default(sum_25, _shape_param_75);  sum_25 = _shape_param_75 = None
        convert_element_type_77: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_65, torch.bfloat16);  view_65 = None
        view_66: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_28, _shape_param_76);  mm_28 = _shape_param_76 = None
        convert_element_type_78: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_66, torch.float32);  view_66 = None
        convert_element_type_79: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_80: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_77, torch.float32);  convert_element_type_77 = None
        mm_30: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_21, arg63_1);  permute_21 = arg63_1 = None
        mm_31: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_60, arg64_1);  view_60 = arg64_1 = None
        view_67: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_31, _shape_param_77);  mm_31 = _shape_param_77 = None
        convert_element_type_81: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_30, torch.float32);  mm_30 = None
        view_68: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_67, _shape_param_78);  view_67 = _shape_param_78 = None
        permute_23: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        view_69: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_23, _shape_param_79);  permute_23 = _shape_param_79 = None
        bmm_8: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg65_1, view_69);  arg65_1 = None
        bmm_9: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_69, arg66_1);  view_69 = arg66_1 = None
        view_70: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_8, _shape_param_80);  bmm_8 = _shape_param_80 = None
        view_71: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_9, _shape_param_81);  bmm_9 = _shape_param_81 = None
        convert_element_type_82: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_71, torch.float32);  view_71 = None
        mul_68: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_82, arg67_1);  convert_element_type_82 = None
        sum_26: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_68, [-1], True)
        neg_6: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg67_1);  arg67_1 = None
        fma_2: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_6, sum_26, mul_68);  neg_6 = sum_26 = mul_68 = None
        convert_element_type_83: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        div_5: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_83, 16.0);  convert_element_type_83 = None
        view_72: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_5, _shape_param_82);  div_5 = _shape_param_82 = None
        bmm_10: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg68_1, view_72);  arg68_1 = None
        bmm_11: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_72, arg69_1);  view_72 = arg69_1 = None
        view_73: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_83);  bmm_10 = _shape_param_83 = None
        view_74: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_11, _shape_param_84);  bmm_11 = _shape_param_84 = None
        convert_element_type_84: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        permute_24: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_84, [0, 1, 3, 2]);  convert_element_type_84 = None
        convert_element_type_85: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_24, torch.bfloat16);  permute_24 = None
        convert_element_type_86: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_74, torch.bfloat16);  view_74 = None
        permute_25: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_86, [0, 2, 1, 3]);  convert_element_type_86 = None
        permute_26: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_85, [0, 2, 1, 3]);  convert_element_type_85 = None
        slice_9: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_25, 3, 0, 64)
        slice_10: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_25, 3, 64, 256);  permute_25 = None
        slice_11: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_26, 3, 0, 64)
        slice_12: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_26, 3, 64, 256);  permute_26 = None
        expand_6: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg70_1, _shape_param_85);  arg70_1 = _shape_param_85 = None
        clone_6: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_75: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_86);  clone_6 = _shape_param_86 = None
        mul_69: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_9, view_75)
        view_76: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_69, _shape_param_87);  mul_69 = _shape_param_87 = None
        select_8: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_76, -1, 0)
        select_9: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_76, -1, 1);  view_76 = None
        neg_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_8);  select_8 = None
        slice_scatter_16: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_7, 3, 1, 9223372036854775807, 2);  neg_7 = None
        slice_scatter_17: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_9, 3, 0, 9223372036854775807, 2);  select_9 = None
        add_34: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_16, slice_scatter_17);  slice_scatter_16 = slice_scatter_17 = None
        expand_7: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg71_1, _shape_param_88);  arg71_1 = _shape_param_88 = None
        clone_7: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_77: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_7, _shape_param_89);  clone_7 = _shape_param_89 = None
        mul_70: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_9, view_77);  slice_9 = None
        add_35: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_34, mul_70);  add_34 = mul_70 = None
        mul_71: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_11, view_75);  view_75 = None
        view_78: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_71, _shape_param_90);  mul_71 = _shape_param_90 = None
        select_10: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_78, -1, 0)
        select_11: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_78, -1, 1);  view_78 = None
        neg_8: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_10);  select_10 = None
        slice_scatter_18: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_8, 3, 1, 9223372036854775807, 2);  neg_8 = None
        slice_scatter_19: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_11, 3, 0, 9223372036854775807, 2);  select_11 = None
        add_36: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_18, slice_scatter_19);  slice_scatter_18 = slice_scatter_19 = None
        mul_72: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_11, view_77);  slice_11 = view_77 = None
        add_37: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_36, mul_72);  add_36 = mul_72 = None
        slice_scatter_20: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_10, 3, 64, 9223372036854775807);  slice_10 = None
        slice_scatter_21: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_35, 3, 0, 64);  add_35 = None
        add_38: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_20, slice_scatter_21);  slice_scatter_20 = slice_scatter_21 = None
        slice_scatter_22: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_12, 3, 64, 9223372036854775807);  slice_12 = None
        slice_scatter_23: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_37, 3, 0, 64);  add_37 = None
        add_39: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_22, slice_scatter_23);  slice_scatter_22 = slice_scatter_23 = None
        permute_27: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_8: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_79: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_8, _shape_param_91);  clone_8 = _shape_param_91 = None
        view_80: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_39, _shape_param_92);  add_39 = _shape_param_92 = None
        view_81: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_38, _shape_param_93);  add_38 = _shape_param_93 = None
        view_82: "bf16[128, 4096]" = torch.ops.aten.view.default(view_79, _shape_param_94);  view_79 = _shape_param_94 = None
        permute_28: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_82, [1, 0])
        mm_32: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_28, arg62_1);  permute_28 = None
        mm_33: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_82, arg72_1);  view_82 = arg72_1 = None
        view_83: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_33, _shape_param_95);  mm_33 = _shape_param_95 = None
        convert_element_type_87: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_83, torch.float32);  view_83 = None
        add_40: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_78, convert_element_type_87);  convert_element_type_78 = convert_element_type_87 = None
        convert_element_type_88: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_32, torch.float32);  mm_32 = None
        view_84: "bf16[128, 4096]" = torch.ops.aten.view.default(view_80, _shape_param_96);  view_80 = _shape_param_96 = None
        permute_29: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_84, [1, 0])
        mm_34: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_29, arg62_1);  permute_29 = None
        mm_35: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_84, arg73_1);  view_84 = arg73_1 = None
        view_85: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_35, _shape_param_97);  mm_35 = _shape_param_97 = None
        convert_element_type_89: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        add_41: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_40, convert_element_type_89);  add_40 = convert_element_type_89 = None
        convert_element_type_90: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        view_86: "bf16[128, 4096]" = torch.ops.aten.view.default(view_81, _shape_param_98);  view_81 = _shape_param_98 = None
        permute_30: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_86, [1, 0])
        mm_36: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_30, arg62_1);  permute_30 = arg62_1 = None
        mm_37: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_86, arg74_1);  view_86 = arg74_1 = None
        view_87: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_37, _shape_param_99);  mm_37 = _shape_param_99 = None
        convert_element_type_91: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_87, torch.float32);  view_87 = None
        add_42: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_41, convert_element_type_91);  add_41 = convert_element_type_91 = None
        convert_element_type_92: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        mul_73: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_42, arg75_1);  arg75_1 = None
        mul_74: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_73, 4096)
        sum_27: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_73, [2], True)
        mul_75: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_73, arg76_1);  mul_73 = None
        sum_28: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_75, [2], True);  mul_75 = None
        mul_76: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg76_1, sum_28);  sum_28 = None
        sub_15: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_74, sum_27);  mul_74 = sum_27 = None
        sub_16: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_15, mul_76);  sub_15 = mul_76 = None
        mul_77: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg77_1, sub_16);  arg77_1 = sub_16 = None
        mul_78: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_42, arg76_1);  arg76_1 = None
        sum_29: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_78, [0, 1]);  mul_78 = None
        sum_30: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_42, [0, 1]);  add_42 = None
        add_43: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_29, mul_77);  add_29 = mul_77 = None
        convert_element_type_93: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16)
        view_88: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_93, _shape_param_100);  convert_element_type_93 = _shape_param_100 = None
        mm_38: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_88, arg78_1);  arg78_1 = None
        permute_31: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_88, [1, 0])
        mm_39: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_31, arg79_1);  arg79_1 = None
        sum_31: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_88, [0], True, dtype = torch.float32)
        view_89: "f32[4096]" = torch.ops.aten.view.default(sum_31, _shape_param_101);  sum_31 = _shape_param_101 = None
        convert_element_type_94: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_89, torch.bfloat16);  view_89 = None
        view_90: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_38, _shape_param_102);  mm_38 = _shape_param_102 = None
        convert_element_type_95: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_90, torch.float32);  view_90 = None
        convert_element_type_96: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_97: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_94, torch.float32);  convert_element_type_94 = None
        view_91: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg80_1, _shape_param_103);  arg80_1 = _shape_param_103 = None
        mul_79: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_91, 0.5)
        mul_80: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_95, mul_79);  mul_79 = None
        convert_element_type_98: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_91, torch.float32)
        pow_7: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_98, 3.0)
        mul_81: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_44: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_91, mul_81);  view_91 = mul_81 = None
        mul_82: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
        tanh_3: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_82);  mul_82 = None
        add_45: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_83: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_95, add_45);  convert_element_type_95 = add_45 = None
        convert_element_type_99: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_83, torch.bfloat16);  mul_83 = None
        mul_84: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_17: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_84);  mul_84 = None
        mul_85: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_80, sub_17);  mul_80 = sub_17 = None
        mul_86: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_85, 0.7978845608028654);  mul_85 = None
        convert_element_type_100: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_86, torch.bfloat16)
        mul_87: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_86, 0.044715);  mul_86 = None
        pow_8: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_98, 2.0);  convert_element_type_98 = None
        mul_88: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_8, 3.0);  pow_8 = None
        mul_89: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_87, mul_88);  mul_87 = mul_88 = None
        convert_element_type_101: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_89, torch.bfloat16);  mul_89 = None
        add_46: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_100, convert_element_type_101);  convert_element_type_100 = convert_element_type_101 = None
        mul_90: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_99, 0.5);  convert_element_type_99 = None
        add_47: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_46, mul_90);  add_46 = mul_90 = None
        view_92: "bf16[128, 16384]" = torch.ops.aten.view.default(add_47, _shape_param_104);  add_47 = _shape_param_104 = None
        mm_40: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_92, arg81_1);  arg81_1 = None
        permute_32: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_92, [1, 0])
        mm_41: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_32, arg82_1);  permute_32 = None
        sum_32: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_92, [0], True, dtype = torch.float32);  view_92 = None
        view_93: "f32[16384]" = torch.ops.aten.view.default(sum_32, _shape_param_105);  sum_32 = _shape_param_105 = None
        convert_element_type_102: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_93, torch.bfloat16);  view_93 = None
        view_94: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_40, _shape_param_106);  mm_40 = _shape_param_106 = None
        convert_element_type_103: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_94, torch.float32);  view_94 = None
        convert_element_type_104: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_105: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_102, torch.float32);  convert_element_type_102 = None
        mm_42: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_31, arg83_1);  permute_31 = arg83_1 = None
        mm_43: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_88, arg84_1);  view_88 = arg84_1 = None
        view_95: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_43, _shape_param_107);  mm_43 = _shape_param_107 = None
        convert_element_type_106: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_42, torch.float32);  mm_42 = None
        view_96: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_95, _shape_param_108);  view_95 = _shape_param_108 = None
        permute_33: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        view_97: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_33, _shape_param_109);  permute_33 = _shape_param_109 = None
        bmm_12: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg85_1, view_97);  arg85_1 = None
        bmm_13: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_97, arg86_1);  view_97 = arg86_1 = None
        view_98: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_12, _shape_param_110);  bmm_12 = _shape_param_110 = None
        view_99: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_13, _shape_param_111);  bmm_13 = _shape_param_111 = None
        convert_element_type_107: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        mul_91: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_107, arg87_1);  convert_element_type_107 = None
        sum_33: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_91, [-1], True)
        neg_9: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg87_1);  arg87_1 = None
        fma_3: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_9, sum_33, mul_91);  neg_9 = sum_33 = mul_91 = None
        convert_element_type_108: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        div_6: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_108, 16.0);  convert_element_type_108 = None
        view_100: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_6, _shape_param_112);  div_6 = _shape_param_112 = None
        bmm_14: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg88_1, view_100);  arg88_1 = None
        bmm_15: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_100, arg89_1);  view_100 = arg89_1 = None
        view_101: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_113);  bmm_14 = _shape_param_113 = None
        view_102: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_15, _shape_param_114);  bmm_15 = _shape_param_114 = None
        convert_element_type_109: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_101, torch.float32);  view_101 = None
        permute_34: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_109, [0, 1, 3, 2]);  convert_element_type_109 = None
        convert_element_type_110: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_34, torch.bfloat16);  permute_34 = None
        convert_element_type_111: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_102, torch.bfloat16);  view_102 = None
        permute_35: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_111, [0, 2, 1, 3]);  convert_element_type_111 = None
        permute_36: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_110, [0, 2, 1, 3]);  convert_element_type_110 = None
        slice_13: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_35, 3, 0, 64)
        slice_14: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_35, 3, 64, 256);  permute_35 = None
        slice_15: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_36, 3, 0, 64)
        slice_16: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_36, 3, 64, 256);  permute_36 = None
        expand_8: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg90_1, _shape_param_115);  arg90_1 = _shape_param_115 = None
        clone_9: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_103: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_9, _shape_param_116);  clone_9 = _shape_param_116 = None
        mul_92: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_13, view_103)
        view_104: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_92, _shape_param_117);  mul_92 = _shape_param_117 = None
        select_12: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_104, -1, 0)
        select_13: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_104, -1, 1);  view_104 = None
        neg_10: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_12);  select_12 = None
        slice_scatter_24: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_10, 3, 1, 9223372036854775807, 2);  neg_10 = None
        slice_scatter_25: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_13, 3, 0, 9223372036854775807, 2);  select_13 = None
        add_48: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_24, slice_scatter_25);  slice_scatter_24 = slice_scatter_25 = None
        expand_9: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg91_1, _shape_param_118);  arg91_1 = _shape_param_118 = None
        clone_10: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_105: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_119);  clone_10 = _shape_param_119 = None
        mul_93: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_13, view_105);  slice_13 = None
        add_49: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_48, mul_93);  add_48 = mul_93 = None
        mul_94: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_15, view_103);  view_103 = None
        view_106: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_94, _shape_param_120);  mul_94 = _shape_param_120 = None
        select_14: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_106, -1, 0)
        select_15: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_106, -1, 1);  view_106 = None
        neg_11: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_14);  select_14 = None
        slice_scatter_26: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_11, 3, 1, 9223372036854775807, 2);  neg_11 = None
        slice_scatter_27: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_15, 3, 0, 9223372036854775807, 2);  select_15 = None
        add_50: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_26, slice_scatter_27);  slice_scatter_26 = slice_scatter_27 = None
        mul_95: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_15, view_105);  slice_15 = view_105 = None
        add_51: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_50, mul_95);  add_50 = mul_95 = None
        slice_scatter_28: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_14, 3, 64, 9223372036854775807);  slice_14 = None
        slice_scatter_29: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_49, 3, 0, 64);  add_49 = None
        add_52: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_28, slice_scatter_29);  slice_scatter_28 = slice_scatter_29 = None
        slice_scatter_30: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_16, 3, 64, 9223372036854775807);  slice_16 = None
        slice_scatter_31: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_51, 3, 0, 64);  add_51 = None
        add_53: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_30, slice_scatter_31);  slice_scatter_30 = slice_scatter_31 = None
        permute_37: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        clone_11: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_107: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_11, _shape_param_121);  clone_11 = _shape_param_121 = None
        view_108: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_53, _shape_param_122);  add_53 = _shape_param_122 = None
        view_109: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_52, _shape_param_123);  add_52 = _shape_param_123 = None
        view_110: "bf16[128, 4096]" = torch.ops.aten.view.default(view_107, _shape_param_124);  view_107 = _shape_param_124 = None
        permute_38: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_110, [1, 0])
        mm_44: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_38, arg82_1);  permute_38 = None
        mm_45: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_110, arg92_1);  view_110 = arg92_1 = None
        view_111: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_45, _shape_param_125);  mm_45 = _shape_param_125 = None
        convert_element_type_112: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_111, torch.float32);  view_111 = None
        add_54: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_103, convert_element_type_112);  convert_element_type_103 = convert_element_type_112 = None
        convert_element_type_113: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        view_112: "bf16[128, 4096]" = torch.ops.aten.view.default(view_108, _shape_param_126);  view_108 = _shape_param_126 = None
        permute_39: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_112, [1, 0])
        mm_46: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_39, arg82_1);  permute_39 = None
        mm_47: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_112, arg93_1);  view_112 = arg93_1 = None
        view_113: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_47, _shape_param_127);  mm_47 = _shape_param_127 = None
        convert_element_type_114: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_113, torch.float32);  view_113 = None
        add_55: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_54, convert_element_type_114);  add_54 = convert_element_type_114 = None
        convert_element_type_115: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        view_114: "bf16[128, 4096]" = torch.ops.aten.view.default(view_109, _shape_param_128);  view_109 = _shape_param_128 = None
        permute_40: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_114, [1, 0])
        mm_48: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_40, arg82_1);  permute_40 = arg82_1 = None
        mm_49: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_114, arg94_1);  view_114 = arg94_1 = None
        view_115: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_49, _shape_param_129);  mm_49 = _shape_param_129 = None
        convert_element_type_116: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_115, torch.float32);  view_115 = None
        add_56: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_55, convert_element_type_116);  add_55 = convert_element_type_116 = None
        convert_element_type_117: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_48, torch.float32);  mm_48 = None
        mul_96: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_56, arg95_1);  arg95_1 = None
        mul_97: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_96, 4096)
        sum_34: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_96, [2], True)
        mul_98: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_96, arg96_1);  mul_96 = None
        sum_35: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_98, [2], True);  mul_98 = None
        mul_99: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg96_1, sum_35);  sum_35 = None
        sub_18: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_97, sum_34);  mul_97 = sum_34 = None
        sub_19: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_18, mul_99);  sub_18 = mul_99 = None
        mul_100: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg97_1, sub_19);  arg97_1 = sub_19 = None
        mul_101: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_56, arg96_1);  arg96_1 = None
        sum_36: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_101, [0, 1]);  mul_101 = None
        sum_37: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_56, [0, 1]);  add_56 = None
        add_57: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_43, mul_100);  add_43 = mul_100 = None
        convert_element_type_118: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16)
        view_116: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_118, _shape_param_130);  convert_element_type_118 = _shape_param_130 = None
        mm_50: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_116, arg98_1);  arg98_1 = None
        permute_41: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_116, [1, 0])
        mm_51: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_41, arg99_1);  arg99_1 = None
        sum_38: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_116, [0], True, dtype = torch.float32)
        view_117: "f32[4096]" = torch.ops.aten.view.default(sum_38, _shape_param_131);  sum_38 = _shape_param_131 = None
        convert_element_type_119: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_117, torch.bfloat16);  view_117 = None
        view_118: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_50, _shape_param_132);  mm_50 = _shape_param_132 = None
        convert_element_type_120: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        convert_element_type_121: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_122: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_119, torch.float32);  convert_element_type_119 = None
        view_119: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg100_1, _shape_param_133);  arg100_1 = _shape_param_133 = None
        mul_102: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_119, 0.5)
        mul_103: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_120, mul_102);  mul_102 = None
        convert_element_type_123: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_119, torch.float32)
        pow_9: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_123, 3.0)
        mul_104: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_58: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_119, mul_104);  view_119 = mul_104 = None
        mul_105: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_58, 0.7978845608028654);  add_58 = None
        tanh_4: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_105);  mul_105 = None
        add_59: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_106: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_120, add_59);  convert_element_type_120 = add_59 = None
        convert_element_type_124: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_106, torch.bfloat16);  mul_106 = None
        mul_107: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_20: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_107);  mul_107 = None
        mul_108: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_103, sub_20);  mul_103 = sub_20 = None
        mul_109: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_108, 0.7978845608028654);  mul_108 = None
        convert_element_type_125: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_109, torch.bfloat16)
        mul_110: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_109, 0.044715);  mul_109 = None
        pow_10: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_123, 2.0);  convert_element_type_123 = None
        mul_111: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_10, 3.0);  pow_10 = None
        mul_112: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_110, mul_111);  mul_110 = mul_111 = None
        convert_element_type_126: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        add_60: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_125, convert_element_type_126);  convert_element_type_125 = convert_element_type_126 = None
        mul_113: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 0.5);  convert_element_type_124 = None
        add_61: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_60, mul_113);  add_60 = mul_113 = None
        view_120: "bf16[128, 16384]" = torch.ops.aten.view.default(add_61, _shape_param_134);  add_61 = _shape_param_134 = None
        mm_52: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_120, arg101_1);  arg101_1 = None
        permute_42: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_120, [1, 0])
        mm_53: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_42, arg102_1);  permute_42 = None
        sum_39: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_120, [0], True, dtype = torch.float32);  view_120 = None
        view_121: "f32[16384]" = torch.ops.aten.view.default(sum_39, _shape_param_135);  sum_39 = _shape_param_135 = None
        convert_element_type_127: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_121, torch.bfloat16);  view_121 = None
        view_122: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_52, _shape_param_136);  mm_52 = _shape_param_136 = None
        convert_element_type_128: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_122, torch.float32);  view_122 = None
        convert_element_type_129: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_130: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_127, torch.float32);  convert_element_type_127 = None
        mm_54: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_41, arg103_1);  permute_41 = arg103_1 = None
        mm_55: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_116, arg104_1);  view_116 = arg104_1 = None
        view_123: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_55, _shape_param_137);  mm_55 = _shape_param_137 = None
        convert_element_type_131: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        view_124: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_123, _shape_param_138);  view_123 = _shape_param_138 = None
        permute_43: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
        view_125: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_43, _shape_param_139);  permute_43 = _shape_param_139 = None
        bmm_16: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg105_1, view_125);  arg105_1 = None
        bmm_17: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_125, arg106_1);  view_125 = arg106_1 = None
        view_126: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_16, _shape_param_140);  bmm_16 = _shape_param_140 = None
        view_127: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_17, _shape_param_141);  bmm_17 = _shape_param_141 = None
        convert_element_type_132: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_127, torch.float32);  view_127 = None
        mul_114: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_132, arg107_1);  convert_element_type_132 = None
        sum_40: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_114, [-1], True)
        neg_12: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg107_1);  arg107_1 = None
        fma_4: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_12, sum_40, mul_114);  neg_12 = sum_40 = mul_114 = None
        convert_element_type_133: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        div_7: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_133, 16.0);  convert_element_type_133 = None
        view_128: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_7, _shape_param_142);  div_7 = _shape_param_142 = None
        bmm_18: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg108_1, view_128);  arg108_1 = None
        bmm_19: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_128, arg109_1);  view_128 = arg109_1 = None
        view_129: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_143);  bmm_18 = _shape_param_143 = None
        view_130: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_19, _shape_param_144);  bmm_19 = _shape_param_144 = None
        convert_element_type_134: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        permute_44: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_134, [0, 1, 3, 2]);  convert_element_type_134 = None
        convert_element_type_135: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_44, torch.bfloat16);  permute_44 = None
        convert_element_type_136: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_130, torch.bfloat16);  view_130 = None
        permute_45: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_136, [0, 2, 1, 3]);  convert_element_type_136 = None
        permute_46: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_135, [0, 2, 1, 3]);  convert_element_type_135 = None
        slice_17: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_45, 3, 0, 64)
        slice_18: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_45, 3, 64, 256);  permute_45 = None
        slice_19: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_46, 3, 0, 64)
        slice_20: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_46, 3, 64, 256);  permute_46 = None
        expand_10: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg110_1, _shape_param_145);  arg110_1 = _shape_param_145 = None
        clone_12: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_131: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_146);  clone_12 = _shape_param_146 = None
        mul_115: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_17, view_131)
        view_132: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_115, _shape_param_147);  mul_115 = _shape_param_147 = None
        select_16: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_132, -1, 0)
        select_17: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_132, -1, 1);  view_132 = None
        neg_13: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_16);  select_16 = None
        slice_scatter_32: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_13, 3, 1, 9223372036854775807, 2);  neg_13 = None
        slice_scatter_33: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_17, 3, 0, 9223372036854775807, 2);  select_17 = None
        add_62: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_33);  slice_scatter_32 = slice_scatter_33 = None
        expand_11: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg111_1, _shape_param_148);  arg111_1 = _shape_param_148 = None
        clone_13: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_133: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_13, _shape_param_149);  clone_13 = _shape_param_149 = None
        mul_116: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_17, view_133);  slice_17 = None
        add_63: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_62, mul_116);  add_62 = mul_116 = None
        mul_117: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_19, view_131);  view_131 = None
        view_134: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_117, _shape_param_150);  mul_117 = _shape_param_150 = None
        select_18: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_134, -1, 0)
        select_19: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_134, -1, 1);  view_134 = None
        neg_14: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_18);  select_18 = None
        slice_scatter_34: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_14, 3, 1, 9223372036854775807, 2);  neg_14 = None
        slice_scatter_35: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_19, 3, 0, 9223372036854775807, 2);  select_19 = None
        add_64: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_34, slice_scatter_35);  slice_scatter_34 = slice_scatter_35 = None
        mul_118: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_19, view_133);  slice_19 = view_133 = None
        add_65: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_64, mul_118);  add_64 = mul_118 = None
        slice_scatter_36: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_18, 3, 64, 9223372036854775807);  slice_18 = None
        slice_scatter_37: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_63, 3, 0, 64);  add_63 = None
        add_66: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_36, slice_scatter_37);  slice_scatter_36 = slice_scatter_37 = None
        slice_scatter_38: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_20, 3, 64, 9223372036854775807);  slice_20 = None
        slice_scatter_39: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_65, 3, 0, 64);  add_65 = None
        add_67: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_38, slice_scatter_39);  slice_scatter_38 = slice_scatter_39 = None
        permute_47: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_14: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_135: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_14, _shape_param_151);  clone_14 = _shape_param_151 = None
        view_136: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_67, _shape_param_152);  add_67 = _shape_param_152 = None
        view_137: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_66, _shape_param_153);  add_66 = _shape_param_153 = None
        view_138: "bf16[128, 4096]" = torch.ops.aten.view.default(view_135, _shape_param_154);  view_135 = _shape_param_154 = None
        permute_48: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_138, [1, 0])
        mm_56: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_48, arg102_1);  permute_48 = None
        mm_57: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_138, arg112_1);  view_138 = arg112_1 = None
        view_139: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_57, _shape_param_155);  mm_57 = _shape_param_155 = None
        convert_element_type_137: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32);  view_139 = None
        add_68: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_128, convert_element_type_137);  convert_element_type_128 = convert_element_type_137 = None
        convert_element_type_138: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        view_140: "bf16[128, 4096]" = torch.ops.aten.view.default(view_136, _shape_param_156);  view_136 = _shape_param_156 = None
        permute_49: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_140, [1, 0])
        mm_58: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_49, arg102_1);  permute_49 = None
        mm_59: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_140, arg113_1);  view_140 = arg113_1 = None
        view_141: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_59, _shape_param_157);  mm_59 = _shape_param_157 = None
        convert_element_type_139: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_141, torch.float32);  view_141 = None
        add_69: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_68, convert_element_type_139);  add_68 = convert_element_type_139 = None
        convert_element_type_140: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        view_142: "bf16[128, 4096]" = torch.ops.aten.view.default(view_137, _shape_param_158);  view_137 = _shape_param_158 = None
        permute_50: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_142, [1, 0])
        mm_60: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_50, arg102_1);  permute_50 = arg102_1 = None
        mm_61: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_142, arg114_1);  view_142 = arg114_1 = None
        view_143: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_61, _shape_param_159);  mm_61 = _shape_param_159 = None
        convert_element_type_141: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        add_70: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_69, convert_element_type_141);  add_69 = convert_element_type_141 = None
        convert_element_type_142: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        mul_119: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_70, arg115_1);  arg115_1 = None
        mul_120: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_119, 4096)
        sum_41: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True)
        mul_121: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_119, arg116_1);  mul_119 = None
        sum_42: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_121, [2], True);  mul_121 = None
        mul_122: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg116_1, sum_42);  sum_42 = None
        sub_21: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_120, sum_41);  mul_120 = sum_41 = None
        sub_22: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_21, mul_122);  sub_21 = mul_122 = None
        mul_123: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg117_1, sub_22);  arg117_1 = sub_22 = None
        mul_124: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_70, arg116_1);  arg116_1 = None
        sum_43: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_124, [0, 1]);  mul_124 = None
        sum_44: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_70, [0, 1]);  add_70 = None
        add_71: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_57, mul_123);  add_57 = mul_123 = None
        convert_element_type_143: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_71, torch.bfloat16)
        view_144: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_143, _shape_param_160);  convert_element_type_143 = _shape_param_160 = None
        mm_62: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_144, arg118_1);  arg118_1 = None
        permute_51: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_144, [1, 0])
        mm_63: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_51, arg119_1);  arg119_1 = None
        sum_45: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True, dtype = torch.float32)
        view_145: "f32[4096]" = torch.ops.aten.view.default(sum_45, _shape_param_161);  sum_45 = _shape_param_161 = None
        convert_element_type_144: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_145, torch.bfloat16);  view_145 = None
        view_146: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_62, _shape_param_162);  mm_62 = _shape_param_162 = None
        convert_element_type_145: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None
        convert_element_type_146: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_147: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_144, torch.float32);  convert_element_type_144 = None
        view_147: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg120_1, _shape_param_163);  arg120_1 = _shape_param_163 = None
        mul_125: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_147, 0.5)
        mul_126: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_145, mul_125);  mul_125 = None
        convert_element_type_148: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_147, torch.float32)
        pow_11: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_148, 3.0)
        mul_127: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_72: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_147, mul_127);  view_147 = mul_127 = None
        mul_128: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_72, 0.7978845608028654);  add_72 = None
        tanh_5: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_128);  mul_128 = None
        add_73: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_129: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_145, add_73);  convert_element_type_145 = add_73 = None
        convert_element_type_149: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_129, torch.bfloat16);  mul_129 = None
        mul_130: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_23: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_130);  mul_130 = None
        mul_131: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_126, sub_23);  mul_126 = sub_23 = None
        mul_132: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_131, 0.7978845608028654);  mul_131 = None
        convert_element_type_150: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_132, torch.bfloat16)
        mul_133: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_132, 0.044715);  mul_132 = None
        pow_12: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_148, 2.0);  convert_element_type_148 = None
        mul_134: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_12, 3.0);  pow_12 = None
        mul_135: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_133, mul_134);  mul_133 = mul_134 = None
        convert_element_type_151: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_135, torch.bfloat16);  mul_135 = None
        add_74: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_150, convert_element_type_151);  convert_element_type_150 = convert_element_type_151 = None
        mul_136: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_149, 0.5);  convert_element_type_149 = None
        add_75: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_74, mul_136);  add_74 = mul_136 = None
        view_148: "bf16[128, 16384]" = torch.ops.aten.view.default(add_75, _shape_param_164);  add_75 = _shape_param_164 = None
        mm_64: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_148, arg121_1);  arg121_1 = None
        permute_52: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_148, [1, 0])
        mm_65: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_52, arg122_1);  permute_52 = None
        sum_46: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_148, [0], True, dtype = torch.float32);  view_148 = None
        view_149: "f32[16384]" = torch.ops.aten.view.default(sum_46, _shape_param_165);  sum_46 = _shape_param_165 = None
        convert_element_type_152: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_149, torch.bfloat16);  view_149 = None
        view_150: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_64, _shape_param_166);  mm_64 = _shape_param_166 = None
        convert_element_type_153: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_150, torch.float32);  view_150 = None
        convert_element_type_154: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_155: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_152, torch.float32);  convert_element_type_152 = None
        mm_66: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_51, arg123_1);  permute_51 = arg123_1 = None
        mm_67: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_144, arg124_1);  view_144 = arg124_1 = None
        view_151: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_67, _shape_param_167);  mm_67 = _shape_param_167 = None
        convert_element_type_156: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        view_152: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_151, _shape_param_168);  view_151 = _shape_param_168 = None
        permute_53: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        view_153: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_53, _shape_param_169);  permute_53 = _shape_param_169 = None
        bmm_20: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg125_1, view_153);  arg125_1 = None
        bmm_21: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_153, arg126_1);  view_153 = arg126_1 = None
        view_154: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_20, _shape_param_170);  bmm_20 = _shape_param_170 = None
        view_155: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_21, _shape_param_171);  bmm_21 = _shape_param_171 = None
        convert_element_type_157: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_155, torch.float32);  view_155 = None
        mul_137: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_157, arg127_1);  convert_element_type_157 = None
        sum_47: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [-1], True)
        neg_15: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg127_1);  arg127_1 = None
        fma_5: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_15, sum_47, mul_137);  neg_15 = sum_47 = mul_137 = None
        convert_element_type_158: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        div_8: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_158, 16.0);  convert_element_type_158 = None
        view_156: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_8, _shape_param_172);  div_8 = _shape_param_172 = None
        bmm_22: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg128_1, view_156);  arg128_1 = None
        bmm_23: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_156, arg129_1);  view_156 = arg129_1 = None
        view_157: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_173);  bmm_22 = _shape_param_173 = None
        view_158: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_23, _shape_param_174);  bmm_23 = _shape_param_174 = None
        convert_element_type_159: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None
        permute_54: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_159, [0, 1, 3, 2]);  convert_element_type_159 = None
        convert_element_type_160: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_54, torch.bfloat16);  permute_54 = None
        convert_element_type_161: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_158, torch.bfloat16);  view_158 = None
        permute_55: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_161, [0, 2, 1, 3]);  convert_element_type_161 = None
        permute_56: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_160, [0, 2, 1, 3]);  convert_element_type_160 = None
        slice_21: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_55, 3, 0, 64)
        slice_22: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_55, 3, 64, 256);  permute_55 = None
        slice_23: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_56, 3, 0, 64)
        slice_24: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_56, 3, 64, 256);  permute_56 = None
        expand_12: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg130_1, _shape_param_175);  arg130_1 = _shape_param_175 = None
        clone_15: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_159: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_15, _shape_param_176);  clone_15 = _shape_param_176 = None
        mul_138: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_21, view_159)
        view_160: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_138, _shape_param_177);  mul_138 = _shape_param_177 = None
        select_20: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_160, -1, 0)
        select_21: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_160, -1, 1);  view_160 = None
        neg_16: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_20);  select_20 = None
        slice_scatter_40: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_16, 3, 1, 9223372036854775807, 2);  neg_16 = None
        slice_scatter_41: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_21, 3, 0, 9223372036854775807, 2);  select_21 = None
        add_76: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_40, slice_scatter_41);  slice_scatter_40 = slice_scatter_41 = None
        expand_13: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg131_1, _shape_param_178);  arg131_1 = _shape_param_178 = None
        clone_16: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_161: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_179);  clone_16 = _shape_param_179 = None
        mul_139: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_21, view_161);  slice_21 = None
        add_77: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_76, mul_139);  add_76 = mul_139 = None
        mul_140: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_23, view_159);  view_159 = None
        view_162: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_140, _shape_param_180);  mul_140 = _shape_param_180 = None
        select_22: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_162, -1, 0)
        select_23: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_162, -1, 1);  view_162 = None
        neg_17: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_22);  select_22 = None
        slice_scatter_42: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_17, 3, 1, 9223372036854775807, 2);  neg_17 = None
        slice_scatter_43: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_23, 3, 0, 9223372036854775807, 2);  select_23 = None
        add_78: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_42, slice_scatter_43);  slice_scatter_42 = slice_scatter_43 = None
        mul_141: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_23, view_161);  slice_23 = view_161 = None
        add_79: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_78, mul_141);  add_78 = mul_141 = None
        slice_scatter_44: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_22, 3, 64, 9223372036854775807);  slice_22 = None
        slice_scatter_45: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_77, 3, 0, 64);  add_77 = None
        add_80: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_44, slice_scatter_45);  slice_scatter_44 = slice_scatter_45 = None
        slice_scatter_46: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_24, 3, 64, 9223372036854775807);  slice_24 = None
        slice_scatter_47: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_79, 3, 0, 64);  add_79 = None
        add_81: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_46, slice_scatter_47);  slice_scatter_46 = slice_scatter_47 = None
        permute_57: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        clone_17: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_163: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_17, _shape_param_181);  clone_17 = _shape_param_181 = None
        view_164: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_81, _shape_param_182);  add_81 = _shape_param_182 = None
        view_165: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_80, _shape_param_183);  add_80 = _shape_param_183 = None
        view_166: "bf16[128, 4096]" = torch.ops.aten.view.default(view_163, _shape_param_184);  view_163 = _shape_param_184 = None
        permute_58: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_166, [1, 0])
        mm_68: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_58, arg122_1);  permute_58 = None
        mm_69: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_166, arg132_1);  view_166 = arg132_1 = None
        view_167: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_69, _shape_param_185);  mm_69 = _shape_param_185 = None
        convert_element_type_162: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_167, torch.float32);  view_167 = None
        add_82: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_153, convert_element_type_162);  convert_element_type_153 = convert_element_type_162 = None
        convert_element_type_163: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        view_168: "bf16[128, 4096]" = torch.ops.aten.view.default(view_164, _shape_param_186);  view_164 = _shape_param_186 = None
        permute_59: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_168, [1, 0])
        mm_70: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_59, arg122_1);  permute_59 = None
        mm_71: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_168, arg133_1);  view_168 = arg133_1 = None
        view_169: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_71, _shape_param_187);  mm_71 = _shape_param_187 = None
        convert_element_type_164: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_169, torch.float32);  view_169 = None
        add_83: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_82, convert_element_type_164);  add_82 = convert_element_type_164 = None
        convert_element_type_165: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        view_170: "bf16[128, 4096]" = torch.ops.aten.view.default(view_165, _shape_param_188);  view_165 = _shape_param_188 = None
        permute_60: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_170, [1, 0])
        mm_72: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_60, arg122_1);  permute_60 = arg122_1 = None
        mm_73: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_170, arg134_1);  view_170 = arg134_1 = None
        view_171: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_73, _shape_param_189);  mm_73 = _shape_param_189 = None
        convert_element_type_166: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_171, torch.float32);  view_171 = None
        add_84: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_83, convert_element_type_166);  add_83 = convert_element_type_166 = None
        convert_element_type_167: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        mul_142: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_84, arg135_1);  arg135_1 = None
        mul_143: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_142, 4096)
        sum_48: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_142, [2], True)
        mul_144: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_142, arg136_1);  mul_142 = None
        sum_49: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [2], True);  mul_144 = None
        mul_145: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg136_1, sum_49);  sum_49 = None
        sub_24: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_143, sum_48);  mul_143 = sum_48 = None
        sub_25: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_24, mul_145);  sub_24 = mul_145 = None
        mul_146: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg137_1, sub_25);  arg137_1 = sub_25 = None
        mul_147: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_84, arg136_1);  arg136_1 = None
        sum_50: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_147, [0, 1]);  mul_147 = None
        sum_51: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_84, [0, 1]);  add_84 = None
        add_85: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_71, mul_146);  add_71 = mul_146 = None
        convert_element_type_168: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16)
        view_172: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_168, _shape_param_190);  convert_element_type_168 = _shape_param_190 = None
        mm_74: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_172, arg138_1);  arg138_1 = None
        permute_61: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_172, [1, 0])
        mm_75: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_61, arg139_1);  arg139_1 = None
        sum_52: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True, dtype = torch.float32)
        view_173: "f32[4096]" = torch.ops.aten.view.default(sum_52, _shape_param_191);  sum_52 = _shape_param_191 = None
        convert_element_type_169: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_173, torch.bfloat16);  view_173 = None
        view_174: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_74, _shape_param_192);  mm_74 = _shape_param_192 = None
        convert_element_type_170: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_174, torch.float32);  view_174 = None
        convert_element_type_171: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_172: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_169, torch.float32);  convert_element_type_169 = None
        view_175: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg140_1, _shape_param_193);  arg140_1 = _shape_param_193 = None
        mul_148: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
        mul_149: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_170, mul_148);  mul_148 = None
        convert_element_type_173: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_175, torch.float32)
        pow_13: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_173, 3.0)
        mul_150: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_86: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_175, mul_150);  view_175 = mul_150 = None
        mul_151: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
        tanh_6: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_151);  mul_151 = None
        add_87: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_152: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_170, add_87);  convert_element_type_170 = add_87 = None
        convert_element_type_174: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        mul_153: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_26: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_153);  mul_153 = None
        mul_154: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_149, sub_26);  mul_149 = sub_26 = None
        mul_155: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_154, 0.7978845608028654);  mul_154 = None
        convert_element_type_175: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_155, torch.bfloat16)
        mul_156: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_155, 0.044715);  mul_155 = None
        pow_14: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_173, 2.0);  convert_element_type_173 = None
        mul_157: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_158: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_156, mul_157);  mul_156 = mul_157 = None
        convert_element_type_176: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_158, torch.bfloat16);  mul_158 = None
        add_88: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_175, convert_element_type_176);  convert_element_type_175 = convert_element_type_176 = None
        mul_159: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_174, 0.5);  convert_element_type_174 = None
        add_89: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_88, mul_159);  add_88 = mul_159 = None
        view_176: "bf16[128, 16384]" = torch.ops.aten.view.default(add_89, _shape_param_194);  add_89 = _shape_param_194 = None
        mm_76: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_176, arg141_1);  arg141_1 = None
        permute_62: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_176, [1, 0])
        mm_77: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_62, arg142_1);  permute_62 = None
        sum_53: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_176, [0], True, dtype = torch.float32);  view_176 = None
        view_177: "f32[16384]" = torch.ops.aten.view.default(sum_53, _shape_param_195);  sum_53 = _shape_param_195 = None
        convert_element_type_177: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_177, torch.bfloat16);  view_177 = None
        view_178: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_76, _shape_param_196);  mm_76 = _shape_param_196 = None
        convert_element_type_178: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_178, torch.float32);  view_178 = None
        convert_element_type_179: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_180: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_177, torch.float32);  convert_element_type_177 = None
        mm_78: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_61, arg143_1);  permute_61 = arg143_1 = None
        mm_79: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_172, arg144_1);  view_172 = arg144_1 = None
        view_179: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_79, _shape_param_197);  mm_79 = _shape_param_197 = None
        convert_element_type_181: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        view_180: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_179, _shape_param_198);  view_179 = _shape_param_198 = None
        permute_63: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        view_181: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_63, _shape_param_199);  permute_63 = _shape_param_199 = None
        bmm_24: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg145_1, view_181);  arg145_1 = None
        bmm_25: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_181, arg146_1);  view_181 = arg146_1 = None
        view_182: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_24, _shape_param_200);  bmm_24 = _shape_param_200 = None
        view_183: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_25, _shape_param_201);  bmm_25 = _shape_param_201 = None
        convert_element_type_182: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_183, torch.float32);  view_183 = None
        mul_160: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_182, arg147_1);  convert_element_type_182 = None
        sum_54: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_160, [-1], True)
        neg_18: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg147_1);  arg147_1 = None
        fma_6: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_18, sum_54, mul_160);  neg_18 = sum_54 = mul_160 = None
        convert_element_type_183: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        div_9: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_183, 16.0);  convert_element_type_183 = None
        view_184: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_9, _shape_param_202);  div_9 = _shape_param_202 = None
        bmm_26: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg148_1, view_184);  arg148_1 = None
        bmm_27: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_184, arg149_1);  view_184 = arg149_1 = None
        view_185: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_203);  bmm_26 = _shape_param_203 = None
        view_186: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_27, _shape_param_204);  bmm_27 = _shape_param_204 = None
        convert_element_type_184: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_185, torch.float32);  view_185 = None
        permute_64: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_184, [0, 1, 3, 2]);  convert_element_type_184 = None
        convert_element_type_185: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_64, torch.bfloat16);  permute_64 = None
        convert_element_type_186: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_186, torch.bfloat16);  view_186 = None
        permute_65: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_186, [0, 2, 1, 3]);  convert_element_type_186 = None
        permute_66: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_185, [0, 2, 1, 3]);  convert_element_type_185 = None
        slice_25: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_65, 3, 0, 64)
        slice_26: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_65, 3, 64, 256);  permute_65 = None
        slice_27: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_66, 3, 0, 64)
        slice_28: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_66, 3, 64, 256);  permute_66 = None
        expand_14: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg150_1, _shape_param_205);  arg150_1 = _shape_param_205 = None
        clone_18: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_187: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_206);  clone_18 = _shape_param_206 = None
        mul_161: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_25, view_187)
        view_188: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_161, _shape_param_207);  mul_161 = _shape_param_207 = None
        select_24: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_188, -1, 0)
        select_25: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_188, -1, 1);  view_188 = None
        neg_19: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_24);  select_24 = None
        slice_scatter_48: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_19, 3, 1, 9223372036854775807, 2);  neg_19 = None
        slice_scatter_49: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_25, 3, 0, 9223372036854775807, 2);  select_25 = None
        add_90: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_48, slice_scatter_49);  slice_scatter_48 = slice_scatter_49 = None
        expand_15: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg151_1, _shape_param_208);  arg151_1 = _shape_param_208 = None
        clone_19: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_189: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_19, _shape_param_209);  clone_19 = _shape_param_209 = None
        mul_162: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_25, view_189);  slice_25 = None
        add_91: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_90, mul_162);  add_90 = mul_162 = None
        mul_163: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_27, view_187);  view_187 = None
        view_190: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_163, _shape_param_210);  mul_163 = _shape_param_210 = None
        select_26: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_190, -1, 0)
        select_27: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_190, -1, 1);  view_190 = None
        neg_20: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_26);  select_26 = None
        slice_scatter_50: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_20, 3, 1, 9223372036854775807, 2);  neg_20 = None
        slice_scatter_51: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_27, 3, 0, 9223372036854775807, 2);  select_27 = None
        add_92: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_50, slice_scatter_51);  slice_scatter_50 = slice_scatter_51 = None
        mul_164: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_27, view_189);  slice_27 = view_189 = None
        add_93: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_92, mul_164);  add_92 = mul_164 = None
        slice_scatter_52: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_26, 3, 64, 9223372036854775807);  slice_26 = None
        slice_scatter_53: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_91, 3, 0, 64);  add_91 = None
        add_94: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_52, slice_scatter_53);  slice_scatter_52 = slice_scatter_53 = None
        slice_scatter_54: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_28, 3, 64, 9223372036854775807);  slice_28 = None
        slice_scatter_55: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_93, 3, 0, 64);  add_93 = None
        add_95: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_54, slice_scatter_55);  slice_scatter_54 = slice_scatter_55 = None
        permute_67: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None
        clone_20: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_191: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_20, _shape_param_211);  clone_20 = _shape_param_211 = None
        view_192: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_95, _shape_param_212);  add_95 = _shape_param_212 = None
        view_193: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_94, _shape_param_213);  add_94 = _shape_param_213 = None
        view_194: "bf16[128, 4096]" = torch.ops.aten.view.default(view_191, _shape_param_214);  view_191 = _shape_param_214 = None
        permute_68: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_194, [1, 0])
        mm_80: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_68, arg142_1);  permute_68 = None
        mm_81: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_194, arg152_1);  view_194 = arg152_1 = None
        view_195: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_81, _shape_param_215);  mm_81 = _shape_param_215 = None
        convert_element_type_187: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        add_96: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_178, convert_element_type_187);  convert_element_type_178 = convert_element_type_187 = None
        convert_element_type_188: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        view_196: "bf16[128, 4096]" = torch.ops.aten.view.default(view_192, _shape_param_216);  view_192 = _shape_param_216 = None
        permute_69: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_196, [1, 0])
        mm_82: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_69, arg142_1);  permute_69 = None
        mm_83: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_196, arg153_1);  view_196 = arg153_1 = None
        view_197: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_83, _shape_param_217);  mm_83 = _shape_param_217 = None
        convert_element_type_189: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_197, torch.float32);  view_197 = None
        add_97: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_96, convert_element_type_189);  add_96 = convert_element_type_189 = None
        convert_element_type_190: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        view_198: "bf16[128, 4096]" = torch.ops.aten.view.default(view_193, _shape_param_218);  view_193 = _shape_param_218 = None
        permute_70: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_198, [1, 0])
        mm_84: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_70, arg142_1);  permute_70 = arg142_1 = None
        mm_85: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_198, arg154_1);  view_198 = arg154_1 = None
        view_199: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_85, _shape_param_219);  mm_85 = _shape_param_219 = None
        convert_element_type_191: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_199, torch.float32);  view_199 = None
        add_98: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_97, convert_element_type_191);  add_97 = convert_element_type_191 = None
        convert_element_type_192: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        mul_165: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_98, arg155_1);  arg155_1 = None
        mul_166: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_165, 4096)
        sum_55: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True)
        mul_167: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_165, arg156_1);  mul_165 = None
        sum_56: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True);  mul_167 = None
        mul_168: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg156_1, sum_56);  sum_56 = None
        sub_27: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_166, sum_55);  mul_166 = sum_55 = None
        sub_28: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_27, mul_168);  sub_27 = mul_168 = None
        mul_169: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg157_1, sub_28);  arg157_1 = sub_28 = None
        mul_170: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_98, arg156_1);  arg156_1 = None
        sum_57: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_170, [0, 1]);  mul_170 = None
        sum_58: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_98, [0, 1]);  add_98 = None
        add_99: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_85, mul_169);  add_85 = mul_169 = None
        convert_element_type_193: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16)
        view_200: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_193, _shape_param_220);  convert_element_type_193 = _shape_param_220 = None
        mm_86: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_200, arg158_1);  arg158_1 = None
        permute_71: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_200, [1, 0])
        mm_87: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_71, arg159_1);  arg159_1 = None
        sum_59: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True, dtype = torch.float32)
        view_201: "f32[4096]" = torch.ops.aten.view.default(sum_59, _shape_param_221);  sum_59 = _shape_param_221 = None
        convert_element_type_194: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_201, torch.bfloat16);  view_201 = None
        view_202: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_86, _shape_param_222);  mm_86 = _shape_param_222 = None
        convert_element_type_195: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_202, torch.float32);  view_202 = None
        convert_element_type_196: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_197: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_194, torch.float32);  convert_element_type_194 = None
        view_203: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg160_1, _shape_param_223);  arg160_1 = _shape_param_223 = None
        mul_171: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_203, 0.5)
        mul_172: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_195, mul_171);  mul_171 = None
        convert_element_type_198: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_203, torch.float32)
        pow_15: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_198, 3.0)
        mul_173: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_100: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_203, mul_173);  view_203 = mul_173 = None
        mul_174: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_7: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
        add_101: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_175: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_195, add_101);  convert_element_type_195 = add_101 = None
        convert_element_type_199: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None
        mul_176: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_29: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_176);  mul_176 = None
        mul_177: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_172, sub_29);  mul_172 = sub_29 = None
        mul_178: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_177, 0.7978845608028654);  mul_177 = None
        convert_element_type_200: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_178, torch.bfloat16)
        mul_179: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_178, 0.044715);  mul_178 = None
        pow_16: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_198, 2.0);  convert_element_type_198 = None
        mul_180: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_181: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_179, mul_180);  mul_179 = mul_180 = None
        convert_element_type_201: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_181, torch.bfloat16);  mul_181 = None
        add_102: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_200, convert_element_type_201);  convert_element_type_200 = convert_element_type_201 = None
        mul_182: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_199, 0.5);  convert_element_type_199 = None
        add_103: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_102, mul_182);  add_102 = mul_182 = None
        view_204: "bf16[128, 16384]" = torch.ops.aten.view.default(add_103, _shape_param_224);  add_103 = _shape_param_224 = None
        mm_88: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_204, arg161_1);  arg161_1 = None
        permute_72: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_204, [1, 0])
        mm_89: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_72, arg162_1);  permute_72 = None
        sum_60: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True, dtype = torch.float32);  view_204 = None
        view_205: "f32[16384]" = torch.ops.aten.view.default(sum_60, _shape_param_225);  sum_60 = _shape_param_225 = None
        convert_element_type_202: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_205, torch.bfloat16);  view_205 = None
        view_206: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_88, _shape_param_226);  mm_88 = _shape_param_226 = None
        convert_element_type_203: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_206, torch.float32);  view_206 = None
        convert_element_type_204: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_205: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_202, torch.float32);  convert_element_type_202 = None
        mm_90: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_71, arg163_1);  permute_71 = arg163_1 = None
        mm_91: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_200, arg164_1);  view_200 = arg164_1 = None
        view_207: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_91, _shape_param_227);  mm_91 = _shape_param_227 = None
        convert_element_type_206: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        view_208: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_207, _shape_param_228);  view_207 = _shape_param_228 = None
        permute_73: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
        view_209: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_73, _shape_param_229);  permute_73 = _shape_param_229 = None
        bmm_28: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg165_1, view_209);  arg165_1 = None
        bmm_29: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_209, arg166_1);  view_209 = arg166_1 = None
        view_210: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_28, _shape_param_230);  bmm_28 = _shape_param_230 = None
        view_211: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_29, _shape_param_231);  bmm_29 = _shape_param_231 = None
        convert_element_type_207: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_211, torch.float32);  view_211 = None
        mul_183: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_207, arg167_1);  convert_element_type_207 = None
        sum_61: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_183, [-1], True)
        neg_21: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg167_1);  arg167_1 = None
        fma_7: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_21, sum_61, mul_183);  neg_21 = sum_61 = mul_183 = None
        convert_element_type_208: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        div_10: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_208, 16.0);  convert_element_type_208 = None
        view_212: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_10, _shape_param_232);  div_10 = _shape_param_232 = None
        bmm_30: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg168_1, view_212);  arg168_1 = None
        bmm_31: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_212, arg169_1);  view_212 = arg169_1 = None
        view_213: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_233);  bmm_30 = _shape_param_233 = None
        view_214: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_31, _shape_param_234);  bmm_31 = _shape_param_234 = None
        convert_element_type_209: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        permute_74: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_209, [0, 1, 3, 2]);  convert_element_type_209 = None
        convert_element_type_210: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_74, torch.bfloat16);  permute_74 = None
        convert_element_type_211: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_214, torch.bfloat16);  view_214 = None
        permute_75: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_211, [0, 2, 1, 3]);  convert_element_type_211 = None
        permute_76: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_210, [0, 2, 1, 3]);  convert_element_type_210 = None
        slice_29: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_75, 3, 0, 64)
        slice_30: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_75, 3, 64, 256);  permute_75 = None
        slice_31: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_76, 3, 0, 64)
        slice_32: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_76, 3, 64, 256);  permute_76 = None
        expand_16: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg170_1, _shape_param_235);  arg170_1 = _shape_param_235 = None
        clone_21: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_215: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_21, _shape_param_236);  clone_21 = _shape_param_236 = None
        mul_184: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_29, view_215)
        view_216: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_184, _shape_param_237);  mul_184 = _shape_param_237 = None
        select_28: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_216, -1, 0)
        select_29: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_216, -1, 1);  view_216 = None
        neg_22: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_28);  select_28 = None
        slice_scatter_56: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_22, 3, 1, 9223372036854775807, 2);  neg_22 = None
        slice_scatter_57: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_29, 3, 0, 9223372036854775807, 2);  select_29 = None
        add_104: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_56, slice_scatter_57);  slice_scatter_56 = slice_scatter_57 = None
        expand_17: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg171_1, _shape_param_238);  arg171_1 = _shape_param_238 = None
        clone_22: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_217: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_239);  clone_22 = _shape_param_239 = None
        mul_185: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_29, view_217);  slice_29 = None
        add_105: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_104, mul_185);  add_104 = mul_185 = None
        mul_186: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_31, view_215);  view_215 = None
        view_218: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_186, _shape_param_240);  mul_186 = _shape_param_240 = None
        select_30: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_218, -1, 0)
        select_31: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_218, -1, 1);  view_218 = None
        neg_23: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_30);  select_30 = None
        slice_scatter_58: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_23, 3, 1, 9223372036854775807, 2);  neg_23 = None
        slice_scatter_59: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_31, 3, 0, 9223372036854775807, 2);  select_31 = None
        add_106: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_58, slice_scatter_59);  slice_scatter_58 = slice_scatter_59 = None
        mul_187: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_31, view_217);  slice_31 = view_217 = None
        add_107: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_106, mul_187);  add_106 = mul_187 = None
        slice_scatter_60: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_30, 3, 64, 9223372036854775807);  slice_30 = None
        slice_scatter_61: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_105, 3, 0, 64);  add_105 = None
        add_108: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_60, slice_scatter_61);  slice_scatter_60 = slice_scatter_61 = None
        slice_scatter_62: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_32, 3, 64, 9223372036854775807);  slice_32 = None
        slice_scatter_63: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_107, 3, 0, 64);  add_107 = None
        add_109: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_62, slice_scatter_63);  slice_scatter_62 = slice_scatter_63 = None
        permute_77: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
        clone_23: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_219: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_23, _shape_param_241);  clone_23 = _shape_param_241 = None
        view_220: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_109, _shape_param_242);  add_109 = _shape_param_242 = None
        view_221: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_108, _shape_param_243);  add_108 = _shape_param_243 = None
        view_222: "bf16[128, 4096]" = torch.ops.aten.view.default(view_219, _shape_param_244);  view_219 = _shape_param_244 = None
        permute_78: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_222, [1, 0])
        mm_92: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_78, arg162_1);  permute_78 = None
        mm_93: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_222, arg172_1);  view_222 = arg172_1 = None
        view_223: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_93, _shape_param_245);  mm_93 = _shape_param_245 = None
        convert_element_type_212: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_223, torch.float32);  view_223 = None
        add_110: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_203, convert_element_type_212);  convert_element_type_203 = convert_element_type_212 = None
        convert_element_type_213: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        view_224: "bf16[128, 4096]" = torch.ops.aten.view.default(view_220, _shape_param_246);  view_220 = _shape_param_246 = None
        permute_79: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_224, [1, 0])
        mm_94: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_79, arg162_1);  permute_79 = None
        mm_95: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_224, arg173_1);  view_224 = arg173_1 = None
        view_225: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_95, _shape_param_247);  mm_95 = _shape_param_247 = None
        convert_element_type_214: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_225, torch.float32);  view_225 = None
        add_111: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_110, convert_element_type_214);  add_110 = convert_element_type_214 = None
        convert_element_type_215: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        view_226: "bf16[128, 4096]" = torch.ops.aten.view.default(view_221, _shape_param_248);  view_221 = _shape_param_248 = None
        permute_80: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_226, [1, 0])
        mm_96: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_80, arg162_1);  permute_80 = arg162_1 = None
        mm_97: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_226, arg174_1);  view_226 = arg174_1 = None
        view_227: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_97, _shape_param_249);  mm_97 = _shape_param_249 = None
        convert_element_type_216: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        add_112: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_111, convert_element_type_216);  add_111 = convert_element_type_216 = None
        convert_element_type_217: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        mul_188: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_112, arg175_1);  arg175_1 = None
        mul_189: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_188, 4096)
        sum_62: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True)
        mul_190: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_188, arg176_1);  mul_188 = None
        sum_63: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_190, [2], True);  mul_190 = None
        mul_191: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg176_1, sum_63);  sum_63 = None
        sub_30: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_189, sum_62);  mul_189 = sum_62 = None
        sub_31: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_30, mul_191);  sub_30 = mul_191 = None
        mul_192: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg177_1, sub_31);  arg177_1 = sub_31 = None
        mul_193: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_112, arg176_1);  arg176_1 = None
        sum_64: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_193, [0, 1]);  mul_193 = None
        sum_65: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_112, [0, 1]);  add_112 = None
        add_113: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_99, mul_192);  add_99 = mul_192 = None
        convert_element_type_218: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_113, torch.bfloat16)
        view_228: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_218, _shape_param_250);  convert_element_type_218 = _shape_param_250 = None
        mm_98: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_228, arg178_1);  arg178_1 = None
        permute_81: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_228, [1, 0])
        mm_99: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_81, arg179_1);  arg179_1 = None
        sum_66: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True, dtype = torch.float32)
        view_229: "f32[4096]" = torch.ops.aten.view.default(sum_66, _shape_param_251);  sum_66 = _shape_param_251 = None
        convert_element_type_219: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_229, torch.bfloat16);  view_229 = None
        view_230: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_98, _shape_param_252);  mm_98 = _shape_param_252 = None
        convert_element_type_220: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_230, torch.float32);  view_230 = None
        convert_element_type_221: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_222: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_219, torch.float32);  convert_element_type_219 = None
        view_231: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg180_1, _shape_param_253);  arg180_1 = _shape_param_253 = None
        mul_194: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_231, 0.5)
        mul_195: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_220, mul_194);  mul_194 = None
        convert_element_type_223: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_231, torch.float32)
        pow_17: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_223, 3.0)
        mul_196: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_114: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_231, mul_196);  view_231 = mul_196 = None
        mul_197: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_114, 0.7978845608028654);  add_114 = None
        tanh_8: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_197);  mul_197 = None
        add_115: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_198: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_220, add_115);  convert_element_type_220 = add_115 = None
        convert_element_type_224: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_198, torch.bfloat16);  mul_198 = None
        mul_199: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_32: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_199);  mul_199 = None
        mul_200: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_195, sub_32);  mul_195 = sub_32 = None
        mul_201: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_200, 0.7978845608028654);  mul_200 = None
        convert_element_type_225: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_201, torch.bfloat16)
        mul_202: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_201, 0.044715);  mul_201 = None
        pow_18: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_223, 2.0);  convert_element_type_223 = None
        mul_203: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_204: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
        convert_element_type_226: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_204, torch.bfloat16);  mul_204 = None
        add_116: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_225, convert_element_type_226);  convert_element_type_225 = convert_element_type_226 = None
        mul_205: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_224, 0.5);  convert_element_type_224 = None
        add_117: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_116, mul_205);  add_116 = mul_205 = None
        view_232: "bf16[128, 16384]" = torch.ops.aten.view.default(add_117, _shape_param_254);  add_117 = _shape_param_254 = None
        mm_100: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_232, arg181_1);  arg181_1 = None
        permute_82: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_232, [1, 0])
        mm_101: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_82, arg182_1);  permute_82 = None
        sum_67: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_232, [0], True, dtype = torch.float32);  view_232 = None
        view_233: "f32[16384]" = torch.ops.aten.view.default(sum_67, _shape_param_255);  sum_67 = _shape_param_255 = None
        convert_element_type_227: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_233, torch.bfloat16);  view_233 = None
        view_234: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_100, _shape_param_256);  mm_100 = _shape_param_256 = None
        convert_element_type_228: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_234, torch.float32);  view_234 = None
        convert_element_type_229: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_230: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_227, torch.float32);  convert_element_type_227 = None
        mm_102: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_81, arg183_1);  permute_81 = arg183_1 = None
        mm_103: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_228, arg184_1);  view_228 = arg184_1 = None
        view_235: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_103, _shape_param_257);  mm_103 = _shape_param_257 = None
        convert_element_type_231: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        view_236: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_235, _shape_param_258);  view_235 = _shape_param_258 = None
        permute_83: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
        view_237: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_83, _shape_param_259);  permute_83 = _shape_param_259 = None
        bmm_32: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg185_1, view_237);  arg185_1 = None
        bmm_33: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_237, arg186_1);  view_237 = arg186_1 = None
        view_238: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_32, _shape_param_260);  bmm_32 = _shape_param_260 = None
        view_239: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_33, _shape_param_261);  bmm_33 = _shape_param_261 = None
        convert_element_type_232: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_206: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_232, arg187_1);  convert_element_type_232 = None
        sum_68: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [-1], True)
        neg_24: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg187_1);  arg187_1 = None
        fma_8: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_24, sum_68, mul_206);  neg_24 = sum_68 = mul_206 = None
        convert_element_type_233: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        div_11: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_233, 16.0);  convert_element_type_233 = None
        view_240: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_11, _shape_param_262);  div_11 = _shape_param_262 = None
        bmm_34: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg188_1, view_240);  arg188_1 = None
        bmm_35: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_240, arg189_1);  view_240 = arg189_1 = None
        view_241: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_263);  bmm_34 = _shape_param_263 = None
        view_242: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_35, _shape_param_264);  bmm_35 = _shape_param_264 = None
        convert_element_type_234: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_241, torch.float32);  view_241 = None
        permute_84: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_234, [0, 1, 3, 2]);  convert_element_type_234 = None
        convert_element_type_235: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_84, torch.bfloat16);  permute_84 = None
        convert_element_type_236: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_242, torch.bfloat16);  view_242 = None
        permute_85: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_236, [0, 2, 1, 3]);  convert_element_type_236 = None
        permute_86: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_235, [0, 2, 1, 3]);  convert_element_type_235 = None
        slice_33: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_85, 3, 0, 64)
        slice_34: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_85, 3, 64, 256);  permute_85 = None
        slice_35: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_86, 3, 0, 64)
        slice_36: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_86, 3, 64, 256);  permute_86 = None
        expand_18: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg190_1, _shape_param_265);  arg190_1 = _shape_param_265 = None
        clone_24: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_243: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_266);  clone_24 = _shape_param_266 = None
        mul_207: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_33, view_243)
        view_244: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_207, _shape_param_267);  mul_207 = _shape_param_267 = None
        select_32: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_244, -1, 0)
        select_33: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_244, -1, 1);  view_244 = None
        neg_25: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_32);  select_32 = None
        slice_scatter_64: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_25, 3, 1, 9223372036854775807, 2);  neg_25 = None
        slice_scatter_65: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_33, 3, 0, 9223372036854775807, 2);  select_33 = None
        add_118: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_64, slice_scatter_65);  slice_scatter_64 = slice_scatter_65 = None
        expand_19: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg191_1, _shape_param_268);  arg191_1 = _shape_param_268 = None
        clone_25: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_245: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_25, _shape_param_269);  clone_25 = _shape_param_269 = None
        mul_208: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_33, view_245);  slice_33 = None
        add_119: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_118, mul_208);  add_118 = mul_208 = None
        mul_209: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_35, view_243);  view_243 = None
        view_246: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_209, _shape_param_270);  mul_209 = _shape_param_270 = None
        select_34: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_246, -1, 0)
        select_35: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_246, -1, 1);  view_246 = None
        neg_26: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_34);  select_34 = None
        slice_scatter_66: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_26, 3, 1, 9223372036854775807, 2);  neg_26 = None
        slice_scatter_67: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_35, 3, 0, 9223372036854775807, 2);  select_35 = None
        add_120: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_66, slice_scatter_67);  slice_scatter_66 = slice_scatter_67 = None
        mul_210: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_35, view_245);  slice_35 = view_245 = None
        add_121: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_120, mul_210);  add_120 = mul_210 = None
        slice_scatter_68: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_34, 3, 64, 9223372036854775807);  slice_34 = None
        slice_scatter_69: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_119, 3, 0, 64);  add_119 = None
        add_122: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_68, slice_scatter_69);  slice_scatter_68 = slice_scatter_69 = None
        slice_scatter_70: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_36, 3, 64, 9223372036854775807);  slice_36 = None
        slice_scatter_71: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_121, 3, 0, 64);  add_121 = None
        add_123: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_70, slice_scatter_71);  slice_scatter_70 = slice_scatter_71 = None
        permute_87: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
        clone_26: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
        view_247: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_26, _shape_param_271);  clone_26 = _shape_param_271 = None
        view_248: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_123, _shape_param_272);  add_123 = _shape_param_272 = None
        view_249: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_122, _shape_param_273);  add_122 = _shape_param_273 = None
        view_250: "bf16[128, 4096]" = torch.ops.aten.view.default(view_247, _shape_param_274);  view_247 = _shape_param_274 = None
        permute_88: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_250, [1, 0])
        mm_104: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_88, arg182_1);  permute_88 = None
        mm_105: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_250, arg192_1);  view_250 = arg192_1 = None
        view_251: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_105, _shape_param_275);  mm_105 = _shape_param_275 = None
        convert_element_type_237: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_251, torch.float32);  view_251 = None
        add_124: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_228, convert_element_type_237);  convert_element_type_228 = convert_element_type_237 = None
        convert_element_type_238: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        view_252: "bf16[128, 4096]" = torch.ops.aten.view.default(view_248, _shape_param_276);  view_248 = _shape_param_276 = None
        permute_89: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_252, [1, 0])
        mm_106: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_89, arg182_1);  permute_89 = None
        mm_107: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_252, arg193_1);  view_252 = arg193_1 = None
        view_253: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_107, _shape_param_277);  mm_107 = _shape_param_277 = None
        convert_element_type_239: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        add_125: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_124, convert_element_type_239);  add_124 = convert_element_type_239 = None
        convert_element_type_240: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        view_254: "bf16[128, 4096]" = torch.ops.aten.view.default(view_249, _shape_param_278);  view_249 = _shape_param_278 = None
        permute_90: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_254, [1, 0])
        mm_108: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_90, arg182_1);  permute_90 = arg182_1 = None
        mm_109: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_254, arg194_1);  view_254 = arg194_1 = None
        view_255: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_109, _shape_param_279);  mm_109 = _shape_param_279 = None
        convert_element_type_241: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_255, torch.float32);  view_255 = None
        add_126: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_125, convert_element_type_241);  add_125 = convert_element_type_241 = None
        convert_element_type_242: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        mul_211: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_126, arg195_1);  arg195_1 = None
        mul_212: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_211, 4096)
        sum_69: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True)
        mul_213: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_211, arg196_1);  mul_211 = None
        sum_70: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True);  mul_213 = None
        mul_214: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg196_1, sum_70);  sum_70 = None
        sub_33: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_212, sum_69);  mul_212 = sum_69 = None
        sub_34: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_33, mul_214);  sub_33 = mul_214 = None
        mul_215: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg197_1, sub_34);  arg197_1 = sub_34 = None
        mul_216: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_126, arg196_1);  arg196_1 = None
        sum_71: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1]);  mul_216 = None
        sum_72: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_126, [0, 1]);  add_126 = None
        add_127: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_113, mul_215);  add_113 = mul_215 = None
        convert_element_type_243: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16)
        view_256: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_243, _shape_param_280);  convert_element_type_243 = _shape_param_280 = None
        mm_110: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_256, arg198_1);  arg198_1 = None
        permute_91: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_256, [1, 0])
        mm_111: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_91, arg199_1);  arg199_1 = None
        sum_73: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_256, [0], True, dtype = torch.float32)
        view_257: "f32[4096]" = torch.ops.aten.view.default(sum_73, _shape_param_281);  sum_73 = _shape_param_281 = None
        convert_element_type_244: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_257, torch.bfloat16);  view_257 = None
        view_258: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_110, _shape_param_282);  mm_110 = _shape_param_282 = None
        convert_element_type_245: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_258, torch.float32);  view_258 = None
        convert_element_type_246: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_247: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_244, torch.float32);  convert_element_type_244 = None
        view_259: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg200_1, _shape_param_283);  arg200_1 = _shape_param_283 = None
        mul_217: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_259, 0.5)
        mul_218: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_245, mul_217);  mul_217 = None
        convert_element_type_248: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_259, torch.float32)
        pow_19: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_248, 3.0)
        mul_219: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_128: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_259, mul_219);  view_259 = mul_219 = None
        mul_220: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_128, 0.7978845608028654);  add_128 = None
        tanh_9: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_220);  mul_220 = None
        add_129: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_221: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_245, add_129);  convert_element_type_245 = add_129 = None
        convert_element_type_249: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_221, torch.bfloat16);  mul_221 = None
        mul_222: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_35: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_222);  mul_222 = None
        mul_223: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_218, sub_35);  mul_218 = sub_35 = None
        mul_224: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_223, 0.7978845608028654);  mul_223 = None
        convert_element_type_250: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_224, torch.bfloat16)
        mul_225: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_224, 0.044715);  mul_224 = None
        pow_20: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_248, 2.0);  convert_element_type_248 = None
        mul_226: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_227: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_225, mul_226);  mul_225 = mul_226 = None
        convert_element_type_251: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_227, torch.bfloat16);  mul_227 = None
        add_130: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_250, convert_element_type_251);  convert_element_type_250 = convert_element_type_251 = None
        mul_228: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_249, 0.5);  convert_element_type_249 = None
        add_131: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_130, mul_228);  add_130 = mul_228 = None
        view_260: "bf16[128, 16384]" = torch.ops.aten.view.default(add_131, _shape_param_284);  add_131 = _shape_param_284 = None
        mm_112: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_260, arg201_1);  arg201_1 = None
        permute_92: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_260, [1, 0])
        mm_113: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_92, arg202_1);  permute_92 = None
        sum_74: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True, dtype = torch.float32);  view_260 = None
        view_261: "f32[16384]" = torch.ops.aten.view.default(sum_74, _shape_param_285);  sum_74 = _shape_param_285 = None
        convert_element_type_252: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_261, torch.bfloat16);  view_261 = None
        view_262: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_112, _shape_param_286);  mm_112 = _shape_param_286 = None
        convert_element_type_253: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_262, torch.float32);  view_262 = None
        convert_element_type_254: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_255: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_252, torch.float32);  convert_element_type_252 = None
        mm_114: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_91, arg203_1);  permute_91 = arg203_1 = None
        mm_115: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_256, arg204_1);  view_256 = arg204_1 = None
        view_263: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_115, _shape_param_287);  mm_115 = _shape_param_287 = None
        convert_element_type_256: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        view_264: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_263, _shape_param_288);  view_263 = _shape_param_288 = None
        permute_93: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
        view_265: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_93, _shape_param_289);  permute_93 = _shape_param_289 = None
        bmm_36: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg205_1, view_265);  arg205_1 = None
        bmm_37: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_265, arg206_1);  view_265 = arg206_1 = None
        view_266: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_36, _shape_param_290);  bmm_36 = _shape_param_290 = None
        view_267: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_37, _shape_param_291);  bmm_37 = _shape_param_291 = None
        convert_element_type_257: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_267, torch.float32);  view_267 = None
        mul_229: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_257, arg207_1);  convert_element_type_257 = None
        sum_75: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [-1], True)
        neg_27: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg207_1);  arg207_1 = None
        fma_9: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_27, sum_75, mul_229);  neg_27 = sum_75 = mul_229 = None
        convert_element_type_258: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        div_12: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_258, 16.0);  convert_element_type_258 = None
        view_268: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_12, _shape_param_292);  div_12 = _shape_param_292 = None
        bmm_38: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg208_1, view_268);  arg208_1 = None
        bmm_39: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_268, arg209_1);  view_268 = arg209_1 = None
        view_269: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_293);  bmm_38 = _shape_param_293 = None
        view_270: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_39, _shape_param_294);  bmm_39 = _shape_param_294 = None
        convert_element_type_259: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_269, torch.float32);  view_269 = None
        permute_94: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_259, [0, 1, 3, 2]);  convert_element_type_259 = None
        convert_element_type_260: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_94, torch.bfloat16);  permute_94 = None
        convert_element_type_261: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_270, torch.bfloat16);  view_270 = None
        permute_95: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_261, [0, 2, 1, 3]);  convert_element_type_261 = None
        permute_96: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_260, [0, 2, 1, 3]);  convert_element_type_260 = None
        slice_37: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_95, 3, 0, 64)
        slice_38: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_95, 3, 64, 256);  permute_95 = None
        slice_39: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_96, 3, 0, 64)
        slice_40: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_96, 3, 64, 256);  permute_96 = None
        expand_20: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg210_1, _shape_param_295);  arg210_1 = _shape_param_295 = None
        clone_27: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_271: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_27, _shape_param_296);  clone_27 = _shape_param_296 = None
        mul_230: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_37, view_271)
        view_272: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_230, _shape_param_297);  mul_230 = _shape_param_297 = None
        select_36: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_272, -1, 0)
        select_37: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_272, -1, 1);  view_272 = None
        neg_28: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_36);  select_36 = None
        slice_scatter_72: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_28, 3, 1, 9223372036854775807, 2);  neg_28 = None
        slice_scatter_73: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_37, 3, 0, 9223372036854775807, 2);  select_37 = None
        add_132: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_72, slice_scatter_73);  slice_scatter_72 = slice_scatter_73 = None
        expand_21: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg211_1, _shape_param_298);  arg211_1 = _shape_param_298 = None
        clone_28: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_273: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_299);  clone_28 = _shape_param_299 = None
        mul_231: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_37, view_273);  slice_37 = None
        add_133: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_132, mul_231);  add_132 = mul_231 = None
        mul_232: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_39, view_271);  view_271 = None
        view_274: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_232, _shape_param_300);  mul_232 = _shape_param_300 = None
        select_38: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_274, -1, 0)
        select_39: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_274, -1, 1);  view_274 = None
        neg_29: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_38);  select_38 = None
        slice_scatter_74: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_29, 3, 1, 9223372036854775807, 2);  neg_29 = None
        slice_scatter_75: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_39, 3, 0, 9223372036854775807, 2);  select_39 = None
        add_134: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_74, slice_scatter_75);  slice_scatter_74 = slice_scatter_75 = None
        mul_233: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_39, view_273);  slice_39 = view_273 = None
        add_135: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_134, mul_233);  add_134 = mul_233 = None
        slice_scatter_76: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_38, 3, 64, 9223372036854775807);  slice_38 = None
        slice_scatter_77: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_133, 3, 0, 64);  add_133 = None
        add_136: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_76, slice_scatter_77);  slice_scatter_76 = slice_scatter_77 = None
        slice_scatter_78: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_40, 3, 64, 9223372036854775807);  slice_40 = None
        slice_scatter_79: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_135, 3, 0, 64);  add_135 = None
        add_137: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_78, slice_scatter_79);  slice_scatter_78 = slice_scatter_79 = None
        permute_97: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        clone_29: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
        view_275: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_29, _shape_param_301);  clone_29 = _shape_param_301 = None
        view_276: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_137, _shape_param_302);  add_137 = _shape_param_302 = None
        view_277: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_136, _shape_param_303);  add_136 = _shape_param_303 = None
        view_278: "bf16[128, 4096]" = torch.ops.aten.view.default(view_275, _shape_param_304);  view_275 = _shape_param_304 = None
        permute_98: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_278, [1, 0])
        mm_116: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_98, arg202_1);  permute_98 = None
        mm_117: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_278, arg212_1);  view_278 = arg212_1 = None
        view_279: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_117, _shape_param_305);  mm_117 = _shape_param_305 = None
        convert_element_type_262: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        add_138: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_253, convert_element_type_262);  convert_element_type_253 = convert_element_type_262 = None
        convert_element_type_263: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        view_280: "bf16[128, 4096]" = torch.ops.aten.view.default(view_276, _shape_param_306);  view_276 = _shape_param_306 = None
        permute_99: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_280, [1, 0])
        mm_118: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_99, arg202_1);  permute_99 = None
        mm_119: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_280, arg213_1);  view_280 = arg213_1 = None
        view_281: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_119, _shape_param_307);  mm_119 = _shape_param_307 = None
        convert_element_type_264: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_281, torch.float32);  view_281 = None
        add_139: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_138, convert_element_type_264);  add_138 = convert_element_type_264 = None
        convert_element_type_265: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        view_282: "bf16[128, 4096]" = torch.ops.aten.view.default(view_277, _shape_param_308);  view_277 = _shape_param_308 = None
        permute_100: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_282, [1, 0])
        mm_120: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_100, arg202_1);  permute_100 = arg202_1 = None
        mm_121: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_282, arg214_1);  view_282 = arg214_1 = None
        view_283: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_121, _shape_param_309);  mm_121 = _shape_param_309 = None
        convert_element_type_266: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        add_140: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_139, convert_element_type_266);  add_139 = convert_element_type_266 = None
        convert_element_type_267: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        mul_234: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_140, arg215_1);  arg215_1 = None
        mul_235: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_234, 4096)
        sum_76: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
        mul_236: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_234, arg216_1);  mul_234 = None
        sum_77: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
        mul_237: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg216_1, sum_77);  sum_77 = None
        sub_36: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_235, sum_76);  mul_235 = sum_76 = None
        sub_37: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_36, mul_237);  sub_36 = mul_237 = None
        mul_238: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg217_1, sub_37);  arg217_1 = sub_37 = None
        mul_239: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_140, arg216_1);  arg216_1 = None
        sum_78: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
        sum_79: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 1]);  add_140 = None
        add_141: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_127, mul_238);  add_127 = mul_238 = None
        convert_element_type_268: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16)
        view_284: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_268, _shape_param_310);  convert_element_type_268 = _shape_param_310 = None
        mm_122: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_284, arg218_1);  arg218_1 = None
        permute_101: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_284, [1, 0])
        mm_123: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_101, arg219_1);  arg219_1 = None
        sum_80: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True, dtype = torch.float32)
        view_285: "f32[4096]" = torch.ops.aten.view.default(sum_80, _shape_param_311);  sum_80 = _shape_param_311 = None
        convert_element_type_269: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_285, torch.bfloat16);  view_285 = None
        view_286: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_122, _shape_param_312);  mm_122 = _shape_param_312 = None
        convert_element_type_270: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_286, torch.float32);  view_286 = None
        convert_element_type_271: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_272: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_269, torch.float32);  convert_element_type_269 = None
        view_287: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg220_1, _shape_param_313);  arg220_1 = _shape_param_313 = None
        mul_240: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_287, 0.5)
        mul_241: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_270, mul_240);  mul_240 = None
        convert_element_type_273: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_287, torch.float32)
        pow_21: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_273, 3.0)
        mul_242: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_142: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_287, mul_242);  view_287 = mul_242 = None
        mul_243: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_142, 0.7978845608028654);  add_142 = None
        tanh_10: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_243);  mul_243 = None
        add_143: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_244: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_270, add_143);  convert_element_type_270 = add_143 = None
        convert_element_type_274: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_244, torch.bfloat16);  mul_244 = None
        mul_245: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_38: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_245);  mul_245 = None
        mul_246: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_241, sub_38);  mul_241 = sub_38 = None
        mul_247: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_246, 0.7978845608028654);  mul_246 = None
        convert_element_type_275: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_247, torch.bfloat16)
        mul_248: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_247, 0.044715);  mul_247 = None
        pow_22: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_273, 2.0);  convert_element_type_273 = None
        mul_249: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_250: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_248, mul_249);  mul_248 = mul_249 = None
        convert_element_type_276: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_250, torch.bfloat16);  mul_250 = None
        add_144: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_275, convert_element_type_276);  convert_element_type_275 = convert_element_type_276 = None
        mul_251: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_274, 0.5);  convert_element_type_274 = None
        add_145: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_144, mul_251);  add_144 = mul_251 = None
        view_288: "bf16[128, 16384]" = torch.ops.aten.view.default(add_145, _shape_param_314);  add_145 = _shape_param_314 = None
        mm_124: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_288, arg221_1);  arg221_1 = None
        permute_102: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_125: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_102, arg222_1);  permute_102 = None
        sum_81: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True, dtype = torch.float32);  view_288 = None
        view_289: "f32[16384]" = torch.ops.aten.view.default(sum_81, _shape_param_315);  sum_81 = _shape_param_315 = None
        convert_element_type_277: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_289, torch.bfloat16);  view_289 = None
        view_290: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_124, _shape_param_316);  mm_124 = _shape_param_316 = None
        convert_element_type_278: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_290, torch.float32);  view_290 = None
        convert_element_type_279: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_280: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_277, torch.float32);  convert_element_type_277 = None
        mm_126: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_101, arg223_1);  permute_101 = arg223_1 = None
        mm_127: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_284, arg224_1);  view_284 = arg224_1 = None
        view_291: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_127, _shape_param_317);  mm_127 = _shape_param_317 = None
        convert_element_type_281: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        view_292: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_291, _shape_param_318);  view_291 = _shape_param_318 = None
        permute_103: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        view_293: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_103, _shape_param_319);  permute_103 = _shape_param_319 = None
        bmm_40: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg225_1, view_293);  arg225_1 = None
        bmm_41: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_293, arg226_1);  view_293 = arg226_1 = None
        view_294: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_40, _shape_param_320);  bmm_40 = _shape_param_320 = None
        view_295: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_41, _shape_param_321);  bmm_41 = _shape_param_321 = None
        convert_element_type_282: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_295, torch.float32);  view_295 = None
        mul_252: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_282, arg227_1);  convert_element_type_282 = None
        sum_82: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [-1], True)
        neg_30: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg227_1);  arg227_1 = None
        fma_10: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_30, sum_82, mul_252);  neg_30 = sum_82 = mul_252 = None
        convert_element_type_283: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        div_13: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_283, 16.0);  convert_element_type_283 = None
        view_296: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_13, _shape_param_322);  div_13 = _shape_param_322 = None
        bmm_42: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg228_1, view_296);  arg228_1 = None
        bmm_43: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_296, arg229_1);  view_296 = arg229_1 = None
        view_297: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_323);  bmm_42 = _shape_param_323 = None
        view_298: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_43, _shape_param_324);  bmm_43 = _shape_param_324 = None
        convert_element_type_284: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_297, torch.float32);  view_297 = None
        permute_104: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_284, [0, 1, 3, 2]);  convert_element_type_284 = None
        convert_element_type_285: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_104, torch.bfloat16);  permute_104 = None
        convert_element_type_286: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_298, torch.bfloat16);  view_298 = None
        permute_105: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_286, [0, 2, 1, 3]);  convert_element_type_286 = None
        permute_106: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_285, [0, 2, 1, 3]);  convert_element_type_285 = None
        slice_41: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_105, 3, 0, 64)
        slice_42: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_105, 3, 64, 256);  permute_105 = None
        slice_43: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_106, 3, 0, 64)
        slice_44: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_106, 3, 64, 256);  permute_106 = None
        expand_22: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg230_1, _shape_param_325);  arg230_1 = _shape_param_325 = None
        clone_30: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_299: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_326);  clone_30 = _shape_param_326 = None
        mul_253: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_41, view_299)
        view_300: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_253, _shape_param_327);  mul_253 = _shape_param_327 = None
        select_40: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_300, -1, 0)
        select_41: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_300, -1, 1);  view_300 = None
        neg_31: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_40);  select_40 = None
        slice_scatter_80: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_31, 3, 1, 9223372036854775807, 2);  neg_31 = None
        slice_scatter_81: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_41, 3, 0, 9223372036854775807, 2);  select_41 = None
        add_146: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_80, slice_scatter_81);  slice_scatter_80 = slice_scatter_81 = None
        expand_23: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg231_1, _shape_param_328);  arg231_1 = _shape_param_328 = None
        clone_31: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_301: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_31, _shape_param_329);  clone_31 = _shape_param_329 = None
        mul_254: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_41, view_301);  slice_41 = None
        add_147: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_146, mul_254);  add_146 = mul_254 = None
        mul_255: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_43, view_299);  view_299 = None
        view_302: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_255, _shape_param_330);  mul_255 = _shape_param_330 = None
        select_42: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_302, -1, 0)
        select_43: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_302, -1, 1);  view_302 = None
        neg_32: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_42);  select_42 = None
        slice_scatter_82: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_32, 3, 1, 9223372036854775807, 2);  neg_32 = None
        slice_scatter_83: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_43, 3, 0, 9223372036854775807, 2);  select_43 = None
        add_148: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_82, slice_scatter_83);  slice_scatter_82 = slice_scatter_83 = None
        mul_256: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_43, view_301);  slice_43 = view_301 = None
        add_149: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_148, mul_256);  add_148 = mul_256 = None
        slice_scatter_84: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_42, 3, 64, 9223372036854775807);  slice_42 = None
        slice_scatter_85: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_147, 3, 0, 64);  add_147 = None
        add_150: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_84, slice_scatter_85);  slice_scatter_84 = slice_scatter_85 = None
        slice_scatter_86: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_44, 3, 64, 9223372036854775807);  slice_44 = None
        slice_scatter_87: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_149, 3, 0, 64);  add_149 = None
        add_151: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_86, slice_scatter_87);  slice_scatter_86 = slice_scatter_87 = None
        permute_107: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        clone_32: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
        view_303: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_32, _shape_param_331);  clone_32 = _shape_param_331 = None
        view_304: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_151, _shape_param_332);  add_151 = _shape_param_332 = None
        view_305: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_150, _shape_param_333);  add_150 = _shape_param_333 = None
        view_306: "bf16[128, 4096]" = torch.ops.aten.view.default(view_303, _shape_param_334);  view_303 = _shape_param_334 = None
        permute_108: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_306, [1, 0])
        mm_128: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_108, arg222_1);  permute_108 = None
        mm_129: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_306, arg232_1);  view_306 = arg232_1 = None
        view_307: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_129, _shape_param_335);  mm_129 = _shape_param_335 = None
        convert_element_type_287: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        add_152: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_278, convert_element_type_287);  convert_element_type_278 = convert_element_type_287 = None
        convert_element_type_288: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        view_308: "bf16[128, 4096]" = torch.ops.aten.view.default(view_304, _shape_param_336);  view_304 = _shape_param_336 = None
        permute_109: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_130: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_109, arg222_1);  permute_109 = None
        mm_131: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_308, arg233_1);  view_308 = arg233_1 = None
        view_309: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_131, _shape_param_337);  mm_131 = _shape_param_337 = None
        convert_element_type_289: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_309, torch.float32);  view_309 = None
        add_153: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_152, convert_element_type_289);  add_152 = convert_element_type_289 = None
        convert_element_type_290: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        view_310: "bf16[128, 4096]" = torch.ops.aten.view.default(view_305, _shape_param_338);  view_305 = _shape_param_338 = None
        permute_110: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_310, [1, 0])
        mm_132: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_110, arg222_1);  permute_110 = arg222_1 = None
        mm_133: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_310, arg234_1);  view_310 = arg234_1 = None
        view_311: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_133, _shape_param_339);  mm_133 = _shape_param_339 = None
        convert_element_type_291: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_311, torch.float32);  view_311 = None
        add_154: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_153, convert_element_type_291);  add_153 = convert_element_type_291 = None
        convert_element_type_292: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None
        mul_257: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_154, arg235_1);  arg235_1 = None
        mul_258: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_257, 4096)
        sum_83: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True)
        mul_259: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_257, arg236_1);  mul_257 = None
        sum_84: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_259, [2], True);  mul_259 = None
        mul_260: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg236_1, sum_84);  sum_84 = None
        sub_39: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_258, sum_83);  mul_258 = sum_83 = None
        sub_40: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_39, mul_260);  sub_39 = mul_260 = None
        mul_261: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg237_1, sub_40);  arg237_1 = sub_40 = None
        mul_262: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_154, arg236_1);  arg236_1 = None
        sum_85: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_262, [0, 1]);  mul_262 = None
        sum_86: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_154, [0, 1]);  add_154 = None
        add_155: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_141, mul_261);  add_141 = mul_261 = None
        convert_element_type_293: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16)
        view_312: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_293, _shape_param_340);  convert_element_type_293 = _shape_param_340 = None
        mm_134: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_312, arg238_1);  arg238_1 = None
        permute_111: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_312, [1, 0])
        mm_135: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_111, arg239_1);  arg239_1 = None
        sum_87: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_312, [0], True, dtype = torch.float32)
        view_313: "f32[4096]" = torch.ops.aten.view.default(sum_87, _shape_param_341);  sum_87 = _shape_param_341 = None
        convert_element_type_294: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_313, torch.bfloat16);  view_313 = None
        view_314: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_134, _shape_param_342);  mm_134 = _shape_param_342 = None
        convert_element_type_295: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_314, torch.float32);  view_314 = None
        convert_element_type_296: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_297: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_294, torch.float32);  convert_element_type_294 = None
        view_315: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg240_1, _shape_param_343);  arg240_1 = _shape_param_343 = None
        mul_263: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_315, 0.5)
        mul_264: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_295, mul_263);  mul_263 = None
        convert_element_type_298: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_315, torch.float32)
        pow_23: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_298, 3.0)
        mul_265: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_156: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_315, mul_265);  view_315 = mul_265 = None
        mul_266: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_156, 0.7978845608028654);  add_156 = None
        tanh_11: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_266);  mul_266 = None
        add_157: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_267: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_295, add_157);  convert_element_type_295 = add_157 = None
        convert_element_type_299: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_267, torch.bfloat16);  mul_267 = None
        mul_268: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_41: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_268);  mul_268 = None
        mul_269: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_264, sub_41);  mul_264 = sub_41 = None
        mul_270: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_269, 0.7978845608028654);  mul_269 = None
        convert_element_type_300: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_270, torch.bfloat16)
        mul_271: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_270, 0.044715);  mul_270 = None
        pow_24: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_298, 2.0);  convert_element_type_298 = None
        mul_272: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_273: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_271, mul_272);  mul_271 = mul_272 = None
        convert_element_type_301: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_273, torch.bfloat16);  mul_273 = None
        add_158: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_300, convert_element_type_301);  convert_element_type_300 = convert_element_type_301 = None
        mul_274: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_299, 0.5);  convert_element_type_299 = None
        add_159: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_158, mul_274);  add_158 = mul_274 = None
        view_316: "bf16[128, 16384]" = torch.ops.aten.view.default(add_159, _shape_param_344);  add_159 = _shape_param_344 = None
        mm_136: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_316, arg241_1);  arg241_1 = None
        permute_112: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_316, [1, 0])
        mm_137: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_112, arg242_1);  permute_112 = None
        sum_88: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_316, [0], True, dtype = torch.float32);  view_316 = None
        view_317: "f32[16384]" = torch.ops.aten.view.default(sum_88, _shape_param_345);  sum_88 = _shape_param_345 = None
        convert_element_type_302: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_317, torch.bfloat16);  view_317 = None
        view_318: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_136, _shape_param_346);  mm_136 = _shape_param_346 = None
        convert_element_type_303: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_318, torch.float32);  view_318 = None
        convert_element_type_304: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_305: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_302, torch.float32);  convert_element_type_302 = None
        mm_138: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_111, arg243_1);  permute_111 = arg243_1 = None
        mm_139: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_312, arg244_1);  view_312 = arg244_1 = None
        view_319: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_139, _shape_param_347);  mm_139 = _shape_param_347 = None
        convert_element_type_306: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        view_320: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_319, _shape_param_348);  view_319 = _shape_param_348 = None
        permute_113: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
        view_321: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_113, _shape_param_349);  permute_113 = _shape_param_349 = None
        bmm_44: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg245_1, view_321);  arg245_1 = None
        bmm_45: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_321, arg246_1);  view_321 = arg246_1 = None
        view_322: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_44, _shape_param_350);  bmm_44 = _shape_param_350 = None
        view_323: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_45, _shape_param_351);  bmm_45 = _shape_param_351 = None
        convert_element_type_307: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_323, torch.float32);  view_323 = None
        mul_275: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_307, arg247_1);  convert_element_type_307 = None
        sum_89: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [-1], True)
        neg_33: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg247_1);  arg247_1 = None
        fma_11: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_33, sum_89, mul_275);  neg_33 = sum_89 = mul_275 = None
        convert_element_type_308: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        div_14: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_308, 16.0);  convert_element_type_308 = None
        view_324: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_14, _shape_param_352);  div_14 = _shape_param_352 = None
        bmm_46: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg248_1, view_324);  arg248_1 = None
        bmm_47: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_324, arg249_1);  view_324 = arg249_1 = None
        view_325: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_353);  bmm_46 = _shape_param_353 = None
        view_326: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_47, _shape_param_354);  bmm_47 = _shape_param_354 = None
        convert_element_type_309: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_325, torch.float32);  view_325 = None
        permute_114: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_309, [0, 1, 3, 2]);  convert_element_type_309 = None
        convert_element_type_310: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_114, torch.bfloat16);  permute_114 = None
        convert_element_type_311: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_326, torch.bfloat16);  view_326 = None
        permute_115: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_311, [0, 2, 1, 3]);  convert_element_type_311 = None
        permute_116: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_310, [0, 2, 1, 3]);  convert_element_type_310 = None
        slice_45: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_115, 3, 0, 64)
        slice_46: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_115, 3, 64, 256);  permute_115 = None
        slice_47: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_116, 3, 0, 64)
        slice_48: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_116, 3, 64, 256);  permute_116 = None
        expand_24: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg250_1, _shape_param_355);  arg250_1 = _shape_param_355 = None
        clone_33: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_327: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_33, _shape_param_356);  clone_33 = _shape_param_356 = None
        mul_276: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_45, view_327)
        view_328: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_276, _shape_param_357);  mul_276 = _shape_param_357 = None
        select_44: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_328, -1, 0)
        select_45: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_328, -1, 1);  view_328 = None
        neg_34: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_44);  select_44 = None
        slice_scatter_88: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_34, 3, 1, 9223372036854775807, 2);  neg_34 = None
        slice_scatter_89: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_45, 3, 0, 9223372036854775807, 2);  select_45 = None
        add_160: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_88, slice_scatter_89);  slice_scatter_88 = slice_scatter_89 = None
        expand_25: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg251_1, _shape_param_358);  arg251_1 = _shape_param_358 = None
        clone_34: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_329: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_359);  clone_34 = _shape_param_359 = None
        mul_277: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_45, view_329);  slice_45 = None
        add_161: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_160, mul_277);  add_160 = mul_277 = None
        mul_278: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_47, view_327);  view_327 = None
        view_330: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_278, _shape_param_360);  mul_278 = _shape_param_360 = None
        select_46: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_330, -1, 0)
        select_47: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_330, -1, 1);  view_330 = None
        neg_35: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_46);  select_46 = None
        slice_scatter_90: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_35, 3, 1, 9223372036854775807, 2);  neg_35 = None
        slice_scatter_91: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_47, 3, 0, 9223372036854775807, 2);  select_47 = None
        add_162: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_90, slice_scatter_91);  slice_scatter_90 = slice_scatter_91 = None
        mul_279: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_47, view_329);  slice_47 = view_329 = None
        add_163: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_162, mul_279);  add_162 = mul_279 = None
        slice_scatter_92: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_46, 3, 64, 9223372036854775807);  slice_46 = None
        slice_scatter_93: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_161, 3, 0, 64);  add_161 = None
        add_164: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_92, slice_scatter_93);  slice_scatter_92 = slice_scatter_93 = None
        slice_scatter_94: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_48, 3, 64, 9223372036854775807);  slice_48 = None
        slice_scatter_95: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_163, 3, 0, 64);  add_163 = None
        add_165: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_94, slice_scatter_95);  slice_scatter_94 = slice_scatter_95 = None
        permute_117: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
        clone_35: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_331: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_35, _shape_param_361);  clone_35 = _shape_param_361 = None
        view_332: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_165, _shape_param_362);  add_165 = _shape_param_362 = None
        view_333: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_164, _shape_param_363);  add_164 = _shape_param_363 = None
        view_334: "bf16[128, 4096]" = torch.ops.aten.view.default(view_331, _shape_param_364);  view_331 = _shape_param_364 = None
        permute_118: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_140: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_118, arg242_1);  permute_118 = None
        mm_141: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_334, arg252_1);  view_334 = arg252_1 = None
        view_335: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_141, _shape_param_365);  mm_141 = _shape_param_365 = None
        convert_element_type_312: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_335, torch.float32);  view_335 = None
        add_166: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_303, convert_element_type_312);  convert_element_type_303 = convert_element_type_312 = None
        convert_element_type_313: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        view_336: "bf16[128, 4096]" = torch.ops.aten.view.default(view_332, _shape_param_366);  view_332 = _shape_param_366 = None
        permute_119: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_336, [1, 0])
        mm_142: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_119, arg242_1);  permute_119 = None
        mm_143: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_336, arg253_1);  view_336 = arg253_1 = None
        view_337: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_143, _shape_param_367);  mm_143 = _shape_param_367 = None
        convert_element_type_314: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_337, torch.float32);  view_337 = None
        add_167: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_166, convert_element_type_314);  add_166 = convert_element_type_314 = None
        convert_element_type_315: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        view_338: "bf16[128, 4096]" = torch.ops.aten.view.default(view_333, _shape_param_368);  view_333 = _shape_param_368 = None
        permute_120: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_144: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_120, arg242_1);  permute_120 = arg242_1 = None
        mm_145: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_338, arg254_1);  view_338 = arg254_1 = None
        view_339: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_145, _shape_param_369);  mm_145 = _shape_param_369 = None
        convert_element_type_316: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        add_168: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_167, convert_element_type_316);  add_167 = convert_element_type_316 = None
        convert_element_type_317: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        mul_280: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_168, arg255_1);  arg255_1 = None
        mul_281: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, 4096)
        sum_90: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True)
        mul_282: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, arg256_1);  mul_280 = None
        sum_91: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True);  mul_282 = None
        mul_283: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg256_1, sum_91);  sum_91 = None
        sub_42: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_281, sum_90);  mul_281 = sum_90 = None
        sub_43: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_42, mul_283);  sub_42 = mul_283 = None
        mul_284: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg257_1, sub_43);  arg257_1 = sub_43 = None
        mul_285: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_168, arg256_1);  arg256_1 = None
        sum_92: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 1]);  mul_285 = None
        sum_93: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_168, [0, 1]);  add_168 = None
        add_169: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_155, mul_284);  add_155 = mul_284 = None
        convert_element_type_318: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_169, torch.bfloat16)
        view_340: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_318, _shape_param_370);  convert_element_type_318 = _shape_param_370 = None
        mm_146: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_340, arg258_1);  arg258_1 = None
        permute_121: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_340, [1, 0])
        mm_147: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_121, arg259_1);  arg259_1 = None
        sum_94: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_340, [0], True, dtype = torch.float32)
        view_341: "f32[4096]" = torch.ops.aten.view.default(sum_94, _shape_param_371);  sum_94 = _shape_param_371 = None
        convert_element_type_319: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_341, torch.bfloat16);  view_341 = None
        view_342: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_146, _shape_param_372);  mm_146 = _shape_param_372 = None
        convert_element_type_320: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_342, torch.float32);  view_342 = None
        convert_element_type_321: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_322: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_319, torch.float32);  convert_element_type_319 = None
        view_343: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg260_1, _shape_param_373);  arg260_1 = _shape_param_373 = None
        mul_286: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
        mul_287: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_320, mul_286);  mul_286 = None
        convert_element_type_323: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_343, torch.float32)
        pow_25: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_323, 3.0)
        mul_288: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_170: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_343, mul_288);  view_343 = mul_288 = None
        mul_289: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_170, 0.7978845608028654);  add_170 = None
        tanh_12: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_289);  mul_289 = None
        add_171: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_290: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_320, add_171);  convert_element_type_320 = add_171 = None
        convert_element_type_324: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16);  mul_290 = None
        mul_291: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_44: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_291);  mul_291 = None
        mul_292: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_287, sub_44);  mul_287 = sub_44 = None
        mul_293: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_292, 0.7978845608028654);  mul_292 = None
        convert_element_type_325: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_293, torch.bfloat16)
        mul_294: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_293, 0.044715);  mul_293 = None
        pow_26: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_323, 2.0);  convert_element_type_323 = None
        mul_295: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
        mul_296: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_294, mul_295);  mul_294 = mul_295 = None
        convert_element_type_326: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16);  mul_296 = None
        add_172: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_325, convert_element_type_326);  convert_element_type_325 = convert_element_type_326 = None
        mul_297: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_324, 0.5);  convert_element_type_324 = None
        add_173: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_172, mul_297);  add_172 = mul_297 = None
        view_344: "bf16[128, 16384]" = torch.ops.aten.view.default(add_173, _shape_param_374);  add_173 = _shape_param_374 = None
        mm_148: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_344, arg261_1);  arg261_1 = None
        permute_122: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_344, [1, 0])
        mm_149: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_122, arg262_1);  permute_122 = None
        sum_95: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True, dtype = torch.float32);  view_344 = None
        view_345: "f32[16384]" = torch.ops.aten.view.default(sum_95, _shape_param_375);  sum_95 = _shape_param_375 = None
        convert_element_type_327: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_345, torch.bfloat16);  view_345 = None
        view_346: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_148, _shape_param_376);  mm_148 = _shape_param_376 = None
        convert_element_type_328: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_346, torch.float32);  view_346 = None
        convert_element_type_329: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_330: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_327, torch.float32);  convert_element_type_327 = None
        mm_150: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_121, arg263_1);  permute_121 = arg263_1 = None
        mm_151: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_340, arg264_1);  view_340 = arg264_1 = None
        view_347: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_151, _shape_param_377);  mm_151 = _shape_param_377 = None
        convert_element_type_331: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        view_348: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_347, _shape_param_378);  view_347 = _shape_param_378 = None
        permute_123: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_348, [0, 2, 1, 3]);  view_348 = None
        view_349: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_123, _shape_param_379);  permute_123 = _shape_param_379 = None
        bmm_48: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg265_1, view_349);  arg265_1 = None
        bmm_49: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_349, arg266_1);  view_349 = arg266_1 = None
        view_350: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_48, _shape_param_380);  bmm_48 = _shape_param_380 = None
        view_351: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_49, _shape_param_381);  bmm_49 = _shape_param_381 = None
        convert_element_type_332: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_351, torch.float32);  view_351 = None
        mul_298: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_332, arg267_1);  convert_element_type_332 = None
        sum_96: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [-1], True)
        neg_36: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg267_1);  arg267_1 = None
        fma_12: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_36, sum_96, mul_298);  neg_36 = sum_96 = mul_298 = None
        convert_element_type_333: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None
        div_15: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_333, 16.0);  convert_element_type_333 = None
        view_352: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_15, _shape_param_382);  div_15 = _shape_param_382 = None
        bmm_50: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg268_1, view_352);  arg268_1 = None
        bmm_51: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_352, arg269_1);  view_352 = arg269_1 = None
        view_353: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_50, _shape_param_383);  bmm_50 = _shape_param_383 = None
        view_354: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_51, _shape_param_384);  bmm_51 = _shape_param_384 = None
        convert_element_type_334: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_353, torch.float32);  view_353 = None
        permute_124: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_334, [0, 1, 3, 2]);  convert_element_type_334 = None
        convert_element_type_335: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_124, torch.bfloat16);  permute_124 = None
        convert_element_type_336: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_354, torch.bfloat16);  view_354 = None
        permute_125: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_336, [0, 2, 1, 3]);  convert_element_type_336 = None
        permute_126: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_335, [0, 2, 1, 3]);  convert_element_type_335 = None
        slice_49: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_125, 3, 0, 64)
        slice_50: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_125, 3, 64, 256);  permute_125 = None
        slice_51: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_126, 3, 0, 64)
        slice_52: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_126, 3, 64, 256);  permute_126 = None
        expand_26: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg270_1, _shape_param_385);  arg270_1 = _shape_param_385 = None
        clone_36: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_355: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_386);  clone_36 = _shape_param_386 = None
        mul_299: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_49, view_355)
        view_356: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_299, _shape_param_387);  mul_299 = _shape_param_387 = None
        select_48: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_356, -1, 0)
        select_49: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_356, -1, 1);  view_356 = None
        neg_37: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_48);  select_48 = None
        slice_scatter_96: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_37, 3, 1, 9223372036854775807, 2);  neg_37 = None
        slice_scatter_97: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_49, 3, 0, 9223372036854775807, 2);  select_49 = None
        add_174: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_96, slice_scatter_97);  slice_scatter_96 = slice_scatter_97 = None
        expand_27: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg271_1, _shape_param_388);  arg271_1 = _shape_param_388 = None
        clone_37: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_357: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_37, _shape_param_389);  clone_37 = _shape_param_389 = None
        mul_300: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_49, view_357);  slice_49 = None
        add_175: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_174, mul_300);  add_174 = mul_300 = None
        mul_301: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_51, view_355);  view_355 = None
        view_358: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_301, _shape_param_390);  mul_301 = _shape_param_390 = None
        select_50: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_358, -1, 0)
        select_51: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_358, -1, 1);  view_358 = None
        neg_38: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_50);  select_50 = None
        slice_scatter_98: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_38, 3, 1, 9223372036854775807, 2);  neg_38 = None
        slice_scatter_99: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_51, 3, 0, 9223372036854775807, 2);  select_51 = None
        add_176: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_98, slice_scatter_99);  slice_scatter_98 = slice_scatter_99 = None
        mul_302: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_51, view_357);  slice_51 = view_357 = None
        add_177: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_176, mul_302);  add_176 = mul_302 = None
        slice_scatter_100: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_50, 3, 64, 9223372036854775807);  slice_50 = None
        slice_scatter_101: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_175, 3, 0, 64);  add_175 = None
        add_178: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_100, slice_scatter_101);  slice_scatter_100 = slice_scatter_101 = None
        slice_scatter_102: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_52, 3, 64, 9223372036854775807);  slice_52 = None
        slice_scatter_103: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_177, 3, 0, 64);  add_177 = None
        add_179: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_102, slice_scatter_103);  slice_scatter_102 = slice_scatter_103 = None
        permute_127: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
        clone_38: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_359: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_38, _shape_param_391);  clone_38 = _shape_param_391 = None
        view_360: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_179, _shape_param_392);  add_179 = _shape_param_392 = None
        view_361: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_178, _shape_param_393);  add_178 = _shape_param_393 = None
        view_362: "bf16[128, 4096]" = torch.ops.aten.view.default(view_359, _shape_param_394);  view_359 = _shape_param_394 = None
        permute_128: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_362, [1, 0])
        mm_152: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_128, arg262_1);  permute_128 = None
        mm_153: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_362, arg272_1);  view_362 = arg272_1 = None
        view_363: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_153, _shape_param_395);  mm_153 = _shape_param_395 = None
        convert_element_type_337: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_363, torch.float32);  view_363 = None
        add_180: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_328, convert_element_type_337);  convert_element_type_328 = convert_element_type_337 = None
        convert_element_type_338: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        view_364: "bf16[128, 4096]" = torch.ops.aten.view.default(view_360, _shape_param_396);  view_360 = _shape_param_396 = None
        permute_129: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_154: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_129, arg262_1);  permute_129 = None
        mm_155: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_364, arg273_1);  view_364 = arg273_1 = None
        view_365: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_155, _shape_param_397);  mm_155 = _shape_param_397 = None
        convert_element_type_339: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_365, torch.float32);  view_365 = None
        add_181: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_180, convert_element_type_339);  add_180 = convert_element_type_339 = None
        convert_element_type_340: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        view_366: "bf16[128, 4096]" = torch.ops.aten.view.default(view_361, _shape_param_398);  view_361 = _shape_param_398 = None
        permute_130: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_156: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_130, arg262_1);  permute_130 = arg262_1 = None
        mm_157: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_366, arg274_1);  view_366 = arg274_1 = None
        view_367: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_157, _shape_param_399);  mm_157 = _shape_param_399 = None
        convert_element_type_341: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_367, torch.float32);  view_367 = None
        add_182: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_181, convert_element_type_341);  add_181 = convert_element_type_341 = None
        convert_element_type_342: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        mul_303: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_182, arg275_1);  arg275_1 = None
        mul_304: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_303, 4096)
        sum_97: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True)
        mul_305: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_303, arg276_1);  mul_303 = None
        sum_98: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_305, [2], True);  mul_305 = None
        mul_306: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg276_1, sum_98);  sum_98 = None
        sub_45: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_304, sum_97);  mul_304 = sum_97 = None
        sub_46: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_45, mul_306);  sub_45 = mul_306 = None
        mul_307: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg277_1, sub_46);  arg277_1 = sub_46 = None
        mul_308: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_182, arg276_1);  arg276_1 = None
        sum_99: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1]);  mul_308 = None
        sum_100: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1]);  add_182 = None
        add_183: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_169, mul_307);  add_169 = mul_307 = None
        convert_element_type_343: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_183, torch.bfloat16)
        view_368: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_343, _shape_param_400);  convert_element_type_343 = _shape_param_400 = None
        mm_158: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_368, arg278_1);  arg278_1 = None
        permute_131: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_368, [1, 0])
        mm_159: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_131, arg279_1);  arg279_1 = None
        sum_101: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True, dtype = torch.float32)
        view_369: "f32[4096]" = torch.ops.aten.view.default(sum_101, _shape_param_401);  sum_101 = _shape_param_401 = None
        convert_element_type_344: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_369, torch.bfloat16);  view_369 = None
        view_370: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_158, _shape_param_402);  mm_158 = _shape_param_402 = None
        convert_element_type_345: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_370, torch.float32);  view_370 = None
        convert_element_type_346: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_347: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_344, torch.float32);  convert_element_type_344 = None
        view_371: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg280_1, _shape_param_403);  arg280_1 = _shape_param_403 = None
        mul_309: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_371, 0.5)
        mul_310: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_345, mul_309);  mul_309 = None
        convert_element_type_348: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_371, torch.float32)
        pow_27: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_348, 3.0)
        mul_311: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_184: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_371, mul_311);  view_371 = mul_311 = None
        mul_312: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_184, 0.7978845608028654);  add_184 = None
        tanh_13: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_312);  mul_312 = None
        add_185: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_313: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_345, add_185);  convert_element_type_345 = add_185 = None
        convert_element_type_349: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_313, torch.bfloat16);  mul_313 = None
        mul_314: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_47: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_314);  mul_314 = None
        mul_315: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_310, sub_47);  mul_310 = sub_47 = None
        mul_316: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_315, 0.7978845608028654);  mul_315 = None
        convert_element_type_350: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_316, torch.bfloat16)
        mul_317: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_316, 0.044715);  mul_316 = None
        pow_28: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_348, 2.0);  convert_element_type_348 = None
        mul_318: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_28, 3.0);  pow_28 = None
        mul_319: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_317, mul_318);  mul_317 = mul_318 = None
        convert_element_type_351: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_319, torch.bfloat16);  mul_319 = None
        add_186: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_350, convert_element_type_351);  convert_element_type_350 = convert_element_type_351 = None
        mul_320: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_349, 0.5);  convert_element_type_349 = None
        add_187: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_186, mul_320);  add_186 = mul_320 = None
        view_372: "bf16[128, 16384]" = torch.ops.aten.view.default(add_187, _shape_param_404);  add_187 = _shape_param_404 = None
        mm_160: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_372, arg281_1);  arg281_1 = None
        permute_132: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_372, [1, 0])
        mm_161: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_132, arg282_1);  permute_132 = None
        sum_102: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_372, [0], True, dtype = torch.float32);  view_372 = None
        view_373: "f32[16384]" = torch.ops.aten.view.default(sum_102, _shape_param_405);  sum_102 = _shape_param_405 = None
        convert_element_type_352: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_373, torch.bfloat16);  view_373 = None
        view_374: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_160, _shape_param_406);  mm_160 = _shape_param_406 = None
        convert_element_type_353: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_374, torch.float32);  view_374 = None
        convert_element_type_354: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None
        convert_element_type_355: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_352, torch.float32);  convert_element_type_352 = None
        mm_162: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_131, arg283_1);  permute_131 = arg283_1 = None
        mm_163: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_368, arg284_1);  view_368 = arg284_1 = None
        view_375: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_163, _shape_param_407);  mm_163 = _shape_param_407 = None
        convert_element_type_356: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        view_376: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_375, _shape_param_408);  view_375 = _shape_param_408 = None
        permute_133: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        view_377: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_133, _shape_param_409);  permute_133 = _shape_param_409 = None
        bmm_52: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg285_1, view_377);  arg285_1 = None
        bmm_53: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_377, arg286_1);  view_377 = arg286_1 = None
        view_378: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_52, _shape_param_410);  bmm_52 = _shape_param_410 = None
        view_379: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_53, _shape_param_411);  bmm_53 = _shape_param_411 = None
        convert_element_type_357: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_379, torch.float32);  view_379 = None
        mul_321: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_357, arg287_1);  convert_element_type_357 = None
        sum_103: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [-1], True)
        neg_39: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg287_1);  arg287_1 = None
        fma_13: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_39, sum_103, mul_321);  neg_39 = sum_103 = mul_321 = None
        convert_element_type_358: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None
        div_16: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_358, 16.0);  convert_element_type_358 = None
        view_380: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_16, _shape_param_412);  div_16 = _shape_param_412 = None
        bmm_54: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg288_1, view_380);  arg288_1 = None
        bmm_55: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_380, arg289_1);  view_380 = arg289_1 = None
        view_381: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_54, _shape_param_413);  bmm_54 = _shape_param_413 = None
        view_382: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_55, _shape_param_414);  bmm_55 = _shape_param_414 = None
        convert_element_type_359: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_381, torch.float32);  view_381 = None
        permute_134: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_359, [0, 1, 3, 2]);  convert_element_type_359 = None
        convert_element_type_360: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_134, torch.bfloat16);  permute_134 = None
        convert_element_type_361: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_382, torch.bfloat16);  view_382 = None
        permute_135: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_361, [0, 2, 1, 3]);  convert_element_type_361 = None
        permute_136: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_360, [0, 2, 1, 3]);  convert_element_type_360 = None
        slice_53: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_135, 3, 0, 64)
        slice_54: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_135, 3, 64, 256);  permute_135 = None
        slice_55: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_136, 3, 0, 64)
        slice_56: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_136, 3, 64, 256);  permute_136 = None
        expand_28: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg290_1, _shape_param_415);  arg290_1 = _shape_param_415 = None
        clone_39: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_383: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_39, _shape_param_416);  clone_39 = _shape_param_416 = None
        mul_322: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_53, view_383)
        view_384: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_322, _shape_param_417);  mul_322 = _shape_param_417 = None
        select_52: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_384, -1, 0)
        select_53: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_384, -1, 1);  view_384 = None
        neg_40: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_52);  select_52 = None
        slice_scatter_104: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_40, 3, 1, 9223372036854775807, 2);  neg_40 = None
        slice_scatter_105: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_53, 3, 0, 9223372036854775807, 2);  select_53 = None
        add_188: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_104, slice_scatter_105);  slice_scatter_104 = slice_scatter_105 = None
        expand_29: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg291_1, _shape_param_418);  arg291_1 = _shape_param_418 = None
        clone_40: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_385: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_419);  clone_40 = _shape_param_419 = None
        mul_323: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_53, view_385);  slice_53 = None
        add_189: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_188, mul_323);  add_188 = mul_323 = None
        mul_324: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_55, view_383);  view_383 = None
        view_386: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_324, _shape_param_420);  mul_324 = _shape_param_420 = None
        select_54: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_386, -1, 0)
        select_55: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_386, -1, 1);  view_386 = None
        neg_41: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_54);  select_54 = None
        slice_scatter_106: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_41, 3, 1, 9223372036854775807, 2);  neg_41 = None
        slice_scatter_107: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_55, 3, 0, 9223372036854775807, 2);  select_55 = None
        add_190: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_106, slice_scatter_107);  slice_scatter_106 = slice_scatter_107 = None
        mul_325: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_55, view_385);  slice_55 = view_385 = None
        add_191: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_190, mul_325);  add_190 = mul_325 = None
        slice_scatter_108: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_54, 3, 64, 9223372036854775807);  slice_54 = None
        slice_scatter_109: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_189, 3, 0, 64);  add_189 = None
        add_192: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_108, slice_scatter_109);  slice_scatter_108 = slice_scatter_109 = None
        slice_scatter_110: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_56, 3, 64, 9223372036854775807);  slice_56 = None
        slice_scatter_111: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_191, 3, 0, 64);  add_191 = None
        add_193: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_110, slice_scatter_111);  slice_scatter_110 = slice_scatter_111 = None
        permute_137: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        clone_41: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_387: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_41, _shape_param_421);  clone_41 = _shape_param_421 = None
        view_388: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_193, _shape_param_422);  add_193 = _shape_param_422 = None
        view_389: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_192, _shape_param_423);  add_192 = _shape_param_423 = None
        view_390: "bf16[128, 4096]" = torch.ops.aten.view.default(view_387, _shape_param_424);  view_387 = _shape_param_424 = None
        permute_138: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_390, [1, 0])
        mm_164: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_138, arg282_1);  permute_138 = None
        mm_165: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_390, arg292_1);  view_390 = arg292_1 = None
        view_391: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_165, _shape_param_425);  mm_165 = _shape_param_425 = None
        convert_element_type_362: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_391, torch.float32);  view_391 = None
        add_194: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_353, convert_element_type_362);  convert_element_type_353 = convert_element_type_362 = None
        convert_element_type_363: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        view_392: "bf16[128, 4096]" = torch.ops.aten.view.default(view_388, _shape_param_426);  view_388 = _shape_param_426 = None
        permute_139: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_166: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_139, arg282_1);  permute_139 = None
        mm_167: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_392, arg293_1);  view_392 = arg293_1 = None
        view_393: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_167, _shape_param_427);  mm_167 = _shape_param_427 = None
        convert_element_type_364: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        add_195: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_194, convert_element_type_364);  add_194 = convert_element_type_364 = None
        convert_element_type_365: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        view_394: "bf16[128, 4096]" = torch.ops.aten.view.default(view_389, _shape_param_428);  view_389 = _shape_param_428 = None
        permute_140: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_394, [1, 0])
        mm_168: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_140, arg282_1);  permute_140 = arg282_1 = None
        mm_169: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_394, arg294_1);  view_394 = arg294_1 = None
        view_395: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_169, _shape_param_429);  mm_169 = _shape_param_429 = None
        convert_element_type_366: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_395, torch.float32);  view_395 = None
        add_196: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_195, convert_element_type_366);  add_195 = convert_element_type_366 = None
        convert_element_type_367: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        mul_326: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_196, arg295_1);  arg295_1 = None
        mul_327: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_326, 4096)
        sum_104: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True)
        mul_328: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_326, arg296_1);  mul_326 = None
        sum_105: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True);  mul_328 = None
        mul_329: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg296_1, sum_105);  sum_105 = None
        sub_48: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_327, sum_104);  mul_327 = sum_104 = None
        sub_49: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_48, mul_329);  sub_48 = mul_329 = None
        mul_330: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg297_1, sub_49);  arg297_1 = sub_49 = None
        mul_331: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_196, arg296_1);  arg296_1 = None
        sum_106: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1]);  mul_331 = None
        sum_107: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_196, [0, 1]);  add_196 = None
        add_197: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_183, mul_330);  add_183 = mul_330 = None
        convert_element_type_368: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16)
        view_396: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_368, _shape_param_430);  convert_element_type_368 = _shape_param_430 = None
        mm_170: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_396, arg298_1);  arg298_1 = None
        permute_141: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_396, [1, 0])
        mm_171: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_141, arg299_1);  arg299_1 = None
        sum_108: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_396, [0], True, dtype = torch.float32)
        view_397: "f32[4096]" = torch.ops.aten.view.default(sum_108, _shape_param_431);  sum_108 = _shape_param_431 = None
        convert_element_type_369: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_397, torch.bfloat16);  view_397 = None
        view_398: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_170, _shape_param_432);  mm_170 = _shape_param_432 = None
        convert_element_type_370: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        convert_element_type_371: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        convert_element_type_372: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_369, torch.float32);  convert_element_type_369 = None
        view_399: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg300_1, _shape_param_433);  arg300_1 = _shape_param_433 = None
        mul_332: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_399, 0.5)
        mul_333: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_370, mul_332);  mul_332 = None
        convert_element_type_373: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_399, torch.float32)
        pow_29: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_373, 3.0)
        mul_334: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_198: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_399, mul_334);  view_399 = mul_334 = None
        mul_335: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_198, 0.7978845608028654);  add_198 = None
        tanh_14: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_335);  mul_335 = None
        add_199: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_336: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_370, add_199);  convert_element_type_370 = add_199 = None
        convert_element_type_374: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_336, torch.bfloat16);  mul_336 = None
        mul_337: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_50: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_337);  mul_337 = None
        mul_338: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_333, sub_50);  mul_333 = sub_50 = None
        mul_339: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_338, 0.7978845608028654);  mul_338 = None
        convert_element_type_375: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_339, torch.bfloat16)
        mul_340: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_339, 0.044715);  mul_339 = None
        pow_30: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_373, 2.0);  convert_element_type_373 = None
        mul_341: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
        mul_342: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_340, mul_341);  mul_340 = mul_341 = None
        convert_element_type_376: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_342, torch.bfloat16);  mul_342 = None
        add_200: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_375, convert_element_type_376);  convert_element_type_375 = convert_element_type_376 = None
        mul_343: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_374, 0.5);  convert_element_type_374 = None
        add_201: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_200, mul_343);  add_200 = mul_343 = None
        view_400: "bf16[128, 16384]" = torch.ops.aten.view.default(add_201, _shape_param_434);  add_201 = _shape_param_434 = None
        mm_172: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_400, arg301_1);  arg301_1 = None
        permute_142: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_400, [1, 0])
        mm_173: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_142, arg302_1);  permute_142 = None
        sum_109: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_400, [0], True, dtype = torch.float32);  view_400 = None
        view_401: "f32[16384]" = torch.ops.aten.view.default(sum_109, _shape_param_435);  sum_109 = _shape_param_435 = None
        convert_element_type_377: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_401, torch.bfloat16);  view_401 = None
        view_402: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_172, _shape_param_436);  mm_172 = _shape_param_436 = None
        convert_element_type_378: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_402, torch.float32);  view_402 = None
        convert_element_type_379: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        convert_element_type_380: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_377, torch.float32);  convert_element_type_377 = None
        mm_174: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_141, arg303_1);  permute_141 = arg303_1 = None
        mm_175: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_396, arg304_1);  view_396 = arg304_1 = None
        view_403: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_175, _shape_param_437);  mm_175 = _shape_param_437 = None
        convert_element_type_381: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        view_404: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_403, _shape_param_438);  view_403 = _shape_param_438 = None
        permute_143: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        view_405: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_143, _shape_param_439);  permute_143 = _shape_param_439 = None
        bmm_56: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg305_1, view_405);  arg305_1 = None
        bmm_57: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_405, arg306_1);  view_405 = arg306_1 = None
        view_406: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_56, _shape_param_440);  bmm_56 = _shape_param_440 = None
        view_407: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_57, _shape_param_441);  bmm_57 = _shape_param_441 = None
        convert_element_type_382: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_407, torch.float32);  view_407 = None
        mul_344: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_382, arg307_1);  convert_element_type_382 = None
        sum_110: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_344, [-1], True)
        neg_42: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg307_1);  arg307_1 = None
        fma_14: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_42, sum_110, mul_344);  neg_42 = sum_110 = mul_344 = None
        convert_element_type_383: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None
        div_17: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_383, 16.0);  convert_element_type_383 = None
        view_408: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_17, _shape_param_442);  div_17 = _shape_param_442 = None
        bmm_58: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg308_1, view_408);  arg308_1 = None
        bmm_59: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_408, arg309_1);  view_408 = arg309_1 = None
        view_409: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_58, _shape_param_443);  bmm_58 = _shape_param_443 = None
        view_410: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_59, _shape_param_444);  bmm_59 = _shape_param_444 = None
        convert_element_type_384: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_409, torch.float32);  view_409 = None
        permute_144: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_384, [0, 1, 3, 2]);  convert_element_type_384 = None
        convert_element_type_385: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_144, torch.bfloat16);  permute_144 = None
        convert_element_type_386: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_410, torch.bfloat16);  view_410 = None
        permute_145: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_386, [0, 2, 1, 3]);  convert_element_type_386 = None
        permute_146: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_385, [0, 2, 1, 3]);  convert_element_type_385 = None
        slice_57: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_145, 3, 0, 64)
        slice_58: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_145, 3, 64, 256);  permute_145 = None
        slice_59: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_146, 3, 0, 64)
        slice_60: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_146, 3, 64, 256);  permute_146 = None
        expand_30: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg310_1, _shape_param_445);  arg310_1 = _shape_param_445 = None
        clone_42: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_411: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_446);  clone_42 = _shape_param_446 = None
        mul_345: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_57, view_411)
        view_412: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_345, _shape_param_447);  mul_345 = _shape_param_447 = None
        select_56: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_412, -1, 0)
        select_57: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_412, -1, 1);  view_412 = None
        neg_43: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_56);  select_56 = None
        slice_scatter_112: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_43, 3, 1, 9223372036854775807, 2);  neg_43 = None
        slice_scatter_113: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_57, 3, 0, 9223372036854775807, 2);  select_57 = None
        add_202: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_112, slice_scatter_113);  slice_scatter_112 = slice_scatter_113 = None
        expand_31: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg311_1, _shape_param_448);  arg311_1 = _shape_param_448 = None
        clone_43: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_413: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_43, _shape_param_449);  clone_43 = _shape_param_449 = None
        mul_346: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_57, view_413);  slice_57 = None
        add_203: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_202, mul_346);  add_202 = mul_346 = None
        mul_347: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_59, view_411);  view_411 = None
        view_414: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_347, _shape_param_450);  mul_347 = _shape_param_450 = None
        select_58: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_414, -1, 0)
        select_59: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_414, -1, 1);  view_414 = None
        neg_44: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_58);  select_58 = None
        slice_scatter_114: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_44, 3, 1, 9223372036854775807, 2);  neg_44 = None
        slice_scatter_115: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_59, 3, 0, 9223372036854775807, 2);  select_59 = None
        add_204: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_114, slice_scatter_115);  slice_scatter_114 = slice_scatter_115 = None
        mul_348: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_59, view_413);  slice_59 = view_413 = None
        add_205: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_204, mul_348);  add_204 = mul_348 = None
        slice_scatter_116: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_58, 3, 64, 9223372036854775807);  slice_58 = None
        slice_scatter_117: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_203, 3, 0, 64);  add_203 = None
        add_206: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_116, slice_scatter_117);  slice_scatter_116 = slice_scatter_117 = None
        slice_scatter_118: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_60, 3, 64, 9223372036854775807);  slice_60 = None
        slice_scatter_119: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_205, 3, 0, 64);  add_205 = None
        add_207: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_118, slice_scatter_119);  slice_scatter_118 = slice_scatter_119 = None
        permute_147: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        clone_44: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_415: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_44, _shape_param_451);  clone_44 = _shape_param_451 = None
        view_416: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_207, _shape_param_452);  add_207 = _shape_param_452 = None
        view_417: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_206, _shape_param_453);  add_206 = _shape_param_453 = None
        view_418: "bf16[128, 4096]" = torch.ops.aten.view.default(view_415, _shape_param_454);  view_415 = _shape_param_454 = None
        permute_148: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_418, [1, 0])
        mm_176: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_148, arg302_1);  permute_148 = None
        mm_177: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_418, arg312_1);  view_418 = arg312_1 = None
        view_419: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_177, _shape_param_455);  mm_177 = _shape_param_455 = None
        convert_element_type_387: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        add_208: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_378, convert_element_type_387);  convert_element_type_378 = convert_element_type_387 = None
        convert_element_type_388: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        view_420: "bf16[128, 4096]" = torch.ops.aten.view.default(view_416, _shape_param_456);  view_416 = _shape_param_456 = None
        permute_149: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_420, [1, 0])
        mm_178: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_149, arg302_1);  permute_149 = None
        mm_179: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_420, arg313_1);  view_420 = arg313_1 = None
        view_421: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_179, _shape_param_457);  mm_179 = _shape_param_457 = None
        convert_element_type_389: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_421, torch.float32);  view_421 = None
        add_209: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_208, convert_element_type_389);  add_208 = convert_element_type_389 = None
        convert_element_type_390: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        view_422: "bf16[128, 4096]" = torch.ops.aten.view.default(view_417, _shape_param_458);  view_417 = _shape_param_458 = None
        permute_150: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_422, [1, 0])
        mm_180: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_150, arg302_1);  permute_150 = arg302_1 = None
        mm_181: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_422, arg314_1);  view_422 = arg314_1 = None
        view_423: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_181, _shape_param_459);  mm_181 = _shape_param_459 = None
        convert_element_type_391: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_423, torch.float32);  view_423 = None
        add_210: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_209, convert_element_type_391);  add_209 = convert_element_type_391 = None
        convert_element_type_392: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None
        mul_349: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_210, arg315_1);  arg315_1 = None
        mul_350: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_349, 4096)
        sum_111: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True)
        mul_351: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_349, arg316_1);  mul_349 = None
        sum_112: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [2], True);  mul_351 = None
        mul_352: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg316_1, sum_112);  sum_112 = None
        sub_51: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_350, sum_111);  mul_350 = sum_111 = None
        sub_52: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_51, mul_352);  sub_51 = mul_352 = None
        mul_353: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg317_1, sub_52);  arg317_1 = sub_52 = None
        mul_354: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_210, arg316_1);  arg316_1 = None
        sum_113: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_354, [0, 1]);  mul_354 = None
        sum_114: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_210, [0, 1]);  add_210 = None
        add_211: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_197, mul_353);  add_197 = mul_353 = None
        convert_element_type_393: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_211, torch.bfloat16)
        view_424: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_393, _shape_param_460);  convert_element_type_393 = _shape_param_460 = None
        mm_182: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_424, arg318_1);  arg318_1 = None
        permute_151: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_424, [1, 0])
        mm_183: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_151, arg319_1);  arg319_1 = None
        sum_115: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_424, [0], True, dtype = torch.float32)
        view_425: "f32[4096]" = torch.ops.aten.view.default(sum_115, _shape_param_461);  sum_115 = _shape_param_461 = None
        convert_element_type_394: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_425, torch.bfloat16);  view_425 = None
        view_426: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_182, _shape_param_462);  mm_182 = _shape_param_462 = None
        convert_element_type_395: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_426, torch.float32);  view_426 = None
        convert_element_type_396: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_397: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_394, torch.float32);  convert_element_type_394 = None
        view_427: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg320_1, _shape_param_463);  arg320_1 = _shape_param_463 = None
        mul_355: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_427, 0.5)
        mul_356: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_395, mul_355);  mul_355 = None
        convert_element_type_398: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_427, torch.float32)
        pow_31: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_398, 3.0)
        mul_357: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_31, 0.044715);  pow_31 = None
        add_212: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_427, mul_357);  view_427 = mul_357 = None
        mul_358: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_212, 0.7978845608028654);  add_212 = None
        tanh_15: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_358);  mul_358 = None
        add_213: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_359: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_395, add_213);  convert_element_type_395 = add_213 = None
        convert_element_type_399: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_359, torch.bfloat16);  mul_359 = None
        mul_360: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_53: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_360);  mul_360 = None
        mul_361: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_356, sub_53);  mul_356 = sub_53 = None
        mul_362: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_361, 0.7978845608028654);  mul_361 = None
        convert_element_type_400: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16)
        mul_363: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_362, 0.044715);  mul_362 = None
        pow_32: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_398, 2.0);  convert_element_type_398 = None
        mul_364: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
        mul_365: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_363, mul_364);  mul_363 = mul_364 = None
        convert_element_type_401: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_365, torch.bfloat16);  mul_365 = None
        add_214: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_400, convert_element_type_401);  convert_element_type_400 = convert_element_type_401 = None
        mul_366: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_399, 0.5);  convert_element_type_399 = None
        add_215: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_214, mul_366);  add_214 = mul_366 = None
        view_428: "bf16[128, 16384]" = torch.ops.aten.view.default(add_215, _shape_param_464);  add_215 = _shape_param_464 = None
        mm_184: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_428, arg321_1);  arg321_1 = None
        permute_152: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_428, [1, 0])
        mm_185: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_152, arg322_1);  permute_152 = None
        sum_116: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_428, [0], True, dtype = torch.float32);  view_428 = None
        view_429: "f32[16384]" = torch.ops.aten.view.default(sum_116, _shape_param_465);  sum_116 = _shape_param_465 = None
        convert_element_type_402: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_429, torch.bfloat16);  view_429 = None
        view_430: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_184, _shape_param_466);  mm_184 = _shape_param_466 = None
        convert_element_type_403: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_430, torch.float32);  view_430 = None
        convert_element_type_404: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None
        convert_element_type_405: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_402, torch.float32);  convert_element_type_402 = None
        mm_186: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_151, arg323_1);  permute_151 = arg323_1 = None
        mm_187: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_424, arg324_1);  view_424 = arg324_1 = None
        view_431: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_187, _shape_param_467);  mm_187 = _shape_param_467 = None
        convert_element_type_406: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        view_432: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_431, _shape_param_468);  view_431 = _shape_param_468 = None
        permute_153: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        view_433: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_153, _shape_param_469);  permute_153 = _shape_param_469 = None
        bmm_60: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg325_1, view_433);  arg325_1 = None
        bmm_61: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_433, arg326_1);  view_433 = arg326_1 = None
        view_434: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_60, _shape_param_470);  bmm_60 = _shape_param_470 = None
        view_435: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_61, _shape_param_471);  bmm_61 = _shape_param_471 = None
        convert_element_type_407: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_435, torch.float32);  view_435 = None
        mul_367: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_407, arg327_1);  convert_element_type_407 = None
        sum_117: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [-1], True)
        neg_45: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg327_1);  arg327_1 = None
        fma_15: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_45, sum_117, mul_367);  neg_45 = sum_117 = mul_367 = None
        convert_element_type_408: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None
        div_18: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_408, 16.0);  convert_element_type_408 = None
        view_436: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_18, _shape_param_472);  div_18 = _shape_param_472 = None
        bmm_62: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg328_1, view_436);  arg328_1 = None
        bmm_63: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_436, arg329_1);  view_436 = arg329_1 = None
        view_437: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_62, _shape_param_473);  bmm_62 = _shape_param_473 = None
        view_438: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_63, _shape_param_474);  bmm_63 = _shape_param_474 = None
        convert_element_type_409: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        permute_154: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_409, [0, 1, 3, 2]);  convert_element_type_409 = None
        convert_element_type_410: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_154, torch.bfloat16);  permute_154 = None
        convert_element_type_411: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_438, torch.bfloat16);  view_438 = None
        permute_155: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_411, [0, 2, 1, 3]);  convert_element_type_411 = None
        permute_156: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_410, [0, 2, 1, 3]);  convert_element_type_410 = None
        slice_61: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_155, 3, 0, 64)
        slice_62: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_155, 3, 64, 256);  permute_155 = None
        slice_63: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_156, 3, 0, 64)
        slice_64: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_156, 3, 64, 256);  permute_156 = None
        expand_32: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg330_1, _shape_param_475);  arg330_1 = _shape_param_475 = None
        clone_45: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_439: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_45, _shape_param_476);  clone_45 = _shape_param_476 = None
        mul_368: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_61, view_439)
        view_440: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_368, _shape_param_477);  mul_368 = _shape_param_477 = None
        select_60: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_440, -1, 0)
        select_61: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_440, -1, 1);  view_440 = None
        neg_46: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_60);  select_60 = None
        slice_scatter_120: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_46, 3, 1, 9223372036854775807, 2);  neg_46 = None
        slice_scatter_121: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_61, 3, 0, 9223372036854775807, 2);  select_61 = None
        add_216: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_120, slice_scatter_121);  slice_scatter_120 = slice_scatter_121 = None
        expand_33: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg331_1, _shape_param_478);  arg331_1 = _shape_param_478 = None
        clone_46: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_441: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_479);  clone_46 = _shape_param_479 = None
        mul_369: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_61, view_441);  slice_61 = None
        add_217: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_216, mul_369);  add_216 = mul_369 = None
        mul_370: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_63, view_439);  view_439 = None
        view_442: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_370, _shape_param_480);  mul_370 = _shape_param_480 = None
        select_62: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_442, -1, 0)
        select_63: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_442, -1, 1);  view_442 = None
        neg_47: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_62);  select_62 = None
        slice_scatter_122: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_47, 3, 1, 9223372036854775807, 2);  neg_47 = None
        slice_scatter_123: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_63, 3, 0, 9223372036854775807, 2);  select_63 = None
        add_218: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_122, slice_scatter_123);  slice_scatter_122 = slice_scatter_123 = None
        mul_371: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_63, view_441);  slice_63 = view_441 = None
        add_219: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_218, mul_371);  add_218 = mul_371 = None
        slice_scatter_124: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_62, 3, 64, 9223372036854775807);  slice_62 = None
        slice_scatter_125: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_217, 3, 0, 64);  add_217 = None
        add_220: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_124, slice_scatter_125);  slice_scatter_124 = slice_scatter_125 = None
        slice_scatter_126: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_64, 3, 64, 9223372036854775807);  slice_64 = None
        slice_scatter_127: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_219, 3, 0, 64);  add_219 = None
        add_221: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_126, slice_scatter_127);  slice_scatter_126 = slice_scatter_127 = None
        permute_157: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
        clone_47: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_443: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_47, _shape_param_481);  clone_47 = _shape_param_481 = None
        view_444: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_221, _shape_param_482);  add_221 = _shape_param_482 = None
        view_445: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_220, _shape_param_483);  add_220 = _shape_param_483 = None
        view_446: "bf16[128, 4096]" = torch.ops.aten.view.default(view_443, _shape_param_484);  view_443 = _shape_param_484 = None
        permute_158: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_446, [1, 0])
        mm_188: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_158, arg322_1);  permute_158 = None
        mm_189: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_446, arg332_1);  view_446 = arg332_1 = None
        view_447: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_189, _shape_param_485);  mm_189 = _shape_param_485 = None
        convert_element_type_412: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_447, torch.float32);  view_447 = None
        add_222: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_403, convert_element_type_412);  convert_element_type_403 = convert_element_type_412 = None
        convert_element_type_413: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        view_448: "bf16[128, 4096]" = torch.ops.aten.view.default(view_444, _shape_param_486);  view_444 = _shape_param_486 = None
        permute_159: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_190: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_159, arg322_1);  permute_159 = None
        mm_191: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_448, arg333_1);  view_448 = arg333_1 = None
        view_449: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_191, _shape_param_487);  mm_191 = _shape_param_487 = None
        convert_element_type_414: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_449, torch.float32);  view_449 = None
        add_223: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_222, convert_element_type_414);  add_222 = convert_element_type_414 = None
        convert_element_type_415: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        view_450: "bf16[128, 4096]" = torch.ops.aten.view.default(view_445, _shape_param_488);  view_445 = _shape_param_488 = None
        permute_160: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_450, [1, 0])
        mm_192: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_160, arg322_1);  permute_160 = arg322_1 = None
        mm_193: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_450, arg334_1);  view_450 = arg334_1 = None
        view_451: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_193, _shape_param_489);  mm_193 = _shape_param_489 = None
        convert_element_type_416: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_451, torch.float32);  view_451 = None
        add_224: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_223, convert_element_type_416);  add_223 = convert_element_type_416 = None
        convert_element_type_417: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        mul_372: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_224, arg335_1);  arg335_1 = None
        mul_373: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_372, 4096)
        sum_118: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True)
        mul_374: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_372, arg336_1);  mul_372 = None
        sum_119: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True);  mul_374 = None
        mul_375: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg336_1, sum_119);  sum_119 = None
        sub_54: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_373, sum_118);  mul_373 = sum_118 = None
        sub_55: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_54, mul_375);  sub_54 = mul_375 = None
        mul_376: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg337_1, sub_55);  arg337_1 = sub_55 = None
        mul_377: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_224, arg336_1);  arg336_1 = None
        sum_120: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 1]);  mul_377 = None
        sum_121: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None
        add_225: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_211, mul_376);  add_211 = mul_376 = None
        convert_element_type_418: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_225, torch.bfloat16)
        view_452: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_418, _shape_param_490);  convert_element_type_418 = _shape_param_490 = None
        mm_194: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_452, arg338_1);  arg338_1 = None
        permute_161: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_452, [1, 0])
        mm_195: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_161, arg339_1);  arg339_1 = None
        sum_122: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_452, [0], True, dtype = torch.float32)
        view_453: "f32[4096]" = torch.ops.aten.view.default(sum_122, _shape_param_491);  sum_122 = _shape_param_491 = None
        convert_element_type_419: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_453, torch.bfloat16);  view_453 = None
        view_454: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_194, _shape_param_492);  mm_194 = _shape_param_492 = None
        convert_element_type_420: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_454, torch.float32);  view_454 = None
        convert_element_type_421: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        convert_element_type_422: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_419, torch.float32);  convert_element_type_419 = None
        view_455: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg340_1, _shape_param_493);  arg340_1 = _shape_param_493 = None
        mul_378: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_455, 0.5)
        mul_379: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_420, mul_378);  mul_378 = None
        convert_element_type_423: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_455, torch.float32)
        pow_33: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_423, 3.0)
        mul_380: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_226: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_455, mul_380);  view_455 = mul_380 = None
        mul_381: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_226, 0.7978845608028654);  add_226 = None
        tanh_16: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_381);  mul_381 = None
        add_227: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_382: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_420, add_227);  convert_element_type_420 = add_227 = None
        convert_element_type_424: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_382, torch.bfloat16);  mul_382 = None
        mul_383: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_56: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_383);  mul_383 = None
        mul_384: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_379, sub_56);  mul_379 = sub_56 = None
        mul_385: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_384, 0.7978845608028654);  mul_384 = None
        convert_element_type_425: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_385, torch.bfloat16)
        mul_386: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_385, 0.044715);  mul_385 = None
        pow_34: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_423, 2.0);  convert_element_type_423 = None
        mul_387: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
        mul_388: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_386, mul_387);  mul_386 = mul_387 = None
        convert_element_type_426: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_388, torch.bfloat16);  mul_388 = None
        add_228: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_425, convert_element_type_426);  convert_element_type_425 = convert_element_type_426 = None
        mul_389: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_424, 0.5);  convert_element_type_424 = None
        add_229: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_228, mul_389);  add_228 = mul_389 = None
        view_456: "bf16[128, 16384]" = torch.ops.aten.view.default(add_229, _shape_param_494);  add_229 = _shape_param_494 = None
        mm_196: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_456, arg341_1);  arg341_1 = None
        permute_162: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_456, [1, 0])
        mm_197: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_162, arg342_1);  permute_162 = None
        sum_123: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_456, [0], True, dtype = torch.float32);  view_456 = None
        view_457: "f32[16384]" = torch.ops.aten.view.default(sum_123, _shape_param_495);  sum_123 = _shape_param_495 = None
        convert_element_type_427: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_457, torch.bfloat16);  view_457 = None
        view_458: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_196, _shape_param_496);  mm_196 = _shape_param_496 = None
        convert_element_type_428: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_458, torch.float32);  view_458 = None
        convert_element_type_429: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None
        convert_element_type_430: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_427, torch.float32);  convert_element_type_427 = None
        mm_198: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_161, arg343_1);  permute_161 = arg343_1 = None
        mm_199: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_452, arg344_1);  view_452 = arg344_1 = None
        view_459: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_199, _shape_param_497);  mm_199 = _shape_param_497 = None
        convert_element_type_431: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_198, torch.float32);  mm_198 = None
        view_460: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_459, _shape_param_498);  view_459 = _shape_param_498 = None
        permute_163: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
        view_461: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_163, _shape_param_499);  permute_163 = _shape_param_499 = None
        bmm_64: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg345_1, view_461);  arg345_1 = None
        bmm_65: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_461, arg346_1);  view_461 = arg346_1 = None
        view_462: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_64, _shape_param_500);  bmm_64 = _shape_param_500 = None
        view_463: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_65, _shape_param_501);  bmm_65 = _shape_param_501 = None
        convert_element_type_432: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_463, torch.float32);  view_463 = None
        mul_390: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_432, arg347_1);  convert_element_type_432 = None
        sum_124: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [-1], True)
        neg_48: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg347_1);  arg347_1 = None
        fma_16: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_48, sum_124, mul_390);  neg_48 = sum_124 = mul_390 = None
        convert_element_type_433: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None
        div_19: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_433, 16.0);  convert_element_type_433 = None
        view_464: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_19, _shape_param_502);  div_19 = _shape_param_502 = None
        bmm_66: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg348_1, view_464);  arg348_1 = None
        bmm_67: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_464, arg349_1);  view_464 = arg349_1 = None
        view_465: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_66, _shape_param_503);  bmm_66 = _shape_param_503 = None
        view_466: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_67, _shape_param_504);  bmm_67 = _shape_param_504 = None
        convert_element_type_434: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_465, torch.float32);  view_465 = None
        permute_164: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_434, [0, 1, 3, 2]);  convert_element_type_434 = None
        convert_element_type_435: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_164, torch.bfloat16);  permute_164 = None
        convert_element_type_436: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_466, torch.bfloat16);  view_466 = None
        permute_165: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_436, [0, 2, 1, 3]);  convert_element_type_436 = None
        permute_166: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_435, [0, 2, 1, 3]);  convert_element_type_435 = None
        slice_65: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_165, 3, 0, 64)
        slice_66: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_165, 3, 64, 256);  permute_165 = None
        slice_67: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_166, 3, 0, 64)
        slice_68: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_166, 3, 64, 256);  permute_166 = None
        expand_34: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg350_1, _shape_param_505);  arg350_1 = _shape_param_505 = None
        clone_48: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_467: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_506);  clone_48 = _shape_param_506 = None
        mul_391: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_65, view_467)
        view_468: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_391, _shape_param_507);  mul_391 = _shape_param_507 = None
        select_64: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_468, -1, 0)
        select_65: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_468, -1, 1);  view_468 = None
        neg_49: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_64);  select_64 = None
        slice_scatter_128: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_49, 3, 1, 9223372036854775807, 2);  neg_49 = None
        slice_scatter_129: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_65, 3, 0, 9223372036854775807, 2);  select_65 = None
        add_230: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_128, slice_scatter_129);  slice_scatter_128 = slice_scatter_129 = None
        expand_35: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg351_1, _shape_param_508);  arg351_1 = _shape_param_508 = None
        clone_49: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_469: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_49, _shape_param_509);  clone_49 = _shape_param_509 = None
        mul_392: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_65, view_469);  slice_65 = None
        add_231: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_230, mul_392);  add_230 = mul_392 = None
        mul_393: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_67, view_467);  view_467 = None
        view_470: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_393, _shape_param_510);  mul_393 = _shape_param_510 = None
        select_66: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_470, -1, 0)
        select_67: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_470, -1, 1);  view_470 = None
        neg_50: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_66);  select_66 = None
        slice_scatter_130: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_50, 3, 1, 9223372036854775807, 2);  neg_50 = None
        slice_scatter_131: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_67, 3, 0, 9223372036854775807, 2);  select_67 = None
        add_232: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_130, slice_scatter_131);  slice_scatter_130 = slice_scatter_131 = None
        mul_394: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_67, view_469);  slice_67 = view_469 = None
        add_233: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_232, mul_394);  add_232 = mul_394 = None
        slice_scatter_132: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_66, 3, 64, 9223372036854775807);  slice_66 = None
        slice_scatter_133: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_231, 3, 0, 64);  add_231 = None
        add_234: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_132, slice_scatter_133);  slice_scatter_132 = slice_scatter_133 = None
        slice_scatter_134: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_68, 3, 64, 9223372036854775807);  slice_68 = None
        slice_scatter_135: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_233, 3, 0, 64);  add_233 = None
        add_235: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_134, slice_scatter_135);  slice_scatter_134 = slice_scatter_135 = None
        permute_167: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
        clone_50: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        view_471: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_50, _shape_param_511);  clone_50 = _shape_param_511 = None
        view_472: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_235, _shape_param_512);  add_235 = _shape_param_512 = None
        view_473: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_234, _shape_param_513);  add_234 = _shape_param_513 = None
        view_474: "bf16[128, 4096]" = torch.ops.aten.view.default(view_471, _shape_param_514);  view_471 = _shape_param_514 = None
        permute_168: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_474, [1, 0])
        mm_200: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_168, arg342_1);  permute_168 = None
        mm_201: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_474, arg352_1);  view_474 = arg352_1 = None
        view_475: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_201, _shape_param_515);  mm_201 = _shape_param_515 = None
        convert_element_type_437: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_475, torch.float32);  view_475 = None
        add_236: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_428, convert_element_type_437);  convert_element_type_428 = convert_element_type_437 = None
        convert_element_type_438: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        view_476: "bf16[128, 4096]" = torch.ops.aten.view.default(view_472, _shape_param_516);  view_472 = _shape_param_516 = None
        permute_169: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_476, [1, 0])
        mm_202: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_169, arg342_1);  permute_169 = None
        mm_203: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_476, arg353_1);  view_476 = arg353_1 = None
        view_477: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_203, _shape_param_517);  mm_203 = _shape_param_517 = None
        convert_element_type_439: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_477, torch.float32);  view_477 = None
        add_237: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_236, convert_element_type_439);  add_236 = convert_element_type_439 = None
        convert_element_type_440: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_202, torch.float32);  mm_202 = None
        view_478: "bf16[128, 4096]" = torch.ops.aten.view.default(view_473, _shape_param_518);  view_473 = _shape_param_518 = None
        permute_170: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_478, [1, 0])
        mm_204: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_170, arg342_1);  permute_170 = arg342_1 = None
        mm_205: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_478, arg354_1);  view_478 = arg354_1 = None
        view_479: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_205, _shape_param_519);  mm_205 = _shape_param_519 = None
        convert_element_type_441: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_479, torch.float32);  view_479 = None
        add_238: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_237, convert_element_type_441);  add_237 = convert_element_type_441 = None
        convert_element_type_442: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_204, torch.float32);  mm_204 = None
        mul_395: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_238, arg355_1);  arg355_1 = None
        mul_396: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_395, 4096)
        sum_125: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
        mul_397: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_395, arg356_1);  mul_395 = None
        sum_126: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
        mul_398: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg356_1, sum_126);  sum_126 = None
        sub_57: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_396, sum_125);  mul_396 = sum_125 = None
        sub_58: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_57, mul_398);  sub_57 = mul_398 = None
        mul_399: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg357_1, sub_58);  arg357_1 = sub_58 = None
        mul_400: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_238, arg356_1);  arg356_1 = None
        sum_127: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
        sum_128: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_238, [0, 1]);  add_238 = None
        add_239: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_225, mul_399);  add_225 = mul_399 = None
        convert_element_type_443: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_239, torch.bfloat16)
        view_480: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_443, _shape_param_520);  convert_element_type_443 = _shape_param_520 = None
        mm_206: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_480, arg358_1);  arg358_1 = None
        permute_171: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_480, [1, 0])
        mm_207: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_171, arg359_1);  arg359_1 = None
        sum_129: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_480, [0], True, dtype = torch.float32)
        view_481: "f32[4096]" = torch.ops.aten.view.default(sum_129, _shape_param_521);  sum_129 = _shape_param_521 = None
        convert_element_type_444: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_481, torch.bfloat16);  view_481 = None
        view_482: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_206, _shape_param_522);  mm_206 = _shape_param_522 = None
        convert_element_type_445: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_482, torch.float32);  view_482 = None
        convert_element_type_446: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        convert_element_type_447: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_444, torch.float32);  convert_element_type_444 = None
        view_483: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg360_1, _shape_param_523);  arg360_1 = _shape_param_523 = None
        mul_401: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_483, 0.5)
        mul_402: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_445, mul_401);  mul_401 = None
        convert_element_type_448: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_483, torch.float32)
        pow_35: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_448, 3.0)
        mul_403: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_35, 0.044715);  pow_35 = None
        add_240: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_483, mul_403);  view_483 = mul_403 = None
        mul_404: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_240, 0.7978845608028654);  add_240 = None
        tanh_17: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_404);  mul_404 = None
        add_241: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_405: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_445, add_241);  convert_element_type_445 = add_241 = None
        convert_element_type_449: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_405, torch.bfloat16);  mul_405 = None
        mul_406: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_59: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_406);  mul_406 = None
        mul_407: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_402, sub_59);  mul_402 = sub_59 = None
        mul_408: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_407, 0.7978845608028654);  mul_407 = None
        convert_element_type_450: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_408, torch.bfloat16)
        mul_409: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_408, 0.044715);  mul_408 = None
        pow_36: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_448, 2.0);  convert_element_type_448 = None
        mul_410: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_411: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_409, mul_410);  mul_409 = mul_410 = None
        convert_element_type_451: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_411, torch.bfloat16);  mul_411 = None
        add_242: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_450, convert_element_type_451);  convert_element_type_450 = convert_element_type_451 = None
        mul_412: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_449, 0.5);  convert_element_type_449 = None
        add_243: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_242, mul_412);  add_242 = mul_412 = None
        view_484: "bf16[128, 16384]" = torch.ops.aten.view.default(add_243, _shape_param_524);  add_243 = _shape_param_524 = None
        mm_208: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_484, arg361_1);  arg361_1 = None
        permute_172: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_484, [1, 0])
        mm_209: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_172, arg362_1);  permute_172 = None
        sum_130: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True, dtype = torch.float32);  view_484 = None
        view_485: "f32[16384]" = torch.ops.aten.view.default(sum_130, _shape_param_525);  sum_130 = _shape_param_525 = None
        convert_element_type_452: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_485, torch.bfloat16);  view_485 = None
        view_486: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_208, _shape_param_526);  mm_208 = _shape_param_526 = None
        convert_element_type_453: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_486, torch.float32);  view_486 = None
        convert_element_type_454: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_209, torch.float32);  mm_209 = None
        convert_element_type_455: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_452, torch.float32);  convert_element_type_452 = None
        mm_210: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_171, arg363_1);  permute_171 = arg363_1 = None
        mm_211: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_480, arg364_1);  view_480 = arg364_1 = None
        view_487: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_211, _shape_param_527);  mm_211 = _shape_param_527 = None
        convert_element_type_456: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_210, torch.float32);  mm_210 = None
        view_488: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_487, _shape_param_528);  view_487 = _shape_param_528 = None
        permute_173: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
        view_489: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_173, _shape_param_529);  permute_173 = _shape_param_529 = None
        bmm_68: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg365_1, view_489);  arg365_1 = None
        bmm_69: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_489, arg366_1);  view_489 = arg366_1 = None
        view_490: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_68, _shape_param_530);  bmm_68 = _shape_param_530 = None
        view_491: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_69, _shape_param_531);  bmm_69 = _shape_param_531 = None
        convert_element_type_457: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_491, torch.float32);  view_491 = None
        mul_413: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_457, arg367_1);  convert_element_type_457 = None
        sum_131: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [-1], True)
        neg_51: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg367_1);  arg367_1 = None
        fma_17: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_51, sum_131, mul_413);  neg_51 = sum_131 = mul_413 = None
        convert_element_type_458: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None
        div_20: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_458, 16.0);  convert_element_type_458 = None
        view_492: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_20, _shape_param_532);  div_20 = _shape_param_532 = None
        bmm_70: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg368_1, view_492);  arg368_1 = None
        bmm_71: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_492, arg369_1);  view_492 = arg369_1 = None
        view_493: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_70, _shape_param_533);  bmm_70 = _shape_param_533 = None
        view_494: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_71, _shape_param_534);  bmm_71 = _shape_param_534 = None
        convert_element_type_459: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_493, torch.float32);  view_493 = None
        permute_174: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_459, [0, 1, 3, 2]);  convert_element_type_459 = None
        convert_element_type_460: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_174, torch.bfloat16);  permute_174 = None
        convert_element_type_461: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_494, torch.bfloat16);  view_494 = None
        permute_175: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_461, [0, 2, 1, 3]);  convert_element_type_461 = None
        permute_176: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_460, [0, 2, 1, 3]);  convert_element_type_460 = None
        slice_69: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_175, 3, 0, 64)
        slice_70: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_175, 3, 64, 256);  permute_175 = None
        slice_71: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_176, 3, 0, 64)
        slice_72: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_176, 3, 64, 256);  permute_176 = None
        expand_36: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg370_1, _shape_param_535);  arg370_1 = _shape_param_535 = None
        clone_51: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_495: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_51, _shape_param_536);  clone_51 = _shape_param_536 = None
        mul_414: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_69, view_495)
        view_496: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_414, _shape_param_537);  mul_414 = _shape_param_537 = None
        select_68: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_496, -1, 0)
        select_69: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_496, -1, 1);  view_496 = None
        neg_52: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_68);  select_68 = None
        slice_scatter_136: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_52, 3, 1, 9223372036854775807, 2);  neg_52 = None
        slice_scatter_137: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_69, 3, 0, 9223372036854775807, 2);  select_69 = None
        add_244: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_136, slice_scatter_137);  slice_scatter_136 = slice_scatter_137 = None
        expand_37: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg371_1, _shape_param_538);  arg371_1 = _shape_param_538 = None
        clone_52: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_497: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_539);  clone_52 = _shape_param_539 = None
        mul_415: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_69, view_497);  slice_69 = None
        add_245: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_244, mul_415);  add_244 = mul_415 = None
        mul_416: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_71, view_495);  view_495 = None
        view_498: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_416, _shape_param_540);  mul_416 = _shape_param_540 = None
        select_70: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_498, -1, 0)
        select_71: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_498, -1, 1);  view_498 = None
        neg_53: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_70);  select_70 = None
        slice_scatter_138: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_53, 3, 1, 9223372036854775807, 2);  neg_53 = None
        slice_scatter_139: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_71, 3, 0, 9223372036854775807, 2);  select_71 = None
        add_246: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_138, slice_scatter_139);  slice_scatter_138 = slice_scatter_139 = None
        mul_417: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_71, view_497);  slice_71 = view_497 = None
        add_247: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_246, mul_417);  add_246 = mul_417 = None
        slice_scatter_140: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_70, 3, 64, 9223372036854775807);  slice_70 = None
        slice_scatter_141: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_245, 3, 0, 64);  add_245 = None
        add_248: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_140, slice_scatter_141);  slice_scatter_140 = slice_scatter_141 = None
        slice_scatter_142: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_72, 3, 64, 9223372036854775807);  slice_72 = None
        slice_scatter_143: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_247, 3, 0, 64);  add_247 = None
        add_249: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_142, slice_scatter_143);  slice_scatter_142 = slice_scatter_143 = None
        permute_177: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
        clone_53: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_499: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_53, _shape_param_541);  clone_53 = _shape_param_541 = None
        view_500: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_249, _shape_param_542);  add_249 = _shape_param_542 = None
        view_501: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_248, _shape_param_543);  add_248 = _shape_param_543 = None
        view_502: "bf16[128, 4096]" = torch.ops.aten.view.default(view_499, _shape_param_544);  view_499 = _shape_param_544 = None
        permute_178: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_502, [1, 0])
        mm_212: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_178, arg362_1);  permute_178 = None
        mm_213: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_502, arg372_1);  view_502 = arg372_1 = None
        view_503: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_213, _shape_param_545);  mm_213 = _shape_param_545 = None
        convert_element_type_462: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None
        add_250: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_453, convert_element_type_462);  convert_element_type_453 = convert_element_type_462 = None
        convert_element_type_463: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_212, torch.float32);  mm_212 = None
        view_504: "bf16[128, 4096]" = torch.ops.aten.view.default(view_500, _shape_param_546);  view_500 = _shape_param_546 = None
        permute_179: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_504, [1, 0])
        mm_214: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_179, arg362_1);  permute_179 = None
        mm_215: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_504, arg373_1);  view_504 = arg373_1 = None
        view_505: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_215, _shape_param_547);  mm_215 = _shape_param_547 = None
        convert_element_type_464: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_505, torch.float32);  view_505 = None
        add_251: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_250, convert_element_type_464);  add_250 = convert_element_type_464 = None
        convert_element_type_465: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_214, torch.float32);  mm_214 = None
        view_506: "bf16[128, 4096]" = torch.ops.aten.view.default(view_501, _shape_param_548);  view_501 = _shape_param_548 = None
        permute_180: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_506, [1, 0])
        mm_216: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_180, arg362_1);  permute_180 = arg362_1 = None
        mm_217: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_506, arg374_1);  view_506 = arg374_1 = None
        view_507: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_217, _shape_param_549);  mm_217 = _shape_param_549 = None
        convert_element_type_466: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_507, torch.float32);  view_507 = None
        add_252: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_251, convert_element_type_466);  add_251 = convert_element_type_466 = None
        convert_element_type_467: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_216, torch.float32);  mm_216 = None
        mul_418: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_252, arg375_1);  arg375_1 = None
        mul_419: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_418, 4096)
        sum_132: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True)
        mul_420: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_418, arg376_1);  mul_418 = None
        sum_133: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_420, [2], True);  mul_420 = None
        mul_421: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg376_1, sum_133);  sum_133 = None
        sub_60: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_419, sum_132);  mul_419 = sum_132 = None
        sub_61: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_60, mul_421);  sub_60 = mul_421 = None
        mul_422: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg377_1, sub_61);  arg377_1 = sub_61 = None
        mul_423: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_252, arg376_1);  arg376_1 = None
        sum_134: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_423, [0, 1]);  mul_423 = None
        sum_135: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_252, [0, 1]);  add_252 = None
        add_253: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_239, mul_422);  add_239 = mul_422 = None
        convert_element_type_468: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_253, torch.bfloat16)
        view_508: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_468, _shape_param_550);  convert_element_type_468 = _shape_param_550 = None
        mm_218: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_508, arg378_1);  arg378_1 = None
        permute_181: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_508, [1, 0])
        mm_219: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_181, arg379_1);  arg379_1 = None
        sum_136: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_508, [0], True, dtype = torch.float32)
        view_509: "f32[4096]" = torch.ops.aten.view.default(sum_136, _shape_param_551);  sum_136 = _shape_param_551 = None
        convert_element_type_469: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_509, torch.bfloat16);  view_509 = None
        view_510: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_218, _shape_param_552);  mm_218 = _shape_param_552 = None
        convert_element_type_470: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_510, torch.float32);  view_510 = None
        convert_element_type_471: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_219, torch.float32);  mm_219 = None
        convert_element_type_472: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_469, torch.float32);  convert_element_type_469 = None
        view_511: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg380_1, _shape_param_553);  arg380_1 = _shape_param_553 = None
        mul_424: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_511, 0.5)
        mul_425: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_470, mul_424);  mul_424 = None
        convert_element_type_473: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_511, torch.float32)
        pow_37: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_473, 3.0)
        mul_426: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_254: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_511, mul_426);  view_511 = mul_426 = None
        mul_427: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_254, 0.7978845608028654);  add_254 = None
        tanh_18: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_427);  mul_427 = None
        add_255: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_428: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_470, add_255);  convert_element_type_470 = add_255 = None
        convert_element_type_474: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_428, torch.bfloat16);  mul_428 = None
        mul_429: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_62: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_429);  mul_429 = None
        mul_430: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_425, sub_62);  mul_425 = sub_62 = None
        mul_431: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_430, 0.7978845608028654);  mul_430 = None
        convert_element_type_475: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_431, torch.bfloat16)
        mul_432: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_431, 0.044715);  mul_431 = None
        pow_38: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_473, 2.0);  convert_element_type_473 = None
        mul_433: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_434: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_432, mul_433);  mul_432 = mul_433 = None
        convert_element_type_476: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_434, torch.bfloat16);  mul_434 = None
        add_256: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_475, convert_element_type_476);  convert_element_type_475 = convert_element_type_476 = None
        mul_435: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_474, 0.5);  convert_element_type_474 = None
        add_257: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_256, mul_435);  add_256 = mul_435 = None
        view_512: "bf16[128, 16384]" = torch.ops.aten.view.default(add_257, _shape_param_554);  add_257 = _shape_param_554 = None
        mm_220: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_512, arg381_1);  arg381_1 = None
        permute_182: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_512, [1, 0])
        mm_221: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_182, arg382_1);  permute_182 = None
        sum_137: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_512, [0], True, dtype = torch.float32);  view_512 = None
        view_513: "f32[16384]" = torch.ops.aten.view.default(sum_137, _shape_param_555);  sum_137 = _shape_param_555 = None
        convert_element_type_477: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_513, torch.bfloat16);  view_513 = None
        view_514: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_220, _shape_param_556);  mm_220 = _shape_param_556 = None
        convert_element_type_478: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_514, torch.float32);  view_514 = None
        convert_element_type_479: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_221, torch.float32);  mm_221 = None
        convert_element_type_480: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_477, torch.float32);  convert_element_type_477 = None
        mm_222: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_181, arg383_1);  permute_181 = arg383_1 = None
        mm_223: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_508, arg384_1);  view_508 = arg384_1 = None
        view_515: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_223, _shape_param_557);  mm_223 = _shape_param_557 = None
        convert_element_type_481: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_222, torch.float32);  mm_222 = None
        view_516: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_515, _shape_param_558);  view_515 = _shape_param_558 = None
        permute_183: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
        view_517: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_183, _shape_param_559);  permute_183 = _shape_param_559 = None
        bmm_72: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg385_1, view_517);  arg385_1 = None
        bmm_73: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_517, arg386_1);  view_517 = arg386_1 = None
        view_518: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_72, _shape_param_560);  bmm_72 = _shape_param_560 = None
        view_519: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_73, _shape_param_561);  bmm_73 = _shape_param_561 = None
        convert_element_type_482: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_519, torch.float32);  view_519 = None
        mul_436: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_482, arg387_1);  convert_element_type_482 = None
        sum_138: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [-1], True)
        neg_54: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg387_1);  arg387_1 = None
        fma_18: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_54, sum_138, mul_436);  neg_54 = sum_138 = mul_436 = None
        convert_element_type_483: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None
        div_21: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_483, 16.0);  convert_element_type_483 = None
        view_520: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_21, _shape_param_562);  div_21 = _shape_param_562 = None
        bmm_74: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg388_1, view_520);  arg388_1 = None
        bmm_75: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_520, arg389_1);  view_520 = arg389_1 = None
        view_521: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_74, _shape_param_563);  bmm_74 = _shape_param_563 = None
        view_522: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_75, _shape_param_564);  bmm_75 = _shape_param_564 = None
        convert_element_type_484: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_521, torch.float32);  view_521 = None
        permute_184: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_484, [0, 1, 3, 2]);  convert_element_type_484 = None
        convert_element_type_485: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_184, torch.bfloat16);  permute_184 = None
        convert_element_type_486: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_522, torch.bfloat16);  view_522 = None
        permute_185: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_486, [0, 2, 1, 3]);  convert_element_type_486 = None
        permute_186: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_485, [0, 2, 1, 3]);  convert_element_type_485 = None
        slice_73: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_185, 3, 0, 64)
        slice_74: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_185, 3, 64, 256);  permute_185 = None
        slice_75: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_186, 3, 0, 64)
        slice_76: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_186, 3, 64, 256);  permute_186 = None
        expand_38: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg390_1, _shape_param_565);  arg390_1 = _shape_param_565 = None
        clone_54: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_523: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_54, _shape_param_566);  clone_54 = _shape_param_566 = None
        mul_437: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_73, view_523)
        view_524: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_437, _shape_param_567);  mul_437 = _shape_param_567 = None
        select_72: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_524, -1, 0)
        select_73: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_524, -1, 1);  view_524 = None
        neg_55: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_72);  select_72 = None
        slice_scatter_144: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_55, 3, 1, 9223372036854775807, 2);  neg_55 = None
        slice_scatter_145: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_73, 3, 0, 9223372036854775807, 2);  select_73 = None
        add_258: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_144, slice_scatter_145);  slice_scatter_144 = slice_scatter_145 = None
        expand_39: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg391_1, _shape_param_568);  arg391_1 = _shape_param_568 = None
        clone_55: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_525: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_55, _shape_param_569);  clone_55 = _shape_param_569 = None
        mul_438: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_73, view_525);  slice_73 = None
        add_259: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_258, mul_438);  add_258 = mul_438 = None
        mul_439: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_75, view_523);  view_523 = None
        view_526: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_439, _shape_param_570);  mul_439 = _shape_param_570 = None
        select_74: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_526, -1, 0)
        select_75: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_526, -1, 1);  view_526 = None
        neg_56: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_74);  select_74 = None
        slice_scatter_146: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_56, 3, 1, 9223372036854775807, 2);  neg_56 = None
        slice_scatter_147: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_75, 3, 0, 9223372036854775807, 2);  select_75 = None
        add_260: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_146, slice_scatter_147);  slice_scatter_146 = slice_scatter_147 = None
        mul_440: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_75, view_525);  slice_75 = view_525 = None
        add_261: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_260, mul_440);  add_260 = mul_440 = None
        slice_scatter_148: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_74, 3, 64, 9223372036854775807);  slice_74 = None
        slice_scatter_149: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_259, 3, 0, 64);  add_259 = None
        add_262: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_148, slice_scatter_149);  slice_scatter_148 = slice_scatter_149 = None
        slice_scatter_150: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_76, 3, 64, 9223372036854775807);  slice_76 = None
        slice_scatter_151: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_261, 3, 0, 64);  add_261 = None
        add_263: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_150, slice_scatter_151);  slice_scatter_150 = slice_scatter_151 = None
        permute_187: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_518, [0, 2, 1, 3]);  view_518 = None
        clone_56: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_527: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_56, _shape_param_571);  clone_56 = _shape_param_571 = None
        view_528: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_263, _shape_param_572);  add_263 = _shape_param_572 = None
        view_529: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_262, _shape_param_573);  add_262 = _shape_param_573 = None
        view_530: "bf16[128, 4096]" = torch.ops.aten.view.default(view_527, _shape_param_574);  view_527 = _shape_param_574 = None
        permute_188: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_530, [1, 0])
        mm_224: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_188, arg382_1);  permute_188 = None
        mm_225: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_530, arg392_1);  view_530 = arg392_1 = None
        view_531: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_225, _shape_param_575);  mm_225 = _shape_param_575 = None
        convert_element_type_487: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_531, torch.float32);  view_531 = None
        add_264: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_478, convert_element_type_487);  convert_element_type_478 = convert_element_type_487 = None
        convert_element_type_488: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_224, torch.float32);  mm_224 = None
        view_532: "bf16[128, 4096]" = torch.ops.aten.view.default(view_528, _shape_param_576);  view_528 = _shape_param_576 = None
        permute_189: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_532, [1, 0])
        mm_226: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_189, arg382_1);  permute_189 = None
        mm_227: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_532, arg393_1);  view_532 = arg393_1 = None
        view_533: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_227, _shape_param_577);  mm_227 = _shape_param_577 = None
        convert_element_type_489: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_533, torch.float32);  view_533 = None
        add_265: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_264, convert_element_type_489);  add_264 = convert_element_type_489 = None
        convert_element_type_490: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_226, torch.float32);  mm_226 = None
        view_534: "bf16[128, 4096]" = torch.ops.aten.view.default(view_529, _shape_param_578);  view_529 = _shape_param_578 = None
        permute_190: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_534, [1, 0])
        mm_228: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_190, arg382_1);  permute_190 = arg382_1 = None
        mm_229: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_534, arg394_1);  view_534 = arg394_1 = None
        view_535: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_229, _shape_param_579);  mm_229 = _shape_param_579 = None
        convert_element_type_491: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_535, torch.float32);  view_535 = None
        add_266: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_265, convert_element_type_491);  add_265 = convert_element_type_491 = None
        convert_element_type_492: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_228, torch.float32);  mm_228 = None
        mul_441: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_266, arg395_1);  arg395_1 = None
        mul_442: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_441, 4096)
        sum_139: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True)
        mul_443: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_441, arg396_1);  mul_441 = None
        sum_140: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True);  mul_443 = None
        mul_444: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg396_1, sum_140);  sum_140 = None
        sub_63: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_442, sum_139);  mul_442 = sum_139 = None
        sub_64: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_63, mul_444);  sub_63 = mul_444 = None
        mul_445: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg397_1, sub_64);  arg397_1 = sub_64 = None
        mul_446: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_266, arg396_1);  arg396_1 = None
        sum_141: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1]);  mul_446 = None
        sum_142: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
        add_267: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_253, mul_445);  add_253 = mul_445 = None
        convert_element_type_493: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_267, torch.bfloat16)
        view_536: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_493, _shape_param_580);  convert_element_type_493 = _shape_param_580 = None
        mm_230: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_536, arg398_1);  arg398_1 = None
        permute_191: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_536, [1, 0])
        mm_231: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_191, arg399_1);  arg399_1 = None
        sum_143: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_536, [0], True, dtype = torch.float32)
        view_537: "f32[4096]" = torch.ops.aten.view.default(sum_143, _shape_param_581);  sum_143 = _shape_param_581 = None
        convert_element_type_494: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_537, torch.bfloat16);  view_537 = None
        view_538: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_230, _shape_param_582);  mm_230 = _shape_param_582 = None
        convert_element_type_495: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_538, torch.float32);  view_538 = None
        convert_element_type_496: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_231, torch.float32);  mm_231 = None
        convert_element_type_497: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_494, torch.float32);  convert_element_type_494 = None
        view_539: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg400_1, _shape_param_583);  arg400_1 = _shape_param_583 = None
        mul_447: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_539, 0.5)
        mul_448: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_495, mul_447);  mul_447 = None
        convert_element_type_498: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_539, torch.float32)
        pow_39: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_498, 3.0)
        mul_449: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_39, 0.044715);  pow_39 = None
        add_268: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_539, mul_449);  view_539 = mul_449 = None
        mul_450: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_268, 0.7978845608028654);  add_268 = None
        tanh_19: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_450);  mul_450 = None
        add_269: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_451: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_495, add_269);  convert_element_type_495 = add_269 = None
        convert_element_type_499: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_451, torch.bfloat16);  mul_451 = None
        mul_452: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_65: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_452);  mul_452 = None
        mul_453: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_448, sub_65);  mul_448 = sub_65 = None
        mul_454: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_453, 0.7978845608028654);  mul_453 = None
        convert_element_type_500: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_454, torch.bfloat16)
        mul_455: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_454, 0.044715);  mul_454 = None
        pow_40: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_498, 2.0);  convert_element_type_498 = None
        mul_456: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_457: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_455, mul_456);  mul_455 = mul_456 = None
        convert_element_type_501: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_457, torch.bfloat16);  mul_457 = None
        add_270: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_500, convert_element_type_501);  convert_element_type_500 = convert_element_type_501 = None
        mul_458: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_499, 0.5);  convert_element_type_499 = None
        add_271: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_270, mul_458);  add_270 = mul_458 = None
        view_540: "bf16[128, 16384]" = torch.ops.aten.view.default(add_271, _shape_param_584);  add_271 = _shape_param_584 = None
        mm_232: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_540, arg401_1);  arg401_1 = None
        permute_192: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_540, [1, 0])
        mm_233: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_192, arg402_1);  permute_192 = None
        sum_144: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_540, [0], True, dtype = torch.float32);  view_540 = None
        view_541: "f32[16384]" = torch.ops.aten.view.default(sum_144, _shape_param_585);  sum_144 = _shape_param_585 = None
        convert_element_type_502: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_541, torch.bfloat16);  view_541 = None
        view_542: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_232, _shape_param_586);  mm_232 = _shape_param_586 = None
        convert_element_type_503: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_542, torch.float32);  view_542 = None
        convert_element_type_504: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_233, torch.float32);  mm_233 = None
        convert_element_type_505: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_502, torch.float32);  convert_element_type_502 = None
        mm_234: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_191, arg403_1);  permute_191 = arg403_1 = None
        mm_235: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_536, arg404_1);  view_536 = arg404_1 = None
        view_543: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_235, _shape_param_587);  mm_235 = _shape_param_587 = None
        convert_element_type_506: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_234, torch.float32);  mm_234 = None
        view_544: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_543, _shape_param_588);  view_543 = _shape_param_588 = None
        permute_193: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None
        view_545: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_193, _shape_param_589);  permute_193 = _shape_param_589 = None
        bmm_76: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg405_1, view_545);  arg405_1 = None
        bmm_77: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_545, arg406_1);  view_545 = arg406_1 = None
        view_546: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_76, _shape_param_590);  bmm_76 = _shape_param_590 = None
        view_547: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_77, _shape_param_591);  bmm_77 = _shape_param_591 = None
        convert_element_type_507: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_547, torch.float32);  view_547 = None
        mul_459: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_507, arg407_1);  convert_element_type_507 = None
        sum_145: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_459, [-1], True)
        neg_57: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg407_1);  arg407_1 = None
        fma_19: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_57, sum_145, mul_459);  neg_57 = sum_145 = mul_459 = None
        convert_element_type_508: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None
        div_22: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_508, 16.0);  convert_element_type_508 = None
        view_548: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_22, _shape_param_592);  div_22 = _shape_param_592 = None
        bmm_78: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg408_1, view_548);  arg408_1 = None
        bmm_79: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_548, arg409_1);  view_548 = arg409_1 = None
        view_549: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_78, _shape_param_593);  bmm_78 = _shape_param_593 = None
        view_550: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_79, _shape_param_594);  bmm_79 = _shape_param_594 = None
        convert_element_type_509: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_549, torch.float32);  view_549 = None
        permute_194: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_509, [0, 1, 3, 2]);  convert_element_type_509 = None
        convert_element_type_510: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_194, torch.bfloat16);  permute_194 = None
        convert_element_type_511: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_550, torch.bfloat16);  view_550 = None
        permute_195: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_511, [0, 2, 1, 3]);  convert_element_type_511 = None
        permute_196: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_510, [0, 2, 1, 3]);  convert_element_type_510 = None
        slice_77: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_195, 3, 0, 64)
        slice_78: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_195, 3, 64, 256);  permute_195 = None
        slice_79: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_196, 3, 0, 64)
        slice_80: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_196, 3, 64, 256);  permute_196 = None
        expand_40: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg410_1, _shape_param_595);  arg410_1 = _shape_param_595 = None
        clone_57: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_551: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_57, _shape_param_596);  clone_57 = _shape_param_596 = None
        mul_460: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_77, view_551)
        view_552: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_460, _shape_param_597);  mul_460 = _shape_param_597 = None
        select_76: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_552, -1, 0)
        select_77: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_552, -1, 1);  view_552 = None
        neg_58: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_76);  select_76 = None
        slice_scatter_152: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_58, 3, 1, 9223372036854775807, 2);  neg_58 = None
        slice_scatter_153: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_77, 3, 0, 9223372036854775807, 2);  select_77 = None
        add_272: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_152, slice_scatter_153);  slice_scatter_152 = slice_scatter_153 = None
        expand_41: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg411_1, _shape_param_598);  arg411_1 = _shape_param_598 = None
        clone_58: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_553: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_58, _shape_param_599);  clone_58 = _shape_param_599 = None
        mul_461: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_77, view_553);  slice_77 = None
        add_273: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_272, mul_461);  add_272 = mul_461 = None
        mul_462: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_79, view_551);  view_551 = None
        view_554: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_462, _shape_param_600);  mul_462 = _shape_param_600 = None
        select_78: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_554, -1, 0)
        select_79: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_554, -1, 1);  view_554 = None
        neg_59: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_78);  select_78 = None
        slice_scatter_154: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_59, 3, 1, 9223372036854775807, 2);  neg_59 = None
        slice_scatter_155: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_79, 3, 0, 9223372036854775807, 2);  select_79 = None
        add_274: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_154, slice_scatter_155);  slice_scatter_154 = slice_scatter_155 = None
        mul_463: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_79, view_553);  slice_79 = view_553 = None
        add_275: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_274, mul_463);  add_274 = mul_463 = None
        slice_scatter_156: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_78, 3, 64, 9223372036854775807);  slice_78 = None
        slice_scatter_157: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_273, 3, 0, 64);  add_273 = None
        add_276: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_156, slice_scatter_157);  slice_scatter_156 = slice_scatter_157 = None
        slice_scatter_158: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_80, 3, 64, 9223372036854775807);  slice_80 = None
        slice_scatter_159: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_275, 3, 0, 64);  add_275 = None
        add_277: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_158, slice_scatter_159);  slice_scatter_158 = slice_scatter_159 = None
        permute_197: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
        clone_59: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
        view_555: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_59, _shape_param_601);  clone_59 = _shape_param_601 = None
        view_556: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_277, _shape_param_602);  add_277 = _shape_param_602 = None
        view_557: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_276, _shape_param_603);  add_276 = _shape_param_603 = None
        view_558: "bf16[128, 4096]" = torch.ops.aten.view.default(view_555, _shape_param_604);  view_555 = _shape_param_604 = None
        permute_198: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_558, [1, 0])
        mm_236: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_198, arg402_1);  permute_198 = None
        mm_237: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_558, arg412_1);  view_558 = arg412_1 = None
        view_559: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_237, _shape_param_605);  mm_237 = _shape_param_605 = None
        convert_element_type_512: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_559, torch.float32);  view_559 = None
        add_278: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_503, convert_element_type_512);  convert_element_type_503 = convert_element_type_512 = None
        convert_element_type_513: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_236, torch.float32);  mm_236 = None
        view_560: "bf16[128, 4096]" = torch.ops.aten.view.default(view_556, _shape_param_606);  view_556 = _shape_param_606 = None
        permute_199: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_560, [1, 0])
        mm_238: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_199, arg402_1);  permute_199 = None
        mm_239: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_560, arg413_1);  view_560 = arg413_1 = None
        view_561: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_239, _shape_param_607);  mm_239 = _shape_param_607 = None
        convert_element_type_514: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_561, torch.float32);  view_561 = None
        add_279: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_278, convert_element_type_514);  add_278 = convert_element_type_514 = None
        convert_element_type_515: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_238, torch.float32);  mm_238 = None
        view_562: "bf16[128, 4096]" = torch.ops.aten.view.default(view_557, _shape_param_608);  view_557 = _shape_param_608 = None
        permute_200: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_240: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_200, arg402_1);  permute_200 = arg402_1 = None
        mm_241: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_562, arg414_1);  view_562 = arg414_1 = None
        view_563: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_241, _shape_param_609);  mm_241 = _shape_param_609 = None
        convert_element_type_516: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_563, torch.float32);  view_563 = None
        add_280: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_279, convert_element_type_516);  add_279 = convert_element_type_516 = None
        convert_element_type_517: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_240, torch.float32);  mm_240 = None
        mul_464: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_280, arg415_1);  arg415_1 = None
        mul_465: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_464, 4096)
        sum_146: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True)
        mul_466: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_464, arg416_1);  mul_464 = None
        sum_147: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True);  mul_466 = None
        mul_467: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg416_1, sum_147);  sum_147 = None
        sub_66: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_465, sum_146);  mul_465 = sum_146 = None
        sub_67: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_66, mul_467);  sub_66 = mul_467 = None
        mul_468: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg417_1, sub_67);  arg417_1 = sub_67 = None
        mul_469: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_280, arg416_1);  arg416_1 = None
        sum_148: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1]);  mul_469 = None
        sum_149: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_280, [0, 1]);  add_280 = None
        add_281: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_267, mul_468);  add_267 = mul_468 = None
        convert_element_type_518: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_281, torch.bfloat16)
        view_564: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_518, _shape_param_610);  convert_element_type_518 = _shape_param_610 = None
        mm_242: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_564, arg418_1);  arg418_1 = None
        permute_201: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_564, [1, 0])
        mm_243: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_201, arg419_1);  arg419_1 = None
        sum_150: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True, dtype = torch.float32)
        view_565: "f32[4096]" = torch.ops.aten.view.default(sum_150, _shape_param_611);  sum_150 = _shape_param_611 = None
        convert_element_type_519: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_565, torch.bfloat16);  view_565 = None
        view_566: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_242, _shape_param_612);  mm_242 = _shape_param_612 = None
        convert_element_type_520: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_566, torch.float32);  view_566 = None
        convert_element_type_521: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_243, torch.float32);  mm_243 = None
        convert_element_type_522: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_519, torch.float32);  convert_element_type_519 = None
        view_567: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg420_1, _shape_param_613);  arg420_1 = _shape_param_613 = None
        mul_470: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_567, 0.5)
        mul_471: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_520, mul_470);  mul_470 = None
        convert_element_type_523: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_567, torch.float32)
        pow_41: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_523, 3.0)
        mul_472: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_282: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_567, mul_472);  view_567 = mul_472 = None
        mul_473: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_282, 0.7978845608028654);  add_282 = None
        tanh_20: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_473);  mul_473 = None
        add_283: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_474: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_520, add_283);  convert_element_type_520 = add_283 = None
        convert_element_type_524: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_474, torch.bfloat16);  mul_474 = None
        mul_475: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_68: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_475);  mul_475 = None
        mul_476: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_471, sub_68);  mul_471 = sub_68 = None
        mul_477: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_476, 0.7978845608028654);  mul_476 = None
        convert_element_type_525: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_477, torch.bfloat16)
        mul_478: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_477, 0.044715);  mul_477 = None
        pow_42: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_523, 2.0);  convert_element_type_523 = None
        mul_479: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_480: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
        convert_element_type_526: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_480, torch.bfloat16);  mul_480 = None
        add_284: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_525, convert_element_type_526);  convert_element_type_525 = convert_element_type_526 = None
        mul_481: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_524, 0.5);  convert_element_type_524 = None
        add_285: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_284, mul_481);  add_284 = mul_481 = None
        view_568: "bf16[128, 16384]" = torch.ops.aten.view.default(add_285, _shape_param_614);  add_285 = _shape_param_614 = None
        mm_244: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_568, arg421_1);  arg421_1 = None
        permute_202: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_568, [1, 0])
        mm_245: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_202, arg422_1);  permute_202 = None
        sum_151: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_568, [0], True, dtype = torch.float32);  view_568 = None
        view_569: "f32[16384]" = torch.ops.aten.view.default(sum_151, _shape_param_615);  sum_151 = _shape_param_615 = None
        convert_element_type_527: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_569, torch.bfloat16);  view_569 = None
        view_570: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_244, _shape_param_616);  mm_244 = _shape_param_616 = None
        convert_element_type_528: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_570, torch.float32);  view_570 = None
        convert_element_type_529: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_245, torch.float32);  mm_245 = None
        convert_element_type_530: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_527, torch.float32);  convert_element_type_527 = None
        mm_246: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_201, arg423_1);  permute_201 = arg423_1 = None
        mm_247: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_564, arg424_1);  view_564 = arg424_1 = None
        view_571: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_247, _shape_param_617);  mm_247 = _shape_param_617 = None
        convert_element_type_531: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_246, torch.float32);  mm_246 = None
        view_572: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_571, _shape_param_618);  view_571 = _shape_param_618 = None
        permute_203: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        view_573: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_203, _shape_param_619);  permute_203 = _shape_param_619 = None
        bmm_80: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg425_1, view_573);  arg425_1 = None
        bmm_81: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_573, arg426_1);  view_573 = arg426_1 = None
        view_574: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_80, _shape_param_620);  bmm_80 = _shape_param_620 = None
        view_575: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_81, _shape_param_621);  bmm_81 = _shape_param_621 = None
        convert_element_type_532: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_575, torch.float32);  view_575 = None
        mul_482: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_532, arg427_1);  convert_element_type_532 = None
        sum_152: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_482, [-1], True)
        neg_60: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg427_1);  arg427_1 = None
        fma_20: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_60, sum_152, mul_482);  neg_60 = sum_152 = mul_482 = None
        convert_element_type_533: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None
        div_23: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_533, 16.0);  convert_element_type_533 = None
        view_576: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_23, _shape_param_622);  div_23 = _shape_param_622 = None
        bmm_82: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg428_1, view_576);  arg428_1 = None
        bmm_83: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_576, arg429_1);  view_576 = arg429_1 = None
        view_577: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_82, _shape_param_623);  bmm_82 = _shape_param_623 = None
        view_578: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_83, _shape_param_624);  bmm_83 = _shape_param_624 = None
        convert_element_type_534: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_577, torch.float32);  view_577 = None
        permute_204: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_534, [0, 1, 3, 2]);  convert_element_type_534 = None
        convert_element_type_535: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_204, torch.bfloat16);  permute_204 = None
        convert_element_type_536: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_578, torch.bfloat16);  view_578 = None
        permute_205: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_536, [0, 2, 1, 3]);  convert_element_type_536 = None
        permute_206: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_535, [0, 2, 1, 3]);  convert_element_type_535 = None
        slice_81: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_205, 3, 0, 64)
        slice_82: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_205, 3, 64, 256);  permute_205 = None
        slice_83: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_206, 3, 0, 64)
        slice_84: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_206, 3, 64, 256);  permute_206 = None
        expand_42: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg430_1, _shape_param_625);  arg430_1 = _shape_param_625 = None
        clone_60: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_579: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_626);  clone_60 = _shape_param_626 = None
        mul_483: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_81, view_579)
        view_580: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_483, _shape_param_627);  mul_483 = _shape_param_627 = None
        select_80: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_580, -1, 0)
        select_81: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_580, -1, 1);  view_580 = None
        neg_61: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_80);  select_80 = None
        slice_scatter_160: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_61, 3, 1, 9223372036854775807, 2);  neg_61 = None
        slice_scatter_161: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_81, 3, 0, 9223372036854775807, 2);  select_81 = None
        add_286: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_160, slice_scatter_161);  slice_scatter_160 = slice_scatter_161 = None
        expand_43: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg431_1, _shape_param_628);  arg431_1 = _shape_param_628 = None
        clone_61: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_581: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_61, _shape_param_629);  clone_61 = _shape_param_629 = None
        mul_484: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_81, view_581);  slice_81 = None
        add_287: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_286, mul_484);  add_286 = mul_484 = None
        mul_485: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_83, view_579);  view_579 = None
        view_582: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_485, _shape_param_630);  mul_485 = _shape_param_630 = None
        select_82: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_582, -1, 0)
        select_83: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_582, -1, 1);  view_582 = None
        neg_62: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_82);  select_82 = None
        slice_scatter_162: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_62, 3, 1, 9223372036854775807, 2);  neg_62 = None
        slice_scatter_163: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_83, 3, 0, 9223372036854775807, 2);  select_83 = None
        add_288: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_162, slice_scatter_163);  slice_scatter_162 = slice_scatter_163 = None
        mul_486: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_83, view_581);  slice_83 = view_581 = None
        add_289: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_288, mul_486);  add_288 = mul_486 = None
        slice_scatter_164: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_82, 3, 64, 9223372036854775807);  slice_82 = None
        slice_scatter_165: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_287, 3, 0, 64);  add_287 = None
        add_290: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_164, slice_scatter_165);  slice_scatter_164 = slice_scatter_165 = None
        slice_scatter_166: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_84, 3, 64, 9223372036854775807);  slice_84 = None
        slice_scatter_167: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_289, 3, 0, 64);  add_289 = None
        add_291: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_166, slice_scatter_167);  slice_scatter_166 = slice_scatter_167 = None
        permute_207: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None
        clone_62: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
        view_583: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_62, _shape_param_631);  clone_62 = _shape_param_631 = None
        view_584: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_291, _shape_param_632);  add_291 = _shape_param_632 = None
        view_585: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_290, _shape_param_633);  add_290 = _shape_param_633 = None
        view_586: "bf16[128, 4096]" = torch.ops.aten.view.default(view_583, _shape_param_634);  view_583 = _shape_param_634 = None
        permute_208: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_586, [1, 0])
        mm_248: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_208, arg422_1);  permute_208 = None
        mm_249: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_586, arg432_1);  view_586 = arg432_1 = None
        view_587: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_249, _shape_param_635);  mm_249 = _shape_param_635 = None
        convert_element_type_537: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_587, torch.float32);  view_587 = None
        add_292: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_528, convert_element_type_537);  convert_element_type_528 = convert_element_type_537 = None
        convert_element_type_538: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_248, torch.float32);  mm_248 = None
        view_588: "bf16[128, 4096]" = torch.ops.aten.view.default(view_584, _shape_param_636);  view_584 = _shape_param_636 = None
        permute_209: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_588, [1, 0])
        mm_250: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_209, arg422_1);  permute_209 = None
        mm_251: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_588, arg433_1);  view_588 = arg433_1 = None
        view_589: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_251, _shape_param_637);  mm_251 = _shape_param_637 = None
        convert_element_type_539: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_589, torch.float32);  view_589 = None
        add_293: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_292, convert_element_type_539);  add_292 = convert_element_type_539 = None
        convert_element_type_540: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_250, torch.float32);  mm_250 = None
        view_590: "bf16[128, 4096]" = torch.ops.aten.view.default(view_585, _shape_param_638);  view_585 = _shape_param_638 = None
        permute_210: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_252: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_210, arg422_1);  permute_210 = arg422_1 = None
        mm_253: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_590, arg434_1);  view_590 = arg434_1 = None
        view_591: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_253, _shape_param_639);  mm_253 = _shape_param_639 = None
        convert_element_type_541: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_591, torch.float32);  view_591 = None
        add_294: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_293, convert_element_type_541);  add_293 = convert_element_type_541 = None
        convert_element_type_542: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_252, torch.float32);  mm_252 = None
        mul_487: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_294, arg435_1);  arg435_1 = None
        mul_488: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_487, 4096)
        sum_153: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True)
        mul_489: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_487, arg436_1);  mul_487 = None
        sum_154: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_489, [2], True);  mul_489 = None
        mul_490: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg436_1, sum_154);  sum_154 = None
        sub_69: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_488, sum_153);  mul_488 = sum_153 = None
        sub_70: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_69, mul_490);  sub_69 = mul_490 = None
        mul_491: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg437_1, sub_70);  arg437_1 = sub_70 = None
        mul_492: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_294, arg436_1);  arg436_1 = None
        sum_155: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_492, [0, 1]);  mul_492 = None
        sum_156: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_294, [0, 1]);  add_294 = None
        add_295: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_281, mul_491);  add_281 = mul_491 = None
        convert_element_type_543: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_295, torch.bfloat16)
        view_592: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_543, _shape_param_640);  convert_element_type_543 = _shape_param_640 = None
        mm_254: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_592, arg438_1);  arg438_1 = None
        permute_211: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_592, [1, 0])
        mm_255: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_211, arg439_1);  arg439_1 = None
        sum_157: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_592, [0], True, dtype = torch.float32)
        view_593: "f32[4096]" = torch.ops.aten.view.default(sum_157, _shape_param_641);  sum_157 = _shape_param_641 = None
        convert_element_type_544: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_593, torch.bfloat16);  view_593 = None
        view_594: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_254, _shape_param_642);  mm_254 = _shape_param_642 = None
        convert_element_type_545: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_594, torch.float32);  view_594 = None
        convert_element_type_546: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_255, torch.float32);  mm_255 = None
        convert_element_type_547: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_544, torch.float32);  convert_element_type_544 = None
        view_595: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg440_1, _shape_param_643);  arg440_1 = _shape_param_643 = None
        mul_493: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_595, 0.5)
        mul_494: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_545, mul_493);  mul_493 = None
        convert_element_type_548: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_595, torch.float32)
        pow_43: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_548, 3.0)
        mul_495: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_43, 0.044715);  pow_43 = None
        add_296: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_595, mul_495);  view_595 = mul_495 = None
        mul_496: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_296, 0.7978845608028654);  add_296 = None
        tanh_21: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_496);  mul_496 = None
        add_297: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_497: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_545, add_297);  convert_element_type_545 = add_297 = None
        convert_element_type_549: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_497, torch.bfloat16);  mul_497 = None
        mul_498: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_71: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_498);  mul_498 = None
        mul_499: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_494, sub_71);  mul_494 = sub_71 = None
        mul_500: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_499, 0.7978845608028654);  mul_499 = None
        convert_element_type_550: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_500, torch.bfloat16)
        mul_501: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_500, 0.044715);  mul_500 = None
        pow_44: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_548, 2.0);  convert_element_type_548 = None
        mul_502: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_503: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
        convert_element_type_551: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_503, torch.bfloat16);  mul_503 = None
        add_298: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_550, convert_element_type_551);  convert_element_type_550 = convert_element_type_551 = None
        mul_504: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_549, 0.5);  convert_element_type_549 = None
        add_299: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_298, mul_504);  add_298 = mul_504 = None
        view_596: "bf16[128, 16384]" = torch.ops.aten.view.default(add_299, _shape_param_644);  add_299 = _shape_param_644 = None
        mm_256: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_596, arg441_1);  arg441_1 = None
        permute_212: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_596, [1, 0])
        mm_257: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_212, arg442_1);  permute_212 = None
        sum_158: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_596, [0], True, dtype = torch.float32);  view_596 = None
        view_597: "f32[16384]" = torch.ops.aten.view.default(sum_158, _shape_param_645);  sum_158 = _shape_param_645 = None
        convert_element_type_552: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_597, torch.bfloat16);  view_597 = None
        view_598: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_256, _shape_param_646);  mm_256 = _shape_param_646 = None
        convert_element_type_553: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_598, torch.float32);  view_598 = None
        convert_element_type_554: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_257, torch.float32);  mm_257 = None
        convert_element_type_555: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_552, torch.float32);  convert_element_type_552 = None
        mm_258: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_211, arg443_1);  permute_211 = arg443_1 = None
        mm_259: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_592, arg444_1);  view_592 = arg444_1 = None
        view_599: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_259, _shape_param_647);  mm_259 = _shape_param_647 = None
        convert_element_type_556: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_258, torch.float32);  mm_258 = None
        view_600: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_599, _shape_param_648);  view_599 = _shape_param_648 = None
        permute_213: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_600, [0, 2, 1, 3]);  view_600 = None
        view_601: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_213, _shape_param_649);  permute_213 = _shape_param_649 = None
        bmm_84: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg445_1, view_601);  arg445_1 = None
        bmm_85: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_601, arg446_1);  view_601 = arg446_1 = None
        view_602: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_84, _shape_param_650);  bmm_84 = _shape_param_650 = None
        view_603: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_85, _shape_param_651);  bmm_85 = _shape_param_651 = None
        convert_element_type_557: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_603, torch.float32);  view_603 = None
        mul_505: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_557, arg447_1);  convert_element_type_557 = None
        sum_159: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_505, [-1], True)
        neg_63: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg447_1);  arg447_1 = None
        fma_21: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_63, sum_159, mul_505);  neg_63 = sum_159 = mul_505 = None
        convert_element_type_558: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None
        div_24: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_558, 16.0);  convert_element_type_558 = None
        view_604: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_24, _shape_param_652);  div_24 = _shape_param_652 = None
        bmm_86: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg448_1, view_604);  arg448_1 = None
        bmm_87: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_604, arg449_1);  view_604 = arg449_1 = None
        view_605: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_86, _shape_param_653);  bmm_86 = _shape_param_653 = None
        view_606: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_87, _shape_param_654);  bmm_87 = _shape_param_654 = None
        convert_element_type_559: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_605, torch.float32);  view_605 = None
        permute_214: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_559, [0, 1, 3, 2]);  convert_element_type_559 = None
        convert_element_type_560: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_214, torch.bfloat16);  permute_214 = None
        convert_element_type_561: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_606, torch.bfloat16);  view_606 = None
        permute_215: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_561, [0, 2, 1, 3]);  convert_element_type_561 = None
        permute_216: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_560, [0, 2, 1, 3]);  convert_element_type_560 = None
        slice_85: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_215, 3, 0, 64)
        slice_86: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_215, 3, 64, 256);  permute_215 = None
        slice_87: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_216, 3, 0, 64)
        slice_88: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_216, 3, 64, 256);  permute_216 = None
        expand_44: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg450_1, _shape_param_655);  arg450_1 = _shape_param_655 = None
        clone_63: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_607: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_63, _shape_param_656);  clone_63 = _shape_param_656 = None
        mul_506: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_85, view_607)
        view_608: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_506, _shape_param_657);  mul_506 = _shape_param_657 = None
        select_84: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_608, -1, 0)
        select_85: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_608, -1, 1);  view_608 = None
        neg_64: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_84);  select_84 = None
        slice_scatter_168: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_64, 3, 1, 9223372036854775807, 2);  neg_64 = None
        slice_scatter_169: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_85, 3, 0, 9223372036854775807, 2);  select_85 = None
        add_300: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_168, slice_scatter_169);  slice_scatter_168 = slice_scatter_169 = None
        expand_45: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg451_1, _shape_param_658);  arg451_1 = _shape_param_658 = None
        clone_64: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_609: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_659);  clone_64 = _shape_param_659 = None
        mul_507: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_85, view_609);  slice_85 = None
        add_301: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_300, mul_507);  add_300 = mul_507 = None
        mul_508: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_87, view_607);  view_607 = None
        view_610: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_508, _shape_param_660);  mul_508 = _shape_param_660 = None
        select_86: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_610, -1, 0)
        select_87: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_610, -1, 1);  view_610 = None
        neg_65: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_86);  select_86 = None
        slice_scatter_170: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_65, 3, 1, 9223372036854775807, 2);  neg_65 = None
        slice_scatter_171: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_87, 3, 0, 9223372036854775807, 2);  select_87 = None
        add_302: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_170, slice_scatter_171);  slice_scatter_170 = slice_scatter_171 = None
        mul_509: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_87, view_609);  slice_87 = view_609 = None
        add_303: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_302, mul_509);  add_302 = mul_509 = None
        slice_scatter_172: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_86, 3, 64, 9223372036854775807);  slice_86 = None
        slice_scatter_173: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_301, 3, 0, 64);  add_301 = None
        add_304: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_172, slice_scatter_173);  slice_scatter_172 = slice_scatter_173 = None
        slice_scatter_174: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_88, 3, 64, 9223372036854775807);  slice_88 = None
        slice_scatter_175: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_303, 3, 0, 64);  add_303 = None
        add_305: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_174, slice_scatter_175);  slice_scatter_174 = slice_scatter_175 = None
        permute_217: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_602, [0, 2, 1, 3]);  view_602 = None
        clone_65: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
        view_611: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_65, _shape_param_661);  clone_65 = _shape_param_661 = None
        view_612: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_305, _shape_param_662);  add_305 = _shape_param_662 = None
        view_613: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_304, _shape_param_663);  add_304 = _shape_param_663 = None
        view_614: "bf16[128, 4096]" = torch.ops.aten.view.default(view_611, _shape_param_664);  view_611 = _shape_param_664 = None
        permute_218: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_614, [1, 0])
        mm_260: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_218, arg442_1);  permute_218 = None
        mm_261: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_614, arg452_1);  view_614 = arg452_1 = None
        view_615: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_261, _shape_param_665);  mm_261 = _shape_param_665 = None
        convert_element_type_562: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_615, torch.float32);  view_615 = None
        add_306: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_553, convert_element_type_562);  convert_element_type_553 = convert_element_type_562 = None
        convert_element_type_563: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_260, torch.float32);  mm_260 = None
        view_616: "bf16[128, 4096]" = torch.ops.aten.view.default(view_612, _shape_param_666);  view_612 = _shape_param_666 = None
        permute_219: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_616, [1, 0])
        mm_262: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_219, arg442_1);  permute_219 = None
        mm_263: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_616, arg453_1);  view_616 = arg453_1 = None
        view_617: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_263, _shape_param_667);  mm_263 = _shape_param_667 = None
        convert_element_type_564: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_617, torch.float32);  view_617 = None
        add_307: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_306, convert_element_type_564);  add_306 = convert_element_type_564 = None
        convert_element_type_565: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_262, torch.float32);  mm_262 = None
        view_618: "bf16[128, 4096]" = torch.ops.aten.view.default(view_613, _shape_param_668);  view_613 = _shape_param_668 = None
        permute_220: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_264: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_220, arg442_1);  permute_220 = arg442_1 = None
        mm_265: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_618, arg454_1);  view_618 = arg454_1 = None
        view_619: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_265, _shape_param_669);  mm_265 = _shape_param_669 = None
        convert_element_type_566: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_619, torch.float32);  view_619 = None
        add_308: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_307, convert_element_type_566);  add_307 = convert_element_type_566 = None
        convert_element_type_567: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_264, torch.float32);  mm_264 = None
        mul_510: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_308, arg455_1);  arg455_1 = None
        mul_511: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_510, 4096)
        sum_160: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_510, [2], True)
        mul_512: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_510, arg456_1);  mul_510 = None
        sum_161: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_512, [2], True);  mul_512 = None
        mul_513: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg456_1, sum_161);  sum_161 = None
        sub_72: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_511, sum_160);  mul_511 = sum_160 = None
        sub_73: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_72, mul_513);  sub_72 = mul_513 = None
        mul_514: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg457_1, sub_73);  arg457_1 = sub_73 = None
        mul_515: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_308, arg456_1);  arg456_1 = None
        sum_162: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 1]);  mul_515 = None
        sum_163: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None
        add_309: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_295, mul_514);  add_295 = mul_514 = None
        convert_element_type_568: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_309, torch.bfloat16)
        view_620: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_568, _shape_param_670);  convert_element_type_568 = _shape_param_670 = None
        mm_266: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_620, arg458_1);  arg458_1 = None
        permute_221: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_620, [1, 0])
        mm_267: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_221, arg459_1);  arg459_1 = None
        sum_164: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_620, [0], True, dtype = torch.float32)
        view_621: "f32[4096]" = torch.ops.aten.view.default(sum_164, _shape_param_671);  sum_164 = _shape_param_671 = None
        convert_element_type_569: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_621, torch.bfloat16);  view_621 = None
        view_622: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_266, _shape_param_672);  mm_266 = _shape_param_672 = None
        convert_element_type_570: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_622, torch.float32);  view_622 = None
        convert_element_type_571: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_267, torch.float32);  mm_267 = None
        convert_element_type_572: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_569, torch.float32);  convert_element_type_569 = None
        view_623: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg460_1, _shape_param_673);  arg460_1 = _shape_param_673 = None
        mul_516: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_623, 0.5)
        mul_517: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_570, mul_516);  mul_516 = None
        convert_element_type_573: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_623, torch.float32)
        pow_45: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_573, 3.0)
        mul_518: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_310: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_623, mul_518);  view_623 = mul_518 = None
        mul_519: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_310, 0.7978845608028654);  add_310 = None
        tanh_22: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_519);  mul_519 = None
        add_311: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_520: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_570, add_311);  convert_element_type_570 = add_311 = None
        convert_element_type_574: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_520, torch.bfloat16);  mul_520 = None
        mul_521: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_74: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_521);  mul_521 = None
        mul_522: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_517, sub_74);  mul_517 = sub_74 = None
        mul_523: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_522, 0.7978845608028654);  mul_522 = None
        convert_element_type_575: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_523, torch.bfloat16)
        mul_524: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_523, 0.044715);  mul_523 = None
        pow_46: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_573, 2.0);  convert_element_type_573 = None
        mul_525: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_526: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_524, mul_525);  mul_524 = mul_525 = None
        convert_element_type_576: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_526, torch.bfloat16);  mul_526 = None
        add_312: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_575, convert_element_type_576);  convert_element_type_575 = convert_element_type_576 = None
        mul_527: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_574, 0.5);  convert_element_type_574 = None
        add_313: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_312, mul_527);  add_312 = mul_527 = None
        view_624: "bf16[128, 16384]" = torch.ops.aten.view.default(add_313, _shape_param_674);  add_313 = _shape_param_674 = None
        mm_268: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_624, arg461_1);  arg461_1 = None
        permute_222: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_624, [1, 0])
        mm_269: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_222, arg462_1);  permute_222 = None
        sum_165: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_624, [0], True, dtype = torch.float32);  view_624 = None
        view_625: "f32[16384]" = torch.ops.aten.view.default(sum_165, _shape_param_675);  sum_165 = _shape_param_675 = None
        convert_element_type_577: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_625, torch.bfloat16);  view_625 = None
        view_626: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_268, _shape_param_676);  mm_268 = _shape_param_676 = None
        convert_element_type_578: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_626, torch.float32);  view_626 = None
        convert_element_type_579: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_269, torch.float32);  mm_269 = None
        convert_element_type_580: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_577, torch.float32);  convert_element_type_577 = None
        mm_270: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_221, arg463_1);  permute_221 = arg463_1 = None
        mm_271: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_620, arg464_1);  view_620 = arg464_1 = None
        view_627: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_271, _shape_param_677);  mm_271 = _shape_param_677 = None
        convert_element_type_581: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_270, torch.float32);  mm_270 = None
        view_628: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_627, _shape_param_678);  view_627 = _shape_param_678 = None
        permute_223: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_628, [0, 2, 1, 3]);  view_628 = None
        view_629: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_223, _shape_param_679);  permute_223 = _shape_param_679 = None
        bmm_88: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg465_1, view_629);  arg465_1 = None
        bmm_89: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_629, arg466_1);  view_629 = arg466_1 = None
        view_630: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_88, _shape_param_680);  bmm_88 = _shape_param_680 = None
        view_631: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_89, _shape_param_681);  bmm_89 = _shape_param_681 = None
        convert_element_type_582: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_631, torch.float32);  view_631 = None
        mul_528: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_582, arg467_1);  convert_element_type_582 = None
        sum_166: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_528, [-1], True)
        neg_66: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg467_1);  arg467_1 = None
        fma_22: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_66, sum_166, mul_528);  neg_66 = sum_166 = mul_528 = None
        convert_element_type_583: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None
        div_25: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_583, 16.0);  convert_element_type_583 = None
        view_632: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_25, _shape_param_682);  div_25 = _shape_param_682 = None
        bmm_90: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg468_1, view_632);  arg468_1 = None
        bmm_91: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_632, arg469_1);  view_632 = arg469_1 = None
        view_633: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_90, _shape_param_683);  bmm_90 = _shape_param_683 = None
        view_634: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_91, _shape_param_684);  bmm_91 = _shape_param_684 = None
        convert_element_type_584: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_633, torch.float32);  view_633 = None
        permute_224: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_584, [0, 1, 3, 2]);  convert_element_type_584 = None
        convert_element_type_585: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_224, torch.bfloat16);  permute_224 = None
        convert_element_type_586: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_634, torch.bfloat16);  view_634 = None
        permute_225: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_586, [0, 2, 1, 3]);  convert_element_type_586 = None
        permute_226: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_585, [0, 2, 1, 3]);  convert_element_type_585 = None
        slice_89: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_225, 3, 0, 64)
        slice_90: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_225, 3, 64, 256);  permute_225 = None
        slice_91: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_226, 3, 0, 64)
        slice_92: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_226, 3, 64, 256);  permute_226 = None
        expand_46: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg470_1, _shape_param_685);  arg470_1 = _shape_param_685 = None
        clone_66: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_635: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_66, _shape_param_686);  clone_66 = _shape_param_686 = None
        mul_529: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_89, view_635)
        view_636: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_529, _shape_param_687);  mul_529 = _shape_param_687 = None
        select_88: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_636, -1, 0)
        select_89: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_636, -1, 1);  view_636 = None
        neg_67: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_88);  select_88 = None
        slice_scatter_176: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_67, 3, 1, 9223372036854775807, 2);  neg_67 = None
        slice_scatter_177: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_89, 3, 0, 9223372036854775807, 2);  select_89 = None
        add_314: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_176, slice_scatter_177);  slice_scatter_176 = slice_scatter_177 = None
        expand_47: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg471_1, _shape_param_688);  arg471_1 = _shape_param_688 = None
        clone_67: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_637: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_67, _shape_param_689);  clone_67 = _shape_param_689 = None
        mul_530: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_89, view_637);  slice_89 = None
        add_315: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_314, mul_530);  add_314 = mul_530 = None
        mul_531: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_91, view_635);  view_635 = None
        view_638: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_531, _shape_param_690);  mul_531 = _shape_param_690 = None
        select_90: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_638, -1, 0)
        select_91: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_638, -1, 1);  view_638 = None
        neg_68: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_90);  select_90 = None
        slice_scatter_178: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_68, 3, 1, 9223372036854775807, 2);  neg_68 = None
        slice_scatter_179: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_91, 3, 0, 9223372036854775807, 2);  select_91 = None
        add_316: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_178, slice_scatter_179);  slice_scatter_178 = slice_scatter_179 = None
        mul_532: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_91, view_637);  slice_91 = view_637 = None
        add_317: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_316, mul_532);  add_316 = mul_532 = None
        slice_scatter_180: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_90, 3, 64, 9223372036854775807);  slice_90 = None
        slice_scatter_181: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_315, 3, 0, 64);  add_315 = None
        add_318: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_180, slice_scatter_181);  slice_scatter_180 = slice_scatter_181 = None
        slice_scatter_182: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_92, 3, 64, 9223372036854775807);  slice_92 = None
        slice_scatter_183: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_317, 3, 0, 64);  add_317 = None
        add_319: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_182, slice_scatter_183);  slice_scatter_182 = slice_scatter_183 = None
        permute_227: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None
        clone_68: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_639: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_68, _shape_param_691);  clone_68 = _shape_param_691 = None
        view_640: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_319, _shape_param_692);  add_319 = _shape_param_692 = None
        view_641: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_318, _shape_param_693);  add_318 = _shape_param_693 = None
        view_642: "bf16[128, 4096]" = torch.ops.aten.view.default(view_639, _shape_param_694);  view_639 = _shape_param_694 = None
        permute_228: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_642, [1, 0])
        mm_272: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_228, arg462_1);  permute_228 = None
        mm_273: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_642, arg472_1);  view_642 = arg472_1 = None
        view_643: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_273, _shape_param_695);  mm_273 = _shape_param_695 = None
        convert_element_type_587: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_643, torch.float32);  view_643 = None
        add_320: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_578, convert_element_type_587);  convert_element_type_578 = convert_element_type_587 = None
        convert_element_type_588: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_272, torch.float32);  mm_272 = None
        view_644: "bf16[128, 4096]" = torch.ops.aten.view.default(view_640, _shape_param_696);  view_640 = _shape_param_696 = None
        permute_229: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_644, [1, 0])
        mm_274: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_229, arg462_1);  permute_229 = None
        mm_275: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_644, arg473_1);  view_644 = arg473_1 = None
        view_645: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_275, _shape_param_697);  mm_275 = _shape_param_697 = None
        convert_element_type_589: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_645, torch.float32);  view_645 = None
        add_321: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_320, convert_element_type_589);  add_320 = convert_element_type_589 = None
        convert_element_type_590: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_274, torch.float32);  mm_274 = None
        view_646: "bf16[128, 4096]" = torch.ops.aten.view.default(view_641, _shape_param_698);  view_641 = _shape_param_698 = None
        permute_230: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_646, [1, 0])
        mm_276: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_230, arg462_1);  permute_230 = arg462_1 = None
        mm_277: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_646, arg474_1);  view_646 = arg474_1 = None
        view_647: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_277, _shape_param_699);  mm_277 = _shape_param_699 = None
        convert_element_type_591: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_647, torch.float32);  view_647 = None
        add_322: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_321, convert_element_type_591);  add_321 = convert_element_type_591 = None
        convert_element_type_592: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_276, torch.float32);  mm_276 = None
        mul_533: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_322, arg475_1);  arg475_1 = None
        mul_534: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_533, 4096)
        sum_167: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_533, [2], True)
        mul_535: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_533, arg476_1);  mul_533 = None
        sum_168: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_535, [2], True);  mul_535 = None
        mul_536: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg476_1, sum_168);  sum_168 = None
        sub_75: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_534, sum_167);  mul_534 = sum_167 = None
        sub_76: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_75, mul_536);  sub_75 = mul_536 = None
        mul_537: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg477_1, sub_76);  arg477_1 = sub_76 = None
        mul_538: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_322, arg476_1);  arg476_1 = None
        sum_169: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_538, [0, 1]);  mul_538 = None
        sum_170: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_322, [0, 1]);  add_322 = None
        add_323: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_309, mul_537);  add_309 = mul_537 = None
        convert_element_type_593: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_323, torch.bfloat16)
        view_648: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_593, _shape_param_700);  convert_element_type_593 = _shape_param_700 = None
        mm_278: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_648, arg478_1);  arg478_1 = None
        permute_231: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_648, [1, 0])
        mm_279: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_231, arg479_1);  arg479_1 = None
        sum_171: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_648, [0], True, dtype = torch.float32)
        view_649: "f32[4096]" = torch.ops.aten.view.default(sum_171, _shape_param_701);  sum_171 = _shape_param_701 = None
        convert_element_type_594: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_649, torch.bfloat16);  view_649 = None
        view_650: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_278, _shape_param_702);  mm_278 = _shape_param_702 = None
        convert_element_type_595: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_650, torch.float32);  view_650 = None
        convert_element_type_596: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None
        convert_element_type_597: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_594, torch.float32);  convert_element_type_594 = None
        view_651: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg480_1, _shape_param_703);  arg480_1 = _shape_param_703 = None
        mul_539: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_651, 0.5)
        mul_540: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_595, mul_539);  mul_539 = None
        convert_element_type_598: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_651, torch.float32)
        pow_47: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_598, 3.0)
        mul_541: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_47, 0.044715);  pow_47 = None
        add_324: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_651, mul_541);  view_651 = mul_541 = None
        mul_542: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_324, 0.7978845608028654);  add_324 = None
        tanh_23: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_542);  mul_542 = None
        add_325: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_543: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_595, add_325);  convert_element_type_595 = add_325 = None
        convert_element_type_599: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_543, torch.bfloat16);  mul_543 = None
        mul_544: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_77: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_544);  mul_544 = None
        mul_545: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_540, sub_77);  mul_540 = sub_77 = None
        mul_546: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_545, 0.7978845608028654);  mul_545 = None
        convert_element_type_600: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_546, torch.bfloat16)
        mul_547: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_546, 0.044715);  mul_546 = None
        pow_48: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_598, 2.0);  convert_element_type_598 = None
        mul_548: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_549: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
        convert_element_type_601: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_549, torch.bfloat16);  mul_549 = None
        add_326: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_600, convert_element_type_601);  convert_element_type_600 = convert_element_type_601 = None
        mul_550: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_599, 0.5);  convert_element_type_599 = None
        add_327: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_326, mul_550);  add_326 = mul_550 = None
        view_652: "bf16[128, 16384]" = torch.ops.aten.view.default(add_327, _shape_param_704);  add_327 = _shape_param_704 = None
        mm_280: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_652, arg481_1);  arg481_1 = None
        permute_232: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_652, [1, 0])
        mm_281: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_232, arg482_1);  permute_232 = None
        sum_172: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_652, [0], True, dtype = torch.float32);  view_652 = None
        view_653: "f32[16384]" = torch.ops.aten.view.default(sum_172, _shape_param_705);  sum_172 = _shape_param_705 = None
        convert_element_type_602: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_653, torch.bfloat16);  view_653 = None
        view_654: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_280, _shape_param_706);  mm_280 = _shape_param_706 = None
        convert_element_type_603: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_654, torch.float32);  view_654 = None
        convert_element_type_604: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_281, torch.float32);  mm_281 = None
        convert_element_type_605: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_602, torch.float32);  convert_element_type_602 = None
        mm_282: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_231, arg483_1);  permute_231 = arg483_1 = None
        mm_283: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_648, arg484_1);  view_648 = arg484_1 = None
        view_655: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_283, _shape_param_707);  mm_283 = _shape_param_707 = None
        convert_element_type_606: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_282, torch.float32);  mm_282 = None
        view_656: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_655, _shape_param_708);  view_655 = _shape_param_708 = None
        permute_233: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_656, [0, 2, 1, 3]);  view_656 = None
        view_657: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_233, _shape_param_709);  permute_233 = _shape_param_709 = None
        bmm_92: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg485_1, view_657);  arg485_1 = None
        bmm_93: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_657, arg486_1);  view_657 = arg486_1 = None
        view_658: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_92, _shape_param_710);  bmm_92 = _shape_param_710 = None
        view_659: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_93, _shape_param_711);  bmm_93 = _shape_param_711 = None
        convert_element_type_607: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_659, torch.float32);  view_659 = None
        mul_551: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_607, arg487_1);  convert_element_type_607 = None
        sum_173: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [-1], True)
        neg_69: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg487_1);  arg487_1 = None
        fma_23: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_69, sum_173, mul_551);  neg_69 = sum_173 = mul_551 = None
        convert_element_type_608: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None
        div_26: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_608, 16.0);  convert_element_type_608 = None
        view_660: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_26, _shape_param_712);  div_26 = _shape_param_712 = None
        bmm_94: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg488_1, view_660);  arg488_1 = None
        bmm_95: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_660, arg489_1);  view_660 = arg489_1 = None
        view_661: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_94, _shape_param_713);  bmm_94 = _shape_param_713 = None
        view_662: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_95, _shape_param_714);  bmm_95 = _shape_param_714 = None
        convert_element_type_609: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_661, torch.float32);  view_661 = None
        permute_234: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_609, [0, 1, 3, 2]);  convert_element_type_609 = None
        convert_element_type_610: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_234, torch.bfloat16);  permute_234 = None
        convert_element_type_611: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_662, torch.bfloat16);  view_662 = None
        permute_235: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_611, [0, 2, 1, 3]);  convert_element_type_611 = None
        permute_236: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_610, [0, 2, 1, 3]);  convert_element_type_610 = None
        slice_93: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_235, 3, 0, 64)
        slice_94: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_235, 3, 64, 256);  permute_235 = None
        slice_95: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_236, 3, 0, 64)
        slice_96: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_236, 3, 64, 256);  permute_236 = None
        expand_48: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg490_1, _shape_param_715);  arg490_1 = _shape_param_715 = None
        clone_69: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_663: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_69, _shape_param_716);  clone_69 = _shape_param_716 = None
        mul_552: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_93, view_663)
        view_664: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_552, _shape_param_717);  mul_552 = _shape_param_717 = None
        select_92: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_664, -1, 0)
        select_93: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_664, -1, 1);  view_664 = None
        neg_70: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_92);  select_92 = None
        slice_scatter_184: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_70, 3, 1, 9223372036854775807, 2);  neg_70 = None
        slice_scatter_185: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_93, 3, 0, 9223372036854775807, 2);  select_93 = None
        add_328: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_184, slice_scatter_185);  slice_scatter_184 = slice_scatter_185 = None
        expand_49: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg491_1, _shape_param_718);  arg491_1 = _shape_param_718 = None
        clone_70: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_665: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_70, _shape_param_719);  clone_70 = _shape_param_719 = None
        mul_553: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_93, view_665);  slice_93 = None
        add_329: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_328, mul_553);  add_328 = mul_553 = None
        mul_554: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_95, view_663);  view_663 = None
        view_666: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_554, _shape_param_720);  mul_554 = _shape_param_720 = None
        select_94: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_666, -1, 0)
        select_95: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_666, -1, 1);  view_666 = None
        neg_71: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_94);  select_94 = None
        slice_scatter_186: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_71, 3, 1, 9223372036854775807, 2);  neg_71 = None
        slice_scatter_187: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_95, 3, 0, 9223372036854775807, 2);  select_95 = None
        add_330: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_186, slice_scatter_187);  slice_scatter_186 = slice_scatter_187 = None
        mul_555: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_95, view_665);  slice_95 = view_665 = None
        add_331: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_330, mul_555);  add_330 = mul_555 = None
        slice_scatter_188: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_94, 3, 64, 9223372036854775807);  slice_94 = None
        slice_scatter_189: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_329, 3, 0, 64);  add_329 = None
        add_332: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_188, slice_scatter_189);  slice_scatter_188 = slice_scatter_189 = None
        slice_scatter_190: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_96, 3, 64, 9223372036854775807);  slice_96 = None
        slice_scatter_191: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_331, 3, 0, 64);  add_331 = None
        add_333: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_190, slice_scatter_191);  slice_scatter_190 = slice_scatter_191 = None
        permute_237: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_658, [0, 2, 1, 3]);  view_658 = None
        clone_71: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_667: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_71, _shape_param_721);  clone_71 = _shape_param_721 = None
        view_668: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_333, _shape_param_722);  add_333 = _shape_param_722 = None
        view_669: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_332, _shape_param_723);  add_332 = _shape_param_723 = None
        view_670: "bf16[128, 4096]" = torch.ops.aten.view.default(view_667, _shape_param_724);  view_667 = _shape_param_724 = None
        permute_238: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_284: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_238, arg482_1);  permute_238 = None
        mm_285: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_670, arg492_1);  view_670 = arg492_1 = None
        view_671: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_285, _shape_param_725);  mm_285 = _shape_param_725 = None
        convert_element_type_612: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_671, torch.float32);  view_671 = None
        add_334: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_603, convert_element_type_612);  convert_element_type_603 = convert_element_type_612 = None
        convert_element_type_613: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_284, torch.float32);  mm_284 = None
        view_672: "bf16[128, 4096]" = torch.ops.aten.view.default(view_668, _shape_param_726);  view_668 = _shape_param_726 = None
        permute_239: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_672, [1, 0])
        mm_286: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_239, arg482_1);  permute_239 = None
        mm_287: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_672, arg493_1);  view_672 = arg493_1 = None
        view_673: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_287, _shape_param_727);  mm_287 = _shape_param_727 = None
        convert_element_type_614: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_673, torch.float32);  view_673 = None
        add_335: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_334, convert_element_type_614);  add_334 = convert_element_type_614 = None
        convert_element_type_615: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_286, torch.float32);  mm_286 = None
        view_674: "bf16[128, 4096]" = torch.ops.aten.view.default(view_669, _shape_param_728);  view_669 = _shape_param_728 = None
        permute_240: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_674, [1, 0])
        mm_288: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_240, arg482_1);  permute_240 = arg482_1 = None
        mm_289: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_674, arg494_1);  view_674 = arg494_1 = None
        view_675: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_289, _shape_param_729);  mm_289 = _shape_param_729 = None
        convert_element_type_616: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_675, torch.float32);  view_675 = None
        add_336: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_335, convert_element_type_616);  add_335 = convert_element_type_616 = None
        convert_element_type_617: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_288, torch.float32);  mm_288 = None
        mul_556: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_336, arg495_1);  arg495_1 = None
        mul_557: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_556, 4096)
        sum_174: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_556, [2], True)
        mul_558: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_556, arg496_1);  mul_556 = None
        sum_175: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_558, [2], True);  mul_558 = None
        mul_559: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg496_1, sum_175);  sum_175 = None
        sub_78: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_557, sum_174);  mul_557 = sum_174 = None
        sub_79: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_78, mul_559);  sub_78 = mul_559 = None
        mul_560: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg497_1, sub_79);  arg497_1 = sub_79 = None
        mul_561: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_336, arg496_1);  arg496_1 = None
        sum_176: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_561, [0, 1]);  mul_561 = None
        sum_177: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_336, [0, 1]);  add_336 = None
        add_337: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_323, mul_560);  add_323 = mul_560 = None
        convert_element_type_618: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_337, torch.bfloat16)
        view_676: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_618, _shape_param_730);  convert_element_type_618 = _shape_param_730 = None
        mm_290: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_676, arg498_1);  arg498_1 = None
        permute_241: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_676, [1, 0])
        mm_291: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_241, arg499_1);  arg499_1 = None
        sum_178: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_676, [0], True, dtype = torch.float32)
        view_677: "f32[4096]" = torch.ops.aten.view.default(sum_178, _shape_param_731);  sum_178 = _shape_param_731 = None
        convert_element_type_619: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_677, torch.bfloat16);  view_677 = None
        view_678: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_290, _shape_param_732);  mm_290 = _shape_param_732 = None
        convert_element_type_620: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_678, torch.float32);  view_678 = None
        convert_element_type_621: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_291, torch.float32);  mm_291 = None
        convert_element_type_622: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_619, torch.float32);  convert_element_type_619 = None
        view_679: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg500_1, _shape_param_733);  arg500_1 = _shape_param_733 = None
        mul_562: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_679, 0.5)
        mul_563: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_620, mul_562);  mul_562 = None
        convert_element_type_623: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_679, torch.float32)
        pow_49: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_623, 3.0)
        mul_564: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_338: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_679, mul_564);  view_679 = mul_564 = None
        mul_565: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_338, 0.7978845608028654);  add_338 = None
        tanh_24: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_565);  mul_565 = None
        add_339: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_24, 1.0)
        mul_566: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_620, add_339);  convert_element_type_620 = add_339 = None
        convert_element_type_624: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_566, torch.bfloat16);  mul_566 = None
        mul_567: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_24, tanh_24);  tanh_24 = None
        sub_80: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_567);  mul_567 = None
        mul_568: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_563, sub_80);  mul_563 = sub_80 = None
        mul_569: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_568, 0.7978845608028654);  mul_568 = None
        convert_element_type_625: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_569, torch.bfloat16)
        mul_570: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_569, 0.044715);  mul_569 = None
        pow_50: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_623, 2.0);  convert_element_type_623 = None
        mul_571: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_50, 3.0);  pow_50 = None
        mul_572: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_570, mul_571);  mul_570 = mul_571 = None
        convert_element_type_626: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_572, torch.bfloat16);  mul_572 = None
        add_340: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_625, convert_element_type_626);  convert_element_type_625 = convert_element_type_626 = None
        mul_573: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_624, 0.5);  convert_element_type_624 = None
        add_341: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_340, mul_573);  add_340 = mul_573 = None
        view_680: "bf16[128, 16384]" = torch.ops.aten.view.default(add_341, _shape_param_734);  add_341 = _shape_param_734 = None
        mm_292: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_680, arg501_1);  arg501_1 = None
        permute_242: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_680, [1, 0])
        mm_293: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_242, arg502_1);  permute_242 = None
        sum_179: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_680, [0], True, dtype = torch.float32);  view_680 = None
        view_681: "f32[16384]" = torch.ops.aten.view.default(sum_179, _shape_param_735);  sum_179 = _shape_param_735 = None
        convert_element_type_627: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_681, torch.bfloat16);  view_681 = None
        view_682: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_292, _shape_param_736);  mm_292 = _shape_param_736 = None
        convert_element_type_628: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_682, torch.float32);  view_682 = None
        convert_element_type_629: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_293, torch.float32);  mm_293 = None
        convert_element_type_630: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_627, torch.float32);  convert_element_type_627 = None
        mm_294: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_241, arg503_1);  permute_241 = arg503_1 = None
        mm_295: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_676, arg504_1);  view_676 = arg504_1 = None
        view_683: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_295, _shape_param_737);  mm_295 = _shape_param_737 = None
        convert_element_type_631: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_294, torch.float32);  mm_294 = None
        view_684: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_683, _shape_param_738);  view_683 = _shape_param_738 = None
        permute_243: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_684, [0, 2, 1, 3]);  view_684 = None
        view_685: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_243, _shape_param_739);  permute_243 = _shape_param_739 = None
        bmm_96: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg505_1, view_685);  arg505_1 = None
        bmm_97: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_685, arg506_1);  view_685 = arg506_1 = None
        view_686: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_96, _shape_param_740);  bmm_96 = _shape_param_740 = None
        view_687: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_97, _shape_param_741);  bmm_97 = _shape_param_741 = None
        convert_element_type_632: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_687, torch.float32);  view_687 = None
        mul_574: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_632, arg507_1);  convert_element_type_632 = None
        sum_180: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_574, [-1], True)
        neg_72: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg507_1);  arg507_1 = None
        fma_24: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_72, sum_180, mul_574);  neg_72 = sum_180 = mul_574 = None
        convert_element_type_633: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_24, torch.bfloat16);  fma_24 = None
        div_27: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_633, 16.0);  convert_element_type_633 = None
        view_688: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_27, _shape_param_742);  div_27 = _shape_param_742 = None
        bmm_98: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg508_1, view_688);  arg508_1 = None
        bmm_99: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_688, arg509_1);  view_688 = arg509_1 = None
        view_689: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_98, _shape_param_743);  bmm_98 = _shape_param_743 = None
        view_690: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_99, _shape_param_744);  bmm_99 = _shape_param_744 = None
        convert_element_type_634: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_689, torch.float32);  view_689 = None
        permute_244: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_634, [0, 1, 3, 2]);  convert_element_type_634 = None
        convert_element_type_635: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_244, torch.bfloat16);  permute_244 = None
        convert_element_type_636: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_690, torch.bfloat16);  view_690 = None
        permute_245: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_636, [0, 2, 1, 3]);  convert_element_type_636 = None
        permute_246: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_635, [0, 2, 1, 3]);  convert_element_type_635 = None
        slice_97: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_245, 3, 0, 64)
        slice_98: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_245, 3, 64, 256);  permute_245 = None
        slice_99: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_246, 3, 0, 64)
        slice_100: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_246, 3, 64, 256);  permute_246 = None
        expand_50: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg510_1, _shape_param_745);  arg510_1 = _shape_param_745 = None
        clone_72: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_691: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_72, _shape_param_746);  clone_72 = _shape_param_746 = None
        mul_575: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_97, view_691)
        view_692: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_575, _shape_param_747);  mul_575 = _shape_param_747 = None
        select_96: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_692, -1, 0)
        select_97: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_692, -1, 1);  view_692 = None
        neg_73: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_96);  select_96 = None
        slice_scatter_192: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_73, 3, 1, 9223372036854775807, 2);  neg_73 = None
        slice_scatter_193: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_97, 3, 0, 9223372036854775807, 2);  select_97 = None
        add_342: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_192, slice_scatter_193);  slice_scatter_192 = slice_scatter_193 = None
        expand_51: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg511_1, _shape_param_748);  arg511_1 = _shape_param_748 = None
        clone_73: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_693: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_73, _shape_param_749);  clone_73 = _shape_param_749 = None
        mul_576: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_97, view_693);  slice_97 = None
        add_343: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_342, mul_576);  add_342 = mul_576 = None
        mul_577: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_99, view_691);  view_691 = None
        view_694: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_577, _shape_param_750);  mul_577 = _shape_param_750 = None
        select_98: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_694, -1, 0)
        select_99: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_694, -1, 1);  view_694 = None
        neg_74: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_98);  select_98 = None
        slice_scatter_194: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_74, 3, 1, 9223372036854775807, 2);  neg_74 = None
        slice_scatter_195: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_99, 3, 0, 9223372036854775807, 2);  select_99 = None
        add_344: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_194, slice_scatter_195);  slice_scatter_194 = slice_scatter_195 = None
        mul_578: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_99, view_693);  slice_99 = view_693 = None
        add_345: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_344, mul_578);  add_344 = mul_578 = None
        slice_scatter_196: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_98, 3, 64, 9223372036854775807);  slice_98 = None
        slice_scatter_197: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_343, 3, 0, 64);  add_343 = None
        add_346: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_196, slice_scatter_197);  slice_scatter_196 = slice_scatter_197 = None
        slice_scatter_198: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_100, 3, 64, 9223372036854775807);  slice_100 = None
        slice_scatter_199: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_345, 3, 0, 64);  add_345 = None
        add_347: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_198, slice_scatter_199);  slice_scatter_198 = slice_scatter_199 = None
        permute_247: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_686, [0, 2, 1, 3]);  view_686 = None
        clone_74: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_695: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_74, _shape_param_751);  clone_74 = _shape_param_751 = None
        view_696: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_347, _shape_param_752);  add_347 = _shape_param_752 = None
        view_697: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_346, _shape_param_753);  add_346 = _shape_param_753 = None
        view_698: "bf16[128, 4096]" = torch.ops.aten.view.default(view_695, _shape_param_754);  view_695 = _shape_param_754 = None
        permute_248: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_698, [1, 0])
        mm_296: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_248, arg502_1);  permute_248 = None
        mm_297: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_698, arg512_1);  view_698 = arg512_1 = None
        view_699: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_297, _shape_param_755);  mm_297 = _shape_param_755 = None
        convert_element_type_637: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_699, torch.float32);  view_699 = None
        add_348: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_628, convert_element_type_637);  convert_element_type_628 = convert_element_type_637 = None
        convert_element_type_638: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_296, torch.float32);  mm_296 = None
        view_700: "bf16[128, 4096]" = torch.ops.aten.view.default(view_696, _shape_param_756);  view_696 = _shape_param_756 = None
        permute_249: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_700, [1, 0])
        mm_298: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_249, arg502_1);  permute_249 = None
        mm_299: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_700, arg513_1);  view_700 = arg513_1 = None
        view_701: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_299, _shape_param_757);  mm_299 = _shape_param_757 = None
        convert_element_type_639: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_701, torch.float32);  view_701 = None
        add_349: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_348, convert_element_type_639);  add_348 = convert_element_type_639 = None
        convert_element_type_640: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_298, torch.float32);  mm_298 = None
        view_702: "bf16[128, 4096]" = torch.ops.aten.view.default(view_697, _shape_param_758);  view_697 = _shape_param_758 = None
        permute_250: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_702, [1, 0])
        mm_300: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_250, arg502_1);  permute_250 = arg502_1 = None
        mm_301: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_702, arg514_1);  view_702 = arg514_1 = None
        view_703: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_301, _shape_param_759);  mm_301 = _shape_param_759 = None
        convert_element_type_641: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_703, torch.float32);  view_703 = None
        add_350: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_349, convert_element_type_641);  add_349 = convert_element_type_641 = None
        convert_element_type_642: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_300, torch.float32);  mm_300 = None
        mul_579: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_350, arg515_1);  arg515_1 = None
        mul_580: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_579, 4096)
        sum_181: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True)
        mul_581: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_579, arg516_1);  mul_579 = None
        sum_182: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True);  mul_581 = None
        mul_582: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg516_1, sum_182);  sum_182 = None
        sub_81: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_580, sum_181);  mul_580 = sum_181 = None
        sub_82: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_81, mul_582);  sub_81 = mul_582 = None
        mul_583: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg517_1, sub_82);  arg517_1 = sub_82 = None
        mul_584: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_350, arg516_1);  arg516_1 = None
        sum_183: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1]);  mul_584 = None
        sum_184: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_350, [0, 1]);  add_350 = None
        add_351: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_337, mul_583);  add_337 = mul_583 = None
        convert_element_type_643: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_351, torch.bfloat16)
        view_704: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_643, _shape_param_760);  convert_element_type_643 = _shape_param_760 = None
        mm_302: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_704, arg518_1);  arg518_1 = None
        permute_251: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_704, [1, 0])
        mm_303: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_251, arg519_1);  arg519_1 = None
        sum_185: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_704, [0], True, dtype = torch.float32)
        view_705: "f32[4096]" = torch.ops.aten.view.default(sum_185, _shape_param_761);  sum_185 = _shape_param_761 = None
        convert_element_type_644: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_705, torch.bfloat16);  view_705 = None
        view_706: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_302, _shape_param_762);  mm_302 = _shape_param_762 = None
        convert_element_type_645: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_706, torch.float32);  view_706 = None
        convert_element_type_646: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_303, torch.float32);  mm_303 = None
        convert_element_type_647: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_644, torch.float32);  convert_element_type_644 = None
        view_707: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg520_1, _shape_param_763);  arg520_1 = _shape_param_763 = None
        mul_585: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_707, 0.5)
        mul_586: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_645, mul_585);  mul_585 = None
        convert_element_type_648: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_707, torch.float32)
        pow_51: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_648, 3.0)
        mul_587: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_51, 0.044715);  pow_51 = None
        add_352: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_707, mul_587);  view_707 = mul_587 = None
        mul_588: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_352, 0.7978845608028654);  add_352 = None
        tanh_25: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_588);  mul_588 = None
        add_353: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_25, 1.0)
        mul_589: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_645, add_353);  convert_element_type_645 = add_353 = None
        convert_element_type_649: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_589, torch.bfloat16);  mul_589 = None
        mul_590: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_25, tanh_25);  tanh_25 = None
        sub_83: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_590);  mul_590 = None
        mul_591: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_586, sub_83);  mul_586 = sub_83 = None
        mul_592: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_591, 0.7978845608028654);  mul_591 = None
        convert_element_type_650: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_592, torch.bfloat16)
        mul_593: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_592, 0.044715);  mul_592 = None
        pow_52: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_648, 2.0);  convert_element_type_648 = None
        mul_594: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_52, 3.0);  pow_52 = None
        mul_595: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_593, mul_594);  mul_593 = mul_594 = None
        convert_element_type_651: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_595, torch.bfloat16);  mul_595 = None
        add_354: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_650, convert_element_type_651);  convert_element_type_650 = convert_element_type_651 = None
        mul_596: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_649, 0.5);  convert_element_type_649 = None
        add_355: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_354, mul_596);  add_354 = mul_596 = None
        view_708: "bf16[128, 16384]" = torch.ops.aten.view.default(add_355, _shape_param_764);  add_355 = _shape_param_764 = None
        mm_304: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_708, arg521_1);  arg521_1 = None
        permute_252: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_708, [1, 0])
        mm_305: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_252, arg522_1);  permute_252 = None
        sum_186: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_708, [0], True, dtype = torch.float32);  view_708 = None
        view_709: "f32[16384]" = torch.ops.aten.view.default(sum_186, _shape_param_765);  sum_186 = _shape_param_765 = None
        convert_element_type_652: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_709, torch.bfloat16);  view_709 = None
        view_710: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_304, _shape_param_766);  mm_304 = _shape_param_766 = None
        convert_element_type_653: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_710, torch.float32);  view_710 = None
        convert_element_type_654: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_305, torch.float32);  mm_305 = None
        convert_element_type_655: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_652, torch.float32);  convert_element_type_652 = None
        mm_306: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_251, arg523_1);  permute_251 = arg523_1 = None
        mm_307: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_704, arg524_1);  view_704 = arg524_1 = None
        view_711: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_307, _shape_param_767);  mm_307 = _shape_param_767 = None
        convert_element_type_656: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_306, torch.float32);  mm_306 = None
        view_712: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_711, _shape_param_768);  view_711 = _shape_param_768 = None
        permute_253: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_712, [0, 2, 1, 3]);  view_712 = None
        view_713: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_253, _shape_param_769);  permute_253 = _shape_param_769 = None
        bmm_100: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg525_1, view_713);  arg525_1 = None
        bmm_101: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_713, arg526_1);  view_713 = arg526_1 = None
        view_714: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_100, _shape_param_770);  bmm_100 = _shape_param_770 = None
        view_715: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_101, _shape_param_771);  bmm_101 = _shape_param_771 = None
        convert_element_type_657: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_715, torch.float32);  view_715 = None
        mul_597: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_657, arg527_1);  convert_element_type_657 = None
        sum_187: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [-1], True)
        neg_75: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg527_1);  arg527_1 = None
        fma_25: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_75, sum_187, mul_597);  neg_75 = sum_187 = mul_597 = None
        convert_element_type_658: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_25, torch.bfloat16);  fma_25 = None
        div_28: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_658, 16.0);  convert_element_type_658 = None
        view_716: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_28, _shape_param_772);  div_28 = _shape_param_772 = None
        bmm_102: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg528_1, view_716);  arg528_1 = None
        bmm_103: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_716, arg529_1);  view_716 = arg529_1 = None
        view_717: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_102, _shape_param_773);  bmm_102 = _shape_param_773 = None
        view_718: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_103, _shape_param_774);  bmm_103 = _shape_param_774 = None
        convert_element_type_659: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_717, torch.float32);  view_717 = None
        permute_254: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_659, [0, 1, 3, 2]);  convert_element_type_659 = None
        convert_element_type_660: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_254, torch.bfloat16);  permute_254 = None
        convert_element_type_661: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_718, torch.bfloat16);  view_718 = None
        permute_255: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_661, [0, 2, 1, 3]);  convert_element_type_661 = None
        permute_256: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_660, [0, 2, 1, 3]);  convert_element_type_660 = None
        slice_101: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_255, 3, 0, 64)
        slice_102: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_255, 3, 64, 256);  permute_255 = None
        slice_103: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_256, 3, 0, 64)
        slice_104: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_256, 3, 64, 256);  permute_256 = None
        expand_52: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg530_1, _shape_param_775);  arg530_1 = _shape_param_775 = None
        clone_75: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_719: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_75, _shape_param_776);  clone_75 = _shape_param_776 = None
        mul_598: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_101, view_719)
        view_720: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_598, _shape_param_777);  mul_598 = _shape_param_777 = None
        select_100: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_720, -1, 0)
        select_101: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_720, -1, 1);  view_720 = None
        neg_76: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_100);  select_100 = None
        slice_scatter_200: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_76, 3, 1, 9223372036854775807, 2);  neg_76 = None
        slice_scatter_201: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_101, 3, 0, 9223372036854775807, 2);  select_101 = None
        add_356: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_200, slice_scatter_201);  slice_scatter_200 = slice_scatter_201 = None
        expand_53: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg531_1, _shape_param_778);  arg531_1 = _shape_param_778 = None
        clone_76: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_721: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_76, _shape_param_779);  clone_76 = _shape_param_779 = None
        mul_599: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_101, view_721);  slice_101 = None
        add_357: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_356, mul_599);  add_356 = mul_599 = None
        mul_600: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_103, view_719);  view_719 = None
        view_722: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_600, _shape_param_780);  mul_600 = _shape_param_780 = None
        select_102: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_722, -1, 0)
        select_103: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_722, -1, 1);  view_722 = None
        neg_77: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_102);  select_102 = None
        slice_scatter_202: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_77, 3, 1, 9223372036854775807, 2);  neg_77 = None
        slice_scatter_203: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_103, 3, 0, 9223372036854775807, 2);  select_103 = None
        add_358: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_202, slice_scatter_203);  slice_scatter_202 = slice_scatter_203 = None
        mul_601: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_103, view_721);  slice_103 = view_721 = None
        add_359: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_358, mul_601);  add_358 = mul_601 = None
        slice_scatter_204: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_102, 3, 64, 9223372036854775807);  slice_102 = None
        slice_scatter_205: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_357, 3, 0, 64);  add_357 = None
        add_360: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_204, slice_scatter_205);  slice_scatter_204 = slice_scatter_205 = None
        slice_scatter_206: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_104, 3, 64, 9223372036854775807);  slice_104 = None
        slice_scatter_207: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_359, 3, 0, 64);  add_359 = None
        add_361: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_206, slice_scatter_207);  slice_scatter_206 = slice_scatter_207 = None
        permute_257: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_714, [0, 2, 1, 3]);  view_714 = None
        clone_77: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
        view_723: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_77, _shape_param_781);  clone_77 = _shape_param_781 = None
        view_724: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_361, _shape_param_782);  add_361 = _shape_param_782 = None
        view_725: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_360, _shape_param_783);  add_360 = _shape_param_783 = None
        view_726: "bf16[128, 4096]" = torch.ops.aten.view.default(view_723, _shape_param_784);  view_723 = _shape_param_784 = None
        permute_258: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_726, [1, 0])
        mm_308: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_258, arg522_1);  permute_258 = None
        mm_309: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_726, arg532_1);  view_726 = arg532_1 = None
        view_727: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_309, _shape_param_785);  mm_309 = _shape_param_785 = None
        convert_element_type_662: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_727, torch.float32);  view_727 = None
        add_362: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_653, convert_element_type_662);  convert_element_type_653 = convert_element_type_662 = None
        convert_element_type_663: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_308, torch.float32);  mm_308 = None
        view_728: "bf16[128, 4096]" = torch.ops.aten.view.default(view_724, _shape_param_786);  view_724 = _shape_param_786 = None
        permute_259: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_728, [1, 0])
        mm_310: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_259, arg522_1);  permute_259 = None
        mm_311: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_728, arg533_1);  view_728 = arg533_1 = None
        view_729: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_311, _shape_param_787);  mm_311 = _shape_param_787 = None
        convert_element_type_664: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_729, torch.float32);  view_729 = None
        add_363: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_362, convert_element_type_664);  add_362 = convert_element_type_664 = None
        convert_element_type_665: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_310, torch.float32);  mm_310 = None
        view_730: "bf16[128, 4096]" = torch.ops.aten.view.default(view_725, _shape_param_788);  view_725 = _shape_param_788 = None
        permute_260: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_730, [1, 0])
        mm_312: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_260, arg522_1);  permute_260 = arg522_1 = None
        mm_313: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_730, arg534_1);  view_730 = arg534_1 = None
        view_731: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_313, _shape_param_789);  mm_313 = _shape_param_789 = None
        convert_element_type_666: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_731, torch.float32);  view_731 = None
        add_364: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_363, convert_element_type_666);  add_363 = convert_element_type_666 = None
        convert_element_type_667: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_312, torch.float32);  mm_312 = None
        mul_602: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_364, arg535_1);  arg535_1 = None
        mul_603: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_602, 4096)
        sum_188: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True)
        mul_604: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_602, arg536_1);  mul_602 = None
        sum_189: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_604, [2], True);  mul_604 = None
        mul_605: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg536_1, sum_189);  sum_189 = None
        sub_84: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_603, sum_188);  mul_603 = sum_188 = None
        sub_85: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_84, mul_605);  sub_84 = mul_605 = None
        mul_606: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg537_1, sub_85);  arg537_1 = sub_85 = None
        mul_607: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_364, arg536_1);  arg536_1 = None
        sum_190: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 1]);  mul_607 = None
        sum_191: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_364, [0, 1]);  add_364 = None
        add_365: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_351, mul_606);  add_351 = mul_606 = None
        convert_element_type_668: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_365, torch.bfloat16)
        view_732: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_668, _shape_param_790);  convert_element_type_668 = _shape_param_790 = None
        mm_314: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_732, arg538_1);  arg538_1 = None
        permute_261: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_732, [1, 0])
        mm_315: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_261, arg539_1);  arg539_1 = None
        sum_192: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_732, [0], True, dtype = torch.float32)
        view_733: "f32[4096]" = torch.ops.aten.view.default(sum_192, _shape_param_791);  sum_192 = _shape_param_791 = None
        convert_element_type_669: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_733, torch.bfloat16);  view_733 = None
        view_734: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_314, _shape_param_792);  mm_314 = _shape_param_792 = None
        convert_element_type_670: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_734, torch.float32);  view_734 = None
        convert_element_type_671: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_315, torch.float32);  mm_315 = None
        convert_element_type_672: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_669, torch.float32);  convert_element_type_669 = None
        view_735: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg540_1, _shape_param_793);  arg540_1 = _shape_param_793 = None
        mul_608: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_735, 0.5)
        mul_609: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_670, mul_608);  mul_608 = None
        convert_element_type_673: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_735, torch.float32)
        pow_53: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_673, 3.0)
        mul_610: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_366: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_735, mul_610);  view_735 = mul_610 = None
        mul_611: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_366, 0.7978845608028654);  add_366 = None
        tanh_26: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_611);  mul_611 = None
        add_367: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_26, 1.0)
        mul_612: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_670, add_367);  convert_element_type_670 = add_367 = None
        convert_element_type_674: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_612, torch.bfloat16);  mul_612 = None
        mul_613: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_26, tanh_26);  tanh_26 = None
        sub_86: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_613);  mul_613 = None
        mul_614: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_609, sub_86);  mul_609 = sub_86 = None
        mul_615: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_614, 0.7978845608028654);  mul_614 = None
        convert_element_type_675: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_615, torch.bfloat16)
        mul_616: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_615, 0.044715);  mul_615 = None
        pow_54: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_673, 2.0);  convert_element_type_673 = None
        mul_617: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_54, 3.0);  pow_54 = None
        mul_618: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_616, mul_617);  mul_616 = mul_617 = None
        convert_element_type_676: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_618, torch.bfloat16);  mul_618 = None
        add_368: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_675, convert_element_type_676);  convert_element_type_675 = convert_element_type_676 = None
        mul_619: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_674, 0.5);  convert_element_type_674 = None
        add_369: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_368, mul_619);  add_368 = mul_619 = None
        view_736: "bf16[128, 16384]" = torch.ops.aten.view.default(add_369, _shape_param_794);  add_369 = _shape_param_794 = None
        mm_316: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_736, arg541_1);  arg541_1 = None
        permute_262: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_736, [1, 0])
        mm_317: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_262, arg542_1);  permute_262 = None
        sum_193: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_736, [0], True, dtype = torch.float32);  view_736 = None
        view_737: "f32[16384]" = torch.ops.aten.view.default(sum_193, _shape_param_795);  sum_193 = _shape_param_795 = None
        convert_element_type_677: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_737, torch.bfloat16);  view_737 = None
        view_738: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_316, _shape_param_796);  mm_316 = _shape_param_796 = None
        convert_element_type_678: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_738, torch.float32);  view_738 = None
        convert_element_type_679: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_317, torch.float32);  mm_317 = None
        convert_element_type_680: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_677, torch.float32);  convert_element_type_677 = None
        mm_318: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_261, arg543_1);  permute_261 = arg543_1 = None
        mm_319: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_732, arg544_1);  view_732 = arg544_1 = None
        view_739: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_319, _shape_param_797);  mm_319 = _shape_param_797 = None
        convert_element_type_681: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_318, torch.float32);  mm_318 = None
        view_740: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_739, _shape_param_798);  view_739 = _shape_param_798 = None
        permute_263: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_740, [0, 2, 1, 3]);  view_740 = None
        view_741: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_263, _shape_param_799);  permute_263 = _shape_param_799 = None
        bmm_104: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg545_1, view_741);  arg545_1 = None
        bmm_105: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_741, arg546_1);  view_741 = arg546_1 = None
        view_742: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_104, _shape_param_800);  bmm_104 = _shape_param_800 = None
        view_743: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_105, _shape_param_801);  bmm_105 = _shape_param_801 = None
        convert_element_type_682: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_743, torch.float32);  view_743 = None
        mul_620: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_682, arg547_1);  convert_element_type_682 = None
        sum_194: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [-1], True)
        neg_78: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg547_1);  arg547_1 = None
        fma_26: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_78, sum_194, mul_620);  neg_78 = sum_194 = mul_620 = None
        convert_element_type_683: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_26, torch.bfloat16);  fma_26 = None
        div_29: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_683, 16.0);  convert_element_type_683 = None
        view_744: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_29, _shape_param_802);  div_29 = _shape_param_802 = None
        bmm_106: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg548_1, view_744);  arg548_1 = None
        bmm_107: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_744, arg549_1);  view_744 = arg549_1 = None
        view_745: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_106, _shape_param_803);  bmm_106 = _shape_param_803 = None
        view_746: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_107, _shape_param_804);  bmm_107 = _shape_param_804 = None
        convert_element_type_684: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_745, torch.float32);  view_745 = None
        permute_264: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_684, [0, 1, 3, 2]);  convert_element_type_684 = None
        convert_element_type_685: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_264, torch.bfloat16);  permute_264 = None
        convert_element_type_686: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_746, torch.bfloat16);  view_746 = None
        permute_265: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_686, [0, 2, 1, 3]);  convert_element_type_686 = None
        permute_266: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_685, [0, 2, 1, 3]);  convert_element_type_685 = None
        slice_105: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_265, 3, 0, 64)
        slice_106: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_265, 3, 64, 256);  permute_265 = None
        slice_107: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_266, 3, 0, 64)
        slice_108: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_266, 3, 64, 256);  permute_266 = None
        expand_54: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg550_1, _shape_param_805);  arg550_1 = _shape_param_805 = None
        clone_78: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_747: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_78, _shape_param_806);  clone_78 = _shape_param_806 = None
        mul_621: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_105, view_747)
        view_748: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_621, _shape_param_807);  mul_621 = _shape_param_807 = None
        select_104: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_748, -1, 0)
        select_105: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_748, -1, 1);  view_748 = None
        neg_79: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_104);  select_104 = None
        slice_scatter_208: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_79, 3, 1, 9223372036854775807, 2);  neg_79 = None
        slice_scatter_209: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_105, 3, 0, 9223372036854775807, 2);  select_105 = None
        add_370: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_208, slice_scatter_209);  slice_scatter_208 = slice_scatter_209 = None
        expand_55: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg551_1, _shape_param_808);  arg551_1 = _shape_param_808 = None
        clone_79: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_749: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_79, _shape_param_809);  clone_79 = _shape_param_809 = None
        mul_622: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_105, view_749);  slice_105 = None
        add_371: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_370, mul_622);  add_370 = mul_622 = None
        mul_623: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_107, view_747);  view_747 = None
        view_750: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_623, _shape_param_810);  mul_623 = _shape_param_810 = None
        select_106: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_750, -1, 0)
        select_107: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_750, -1, 1);  view_750 = None
        neg_80: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_106);  select_106 = None
        slice_scatter_210: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_80, 3, 1, 9223372036854775807, 2);  neg_80 = None
        slice_scatter_211: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_107, 3, 0, 9223372036854775807, 2);  select_107 = None
        add_372: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_210, slice_scatter_211);  slice_scatter_210 = slice_scatter_211 = None
        mul_624: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_107, view_749);  slice_107 = view_749 = None
        add_373: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_372, mul_624);  add_372 = mul_624 = None
        slice_scatter_212: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_106, 3, 64, 9223372036854775807);  slice_106 = None
        slice_scatter_213: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_371, 3, 0, 64);  add_371 = None
        add_374: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_212, slice_scatter_213);  slice_scatter_212 = slice_scatter_213 = None
        slice_scatter_214: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_108, 3, 64, 9223372036854775807);  slice_108 = None
        slice_scatter_215: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_373, 3, 0, 64);  add_373 = None
        add_375: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_214, slice_scatter_215);  slice_scatter_214 = slice_scatter_215 = None
        permute_267: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_742, [0, 2, 1, 3]);  view_742 = None
        clone_80: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_751: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_80, _shape_param_811);  clone_80 = _shape_param_811 = None
        view_752: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_375, _shape_param_812);  add_375 = _shape_param_812 = None
        view_753: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_374, _shape_param_813);  add_374 = _shape_param_813 = None
        view_754: "bf16[128, 4096]" = torch.ops.aten.view.default(view_751, _shape_param_814);  view_751 = _shape_param_814 = None
        permute_268: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_754, [1, 0])
        mm_320: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_268, arg542_1);  permute_268 = None
        mm_321: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_754, arg552_1);  view_754 = arg552_1 = None
        view_755: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_321, _shape_param_815);  mm_321 = _shape_param_815 = None
        convert_element_type_687: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_755, torch.float32);  view_755 = None
        add_376: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_678, convert_element_type_687);  convert_element_type_678 = convert_element_type_687 = None
        convert_element_type_688: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_320, torch.float32);  mm_320 = None
        view_756: "bf16[128, 4096]" = torch.ops.aten.view.default(view_752, _shape_param_816);  view_752 = _shape_param_816 = None
        permute_269: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_756, [1, 0])
        mm_322: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_269, arg542_1);  permute_269 = None
        mm_323: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_756, arg553_1);  view_756 = arg553_1 = None
        view_757: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_323, _shape_param_817);  mm_323 = _shape_param_817 = None
        convert_element_type_689: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_757, torch.float32);  view_757 = None
        add_377: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_376, convert_element_type_689);  add_376 = convert_element_type_689 = None
        convert_element_type_690: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_322, torch.float32);  mm_322 = None
        view_758: "bf16[128, 4096]" = torch.ops.aten.view.default(view_753, _shape_param_818);  view_753 = _shape_param_818 = None
        permute_270: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_758, [1, 0])
        mm_324: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_270, arg542_1);  permute_270 = arg542_1 = None
        mm_325: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_758, arg554_1);  view_758 = arg554_1 = None
        view_759: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_325, _shape_param_819);  mm_325 = _shape_param_819 = None
        convert_element_type_691: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_759, torch.float32);  view_759 = None
        add_378: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_377, convert_element_type_691);  add_377 = convert_element_type_691 = None
        convert_element_type_692: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_324, torch.float32);  mm_324 = None
        mul_625: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_378, arg555_1);  arg555_1 = None
        mul_626: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_625, 4096)
        sum_195: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_625, [2], True)
        add_379: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(arg556_1, arg557_1);  arg556_1 = None
        sub_87: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_379, arg558_1);  add_379 = arg558_1 = None
        mul_627: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_87, arg559_1);  sub_87 = None
        mul_628: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_625, mul_627);  mul_625 = None
        sum_196: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_628, [2], True);  mul_628 = None
        mul_629: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_627, sum_196);  sum_196 = None
        sub_88: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_626, sum_195);  mul_626 = sum_195 = None
        sub_89: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_88, mul_629);  sub_88 = mul_629 = None
        div_30: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(arg559_1, 4096);  arg559_1 = None
        mul_630: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_30, sub_89);  div_30 = sub_89 = None
        mul_631: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_378, mul_627);  mul_627 = None
        sum_197: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_631, [0, 1]);  mul_631 = None
        sum_198: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_378, [0, 1]);  add_378 = None
        add_380: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_365, mul_630);  add_365 = mul_630 = None
        convert_element_type_693: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_380, torch.bfloat16)
        view_760: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_693, _shape_param_820);  convert_element_type_693 = _shape_param_820 = None
        mm_326: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_760, arg560_1);  arg560_1 = None
        permute_271: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_760, [1, 0])
        mm_327: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_271, arg561_1);  arg561_1 = None
        sum_199: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_760, [0], True, dtype = torch.float32)
        view_761: "f32[4096]" = torch.ops.aten.view.default(sum_199, _shape_param_821);  sum_199 = _shape_param_821 = None
        convert_element_type_694: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_761, torch.bfloat16);  view_761 = None
        view_762: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_326, _shape_param_822);  mm_326 = _shape_param_822 = None
        convert_element_type_695: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_762, torch.float32);  view_762 = None
        convert_element_type_696: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_327, torch.float32);  mm_327 = None
        convert_element_type_697: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_694, torch.float32);  convert_element_type_694 = None
        view_763: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg562_1, _shape_param_823);  arg562_1 = _shape_param_823 = None
        mul_632: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_763, 0.5)
        mul_633: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_695, mul_632);  mul_632 = None
        convert_element_type_698: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_763, torch.float32)
        pow_55: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_698, 3.0)
        mul_634: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_55, 0.044715);  pow_55 = None
        add_381: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_763, mul_634);  view_763 = mul_634 = None
        mul_635: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_381, 0.7978845608028654);  add_381 = None
        tanh_27: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_635);  mul_635 = None
        add_382: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_27, 1.0)
        mul_636: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_695, add_382);  convert_element_type_695 = add_382 = None
        convert_element_type_699: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_636, torch.bfloat16);  mul_636 = None
        mul_637: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_27, tanh_27);  tanh_27 = None
        sub_90: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_637);  mul_637 = None
        mul_638: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_633, sub_90);  mul_633 = sub_90 = None
        mul_639: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_638, 0.7978845608028654);  mul_638 = None
        convert_element_type_700: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_639, torch.bfloat16)
        mul_640: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_639, 0.044715);  mul_639 = None
        pow_56: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_698, 2.0);  convert_element_type_698 = None
        mul_641: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_56, 3.0);  pow_56 = None
        mul_642: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_640, mul_641);  mul_640 = mul_641 = None
        convert_element_type_701: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_642, torch.bfloat16);  mul_642 = None
        add_383: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_700, convert_element_type_701);  convert_element_type_700 = convert_element_type_701 = None
        mul_643: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_699, 0.5);  convert_element_type_699 = None
        add_384: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_383, mul_643);  add_383 = mul_643 = None
        view_764: "bf16[128, 16384]" = torch.ops.aten.view.default(add_384, _shape_param_824);  add_384 = _shape_param_824 = None
        mm_328: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_764, arg563_1);  arg563_1 = None
        permute_272: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_764, [1, 0])
        mm_329: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_272, arg564_1);  permute_272 = None
        sum_200: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_764, [0], True, dtype = torch.float32);  view_764 = None
        view_765: "f32[16384]" = torch.ops.aten.view.default(sum_200, _shape_param_825);  sum_200 = _shape_param_825 = None
        convert_element_type_702: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_765, torch.bfloat16);  view_765 = None
        view_766: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_328, _shape_param_826);  mm_328 = _shape_param_826 = None
        convert_element_type_703: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_766, torch.float32);  view_766 = None
        convert_element_type_704: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_329, torch.float32);  mm_329 = None
        convert_element_type_705: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_702, torch.float32);  convert_element_type_702 = None
        mm_330: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_271, arg565_1);  permute_271 = arg565_1 = None
        mm_331: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_760, arg566_1);  view_760 = arg566_1 = None
        view_767: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_331, _shape_param_827);  mm_331 = _shape_param_827 = None
        convert_element_type_706: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_330, torch.float32);  mm_330 = None
        view_768: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_767, _shape_param_828);  view_767 = _shape_param_828 = None
        permute_273: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_768, [0, 2, 1, 3]);  view_768 = None
        view_769: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_273, _shape_param_829);  permute_273 = _shape_param_829 = None
        bmm_108: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg567_1, view_769);  arg567_1 = None
        bmm_109: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_769, arg568_1);  view_769 = arg568_1 = None
        view_770: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_108, _shape_param_830);  bmm_108 = _shape_param_830 = None
        view_771: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_109, _shape_param_831);  bmm_109 = _shape_param_831 = None
        convert_element_type_707: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_771, torch.float32);  view_771 = None
        mul_644: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_707, arg569_1);  convert_element_type_707 = None
        sum_201: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [-1], True)
        neg_81: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg569_1);  arg569_1 = None
        fma_27: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_81, sum_201, mul_644);  neg_81 = sum_201 = mul_644 = None
        convert_element_type_708: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_27, torch.bfloat16);  fma_27 = None
        div_31: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_708, 16.0);  convert_element_type_708 = None
        view_772: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_31, _shape_param_832);  div_31 = _shape_param_832 = None
        bmm_110: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg570_1, view_772);  arg570_1 = None
        bmm_111: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_772, arg571_1);  view_772 = arg571_1 = None
        view_773: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_110, _shape_param_833);  bmm_110 = _shape_param_833 = None
        view_774: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_111, _shape_param_834);  bmm_111 = _shape_param_834 = None
        convert_element_type_709: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_773, torch.float32);  view_773 = None
        permute_274: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_709, [0, 1, 3, 2]);  convert_element_type_709 = None
        convert_element_type_710: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_274, torch.bfloat16);  permute_274 = None
        convert_element_type_711: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_774, torch.bfloat16);  view_774 = None
        permute_275: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_711, [0, 2, 1, 3]);  convert_element_type_711 = None
        permute_276: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_710, [0, 2, 1, 3]);  convert_element_type_710 = None
        slice_109: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_275, 3, 0, 64)
        slice_110: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_275, 3, 64, 256);  permute_275 = None
        slice_111: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_276, 3, 0, 64)
        slice_112: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_276, 3, 64, 256);  permute_276 = None
        expand_56: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg572_1, _shape_param_835);  arg572_1 = _shape_param_835 = None
        clone_81: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_775: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_81, _shape_param_836);  clone_81 = _shape_param_836 = None
        mul_645: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_109, view_775)
        view_776: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_645, _shape_param_837);  mul_645 = _shape_param_837 = None
        select_108: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_776, -1, 0)
        select_109: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_776, -1, 1);  view_776 = None
        neg_82: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_108);  select_108 = None
        slice_scatter_216: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_82, 3, 1, 9223372036854775807, 2);  neg_82 = None
        slice_scatter_217: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_109, 3, 0, 9223372036854775807, 2);  select_109 = None
        add_385: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_216, slice_scatter_217);  slice_scatter_216 = slice_scatter_217 = None
        expand_57: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg573_1, _shape_param_838);  arg573_1 = _shape_param_838 = None
        clone_82: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_777: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_82, _shape_param_839);  clone_82 = _shape_param_839 = None
        mul_646: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_109, view_777);  slice_109 = None
        add_386: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_385, mul_646);  add_385 = mul_646 = None
        mul_647: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_111, view_775);  view_775 = None
        view_778: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_647, _shape_param_840);  mul_647 = _shape_param_840 = None
        select_110: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_778, -1, 0)
        select_111: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_778, -1, 1);  view_778 = None
        neg_83: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_110);  select_110 = None
        slice_scatter_218: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_83, 3, 1, 9223372036854775807, 2);  neg_83 = None
        slice_scatter_219: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_111, 3, 0, 9223372036854775807, 2);  full_2 = select_111 = None
        add_387: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_218, slice_scatter_219);  slice_scatter_218 = slice_scatter_219 = None
        mul_648: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_111, view_777);  slice_111 = view_777 = None
        add_388: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_387, mul_648);  add_387 = mul_648 = None
        slice_scatter_220: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_110, 3, 64, 9223372036854775807);  slice_110 = None
        slice_scatter_221: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_386, 3, 0, 64);  add_386 = None
        add_389: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_220, slice_scatter_221);  slice_scatter_220 = slice_scatter_221 = None
        slice_scatter_222: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_112, 3, 64, 9223372036854775807);  slice_112 = None
        slice_scatter_223: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_388, 3, 0, 64);  full_3 = add_388 = None
        add_390: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_222, slice_scatter_223);  slice_scatter_222 = slice_scatter_223 = None
        permute_277: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_770, [0, 2, 1, 3]);  view_770 = None
        clone_83: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
        view_779: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_83, _shape_param_841);  clone_83 = _shape_param_841 = None
        view_780: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_390, _shape_param_842);  add_390 = _shape_param_842 = None
        view_781: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_389, _shape_param_843);  add_389 = _shape_param_843 = None
        view_782: "bf16[128, 4096]" = torch.ops.aten.view.default(view_779, _shape_param_844);  view_779 = _shape_param_844 = None
        permute_278: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_782, [1, 0])
        mm_332: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_278, arg564_1);  permute_278 = None
        mm_333: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_782, arg574_1);  view_782 = arg574_1 = None
        view_783: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_333, _shape_param_845);  mm_333 = _shape_param_845 = None
        convert_element_type_712: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_783, torch.float32);  view_783 = None
        add_391: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_703, convert_element_type_712);  convert_element_type_703 = convert_element_type_712 = None
        convert_element_type_713: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_332, torch.float32);  mm_332 = None
        view_784: "bf16[128, 4096]" = torch.ops.aten.view.default(view_780, _shape_param_846);  view_780 = _shape_param_846 = None
        permute_279: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_784, [1, 0])
        mm_334: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_279, arg564_1);  permute_279 = None
        mm_335: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_784, arg575_1);  view_784 = arg575_1 = None
        view_785: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_335, _shape_param_847);  mm_335 = _shape_param_847 = None
        convert_element_type_714: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_785, torch.float32);  view_785 = None
        add_392: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_391, convert_element_type_714);  add_391 = convert_element_type_714 = None
        convert_element_type_715: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_334, torch.float32);  mm_334 = None
        view_786: "bf16[128, 4096]" = torch.ops.aten.view.default(view_781, _shape_param_848);  view_781 = _shape_param_848 = None
        permute_280: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_786, [1, 0])
        mm_336: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_280, arg564_1);  permute_280 = arg564_1 = None
        mm_337: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_786, arg576_1);  view_786 = arg576_1 = None
        view_787: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_337, _shape_param_849);  mm_337 = _shape_param_849 = None
        convert_element_type_716: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_787, torch.float32);  view_787 = None
        add_393: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_392, convert_element_type_716);  add_392 = convert_element_type_716 = None
        convert_element_type_717: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_336, torch.float32);  mm_336 = None
        mul_649: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_393, arg577_1);  arg577_1 = None
        mul_650: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_649, 4096)
        sum_202: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [2], True)
        sub_91: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(arg557_1, arg578_1);  arg557_1 = arg578_1 = None
        mul_651: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_91, arg579_1);  sub_91 = None
        mul_652: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_649, mul_651);  mul_649 = None
        sum_203: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_652, [2], True);  mul_652 = None
        mul_653: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_651, sum_203);  sum_203 = None
        sub_92: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_650, sum_202);  mul_650 = sum_202 = None
        sub_93: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_92, mul_653);  sub_92 = mul_653 = None
        div_32: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(arg579_1, 4096);  arg579_1 = None
        mul_654: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_32, sub_93);  div_32 = sub_93 = None
        mul_655: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_393, mul_651);  mul_651 = None
        sum_204: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_655, [0, 1]);  mul_655 = None
        sum_205: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_393, [0, 1]);  add_393 = None
        add_394: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_380, mul_654);  add_380 = mul_654 = None
        ge: "b8[1, 128]" = torch.ops.aten.ge.Scalar(arg580_1, 0)
        lt: "b8[1, 128]" = torch.ops.aten.lt.Scalar(arg580_1, 50400)
        bitwise_and: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_2: "b8[1, 128]" = torch.ops.aten.ne.Scalar(arg580_1, -1)
        bitwise_and_1: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_2);  bitwise_and = ne_2 = None
        unsqueeze_4: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_4: "f32[50400, 4096]" = torch.ops.aten.full.default(_shape_param_850, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_850 = None
        _unsafe_masked_index_put_accumulate: "f32[50400, 4096]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_4, [arg580_1], add_394);  full_4 = unsqueeze_4 = arg580_1 = add_394 = None
        return (convert_element_type_16, convert_element_type_17, sum_8, sum_9, convert_element_type_21, convert_element_type_22, convert_element_type_29, convert_element_type_30, convert_element_type_31, convert_element_type_38, convert_element_type_40, convert_element_type_42, sum_15, sum_16, convert_element_type_46, convert_element_type_47, convert_element_type_54, convert_element_type_55, convert_element_type_56, convert_element_type_63, convert_element_type_65, convert_element_type_67, sum_22, sum_23, convert_element_type_71, convert_element_type_72, convert_element_type_79, convert_element_type_80, convert_element_type_81, convert_element_type_88, convert_element_type_90, convert_element_type_92, sum_29, sum_30, convert_element_type_96, convert_element_type_97, convert_element_type_104, convert_element_type_105, convert_element_type_106, convert_element_type_113, convert_element_type_115, convert_element_type_117, sum_36, sum_37, convert_element_type_121, convert_element_type_122, convert_element_type_129, convert_element_type_130, convert_element_type_131, convert_element_type_138, convert_element_type_140, convert_element_type_142, sum_43, sum_44, convert_element_type_146, convert_element_type_147, convert_element_type_154, convert_element_type_155, convert_element_type_156, convert_element_type_163, convert_element_type_165, convert_element_type_167, sum_50, sum_51, convert_element_type_171, convert_element_type_172, convert_element_type_179, convert_element_type_180, convert_element_type_181, convert_element_type_188, convert_element_type_190, convert_element_type_192, sum_57, sum_58, convert_element_type_196, convert_element_type_197, convert_element_type_204, convert_element_type_205, convert_element_type_206, convert_element_type_213, convert_element_type_215, convert_element_type_217, sum_64, sum_65, convert_element_type_221, convert_element_type_222, convert_element_type_229, convert_element_type_230, convert_element_type_231, convert_element_type_238, convert_element_type_240, convert_element_type_242, sum_71, sum_72, convert_element_type_246, convert_element_type_247, convert_element_type_254, convert_element_type_255, convert_element_type_256, convert_element_type_263, convert_element_type_265, convert_element_type_267, sum_78, sum_79, convert_element_type_271, convert_element_type_272, convert_element_type_279, convert_element_type_280, convert_element_type_281, convert_element_type_288, convert_element_type_290, convert_element_type_292, sum_85, sum_86, convert_element_type_296, convert_element_type_297, convert_element_type_304, convert_element_type_305, convert_element_type_306, convert_element_type_313, convert_element_type_315, convert_element_type_317, sum_92, sum_93, convert_element_type_321, convert_element_type_322, convert_element_type_329, convert_element_type_330, convert_element_type_331, convert_element_type_338, convert_element_type_340, convert_element_type_342, sum_99, sum_100, convert_element_type_346, convert_element_type_347, convert_element_type_354, convert_element_type_355, convert_element_type_356, convert_element_type_363, convert_element_type_365, convert_element_type_367, sum_106, sum_107, convert_element_type_371, convert_element_type_372, convert_element_type_379, convert_element_type_380, convert_element_type_381, convert_element_type_388, convert_element_type_390, convert_element_type_392, sum_113, sum_114, convert_element_type_396, convert_element_type_397, convert_element_type_404, convert_element_type_405, convert_element_type_406, convert_element_type_413, convert_element_type_415, convert_element_type_417, sum_120, sum_121, convert_element_type_421, convert_element_type_422, convert_element_type_429, convert_element_type_430, convert_element_type_431, convert_element_type_438, convert_element_type_440, convert_element_type_442, sum_127, sum_128, convert_element_type_446, convert_element_type_447, convert_element_type_454, convert_element_type_455, convert_element_type_456, convert_element_type_463, convert_element_type_465, convert_element_type_467, sum_134, sum_135, convert_element_type_471, convert_element_type_472, convert_element_type_479, convert_element_type_480, convert_element_type_481, convert_element_type_488, convert_element_type_490, convert_element_type_492, sum_141, sum_142, convert_element_type_496, convert_element_type_497, convert_element_type_504, convert_element_type_505, convert_element_type_506, convert_element_type_513, convert_element_type_515, convert_element_type_517, sum_148, sum_149, convert_element_type_521, convert_element_type_522, convert_element_type_529, convert_element_type_530, convert_element_type_531, convert_element_type_538, convert_element_type_540, convert_element_type_542, sum_155, sum_156, convert_element_type_546, convert_element_type_547, convert_element_type_554, convert_element_type_555, convert_element_type_556, convert_element_type_563, convert_element_type_565, convert_element_type_567, sum_162, sum_163, convert_element_type_571, convert_element_type_572, convert_element_type_579, convert_element_type_580, convert_element_type_581, convert_element_type_588, convert_element_type_590, convert_element_type_592, sum_169, sum_170, convert_element_type_596, convert_element_type_597, convert_element_type_604, convert_element_type_605, convert_element_type_606, convert_element_type_613, convert_element_type_615, convert_element_type_617, sum_176, sum_177, convert_element_type_621, convert_element_type_622, convert_element_type_629, convert_element_type_630, convert_element_type_631, convert_element_type_638, convert_element_type_640, convert_element_type_642, sum_183, sum_184, convert_element_type_646, convert_element_type_647, convert_element_type_654, convert_element_type_655, convert_element_type_656, convert_element_type_663, convert_element_type_665, convert_element_type_667, sum_190, sum_191, convert_element_type_671, convert_element_type_672, convert_element_type_679, convert_element_type_680, convert_element_type_681, convert_element_type_688, convert_element_type_690, convert_element_type_692, sum_197, sum_198, convert_element_type_696, convert_element_type_697, convert_element_type_704, convert_element_type_705, convert_element_type_706, convert_element_type_713, convert_element_type_715, convert_element_type_717, sum_204, sum_205, _unsafe_masked_index_put_accumulate)



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
