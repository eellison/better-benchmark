"""
Standalone repro captured via capture_hook.
Label: T5ForConditionalGeneration
Pattern hash: 17abcf417d41
Shape hash: 294eea8b
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[8, 1024]", arg3_1: "bf16[8, 1024, 32128]", arg4_1: "f32[8192, 1]", arg5_1: "f32[8192, 1]", arg6_1: "bf16[8, 1024, 32128]", arg7_1: "bf16[8192, 512]", arg8_1: "bf16[32128, 512]", arg9_1: "b8[8, 1024, 512]", arg10_1: "f32[512]", arg11_1: "f32[8, 1024, 512]", arg12_1: "f32[8, 1024, 1]", arg13_1: "b8[8, 1024, 512]", arg14_1: "bf16[8192, 2048]", arg15_1: "bf16[512, 2048]", arg16_1: "b8[8, 1024, 2048]", arg17_1: "b8[8, 1024, 2048]", arg18_1: "bf16[8192, 512]", arg19_1: "bf16[2048, 512]", arg20_1: "f32[512]", arg21_1: "f32[8, 1024, 512]", arg22_1: "f32[8, 1024, 1]", arg23_1: "b8[8, 1024, 512]", arg24_1: "bf16[8192, 512]", arg25_1: "bf16[512, 512]", arg26_1: "bf16[64, 1024, 1024]", arg27_1: "bf16[64, 64, 1024]", arg28_1: "b8[8, 8, 1024, 1024]", arg29_1: "bf16[64, 1024, 1024]", arg30_1: "f32[8, 8, 1024, 1]", arg31_1: "f32[8, 8, 1024, 1]", arg32_1: "bf16[64, 64, 1024]", arg33_1: "bf16[64, 1024, 64]", arg34_1: "bf16[8192, 512]", arg35_1: "bf16[512, 512]", arg36_1: "f32[8, 1024, 512]", arg37_1: "bf16[512, 512]", arg38_1: "bf16[8192, 512]", arg39_1: "bf16[512, 512]", arg40_1: "f32[512]", arg41_1: "f32[8, 1024, 512]", arg42_1: "f32[8, 1024, 1]", arg43_1: "b8[8, 1024, 512]", arg44_1: "bf16[8192, 512]", arg45_1: "bf16[512, 512]", arg46_1: "bf16[64, 1024, 1024]", arg47_1: "bf16[64, 64, 1024]", arg48_1: "b8[8, 8, 1024, 1024]", arg49_1: "bf16[8, 8, 1024, 1024]", arg50_1: "f32[8, 8, 1024, 1]", arg51_1: "f32[8, 8, 1024, 1]", arg52_1: "bf16[64, 64, 1024]", arg53_1: "bf16[64, 1024, 64]", arg54_1: "bf16[8192, 512]", arg55_1: "bf16[512, 512]", arg56_1: "bf16[512, 512]", arg57_1: "bf16[512, 512]", arg58_1: "f32[512]", arg59_1: "f32[8, 1024, 512]", arg60_1: "f32[8, 1024, 1]", arg61_1: "b8[8, 1024, 512]", arg62_1: "bf16[8192, 2048]", arg63_1: "bf16[512, 2048]", arg64_1: "b8[8, 1024, 2048]", arg65_1: "b8[8, 1024, 2048]", arg66_1: "bf16[8192, 512]", arg67_1: "bf16[2048, 512]", arg68_1: "f32[512]", arg69_1: "f32[8, 1024, 512]", arg70_1: "f32[8, 1024, 1]", arg71_1: "b8[8, 1024, 512]", arg72_1: "bf16[8192, 512]", arg73_1: "bf16[512, 512]", arg74_1: "bf16[64, 1024, 1024]", arg75_1: "bf16[64, 64, 1024]", arg76_1: "b8[8, 8, 1024, 1024]", arg77_1: "bf16[64, 1024, 1024]", arg78_1: "f32[8, 8, 1024, 1]", arg79_1: "f32[8, 8, 1024, 1]", arg80_1: "bf16[64, 64, 1024]", arg81_1: "bf16[64, 1024, 64]", arg82_1: "bf16[512, 512]", arg83_1: "bf16[512, 512]", arg84_1: "bf16[8192, 512]", arg85_1: "bf16[512, 512]", arg86_1: "f32[512]", arg87_1: "f32[8, 1024, 512]", arg88_1: "f32[8, 1024, 1]", arg89_1: "b8[8, 1024, 512]", arg90_1: "bf16[8192, 512]", arg91_1: "bf16[512, 512]", arg92_1: "bf16[64, 1024, 1024]", arg93_1: "bf16[64, 64, 1024]", arg94_1: "b8[8, 8, 1024, 1024]", arg95_1: "bf16[8, 8, 1024, 1024]", arg96_1: "f32[8, 8, 1024, 1]", arg97_1: "f32[8, 8, 1024, 1]", arg98_1: "bf16[64, 64, 1024]", arg99_1: "bf16[64, 1024, 64]", arg100_1: "bf16[8192, 512]", arg101_1: "bf16[512, 512]", arg102_1: "bf16[512, 512]", arg103_1: "bf16[512, 512]", arg104_1: "f32[512]", arg105_1: "f32[8, 1024, 512]", arg106_1: "f32[8, 1024, 1]", arg107_1: "b8[8, 1024, 512]", arg108_1: "bf16[8192, 2048]", arg109_1: "bf16[512, 2048]", arg110_1: "b8[8, 1024, 2048]", arg111_1: "b8[8, 1024, 2048]", arg112_1: "bf16[8192, 512]", arg113_1: "bf16[2048, 512]", arg114_1: "f32[512]", arg115_1: "f32[8, 1024, 512]", arg116_1: "f32[8, 1024, 1]", arg117_1: "b8[8, 1024, 512]", arg118_1: "bf16[8192, 512]", arg119_1: "bf16[512, 512]", arg120_1: "bf16[64, 1024, 1024]", arg121_1: "bf16[64, 64, 1024]", arg122_1: "b8[8, 8, 1024, 1024]", arg123_1: "bf16[64, 1024, 1024]", arg124_1: "f32[8, 8, 1024, 1]", arg125_1: "f32[8, 8, 1024, 1]", arg126_1: "bf16[64, 64, 1024]", arg127_1: "bf16[64, 1024, 64]", arg128_1: "bf16[512, 512]", arg129_1: "bf16[512, 512]", arg130_1: "bf16[8192, 512]", arg131_1: "bf16[512, 512]", arg132_1: "f32[512]", arg133_1: "f32[8, 1024, 512]", arg134_1: "f32[8, 1024, 1]", arg135_1: "b8[8, 1024, 512]", arg136_1: "bf16[8192, 512]", arg137_1: "bf16[512, 512]", arg138_1: "bf16[64, 1024, 1024]", arg139_1: "bf16[64, 64, 1024]", arg140_1: "b8[8, 8, 1024, 1024]", arg141_1: "bf16[8, 8, 1024, 1024]", arg142_1: "f32[8, 8, 1024, 1]", arg143_1: "f32[8, 8, 1024, 1]", arg144_1: "bf16[64, 64, 1024]", arg145_1: "bf16[64, 1024, 64]", arg146_1: "bf16[8192, 512]", arg147_1: "bf16[512, 512]", arg148_1: "bf16[512, 512]", arg149_1: "bf16[512, 512]", arg150_1: "f32[512]", arg151_1: "f32[8, 1024, 512]", arg152_1: "f32[8, 1024, 1]", arg153_1: "b8[8, 1024, 512]", arg154_1: "bf16[8192, 2048]", arg155_1: "bf16[512, 2048]", arg156_1: "b8[8, 1024, 2048]", arg157_1: "b8[8, 1024, 2048]", arg158_1: "bf16[8192, 512]", arg159_1: "bf16[2048, 512]", arg160_1: "f32[512]", arg161_1: "f32[8, 1024, 512]", arg162_1: "f32[8, 1024, 1]", arg163_1: "b8[8, 1024, 512]", arg164_1: "bf16[8192, 512]", arg165_1: "bf16[512, 512]", arg166_1: "bf16[64, 1024, 1024]", arg167_1: "bf16[64, 64, 1024]", arg168_1: "b8[8, 8, 1024, 1024]", arg169_1: "bf16[64, 1024, 1024]", arg170_1: "f32[8, 8, 1024, 1]", arg171_1: "f32[8, 8, 1024, 1]", arg172_1: "bf16[64, 64, 1024]", arg173_1: "bf16[64, 1024, 64]", arg174_1: "bf16[512, 512]", arg175_1: "bf16[512, 512]", arg176_1: "bf16[8192, 512]", arg177_1: "bf16[512, 512]", arg178_1: "f32[512]", arg179_1: "f32[8, 1024, 512]", arg180_1: "f32[8, 1024, 1]", arg181_1: "b8[8, 1024, 512]", arg182_1: "bf16[8192, 512]", arg183_1: "bf16[512, 512]", arg184_1: "bf16[64, 1024, 1024]", arg185_1: "bf16[64, 64, 1024]", arg186_1: "b8[8, 8, 1024, 1024]", arg187_1: "bf16[8, 8, 1024, 1024]", arg188_1: "f32[8, 8, 1024, 1]", arg189_1: "f32[8, 8, 1024, 1]", arg190_1: "bf16[64, 64, 1024]", arg191_1: "bf16[64, 1024, 64]", arg192_1: "bf16[8192, 512]", arg193_1: "bf16[512, 512]", arg194_1: "bf16[512, 512]", arg195_1: "bf16[512, 512]", arg196_1: "f32[512]", arg197_1: "f32[8, 1024, 512]", arg198_1: "f32[8, 1024, 1]", arg199_1: "b8[8, 1024, 512]", arg200_1: "bf16[8192, 2048]", arg201_1: "bf16[512, 2048]", arg202_1: "b8[8, 1024, 2048]", arg203_1: "b8[8, 1024, 2048]", arg204_1: "bf16[8192, 512]", arg205_1: "bf16[2048, 512]", arg206_1: "f32[512]", arg207_1: "f32[8, 1024, 512]", arg208_1: "f32[8, 1024, 1]", arg209_1: "b8[8, 1024, 512]", arg210_1: "bf16[8192, 512]", arg211_1: "bf16[512, 512]", arg212_1: "bf16[64, 1024, 1024]", arg213_1: "bf16[64, 64, 1024]", arg214_1: "b8[8, 8, 1024, 1024]", arg215_1: "bf16[64, 1024, 1024]", arg216_1: "f32[8, 8, 1024, 1]", arg217_1: "f32[8, 8, 1024, 1]", arg218_1: "bf16[64, 64, 1024]", arg219_1: "bf16[64, 1024, 64]", arg220_1: "bf16[512, 512]", arg221_1: "bf16[512, 512]", arg222_1: "bf16[8192, 512]", arg223_1: "bf16[512, 512]", arg224_1: "f32[512]", arg225_1: "f32[8, 1024, 512]", arg226_1: "f32[8, 1024, 1]", arg227_1: "b8[8, 1024, 512]", arg228_1: "bf16[8192, 512]", arg229_1: "bf16[512, 512]", arg230_1: "bf16[64, 1024, 1024]", arg231_1: "bf16[64, 64, 1024]", arg232_1: "b8[8, 8, 1024, 1024]", arg233_1: "bf16[8, 8, 1024, 1024]", arg234_1: "f32[8, 8, 1024, 1]", arg235_1: "f32[8, 8, 1024, 1]", arg236_1: "bf16[64, 64, 1024]", arg237_1: "bf16[64, 1024, 64]", arg238_1: "bf16[8192, 512]", arg239_1: "bf16[512, 512]", arg240_1: "bf16[512, 512]", arg241_1: "bf16[512, 512]", arg242_1: "f32[512]", arg243_1: "f32[8, 1024, 512]", arg244_1: "f32[8, 1024, 1]", arg245_1: "b8[8, 1024, 512]", arg246_1: "bf16[8192, 2048]", arg247_1: "bf16[512, 2048]", arg248_1: "b8[8, 1024, 2048]", arg249_1: "b8[8, 1024, 2048]", arg250_1: "bf16[8192, 512]", arg251_1: "bf16[2048, 512]", arg252_1: "f32[512]", arg253_1: "f32[8, 1024, 512]", arg254_1: "f32[8, 1024, 1]", arg255_1: "b8[8, 1024, 512]", arg256_1: "bf16[8192, 512]", arg257_1: "bf16[512, 512]", arg258_1: "bf16[64, 1024, 1024]", arg259_1: "bf16[64, 64, 1024]", arg260_1: "b8[8, 8, 1024, 1024]", arg261_1: "bf16[64, 1024, 1024]", arg262_1: "f32[8, 8, 1024, 1]", arg263_1: "f32[8, 8, 1024, 1]", arg264_1: "bf16[64, 64, 1024]", arg265_1: "bf16[64, 1024, 64]", arg266_1: "bf16[512, 512]", arg267_1: "bf16[512, 512]", arg268_1: "bf16[8192, 512]", arg269_1: "bf16[512, 512]", arg270_1: "f32[512]", arg271_1: "f32[8, 1024, 512]", arg272_1: "f32[8, 1024, 1]", arg273_1: "b8[8, 1024, 512]", arg274_1: "bf16[8192, 512]", arg275_1: "bf16[512, 512]", arg276_1: "bf16[64, 1024, 1024]", arg277_1: "bf16[64, 64, 1024]", arg278_1: "b8[8, 8, 1024, 1024]", arg279_1: "i64[1, 1, 1024]", arg280_1: "bf16[64, 1024, 1024]", arg281_1: "f32[1024, 1024, 8]", arg282_1: "f32[8, 8, 1024, 1]", arg283_1: "f32[8, 8, 1024, 1]", arg284_1: "i64[1024, 1024]", arg285_1: "bf16[64, 64, 1024]", arg286_1: "bf16[64, 1024, 64]", arg287_1: "bf16[8192, 512]", arg288_1: "bf16[512, 512]", arg289_1: "bf16[512, 512]", arg290_1: "bf16[512, 512]", arg291_1: "f32[512]", arg292_1: "b8[8, 1024, 512]", arg293_1: "f32[8, 1024, 512]", arg294_1: "f32[8, 1024, 1]", arg295_1: "i64[8, 1024]", arg296_1: "b8[8, 1024, 512]", arg297_1: "f32[512]", arg298_1: "f32[8, 1024, 512]", arg299_1: "f32[8, 1024, 1]", arg300_1: "b8[8, 1024, 512]", arg301_1: "bf16[8192, 2048]", arg302_1: "bf16[512, 2048]", arg303_1: "b8[8, 1024, 2048]", arg304_1: "b8[8, 1024, 2048]", arg305_1: "bf16[8192, 512]", arg306_1: "bf16[2048, 512]", arg307_1: "f32[512]", arg308_1: "f32[8, 1024, 512]", arg309_1: "f32[8, 1024, 1]", arg310_1: "b8[8, 1024, 512]", arg311_1: "bf16[8192, 512]", arg312_1: "bf16[512, 512]", arg313_1: "bf16[64, 1024, 1024]", arg314_1: "bf16[64, 64, 1024]", arg315_1: "b8[8, 8, 1024, 1024]", arg316_1: "bf16[8, 8, 1024, 1024]", arg317_1: "f32[8, 8, 1024, 1]", arg318_1: "f32[8, 8, 1024, 1]", arg319_1: "bf16[64, 64, 1024]", arg320_1: "bf16[64, 1024, 64]", arg321_1: "bf16[8192, 512]", arg322_1: "bf16[512, 512]", arg323_1: "bf16[512, 512]", arg324_1: "bf16[512, 512]", arg325_1: "f32[512]", arg326_1: "f32[8, 1024, 512]", arg327_1: "f32[8, 1024, 1]", arg328_1: "b8[8, 1024, 512]", arg329_1: "bf16[8192, 2048]", arg330_1: "bf16[512, 2048]", arg331_1: "b8[8, 1024, 2048]", arg332_1: "b8[8, 1024, 2048]", arg333_1: "bf16[8192, 512]", arg334_1: "bf16[2048, 512]", arg335_1: "f32[512]", arg336_1: "f32[8, 1024, 512]", arg337_1: "f32[8, 1024, 1]", arg338_1: "b8[8, 1024, 512]", arg339_1: "bf16[8192, 512]", arg340_1: "bf16[512, 512]", arg341_1: "bf16[64, 1024, 1024]", arg342_1: "bf16[64, 64, 1024]", arg343_1: "b8[8, 8, 1024, 1024]", arg344_1: "bf16[8, 8, 1024, 1024]", arg345_1: "f32[8, 8, 1024, 1]", arg346_1: "f32[8, 8, 1024, 1]", arg347_1: "bf16[64, 64, 1024]", arg348_1: "bf16[64, 1024, 64]", arg349_1: "bf16[8192, 512]", arg350_1: "bf16[512, 512]", arg351_1: "bf16[512, 512]", arg352_1: "bf16[512, 512]", arg353_1: "f32[512]", arg354_1: "f32[8, 1024, 512]", arg355_1: "f32[8, 1024, 1]", arg356_1: "b8[8, 1024, 512]", arg357_1: "bf16[8192, 2048]", arg358_1: "bf16[512, 2048]", arg359_1: "b8[8, 1024, 2048]", arg360_1: "b8[8, 1024, 2048]", arg361_1: "bf16[8192, 512]", arg362_1: "bf16[2048, 512]", arg363_1: "f32[512]", arg364_1: "f32[8, 1024, 512]", arg365_1: "f32[8, 1024, 1]", arg366_1: "b8[8, 1024, 512]", arg367_1: "bf16[8192, 512]", arg368_1: "bf16[512, 512]", arg369_1: "bf16[64, 1024, 1024]", arg370_1: "bf16[64, 64, 1024]", arg371_1: "b8[8, 8, 1024, 1024]", arg372_1: "bf16[8, 8, 1024, 1024]", arg373_1: "f32[8, 8, 1024, 1]", arg374_1: "f32[8, 8, 1024, 1]", arg375_1: "bf16[64, 64, 1024]", arg376_1: "bf16[64, 1024, 64]", arg377_1: "bf16[8192, 512]", arg378_1: "bf16[512, 512]", arg379_1: "bf16[512, 512]", arg380_1: "bf16[512, 512]", arg381_1: "f32[512]", arg382_1: "f32[8, 1024, 512]", arg383_1: "f32[8, 1024, 1]", arg384_1: "b8[8, 1024, 512]", arg385_1: "bf16[8192, 2048]", arg386_1: "bf16[512, 2048]", arg387_1: "b8[8, 1024, 2048]", arg388_1: "b8[8, 1024, 2048]", arg389_1: "bf16[8192, 512]", arg390_1: "bf16[2048, 512]", arg391_1: "f32[512]", arg392_1: "f32[8, 1024, 512]", arg393_1: "f32[8, 1024, 1]", arg394_1: "b8[8, 1024, 512]", arg395_1: "bf16[8192, 512]", arg396_1: "bf16[512, 512]", arg397_1: "bf16[64, 1024, 1024]", arg398_1: "bf16[64, 64, 1024]", arg399_1: "b8[8, 8, 1024, 1024]", arg400_1: "bf16[8, 8, 1024, 1024]", arg401_1: "f32[8, 8, 1024, 1]", arg402_1: "f32[8, 8, 1024, 1]", arg403_1: "bf16[64, 64, 1024]", arg404_1: "bf16[64, 1024, 64]", arg405_1: "bf16[8192, 512]", arg406_1: "bf16[512, 512]", arg407_1: "bf16[512, 512]", arg408_1: "bf16[512, 512]", arg409_1: "f32[512]", arg410_1: "f32[8, 1024, 512]", arg411_1: "f32[8, 1024, 1]", arg412_1: "b8[8, 1024, 512]", arg413_1: "bf16[8192, 2048]", arg414_1: "bf16[512, 2048]", arg415_1: "b8[8, 1024, 2048]", arg416_1: "b8[8, 1024, 2048]", arg417_1: "bf16[8192, 512]", arg418_1: "bf16[2048, 512]", arg419_1: "f32[512]", arg420_1: "f32[8, 1024, 512]", arg421_1: "f32[8, 1024, 1]", arg422_1: "b8[8, 1024, 512]", arg423_1: "bf16[8192, 512]", arg424_1: "bf16[512, 512]", arg425_1: "bf16[64, 1024, 1024]", arg426_1: "bf16[64, 64, 1024]", arg427_1: "b8[8, 8, 1024, 1024]", arg428_1: "bf16[8, 8, 1024, 1024]", arg429_1: "f32[8, 8, 1024, 1]", arg430_1: "f32[8, 8, 1024, 1]", arg431_1: "bf16[64, 64, 1024]", arg432_1: "bf16[64, 1024, 64]", arg433_1: "bf16[8192, 512]", arg434_1: "bf16[512, 512]", arg435_1: "bf16[512, 512]", arg436_1: "bf16[512, 512]", arg437_1: "f32[512]", arg438_1: "f32[8, 1024, 512]", arg439_1: "f32[8, 1024, 1]", arg440_1: "b8[8, 1024, 512]", arg441_1: "bf16[8192, 2048]", arg442_1: "bf16[512, 2048]", arg443_1: "b8[8, 1024, 2048]", arg444_1: "b8[8, 1024, 2048]", arg445_1: "bf16[8192, 512]", arg446_1: "bf16[2048, 512]", arg447_1: "f32[512]", arg448_1: "f32[8, 1024, 512]", arg449_1: "f32[8, 1024, 1]", arg450_1: "b8[8, 1024, 512]", arg451_1: "bf16[8192, 512]", arg452_1: "bf16[512, 512]", arg453_1: "bf16[64, 1024, 1024]", arg454_1: "bf16[64, 64, 1024]", arg455_1: "b8[8, 8, 1024, 1024]", arg456_1: "bf16[64, 1024, 1024]", arg457_1: "f32[1024, 1024, 8]", arg458_1: "f32[8, 8, 1024, 1]", arg459_1: "f32[8, 8, 1024, 1]", arg460_1: "i64[1024, 1024]", arg461_1: "bf16[64, 64, 1024]", arg462_1: "bf16[64, 1024, 64]", arg463_1: "bf16[8192, 512]", arg464_1: "bf16[512, 512]", arg465_1: "bf16[512, 512]", arg466_1: "bf16[512, 512]", arg467_1: "f32[512]", arg468_1: "b8[8, 1024, 512]", arg469_1: "f32[8, 1024, 1]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        view: "i64[8192]" = torch.ops.aten.view.default(arg2_1, [-1]);  arg2_1 = None
        unsqueeze: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[32128]" = torch.ops.prims.iota.default(32128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 32128]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[8192, 32128]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[8192, 32128]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[8192, 32128]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[8192, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = None
        mul: "f32[8192, 32128]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "bf16[8192, 32128]" = torch.ops.prims.convert_element_type.default(mul, torch.bfloat16);  mul = None
        convert_element_type_1: "f32[8192, 32128]" = torch.ops.prims.convert_element_type.default(convert_element_type, torch.float32);  convert_element_type = None
        view_2: "bf16[8192, 32128]" = torch.ops.aten.view.default(arg3_1, _shape_param_2);  arg3_1 = _shape_param_2 = None
        convert_element_type_2: "f32[8192, 32128]" = torch.ops.prims.convert_element_type.default(view_2, torch.float32);  view_2 = None
        sub: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, arg4_1);  convert_element_type_2 = arg4_1 = None
        sub_1: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        convert_element_type_3: "bf16[8192, 32128]" = torch.ops.prims.convert_element_type.default(sub_1, torch.bfloat16);  sub_1 = None
        convert_element_type_4: "f32[8192, 32128]" = torch.ops.prims.convert_element_type.default(convert_element_type_3, torch.float32);  convert_element_type_3 = None
        exp: "f32[8192, 32128]" = torch.ops.aten.exp.default(convert_element_type_4);  convert_element_type_4 = None
        sum_1: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(convert_element_type_1, [1], True)
        mul_1: "f32[8192, 32128]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[8192, 32128]" = torch.ops.aten.sub.Tensor(convert_element_type_1, mul_1);  convert_element_type_1 = mul_1 = None
        convert_element_type_5: "bf16[8192, 32128]" = torch.ops.prims.convert_element_type.default(sub_2, torch.bfloat16);  sub_2 = None
        view_3: "bf16[8, 1024, 32128]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        add: "bf16[8, 1024, 32128]" = torch.ops.aten.add.Tensor(arg6_1, view_3);  arg6_1 = view_3 = None
        view_4: "bf16[8192, 32128]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        permute: "bf16[32128, 8192]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm: "bf16[32128, 512]" = torch.ops.aten.mm.default(permute, arg7_1);  permute = arg7_1 = None
        mm_1: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_4, arg8_1);  view_4 = arg8_1 = None
        view_5: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_1, _shape_param_5);  mm_1 = _shape_param_5 = None
        convert_element_type_6: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        convert_element_type_7: "f32[32128, 512]" = torch.ops.prims.convert_element_type.default(mm, torch.float32);  mm = None
        mul_2: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 0.04419417382415922);  convert_element_type_6 = None
        convert_element_type_8: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.float32);  arg9_1 = None
        mul_3: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_4: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_5: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_4, arg10_1);  arg10_1 = None
        mul_6: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg11_1, arg12_1)
        mul_7: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_4, mul_6);  mul_4 = mul_6 = None
        sum_2: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1], True, dtype = torch.float32);  mul_7 = None
        view_6: "f32[512]" = torch.ops.aten.view.default(sum_2, _shape_param_6);  sum_2 = _shape_param_6 = None
        mul_8: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_5, arg11_1)
        mul_9: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_5, arg12_1);  mul_5 = None
        sum_3: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_8, [2], True, dtype = torch.float32);  mul_8 = None
        pow_1: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg12_1, 3);  arg12_1 = None
        mul_10: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_3, -0.5);  sum_3 = None
        mul_11: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_10, pow_1);  mul_10 = pow_1 = None
        expand_1: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_11, _shape_param_7);  mul_11 = _shape_param_7 = None
        div_1: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_1, 512);  expand_1 = None
        pow_2: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg11_1, 1.0);  arg11_1 = None
        mul_12: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_2, 2.0);  pow_2 = None
        mul_13: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_1, mul_12);  div_1 = mul_12 = None
        add_1: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_9, mul_13);  mul_9 = mul_13 = None
        convert_element_type_9: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16)
        convert_element_type_10: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        mul_14: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_15: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, mul_14);  convert_element_type_9 = mul_14 = None
        view_7: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_15, _shape_param_8);  mul_15 = _shape_param_8 = None
        permute_1: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_7, [1, 0])
        mm_2: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_1, arg14_1);  permute_1 = arg14_1 = None
        mm_3: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_7, arg15_1);  view_7 = arg15_1 = None
        view_8: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_3, _shape_param_9);  mm_3 = _shape_param_9 = None
        convert_element_type_11: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_2, torch.float32);  mm_2 = None
        convert_element_type_12: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        convert_element_type_13: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg16_1, torch.bfloat16);  arg16_1 = None
        mul_16: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_17: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_12, mul_16);  convert_element_type_12 = mul_16 = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg17_1, full_2, mul_17);  arg17_1 = mul_17 = None
        view_9: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_3, _shape_param_10);  where_3 = _shape_param_10 = None
        permute_2: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_9, [1, 0])
        mm_4: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_2, arg18_1);  permute_2 = arg18_1 = None
        mm_5: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_9, arg19_1);  view_9 = arg19_1 = None
        view_10: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_5, _shape_param_11);  mm_5 = _shape_param_11 = None
        convert_element_type_14: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32);  view_10 = None
        convert_element_type_15: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_4, torch.float32);  mm_4 = None
        mul_18: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_14, arg20_1);  arg20_1 = None
        mul_19: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg21_1, arg22_1)
        mul_20: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_14, mul_19);  convert_element_type_14 = mul_19 = None
        sum_4: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_20, [0, 1], True, dtype = torch.float32);  mul_20 = None
        view_11: "f32[512]" = torch.ops.aten.view.default(sum_4, _shape_param_12);  sum_4 = _shape_param_12 = None
        mul_21: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_18, arg21_1)
        mul_22: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_18, arg22_1);  mul_18 = None
        sum_5: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_21, [2], True, dtype = torch.float32);  mul_21 = None
        add_2: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_1, mul_22);  add_1 = mul_22 = None
        pow_3: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg22_1, 3);  arg22_1 = None
        mul_23: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_5, -0.5);  sum_5 = None
        mul_24: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_23, pow_3);  mul_23 = pow_3 = None
        expand_2: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_24, _shape_param_13);  mul_24 = _shape_param_13 = None
        div_2: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_2, 512);  expand_2 = None
        pow_4: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg21_1, 1.0);  arg21_1 = None
        mul_25: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_4, 2.0);  pow_4 = None
        mul_26: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_2, mul_25);  div_2 = mul_25 = None
        add_3: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_2, mul_26);  add_2 = mul_26 = None
        convert_element_type_16: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_3, torch.bfloat16)
        convert_element_type_17: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        mul_27: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_28: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_16, mul_27);  convert_element_type_16 = mul_27 = None
        view_12: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_28, _shape_param_14);  mul_28 = _shape_param_14 = None
        permute_3: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_12, [1, 0])
        mm_6: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_3, arg24_1);  permute_3 = arg24_1 = None
        mm_7: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_12, arg25_1);  view_12 = arg25_1 = None
        view_13: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_7, _shape_param_15);  mm_7 = _shape_param_15 = None
        convert_element_type_18: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        view_14: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_13, _shape_param_16);  view_13 = _shape_param_16 = None
        permute_4: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_15: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone, _shape_param_17);  clone = _shape_param_17 = None
        bmm: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg26_1, view_15);  arg26_1 = None
        bmm_1: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_15, arg27_1);  view_15 = arg27_1 = None
        view_16: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm, _shape_param_18);  bmm = _shape_param_18 = None
        view_17: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_1, _shape_param_19);  bmm_1 = _shape_param_19 = None
        convert_element_type_19: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        mul_29: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_30: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_17, mul_29);  view_17 = mul_29 = None
        convert_element_type_20: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_30, torch.float32);  mul_30 = None
        view_18: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg29_1, _shape_param_20);  arg29_1 = _shape_param_20 = None
        convert_element_type_21: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_18, torch.bfloat16);  view_18 = None
        convert_element_type_22: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_21, torch.float32);  convert_element_type_21 = None
        sub_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_22, arg30_1);  convert_element_type_22 = arg30_1 = None
        exp_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        div_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_1, arg31_1);  exp_1 = arg31_1 = None
        mul_31: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, div_3);  convert_element_type_20 = None
        sum_6: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_31, [-1], True)
        neg: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg, sum_6, mul_31);  neg = sum_6 = mul_31 = None
        convert_element_type_23: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        view_19: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_23, _shape_param_21);  convert_element_type_23 = _shape_param_21 = None
        bmm_2: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg32_1, view_19);  arg32_1 = None
        bmm_3: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_19, arg33_1);  view_19 = arg33_1 = None
        view_20: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_2, _shape_param_22);  bmm_2 = _shape_param_22 = None
        view_21: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_3, _shape_param_23);  bmm_3 = _shape_param_23 = None
        permute_5: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_20, [0, 1, 3, 2]);  view_20 = None
        permute_6: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_1: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_22: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_1, _shape_param_24);  clone_1 = _shape_param_24 = None
        view_23: "bf16[8192, 512]" = torch.ops.aten.view.default(view_22, _shape_param_25);  view_22 = _shape_param_25 = None
        permute_7: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_23, [1, 0])
        mm_8: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_7, arg34_1);  permute_7 = None
        mm_9: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_23, arg35_1);  view_23 = arg35_1 = None
        view_24: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_9, _shape_param_26);  mm_9 = _shape_param_26 = None
        convert_element_type_24: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_24, torch.float32);  view_24 = None
        add_4: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(arg36_1, convert_element_type_24);  arg36_1 = convert_element_type_24 = None
        convert_element_type_25: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        permute_8: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_5, [0, 2, 1, 3]);  permute_5 = None
        view_25: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_8, _shape_param_27);  permute_8 = _shape_param_27 = None
        clone_2: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_25, memory_format = torch.contiguous_format);  view_25 = None
        view_26: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_2, _shape_param_28);  clone_2 = _shape_param_28 = None
        permute_9: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_26, [1, 0])
        mm_10: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_9, arg34_1);  permute_9 = None
        mm_11: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_26, arg37_1);  view_26 = arg37_1 = None
        view_27: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_11, _shape_param_29);  mm_11 = _shape_param_29 = None
        convert_element_type_26: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        add_5: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_4, convert_element_type_26);  add_4 = convert_element_type_26 = None
        convert_element_type_27: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        permute_10: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_21, [0, 2, 1, 3]);  view_21 = None
        clone_3: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        view_28: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_3, _shape_param_30);  clone_3 = _shape_param_30 = None
        view_29: "bf16[8192, 512]" = torch.ops.aten.view.default(view_28, _shape_param_31);  view_28 = _shape_param_31 = None
        permute_11: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_29, [1, 0])
        mm_12: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_11, arg38_1);  permute_11 = arg38_1 = None
        mm_13: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_29, arg39_1);  view_29 = arg39_1 = None
        view_30: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_13, _shape_param_32);  mm_13 = _shape_param_32 = None
        convert_element_type_28: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        convert_element_type_29: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        mul_32: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_28, arg40_1);  arg40_1 = None
        mul_33: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg41_1, arg42_1)
        mul_34: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_28, mul_33);  convert_element_type_28 = mul_33 = None
        sum_7: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_34, [0, 1], True, dtype = torch.float32);  mul_34 = None
        view_31: "f32[512]" = torch.ops.aten.view.default(sum_7, _shape_param_33);  sum_7 = _shape_param_33 = None
        mul_35: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_32, arg41_1)
        mul_36: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_32, arg42_1);  mul_32 = None
        sum_8: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_35, [2], True, dtype = torch.float32);  mul_35 = None
        add_6: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_3, mul_36);  add_3 = mul_36 = None
        pow_5: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg42_1, 3);  arg42_1 = None
        mul_37: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_8, -0.5);  sum_8 = None
        mul_38: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_37, pow_5);  mul_37 = pow_5 = None
        expand_3: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_38, _shape_param_34);  mul_38 = _shape_param_34 = None
        div_4: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_3, 512);  expand_3 = None
        pow_6: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg41_1, 1.0);  arg41_1 = None
        mul_39: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_6, 2.0);  pow_6 = None
        mul_40: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_4, mul_39);  div_4 = mul_39 = None
        add_7: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_6, mul_40);  add_6 = mul_40 = None
        convert_element_type_30: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_7, torch.bfloat16)
        convert_element_type_31: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg43_1, torch.bfloat16);  arg43_1 = None
        mul_41: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_42: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_30, mul_41);  convert_element_type_30 = mul_41 = None
        view_32: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_42, _shape_param_35);  mul_42 = _shape_param_35 = None
        permute_12: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_14: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_12, arg44_1);  permute_12 = arg44_1 = None
        mm_15: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_32, arg45_1);  view_32 = arg45_1 = None
        view_33: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_15, _shape_param_36);  mm_15 = _shape_param_36 = None
        convert_element_type_32: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None
        view_34: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_33, _shape_param_37);  view_33 = _shape_param_37 = None
        permute_13: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_34, [0, 2, 1, 3]);  view_34 = None
        clone_4: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
        view_35: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_38);  clone_4 = _shape_param_38 = None
        bmm_4: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg46_1, view_35);  arg46_1 = None
        bmm_5: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_35, arg47_1);  view_35 = arg47_1 = None
        view_36: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_4, _shape_param_39);  bmm_4 = _shape_param_39 = None
        view_37: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_5, _shape_param_40);  bmm_5 = _shape_param_40 = None
        convert_element_type_33: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg48_1, torch.bfloat16);  arg48_1 = None
        mul_43: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_44: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_37, mul_43);  view_37 = mul_43 = None
        convert_element_type_34: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_44, torch.float32);  mul_44 = None
        convert_element_type_35: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg49_1, torch.float32);  arg49_1 = None
        sub_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_35, arg50_1);  convert_element_type_35 = arg50_1 = None
        exp_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        div_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_2, arg51_1);  exp_2 = arg51_1 = None
        mul_45: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, div_5);  convert_element_type_34 = None
        sum_9: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_45, [-1], True)
        neg_1: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_1: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_1, sum_9, mul_45);  neg_1 = sum_9 = mul_45 = None
        convert_element_type_36: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        view_38: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_36, _shape_param_41);  convert_element_type_36 = _shape_param_41 = None
        view_39: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_38, _shape_param_42);  view_38 = _shape_param_42 = None
        view_40: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_39, _shape_param_43);  _shape_param_43 = None
        convert_element_type_37: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_39, torch.float32);  view_39 = None
        bmm_6: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg52_1, view_40);  arg52_1 = None
        bmm_7: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_40, arg53_1);  view_40 = arg53_1 = None
        view_41: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_6, _shape_param_44);  bmm_6 = _shape_param_44 = None
        view_42: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_7, _shape_param_45);  bmm_7 = _shape_param_45 = None
        permute_14: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_41, [0, 1, 3, 2]);  view_41 = None
        permute_15: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_5: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_43: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_5, _shape_param_46);  clone_5 = _shape_param_46 = None
        view_44: "bf16[8192, 512]" = torch.ops.aten.view.default(view_43, _shape_param_47);  view_43 = _shape_param_47 = None
        permute_16: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_44, [1, 0])
        mm_16: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_16, arg54_1);  permute_16 = None
        mm_17: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_44, arg55_1);  view_44 = arg55_1 = None
        view_45: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_17, _shape_param_48);  mm_17 = _shape_param_48 = None
        convert_element_type_38: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        convert_element_type_39: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_16, torch.float32);  mm_16 = None
        permute_17: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_14, [0, 2, 1, 3]);  permute_14 = None
        view_46: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_17, _shape_param_49);  permute_17 = _shape_param_49 = None
        clone_6: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_46, memory_format = torch.contiguous_format);  view_46 = None
        view_47: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_6, _shape_param_50);  clone_6 = _shape_param_50 = None
        permute_18: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_47, [1, 0])
        mm_18: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_18, arg54_1);  permute_18 = None
        mm_19: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_47, arg56_1);  view_47 = arg56_1 = None
        view_48: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_19, _shape_param_51);  mm_19 = _shape_param_51 = None
        convert_element_type_40: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_48, torch.float32);  view_48 = None
        add_8: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_38, convert_element_type_40);  convert_element_type_38 = convert_element_type_40 = None
        convert_element_type_41: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        permute_19: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        clone_7: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        view_49: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_7, _shape_param_52);  clone_7 = _shape_param_52 = None
        view_50: "bf16[8192, 512]" = torch.ops.aten.view.default(view_49, _shape_param_53);  view_49 = _shape_param_53 = None
        permute_20: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_50, [1, 0])
        mm_20: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_20, arg54_1);  permute_20 = arg54_1 = None
        mm_21: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_50, arg57_1);  view_50 = arg57_1 = None
        view_51: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_21, _shape_param_54);  mm_21 = _shape_param_54 = None
        convert_element_type_42: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_51, torch.float32);  view_51 = None
        add_9: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_8, convert_element_type_42);  add_8 = convert_element_type_42 = None
        convert_element_type_43: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        mul_46: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_9, arg58_1);  arg58_1 = None
        mul_47: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg59_1, arg60_1)
        mul_48: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_9, mul_47);  add_9 = mul_47 = None
        sum_10: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_48, [0, 1], True, dtype = torch.float32);  mul_48 = None
        view_52: "f32[512]" = torch.ops.aten.view.default(sum_10, _shape_param_55);  sum_10 = _shape_param_55 = None
        mul_49: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_46, arg59_1)
        mul_50: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_46, arg60_1);  mul_46 = None
        sum_11: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_49, [2], True, dtype = torch.float32);  mul_49 = None
        add_10: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_7, mul_50);  add_7 = mul_50 = None
        pow_7: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg60_1, 3);  arg60_1 = None
        mul_51: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_11, -0.5);  sum_11 = None
        mul_52: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_51, pow_7);  mul_51 = pow_7 = None
        expand_4: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_52, _shape_param_56);  mul_52 = _shape_param_56 = None
        div_6: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_4, 512);  expand_4 = None
        pow_8: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg59_1, 1.0);  arg59_1 = None
        mul_53: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_8, 2.0);  pow_8 = None
        mul_54: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_6, mul_53);  div_6 = mul_53 = None
        add_11: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_10, mul_54);  add_10 = mul_54 = None
        convert_element_type_44: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16)
        convert_element_type_45: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg61_1, torch.bfloat16);  arg61_1 = None
        mul_55: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
        mul_56: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_44, mul_55);  convert_element_type_44 = mul_55 = None
        view_53: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_56, _shape_param_57);  mul_56 = _shape_param_57 = None
        permute_21: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_53, [1, 0])
        mm_22: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_21, arg62_1);  permute_21 = arg62_1 = None
        mm_23: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_53, arg63_1);  view_53 = arg63_1 = None
        view_54: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_23, _shape_param_58);  mm_23 = _shape_param_58 = None
        convert_element_type_46: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        convert_element_type_47: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_54, torch.bfloat16);  view_54 = None
        convert_element_type_48: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        mul_57: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
        mul_58: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_47, mul_57);  convert_element_type_47 = mul_57 = None
        where_4: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg65_1, full_2, mul_58);  arg65_1 = mul_58 = None
        view_55: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_4, _shape_param_59);  where_4 = _shape_param_59 = None
        permute_22: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_55, [1, 0])
        mm_24: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_22, arg66_1);  permute_22 = arg66_1 = None
        mm_25: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_55, arg67_1);  view_55 = arg67_1 = None
        view_56: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_25, _shape_param_60);  mm_25 = _shape_param_60 = None
        convert_element_type_49: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_56, torch.float32);  view_56 = None
        convert_element_type_50: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        mul_59: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_49, arg68_1);  arg68_1 = None
        mul_60: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg69_1, arg70_1)
        mul_61: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_49, mul_60);  convert_element_type_49 = mul_60 = None
        sum_12: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_61, [0, 1], True, dtype = torch.float32);  mul_61 = None
        view_57: "f32[512]" = torch.ops.aten.view.default(sum_12, _shape_param_61);  sum_12 = _shape_param_61 = None
        mul_62: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_59, arg69_1)
        mul_63: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_59, arg70_1);  mul_59 = None
        sum_13: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_62, [2], True, dtype = torch.float32);  mul_62 = None
        add_12: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_11, mul_63);  add_11 = mul_63 = None
        pow_9: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg70_1, 3);  arg70_1 = None
        mul_64: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_13, -0.5);  sum_13 = None
        mul_65: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_64, pow_9);  mul_64 = pow_9 = None
        expand_5: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_65, _shape_param_62);  mul_65 = _shape_param_62 = None
        div_7: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_5, 512);  expand_5 = None
        pow_10: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg69_1, 1.0);  arg69_1 = None
        mul_66: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_10, 2.0);  pow_10 = None
        mul_67: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_7, mul_66);  div_7 = mul_66 = None
        add_13: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_12, mul_67);  add_12 = mul_67 = None
        convert_element_type_51: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16)
        convert_element_type_52: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg71_1, torch.bfloat16);  arg71_1 = None
        mul_68: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_69: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_51, mul_68);  convert_element_type_51 = mul_68 = None
        view_58: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_69, _shape_param_63);  mul_69 = _shape_param_63 = None
        permute_23: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_58, [1, 0])
        mm_26: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_23, arg72_1);  permute_23 = arg72_1 = None
        mm_27: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_58, arg73_1);  view_58 = arg73_1 = None
        view_59: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_27, _shape_param_64);  mm_27 = _shape_param_64 = None
        convert_element_type_53: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_26, torch.float32);  mm_26 = None
        view_60: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_59, _shape_param_65);  view_59 = _shape_param_65 = None
        permute_24: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
        clone_8: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
        view_61: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_8, _shape_param_66);  clone_8 = _shape_param_66 = None
        bmm_8: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg74_1, view_61);  arg74_1 = None
        bmm_9: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_61, arg75_1);  view_61 = arg75_1 = None
        view_62: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_8, _shape_param_67);  bmm_8 = _shape_param_67 = None
        view_63: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_9, _shape_param_68);  bmm_9 = _shape_param_68 = None
        convert_element_type_54: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg76_1, torch.bfloat16);  arg76_1 = None
        mul_70: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
        mul_71: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_63, mul_70);  view_63 = mul_70 = None
        convert_element_type_55: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_71, torch.float32);  mul_71 = None
        view_64: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg77_1, _shape_param_69);  arg77_1 = _shape_param_69 = None
        convert_element_type_56: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_64, torch.bfloat16);  view_64 = None
        convert_element_type_57: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_56, torch.float32);  convert_element_type_56 = None
        sub_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_57, arg78_1);  convert_element_type_57 = arg78_1 = None
        exp_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        div_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_3, arg79_1);  exp_3 = arg79_1 = None
        mul_72: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_55, div_8);  convert_element_type_55 = None
        sum_14: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_72, [-1], True)
        neg_2: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_2: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_2, sum_14, mul_72);  neg_2 = sum_14 = mul_72 = None
        convert_element_type_58: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        view_65: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_58, _shape_param_70);  convert_element_type_58 = _shape_param_70 = None
        bmm_10: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg80_1, view_65);  arg80_1 = None
        bmm_11: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_65, arg81_1);  view_65 = arg81_1 = None
        view_66: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_10, _shape_param_71);  bmm_10 = _shape_param_71 = None
        view_67: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_11, _shape_param_72);  bmm_11 = _shape_param_72 = None
        permute_25: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_66, [0, 1, 3, 2]);  view_66 = None
        permute_26: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_62, [0, 2, 1, 3]);  view_62 = None
        clone_9: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_68: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_9, _shape_param_73);  clone_9 = _shape_param_73 = None
        view_69: "bf16[8192, 512]" = torch.ops.aten.view.default(view_68, _shape_param_74);  view_68 = _shape_param_74 = None
        permute_27: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_69, [1, 0])
        mm_28: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_27, arg34_1);  permute_27 = None
        mm_29: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_69, arg82_1);  view_69 = arg82_1 = None
        view_70: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_29, _shape_param_75);  mm_29 = _shape_param_75 = None
        convert_element_type_59: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_70, torch.float32);  view_70 = None
        add_14: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_5, convert_element_type_59);  add_5 = convert_element_type_59 = None
        convert_element_type_60: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_28, torch.float32);  mm_28 = None
        permute_28: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_25, [0, 2, 1, 3]);  permute_25 = None
        view_71: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_28, _shape_param_76);  permute_28 = _shape_param_76 = None
        clone_10: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_71, memory_format = torch.contiguous_format);  view_71 = None
        view_72: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_10, _shape_param_77);  clone_10 = _shape_param_77 = None
        permute_29: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_72, [1, 0])
        mm_30: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_29, arg34_1);  permute_29 = None
        mm_31: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_72, arg83_1);  view_72 = arg83_1 = None
        view_73: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_31, _shape_param_78);  mm_31 = _shape_param_78 = None
        convert_element_type_61: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        add_15: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_14, convert_element_type_61);  add_14 = convert_element_type_61 = None
        convert_element_type_62: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_30, torch.float32);  mm_30 = None
        permute_30: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
        clone_11: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_74: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_11, _shape_param_79);  clone_11 = _shape_param_79 = None
        view_75: "bf16[8192, 512]" = torch.ops.aten.view.default(view_74, _shape_param_80);  view_74 = _shape_param_80 = None
        permute_31: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_75, [1, 0])
        mm_32: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_31, arg84_1);  permute_31 = arg84_1 = None
        mm_33: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_75, arg85_1);  view_75 = arg85_1 = None
        view_76: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_33, _shape_param_81);  mm_33 = _shape_param_81 = None
        convert_element_type_63: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_76, torch.float32);  view_76 = None
        convert_element_type_64: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_32, torch.float32);  mm_32 = None
        mul_73: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, arg86_1);  arg86_1 = None
        mul_74: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg87_1, arg88_1)
        mul_75: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, mul_74);  convert_element_type_63 = mul_74 = None
        sum_15: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_75, [0, 1], True, dtype = torch.float32);  mul_75 = None
        view_77: "f32[512]" = torch.ops.aten.view.default(sum_15, _shape_param_82);  sum_15 = _shape_param_82 = None
        mul_76: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_73, arg87_1)
        mul_77: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_73, arg88_1);  mul_73 = None
        sum_16: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_76, [2], True, dtype = torch.float32);  mul_76 = None
        add_16: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_13, mul_77);  add_13 = mul_77 = None
        pow_11: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg88_1, 3);  arg88_1 = None
        mul_78: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_16, -0.5);  sum_16 = None
        mul_79: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_78, pow_11);  mul_78 = pow_11 = None
        expand_6: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_79, _shape_param_83);  mul_79 = _shape_param_83 = None
        div_9: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_6, 512);  expand_6 = None
        pow_12: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg87_1, 1.0);  arg87_1 = None
        mul_80: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_12, 2.0);  pow_12 = None
        mul_81: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_9, mul_80);  div_9 = mul_80 = None
        add_17: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_16, mul_81);  add_16 = mul_81 = None
        convert_element_type_65: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16)
        convert_element_type_66: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg89_1, torch.bfloat16);  arg89_1 = None
        mul_82: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_83: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_65, mul_82);  convert_element_type_65 = mul_82 = None
        view_78: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_83, _shape_param_84);  mul_83 = _shape_param_84 = None
        permute_32: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_78, [1, 0])
        mm_34: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_32, arg90_1);  permute_32 = arg90_1 = None
        mm_35: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_78, arg91_1);  view_78 = arg91_1 = None
        view_79: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_35, _shape_param_85);  mm_35 = _shape_param_85 = None
        convert_element_type_67: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        view_80: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_79, _shape_param_86);  view_79 = _shape_param_86 = None
        permute_33: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
        clone_12: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_33, memory_format = torch.contiguous_format);  permute_33 = None
        view_81: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_87);  clone_12 = _shape_param_87 = None
        bmm_12: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg92_1, view_81);  arg92_1 = None
        bmm_13: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_81, arg93_1);  view_81 = arg93_1 = None
        view_82: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_12, _shape_param_88);  bmm_12 = _shape_param_88 = None
        view_83: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_13, _shape_param_89);  bmm_13 = _shape_param_89 = None
        convert_element_type_68: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg94_1, torch.bfloat16);  arg94_1 = None
        mul_84: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_85: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_83, mul_84);  view_83 = mul_84 = None
        convert_element_type_69: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_85, torch.float32);  mul_85 = None
        convert_element_type_70: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg95_1, torch.float32);  arg95_1 = None
        sub_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_70, arg96_1);  convert_element_type_70 = arg96_1 = None
        exp_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        div_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_4, arg97_1);  exp_4 = arg97_1 = None
        mul_86: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_69, div_10);  convert_element_type_69 = None
        sum_17: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_86, [-1], True)
        neg_3: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_10);  div_10 = None
        fma_3: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_3, sum_17, mul_86);  neg_3 = sum_17 = mul_86 = None
        convert_element_type_71: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        view_84: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_71, _shape_param_90);  convert_element_type_71 = _shape_param_90 = None
        view_85: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_84, _shape_param_91);  view_84 = _shape_param_91 = None
        view_86: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_85, _shape_param_92);  _shape_param_92 = None
        convert_element_type_72: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        add_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_37, convert_element_type_72);  convert_element_type_37 = convert_element_type_72 = None
        bmm_14: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg98_1, view_86);  arg98_1 = None
        bmm_15: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_86, arg99_1);  view_86 = arg99_1 = None
        view_87: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_14, _shape_param_93);  bmm_14 = _shape_param_93 = None
        view_88: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_15, _shape_param_94);  bmm_15 = _shape_param_94 = None
        permute_34: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_87, [0, 1, 3, 2]);  view_87 = None
        permute_35: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
        clone_13: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
        view_89: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_13, _shape_param_95);  clone_13 = _shape_param_95 = None
        view_90: "bf16[8192, 512]" = torch.ops.aten.view.default(view_89, _shape_param_96);  view_89 = _shape_param_96 = None
        permute_36: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_90, [1, 0])
        mm_36: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_36, arg100_1);  permute_36 = None
        mm_37: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_90, arg101_1);  view_90 = arg101_1 = None
        view_91: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_37, _shape_param_97);  mm_37 = _shape_param_97 = None
        convert_element_type_73: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_91, torch.float32);  view_91 = None
        convert_element_type_74: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        permute_37: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_34, [0, 2, 1, 3]);  permute_34 = None
        view_92: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_37, _shape_param_98);  permute_37 = _shape_param_98 = None
        clone_14: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_92, memory_format = torch.contiguous_format);  view_92 = None
        view_93: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_14, _shape_param_99);  clone_14 = _shape_param_99 = None
        permute_38: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_93, [1, 0])
        mm_38: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_38, arg100_1);  permute_38 = None
        mm_39: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_93, arg102_1);  view_93 = arg102_1 = None
        view_94: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_39, _shape_param_100);  mm_39 = _shape_param_100 = None
        convert_element_type_75: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_94, torch.float32);  view_94 = None
        add_19: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_73, convert_element_type_75);  convert_element_type_73 = convert_element_type_75 = None
        convert_element_type_76: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_38, torch.float32);  mm_38 = None
        permute_39: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
        clone_15: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_95: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_15, _shape_param_101);  clone_15 = _shape_param_101 = None
        view_96: "bf16[8192, 512]" = torch.ops.aten.view.default(view_95, _shape_param_102);  view_95 = _shape_param_102 = None
        permute_40: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_96, [1, 0])
        mm_40: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_40, arg100_1);  permute_40 = arg100_1 = None
        mm_41: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_96, arg103_1);  view_96 = arg103_1 = None
        view_97: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_41, _shape_param_103);  mm_41 = _shape_param_103 = None
        convert_element_type_77: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None
        add_20: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_19, convert_element_type_77);  add_19 = convert_element_type_77 = None
        convert_element_type_78: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_40, torch.float32);  mm_40 = None
        mul_87: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_20, arg104_1);  arg104_1 = None
        mul_88: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg105_1, arg106_1)
        mul_89: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_20, mul_88);  add_20 = mul_88 = None
        sum_18: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_89, [0, 1], True, dtype = torch.float32);  mul_89 = None
        view_98: "f32[512]" = torch.ops.aten.view.default(sum_18, _shape_param_104);  sum_18 = _shape_param_104 = None
        mul_90: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_87, arg105_1)
        mul_91: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_87, arg106_1);  mul_87 = None
        sum_19: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_90, [2], True, dtype = torch.float32);  mul_90 = None
        add_21: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_17, mul_91);  add_17 = mul_91 = None
        pow_13: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg106_1, 3);  arg106_1 = None
        mul_92: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_19, -0.5);  sum_19 = None
        mul_93: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_92, pow_13);  mul_92 = pow_13 = None
        expand_7: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_93, _shape_param_105);  mul_93 = _shape_param_105 = None
        div_11: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_7, 512);  expand_7 = None
        pow_14: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg105_1, 1.0);  arg105_1 = None
        mul_94: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_14, 2.0);  pow_14 = None
        mul_95: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_11, mul_94);  div_11 = mul_94 = None
        add_22: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_21, mul_95);  add_21 = mul_95 = None
        convert_element_type_79: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16)
        convert_element_type_80: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.bfloat16);  arg107_1 = None
        mul_96: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 1.1111111111111112);  convert_element_type_80 = None
        mul_97: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_79, mul_96);  convert_element_type_79 = mul_96 = None
        view_99: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_97, _shape_param_106);  mul_97 = _shape_param_106 = None
        permute_41: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_99, [1, 0])
        mm_42: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_41, arg108_1);  permute_41 = arg108_1 = None
        mm_43: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_99, arg109_1);  view_99 = arg109_1 = None
        view_100: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_43, _shape_param_107);  mm_43 = _shape_param_107 = None
        convert_element_type_81: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_42, torch.float32);  mm_42 = None
        convert_element_type_82: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_100, torch.bfloat16);  view_100 = None
        convert_element_type_83: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg110_1, torch.bfloat16);  arg110_1 = None
        mul_98: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_83, 1.1111111111111112);  convert_element_type_83 = None
        mul_99: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_82, mul_98);  convert_element_type_82 = mul_98 = None
        where_5: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg111_1, full_2, mul_99);  arg111_1 = mul_99 = None
        view_101: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_5, _shape_param_108);  where_5 = _shape_param_108 = None
        permute_42: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_101, [1, 0])
        mm_44: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_42, arg112_1);  permute_42 = arg112_1 = None
        mm_45: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_101, arg113_1);  view_101 = arg113_1 = None
        view_102: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_45, _shape_param_109);  mm_45 = _shape_param_109 = None
        convert_element_type_84: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_102, torch.float32);  view_102 = None
        convert_element_type_85: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        mul_100: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_84, arg114_1);  arg114_1 = None
        mul_101: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg115_1, arg116_1)
        mul_102: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_84, mul_101);  convert_element_type_84 = mul_101 = None
        sum_20: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_102, [0, 1], True, dtype = torch.float32);  mul_102 = None
        view_103: "f32[512]" = torch.ops.aten.view.default(sum_20, _shape_param_110);  sum_20 = _shape_param_110 = None
        mul_103: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_100, arg115_1)
        mul_104: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_100, arg116_1);  mul_100 = None
        sum_21: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True, dtype = torch.float32);  mul_103 = None
        add_23: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_22, mul_104);  add_22 = mul_104 = None
        pow_15: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg116_1, 3);  arg116_1 = None
        mul_105: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_21, -0.5);  sum_21 = None
        mul_106: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_105, pow_15);  mul_105 = pow_15 = None
        expand_8: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_106, _shape_param_111);  mul_106 = _shape_param_111 = None
        div_12: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_8, 512);  expand_8 = None
        pow_16: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg115_1, 1.0);  arg115_1 = None
        mul_107: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_16, 2.0);  pow_16 = None
        mul_108: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_12, mul_107);  div_12 = mul_107 = None
        add_24: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_23, mul_108);  add_23 = mul_108 = None
        convert_element_type_86: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16)
        convert_element_type_87: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.bfloat16);  arg117_1 = None
        mul_109: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 1.1111111111111112);  convert_element_type_87 = None
        mul_110: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_86, mul_109);  convert_element_type_86 = mul_109 = None
        view_104: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_110, _shape_param_112);  mul_110 = _shape_param_112 = None
        permute_43: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_104, [1, 0])
        mm_46: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_43, arg118_1);  permute_43 = arg118_1 = None
        mm_47: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_104, arg119_1);  view_104 = arg119_1 = None
        view_105: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_47, _shape_param_113);  mm_47 = _shape_param_113 = None
        convert_element_type_88: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        view_106: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_105, _shape_param_114);  view_105 = _shape_param_114 = None
        permute_44: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
        clone_16: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
        view_107: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_115);  clone_16 = _shape_param_115 = None
        bmm_16: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg120_1, view_107);  arg120_1 = None
        bmm_17: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_107, arg121_1);  view_107 = arg121_1 = None
        view_108: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_16, _shape_param_116);  bmm_16 = _shape_param_116 = None
        view_109: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_17, _shape_param_117);  bmm_17 = _shape_param_117 = None
        convert_element_type_89: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        mul_111: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_89, 1.1111111111111112);  convert_element_type_89 = None
        mul_112: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_109, mul_111);  view_109 = mul_111 = None
        convert_element_type_90: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_112, torch.float32);  mul_112 = None
        view_110: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg123_1, _shape_param_118);  arg123_1 = _shape_param_118 = None
        convert_element_type_91: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_110, torch.bfloat16);  view_110 = None
        convert_element_type_92: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_91, torch.float32);  convert_element_type_91 = None
        sub_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_92, arg124_1);  convert_element_type_92 = arg124_1 = None
        exp_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        div_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_5, arg125_1);  exp_5 = arg125_1 = None
        mul_113: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_90, div_13);  convert_element_type_90 = None
        sum_22: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_113, [-1], True)
        neg_4: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_4: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_4, sum_22, mul_113);  neg_4 = sum_22 = mul_113 = None
        convert_element_type_93: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        view_111: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_93, _shape_param_119);  convert_element_type_93 = _shape_param_119 = None
        bmm_18: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg126_1, view_111);  arg126_1 = None
        bmm_19: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_111, arg127_1);  view_111 = arg127_1 = None
        view_112: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_18, _shape_param_120);  bmm_18 = _shape_param_120 = None
        view_113: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_19, _shape_param_121);  bmm_19 = _shape_param_121 = None
        permute_45: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_112, [0, 1, 3, 2]);  view_112 = None
        permute_46: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
        clone_17: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
        view_114: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_17, _shape_param_122);  clone_17 = _shape_param_122 = None
        view_115: "bf16[8192, 512]" = torch.ops.aten.view.default(view_114, _shape_param_123);  view_114 = _shape_param_123 = None
        permute_47: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_115, [1, 0])
        mm_48: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_47, arg34_1);  permute_47 = None
        mm_49: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_115, arg128_1);  view_115 = arg128_1 = None
        view_116: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_49, _shape_param_124);  mm_49 = _shape_param_124 = None
        convert_element_type_94: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_116, torch.float32);  view_116 = None
        add_25: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_15, convert_element_type_94);  add_15 = convert_element_type_94 = None
        convert_element_type_95: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_48, torch.float32);  mm_48 = None
        permute_48: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_45, [0, 2, 1, 3]);  permute_45 = None
        view_117: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_48, _shape_param_125);  permute_48 = _shape_param_125 = None
        clone_18: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_117, memory_format = torch.contiguous_format);  view_117 = None
        view_118: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_18, _shape_param_126);  clone_18 = _shape_param_126 = None
        permute_49: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_118, [1, 0])
        mm_50: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_49, arg34_1);  permute_49 = None
        mm_51: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_118, arg129_1);  view_118 = arg129_1 = None
        view_119: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_51, _shape_param_127);  mm_51 = _shape_param_127 = None
        convert_element_type_96: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_119, torch.float32);  view_119 = None
        add_26: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_25, convert_element_type_96);  add_25 = convert_element_type_96 = None
        convert_element_type_97: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_50, torch.float32);  mm_50 = None
        permute_50: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
        clone_19: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_120: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_19, _shape_param_128);  clone_19 = _shape_param_128 = None
        view_121: "bf16[8192, 512]" = torch.ops.aten.view.default(view_120, _shape_param_129);  view_120 = _shape_param_129 = None
        permute_51: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_121, [1, 0])
        mm_52: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_51, arg130_1);  permute_51 = arg130_1 = None
        mm_53: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_121, arg131_1);  view_121 = arg131_1 = None
        view_122: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_53, _shape_param_130);  mm_53 = _shape_param_130 = None
        convert_element_type_98: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_122, torch.float32);  view_122 = None
        convert_element_type_99: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_52, torch.float32);  mm_52 = None
        mul_114: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_98, arg132_1);  arg132_1 = None
        mul_115: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg133_1, arg134_1)
        mul_116: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_98, mul_115);  convert_element_type_98 = mul_115 = None
        sum_23: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_116, [0, 1], True, dtype = torch.float32);  mul_116 = None
        view_123: "f32[512]" = torch.ops.aten.view.default(sum_23, _shape_param_131);  sum_23 = _shape_param_131 = None
        mul_117: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_114, arg133_1)
        mul_118: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_114, arg134_1);  mul_114 = None
        sum_24: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [2], True, dtype = torch.float32);  mul_117 = None
        add_27: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_24, mul_118);  add_24 = mul_118 = None
        pow_17: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg134_1, 3);  arg134_1 = None
        mul_119: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_24, -0.5);  sum_24 = None
        mul_120: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_119, pow_17);  mul_119 = pow_17 = None
        expand_9: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_120, _shape_param_132);  mul_120 = _shape_param_132 = None
        div_14: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_9, 512);  expand_9 = None
        pow_18: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg133_1, 1.0);  arg133_1 = None
        mul_121: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_18, 2.0);  pow_18 = None
        mul_122: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_14, mul_121);  div_14 = mul_121 = None
        add_28: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_27, mul_122);  add_27 = mul_122 = None
        convert_element_type_100: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16)
        convert_element_type_101: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg135_1, torch.bfloat16);  arg135_1 = None
        mul_123: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_101, 1.1111111111111112);  convert_element_type_101 = None
        mul_124: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_100, mul_123);  convert_element_type_100 = mul_123 = None
        view_124: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_124, _shape_param_133);  mul_124 = _shape_param_133 = None
        permute_52: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_124, [1, 0])
        mm_54: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_52, arg136_1);  permute_52 = arg136_1 = None
        mm_55: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_124, arg137_1);  view_124 = arg137_1 = None
        view_125: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_55, _shape_param_134);  mm_55 = _shape_param_134 = None
        convert_element_type_102: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        view_126: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_125, _shape_param_135);  view_125 = _shape_param_135 = None
        permute_53: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        clone_20: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
        view_127: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_20, _shape_param_136);  clone_20 = _shape_param_136 = None
        bmm_20: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg138_1, view_127);  arg138_1 = None
        bmm_21: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_127, arg139_1);  view_127 = arg139_1 = None
        view_128: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_20, _shape_param_137);  bmm_20 = _shape_param_137 = None
        view_129: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_21, _shape_param_138);  bmm_21 = _shape_param_138 = None
        convert_element_type_103: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.bfloat16);  arg140_1 = None
        mul_125: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_103, 1.1111111111111112);  convert_element_type_103 = None
        mul_126: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_129, mul_125);  view_129 = mul_125 = None
        convert_element_type_104: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_126, torch.float32);  mul_126 = None
        convert_element_type_105: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg141_1, torch.float32);  arg141_1 = None
        sub_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_105, arg142_1);  convert_element_type_105 = arg142_1 = None
        exp_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        div_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_6, arg143_1);  exp_6 = arg143_1 = None
        mul_127: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_104, div_15);  convert_element_type_104 = None
        sum_25: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [-1], True)
        neg_5: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_5: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_5, sum_25, mul_127);  neg_5 = sum_25 = mul_127 = None
        convert_element_type_106: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        view_130: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_139);  convert_element_type_106 = _shape_param_139 = None
        view_131: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_130, _shape_param_140);  view_130 = _shape_param_140 = None
        view_132: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_131, _shape_param_141);  _shape_param_141 = None
        convert_element_type_107: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None
        add_29: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_18, convert_element_type_107);  add_18 = convert_element_type_107 = None
        bmm_22: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg144_1, view_132);  arg144_1 = None
        bmm_23: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_132, arg145_1);  view_132 = arg145_1 = None
        view_133: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_22, _shape_param_142);  bmm_22 = _shape_param_142 = None
        view_134: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_23, _shape_param_143);  bmm_23 = _shape_param_143 = None
        permute_54: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_133, [0, 1, 3, 2]);  view_133 = None
        permute_55: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        clone_21: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_55, memory_format = torch.contiguous_format);  permute_55 = None
        view_135: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_21, _shape_param_144);  clone_21 = _shape_param_144 = None
        view_136: "bf16[8192, 512]" = torch.ops.aten.view.default(view_135, _shape_param_145);  view_135 = _shape_param_145 = None
        permute_56: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_136, [1, 0])
        mm_56: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_56, arg146_1);  permute_56 = None
        mm_57: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_136, arg147_1);  view_136 = arg147_1 = None
        view_137: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_57, _shape_param_146);  mm_57 = _shape_param_146 = None
        convert_element_type_108: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_137, torch.float32);  view_137 = None
        convert_element_type_109: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        permute_57: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_54, [0, 2, 1, 3]);  permute_54 = None
        view_138: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_57, _shape_param_147);  permute_57 = _shape_param_147 = None
        clone_22: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_138, memory_format = torch.contiguous_format);  view_138 = None
        view_139: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_22, _shape_param_148);  clone_22 = _shape_param_148 = None
        permute_58: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_139, [1, 0])
        mm_58: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_58, arg146_1);  permute_58 = None
        mm_59: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_139, arg148_1);  view_139 = arg148_1 = None
        view_140: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_59, _shape_param_149);  mm_59 = _shape_param_149 = None
        convert_element_type_110: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_140, torch.float32);  view_140 = None
        add_30: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_108, convert_element_type_110);  convert_element_type_108 = convert_element_type_110 = None
        convert_element_type_111: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        permute_59: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
        clone_23: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_141: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_23, _shape_param_150);  clone_23 = _shape_param_150 = None
        view_142: "bf16[8192, 512]" = torch.ops.aten.view.default(view_141, _shape_param_151);  view_141 = _shape_param_151 = None
        permute_60: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_142, [1, 0])
        mm_60: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_60, arg146_1);  permute_60 = arg146_1 = None
        mm_61: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_142, arg149_1);  view_142 = arg149_1 = None
        view_143: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_61, _shape_param_152);  mm_61 = _shape_param_152 = None
        convert_element_type_112: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_143, torch.float32);  view_143 = None
        add_31: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_30, convert_element_type_112);  add_30 = convert_element_type_112 = None
        convert_element_type_113: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        mul_128: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_31, arg150_1);  arg150_1 = None
        mul_129: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg151_1, arg152_1)
        mul_130: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_31, mul_129);  add_31 = mul_129 = None
        sum_26: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_130, [0, 1], True, dtype = torch.float32);  mul_130 = None
        view_144: "f32[512]" = torch.ops.aten.view.default(sum_26, _shape_param_153);  sum_26 = _shape_param_153 = None
        mul_131: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_128, arg151_1)
        mul_132: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_128, arg152_1);  mul_128 = None
        sum_27: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True, dtype = torch.float32);  mul_131 = None
        add_32: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_28, mul_132);  add_28 = mul_132 = None
        pow_19: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg152_1, 3);  arg152_1 = None
        mul_133: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_27, -0.5);  sum_27 = None
        mul_134: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_133, pow_19);  mul_133 = pow_19 = None
        expand_10: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_134, _shape_param_154);  mul_134 = _shape_param_154 = None
        div_16: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_10, 512);  expand_10 = None
        pow_20: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg151_1, 1.0);  arg151_1 = None
        mul_135: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_20, 2.0);  pow_20 = None
        mul_136: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_16, mul_135);  div_16 = mul_135 = None
        add_33: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_32, mul_136);  add_32 = mul_136 = None
        convert_element_type_114: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16)
        convert_element_type_115: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.bfloat16);  arg153_1 = None
        mul_137: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_115, 1.1111111111111112);  convert_element_type_115 = None
        mul_138: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_114, mul_137);  convert_element_type_114 = mul_137 = None
        view_145: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_138, _shape_param_155);  mul_138 = _shape_param_155 = None
        permute_61: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_145, [1, 0])
        mm_62: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_61, arg154_1);  permute_61 = arg154_1 = None
        mm_63: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_145, arg155_1);  view_145 = arg155_1 = None
        view_146: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_63, _shape_param_156);  mm_63 = _shape_param_156 = None
        convert_element_type_116: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_62, torch.float32);  mm_62 = None
        convert_element_type_117: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_146, torch.bfloat16);  view_146 = None
        convert_element_type_118: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg156_1, torch.bfloat16);  arg156_1 = None
        mul_139: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_118, 1.1111111111111112);  convert_element_type_118 = None
        mul_140: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_117, mul_139);  convert_element_type_117 = mul_139 = None
        where_6: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg157_1, full_2, mul_140);  arg157_1 = mul_140 = None
        view_147: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_6, _shape_param_157);  where_6 = _shape_param_157 = None
        permute_62: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_147, [1, 0])
        mm_64: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_62, arg158_1);  permute_62 = arg158_1 = None
        mm_65: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_147, arg159_1);  view_147 = arg159_1 = None
        view_148: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_65, _shape_param_158);  mm_65 = _shape_param_158 = None
        convert_element_type_119: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_148, torch.float32);  view_148 = None
        convert_element_type_120: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_64, torch.float32);  mm_64 = None
        mul_141: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_119, arg160_1);  arg160_1 = None
        mul_142: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg161_1, arg162_1)
        mul_143: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_119, mul_142);  convert_element_type_119 = mul_142 = None
        sum_28: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_143, [0, 1], True, dtype = torch.float32);  mul_143 = None
        view_149: "f32[512]" = torch.ops.aten.view.default(sum_28, _shape_param_159);  sum_28 = _shape_param_159 = None
        mul_144: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_141, arg161_1)
        mul_145: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_141, arg162_1);  mul_141 = None
        sum_29: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [2], True, dtype = torch.float32);  mul_144 = None
        add_34: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_33, mul_145);  add_33 = mul_145 = None
        pow_21: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg162_1, 3);  arg162_1 = None
        mul_146: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_29, -0.5);  sum_29 = None
        mul_147: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_146, pow_21);  mul_146 = pow_21 = None
        expand_11: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_147, _shape_param_160);  mul_147 = _shape_param_160 = None
        div_17: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_11, 512);  expand_11 = None
        pow_22: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg161_1, 1.0);  arg161_1 = None
        mul_148: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_22, 2.0);  pow_22 = None
        mul_149: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_17, mul_148);  div_17 = mul_148 = None
        add_35: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_34, mul_149);  add_34 = mul_149 = None
        convert_element_type_121: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_35, torch.bfloat16)
        convert_element_type_122: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.bfloat16);  arg163_1 = None
        mul_150: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_122, 1.1111111111111112);  convert_element_type_122 = None
        mul_151: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_121, mul_150);  convert_element_type_121 = mul_150 = None
        view_150: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_151, _shape_param_161);  mul_151 = _shape_param_161 = None
        permute_63: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_150, [1, 0])
        mm_66: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_63, arg164_1);  permute_63 = arg164_1 = None
        mm_67: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_150, arg165_1);  view_150 = arg165_1 = None
        view_151: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_67, _shape_param_162);  mm_67 = _shape_param_162 = None
        convert_element_type_123: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        view_152: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_151, _shape_param_163);  view_151 = _shape_param_163 = None
        permute_64: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_152, [0, 2, 1, 3]);  view_152 = None
        clone_24: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
        view_153: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_164);  clone_24 = _shape_param_164 = None
        bmm_24: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg166_1, view_153);  arg166_1 = None
        bmm_25: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_153, arg167_1);  view_153 = arg167_1 = None
        view_154: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_24, _shape_param_165);  bmm_24 = _shape_param_165 = None
        view_155: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_25, _shape_param_166);  bmm_25 = _shape_param_166 = None
        convert_element_type_124: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.bfloat16);  arg168_1 = None
        mul_152: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_124, 1.1111111111111112);  convert_element_type_124 = None
        mul_153: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_155, mul_152);  view_155 = mul_152 = None
        convert_element_type_125: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_153, torch.float32);  mul_153 = None
        view_156: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg169_1, _shape_param_167);  arg169_1 = _shape_param_167 = None
        convert_element_type_126: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_156, torch.bfloat16);  view_156 = None
        convert_element_type_127: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_126, torch.float32);  convert_element_type_126 = None
        sub_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_127, arg170_1);  convert_element_type_127 = arg170_1 = None
        exp_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        div_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_7, arg171_1);  exp_7 = arg171_1 = None
        mul_154: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_125, div_18);  convert_element_type_125 = None
        sum_30: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [-1], True)
        neg_6: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_6: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_6, sum_30, mul_154);  neg_6 = sum_30 = mul_154 = None
        convert_element_type_128: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        view_157: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_128, _shape_param_168);  convert_element_type_128 = _shape_param_168 = None
        bmm_26: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg172_1, view_157);  arg172_1 = None
        bmm_27: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_157, arg173_1);  view_157 = arg173_1 = None
        view_158: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_26, _shape_param_169);  bmm_26 = _shape_param_169 = None
        view_159: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_27, _shape_param_170);  bmm_27 = _shape_param_170 = None
        permute_65: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_158, [0, 1, 3, 2]);  view_158 = None
        permute_66: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None
        clone_25: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
        view_160: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_25, _shape_param_171);  clone_25 = _shape_param_171 = None
        view_161: "bf16[8192, 512]" = torch.ops.aten.view.default(view_160, _shape_param_172);  view_160 = _shape_param_172 = None
        permute_67: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_161, [1, 0])
        mm_68: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_67, arg34_1);  permute_67 = None
        mm_69: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_161, arg174_1);  view_161 = arg174_1 = None
        view_162: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_69, _shape_param_173);  mm_69 = _shape_param_173 = None
        convert_element_type_129: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_162, torch.float32);  view_162 = None
        add_36: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_26, convert_element_type_129);  add_26 = convert_element_type_129 = None
        convert_element_type_130: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        permute_68: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_65, [0, 2, 1, 3]);  permute_65 = None
        view_163: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_68, _shape_param_174);  permute_68 = _shape_param_174 = None
        clone_26: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_163, memory_format = torch.contiguous_format);  view_163 = None
        view_164: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_26, _shape_param_175);  clone_26 = _shape_param_175 = None
        permute_69: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_164, [1, 0])
        mm_70: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_69, arg34_1);  permute_69 = None
        mm_71: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_164, arg175_1);  view_164 = arg175_1 = None
        view_165: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_71, _shape_param_176);  mm_71 = _shape_param_176 = None
        convert_element_type_131: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_165, torch.float32);  view_165 = None
        add_37: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_36, convert_element_type_131);  add_36 = convert_element_type_131 = None
        convert_element_type_132: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        permute_70: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        clone_27: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_166: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_27, _shape_param_177);  clone_27 = _shape_param_177 = None
        view_167: "bf16[8192, 512]" = torch.ops.aten.view.default(view_166, _shape_param_178);  view_166 = _shape_param_178 = None
        permute_71: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_167, [1, 0])
        mm_72: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_71, arg176_1);  permute_71 = arg176_1 = None
        mm_73: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_167, arg177_1);  view_167 = arg177_1 = None
        view_168: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_73, _shape_param_179);  mm_73 = _shape_param_179 = None
        convert_element_type_133: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_168, torch.float32);  view_168 = None
        convert_element_type_134: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        mul_155: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_133, arg178_1);  arg178_1 = None
        mul_156: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg179_1, arg180_1)
        mul_157: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_133, mul_156);  convert_element_type_133 = mul_156 = None
        sum_31: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_157, [0, 1], True, dtype = torch.float32);  mul_157 = None
        view_169: "f32[512]" = torch.ops.aten.view.default(sum_31, _shape_param_180);  sum_31 = _shape_param_180 = None
        mul_158: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_155, arg179_1)
        mul_159: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_155, arg180_1);  mul_155 = None
        sum_32: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_158, [2], True, dtype = torch.float32);  mul_158 = None
        add_38: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_35, mul_159);  add_35 = mul_159 = None
        pow_23: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg180_1, 3);  arg180_1 = None
        mul_160: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_32, -0.5);  sum_32 = None
        mul_161: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_160, pow_23);  mul_160 = pow_23 = None
        expand_12: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_161, _shape_param_181);  mul_161 = _shape_param_181 = None
        div_19: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_12, 512);  expand_12 = None
        pow_24: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg179_1, 1.0);  arg179_1 = None
        mul_162: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_24, 2.0);  pow_24 = None
        mul_163: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_19, mul_162);  div_19 = mul_162 = None
        add_39: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_38, mul_163);  add_38 = mul_163 = None
        convert_element_type_135: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16)
        convert_element_type_136: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg181_1, torch.bfloat16);  arg181_1 = None
        mul_164: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_136, 1.1111111111111112);  convert_element_type_136 = None
        mul_165: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_135, mul_164);  convert_element_type_135 = mul_164 = None
        view_170: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_165, _shape_param_182);  mul_165 = _shape_param_182 = None
        permute_72: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_170, [1, 0])
        mm_74: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_72, arg182_1);  permute_72 = arg182_1 = None
        mm_75: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_170, arg183_1);  view_170 = arg183_1 = None
        view_171: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_75, _shape_param_183);  mm_75 = _shape_param_183 = None
        convert_element_type_137: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_74, torch.float32);  mm_74 = None
        view_172: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_171, _shape_param_184);  view_171 = _shape_param_184 = None
        permute_73: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        clone_28: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
        view_173: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_185);  clone_28 = _shape_param_185 = None
        bmm_28: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg184_1, view_173);  arg184_1 = None
        bmm_29: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_173, arg185_1);  view_173 = arg185_1 = None
        view_174: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_28, _shape_param_186);  bmm_28 = _shape_param_186 = None
        view_175: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_29, _shape_param_187);  bmm_29 = _shape_param_187 = None
        convert_element_type_138: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.bfloat16);  arg186_1 = None
        mul_166: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_138, 1.1111111111111112);  convert_element_type_138 = None
        mul_167: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_175, mul_166);  view_175 = mul_166 = None
        convert_element_type_139: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_167, torch.float32);  mul_167 = None
        convert_element_type_140: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.float32);  arg187_1 = None
        sub_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_140, arg188_1);  convert_element_type_140 = arg188_1 = None
        exp_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        div_20: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_8, arg189_1);  exp_8 = arg189_1 = None
        mul_168: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_139, div_20);  convert_element_type_139 = None
        sum_33: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [-1], True)
        neg_7: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_20);  div_20 = None
        fma_7: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_7, sum_33, mul_168);  neg_7 = sum_33 = mul_168 = None
        convert_element_type_141: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        view_176: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_141, _shape_param_188);  convert_element_type_141 = _shape_param_188 = None
        view_177: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_176, _shape_param_189);  view_176 = _shape_param_189 = None
        view_178: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_177, _shape_param_190);  _shape_param_190 = None
        convert_element_type_142: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        add_40: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_29, convert_element_type_142);  add_29 = convert_element_type_142 = None
        bmm_30: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg190_1, view_178);  arg190_1 = None
        bmm_31: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_178, arg191_1);  view_178 = arg191_1 = None
        view_179: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_30, _shape_param_191);  bmm_30 = _shape_param_191 = None
        view_180: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_31, _shape_param_192);  bmm_31 = _shape_param_192 = None
        permute_74: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_179, [0, 1, 3, 2]);  view_179 = None
        permute_75: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        clone_29: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
        view_181: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_29, _shape_param_193);  clone_29 = _shape_param_193 = None
        view_182: "bf16[8192, 512]" = torch.ops.aten.view.default(view_181, _shape_param_194);  view_181 = _shape_param_194 = None
        permute_76: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_182, [1, 0])
        mm_76: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_76, arg192_1);  permute_76 = None
        mm_77: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_182, arg193_1);  view_182 = arg193_1 = None
        view_183: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_77, _shape_param_195);  mm_77 = _shape_param_195 = None
        convert_element_type_143: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_183, torch.float32);  view_183 = None
        convert_element_type_144: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_76, torch.float32);  mm_76 = None
        permute_77: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_74, [0, 2, 1, 3]);  permute_74 = None
        view_184: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_77, _shape_param_196);  permute_77 = _shape_param_196 = None
        clone_30: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_184, memory_format = torch.contiguous_format);  view_184 = None
        view_185: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_30, _shape_param_197);  clone_30 = _shape_param_197 = None
        permute_78: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_185, [1, 0])
        mm_78: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_78, arg192_1);  permute_78 = None
        mm_79: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_185, arg194_1);  view_185 = arg194_1 = None
        view_186: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_79, _shape_param_198);  mm_79 = _shape_param_198 = None
        convert_element_type_145: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_186, torch.float32);  view_186 = None
        add_41: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_143, convert_element_type_145);  convert_element_type_143 = convert_element_type_145 = None
        convert_element_type_146: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        permute_79: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_180, [0, 2, 1, 3]);  view_180 = None
        clone_31: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
        view_187: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_31, _shape_param_199);  clone_31 = _shape_param_199 = None
        view_188: "bf16[8192, 512]" = torch.ops.aten.view.default(view_187, _shape_param_200);  view_187 = _shape_param_200 = None
        permute_80: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_188, [1, 0])
        mm_80: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_80, arg192_1);  permute_80 = arg192_1 = None
        mm_81: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_188, arg195_1);  view_188 = arg195_1 = None
        view_189: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_81, _shape_param_201);  mm_81 = _shape_param_201 = None
        convert_element_type_147: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_189, torch.float32);  view_189 = None
        add_42: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_41, convert_element_type_147);  add_41 = convert_element_type_147 = None
        convert_element_type_148: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        mul_169: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_42, arg196_1);  arg196_1 = None
        mul_170: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg197_1, arg198_1)
        mul_171: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_42, mul_170);  add_42 = mul_170 = None
        sum_34: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1], True, dtype = torch.float32);  mul_171 = None
        view_190: "f32[512]" = torch.ops.aten.view.default(sum_34, _shape_param_202);  sum_34 = _shape_param_202 = None
        mul_172: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_169, arg197_1)
        mul_173: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_169, arg198_1);  mul_169 = None
        sum_35: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True, dtype = torch.float32);  mul_172 = None
        add_43: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_39, mul_173);  add_39 = mul_173 = None
        pow_25: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg198_1, 3);  arg198_1 = None
        mul_174: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_35, -0.5);  sum_35 = None
        mul_175: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_174, pow_25);  mul_174 = pow_25 = None
        expand_13: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_175, _shape_param_203);  mul_175 = _shape_param_203 = None
        div_21: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_13, 512);  expand_13 = None
        pow_26: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg197_1, 1.0);  arg197_1 = None
        mul_176: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_26, 2.0);  pow_26 = None
        mul_177: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_21, mul_176);  div_21 = mul_176 = None
        add_44: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_43, mul_177);  add_43 = mul_177 = None
        convert_element_type_149: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16)
        convert_element_type_150: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg199_1, torch.bfloat16);  arg199_1 = None
        mul_178: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_150, 1.1111111111111112);  convert_element_type_150 = None
        mul_179: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_149, mul_178);  convert_element_type_149 = mul_178 = None
        view_191: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_179, _shape_param_204);  mul_179 = _shape_param_204 = None
        permute_81: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_191, [1, 0])
        mm_82: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_81, arg200_1);  permute_81 = arg200_1 = None
        mm_83: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_191, arg201_1);  view_191 = arg201_1 = None
        view_192: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_83, _shape_param_205);  mm_83 = _shape_param_205 = None
        convert_element_type_151: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        convert_element_type_152: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_192, torch.bfloat16);  view_192 = None
        convert_element_type_153: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg202_1, torch.bfloat16);  arg202_1 = None
        mul_180: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_153, 1.1111111111111112);  convert_element_type_153 = None
        mul_181: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_152, mul_180);  convert_element_type_152 = mul_180 = None
        where_7: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg203_1, full_2, mul_181);  arg203_1 = mul_181 = None
        view_193: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_7, _shape_param_206);  where_7 = _shape_param_206 = None
        permute_82: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_193, [1, 0])
        mm_84: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_82, arg204_1);  permute_82 = arg204_1 = None
        mm_85: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_193, arg205_1);  view_193 = arg205_1 = None
        view_194: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_85, _shape_param_207);  mm_85 = _shape_param_207 = None
        convert_element_type_154: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_194, torch.float32);  view_194 = None
        convert_element_type_155: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        mul_182: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_154, arg206_1);  arg206_1 = None
        mul_183: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg207_1, arg208_1)
        mul_184: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_154, mul_183);  convert_element_type_154 = mul_183 = None
        sum_36: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_184, [0, 1], True, dtype = torch.float32);  mul_184 = None
        view_195: "f32[512]" = torch.ops.aten.view.default(sum_36, _shape_param_208);  sum_36 = _shape_param_208 = None
        mul_185: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_182, arg207_1)
        mul_186: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_182, arg208_1);  mul_182 = None
        sum_37: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True, dtype = torch.float32);  mul_185 = None
        add_45: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_44, mul_186);  add_44 = mul_186 = None
        pow_27: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg208_1, 3);  arg208_1 = None
        mul_187: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_37, -0.5);  sum_37 = None
        mul_188: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_187, pow_27);  mul_187 = pow_27 = None
        expand_14: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_188, _shape_param_209);  mul_188 = _shape_param_209 = None
        div_22: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_14, 512);  expand_14 = None
        pow_28: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg207_1, 1.0);  arg207_1 = None
        mul_189: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_28, 2.0);  pow_28 = None
        mul_190: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_22, mul_189);  div_22 = mul_189 = None
        add_46: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_45, mul_190);  add_45 = mul_190 = None
        convert_element_type_156: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16)
        convert_element_type_157: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg209_1, torch.bfloat16);  arg209_1 = None
        mul_191: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_157, 1.1111111111111112);  convert_element_type_157 = None
        mul_192: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_156, mul_191);  convert_element_type_156 = mul_191 = None
        view_196: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_192, _shape_param_210);  mul_192 = _shape_param_210 = None
        permute_83: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_196, [1, 0])
        mm_86: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_83, arg210_1);  permute_83 = arg210_1 = None
        mm_87: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_196, arg211_1);  view_196 = arg211_1 = None
        view_197: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_87, _shape_param_211);  mm_87 = _shape_param_211 = None
        convert_element_type_158: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_86, torch.float32);  mm_86 = None
        view_198: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_197, _shape_param_212);  view_197 = _shape_param_212 = None
        permute_84: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
        clone_32: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
        view_199: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_32, _shape_param_213);  clone_32 = _shape_param_213 = None
        bmm_32: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg212_1, view_199);  arg212_1 = None
        bmm_33: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_199, arg213_1);  view_199 = arg213_1 = None
        view_200: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_32, _shape_param_214);  bmm_32 = _shape_param_214 = None
        view_201: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_33, _shape_param_215);  bmm_33 = _shape_param_215 = None
        convert_element_type_159: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg214_1, torch.bfloat16);  arg214_1 = None
        mul_193: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_159, 1.1111111111111112);  convert_element_type_159 = None
        mul_194: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_201, mul_193);  view_201 = mul_193 = None
        convert_element_type_160: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_194, torch.float32);  mul_194 = None
        view_202: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg215_1, _shape_param_216);  arg215_1 = _shape_param_216 = None
        convert_element_type_161: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_202, torch.bfloat16);  view_202 = None
        convert_element_type_162: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_161, torch.float32);  convert_element_type_161 = None
        sub_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_162, arg216_1);  convert_element_type_162 = arg216_1 = None
        exp_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        div_23: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_9, arg217_1);  exp_9 = arg217_1 = None
        mul_195: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_160, div_23);  convert_element_type_160 = None
        sum_38: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [-1], True)
        neg_8: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma_8: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_8, sum_38, mul_195);  neg_8 = sum_38 = mul_195 = None
        convert_element_type_163: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        view_203: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_163, _shape_param_217);  convert_element_type_163 = _shape_param_217 = None
        bmm_34: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg218_1, view_203);  arg218_1 = None
        bmm_35: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_203, arg219_1);  view_203 = arg219_1 = None
        view_204: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_34, _shape_param_218);  bmm_34 = _shape_param_218 = None
        view_205: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_35, _shape_param_219);  bmm_35 = _shape_param_219 = None
        permute_85: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_204, [0, 1, 3, 2]);  view_204 = None
        permute_86: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        clone_33: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_86, memory_format = torch.contiguous_format);  permute_86 = None
        view_206: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_33, _shape_param_220);  clone_33 = _shape_param_220 = None
        view_207: "bf16[8192, 512]" = torch.ops.aten.view.default(view_206, _shape_param_221);  view_206 = _shape_param_221 = None
        permute_87: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_88: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_87, arg34_1);  permute_87 = None
        mm_89: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_207, arg220_1);  view_207 = arg220_1 = None
        view_208: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_89, _shape_param_222);  mm_89 = _shape_param_222 = None
        convert_element_type_164: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_208, torch.float32);  view_208 = None
        add_47: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_37, convert_element_type_164);  add_37 = convert_element_type_164 = None
        convert_element_type_165: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_88, torch.float32);  mm_88 = None
        permute_88: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_85, [0, 2, 1, 3]);  permute_85 = None
        view_209: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_88, _shape_param_223);  permute_88 = _shape_param_223 = None
        clone_34: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_209, memory_format = torch.contiguous_format);  view_209 = None
        view_210: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_34, _shape_param_224);  clone_34 = _shape_param_224 = None
        permute_89: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_90: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_89, arg34_1);  permute_89 = None
        mm_91: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_210, arg221_1);  view_210 = arg221_1 = None
        view_211: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_91, _shape_param_225);  mm_91 = _shape_param_225 = None
        convert_element_type_166: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_211, torch.float32);  view_211 = None
        add_48: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_47, convert_element_type_166);  add_47 = convert_element_type_166 = None
        convert_element_type_167: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        permute_90: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
        clone_35: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_212: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_35, _shape_param_226);  clone_35 = _shape_param_226 = None
        view_213: "bf16[8192, 512]" = torch.ops.aten.view.default(view_212, _shape_param_227);  view_212 = _shape_param_227 = None
        permute_91: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_213, [1, 0])
        mm_92: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_91, arg222_1);  permute_91 = arg222_1 = None
        mm_93: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_213, arg223_1);  view_213 = arg223_1 = None
        view_214: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_93, _shape_param_228);  mm_93 = _shape_param_228 = None
        convert_element_type_168: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_214, torch.float32);  view_214 = None
        convert_element_type_169: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        mul_196: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_168, arg224_1);  arg224_1 = None
        mul_197: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg225_1, arg226_1)
        mul_198: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_168, mul_197);  convert_element_type_168 = mul_197 = None
        sum_39: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1], True, dtype = torch.float32);  mul_198 = None
        view_215: "f32[512]" = torch.ops.aten.view.default(sum_39, _shape_param_229);  sum_39 = _shape_param_229 = None
        mul_199: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_196, arg225_1)
        mul_200: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_196, arg226_1);  mul_196 = None
        sum_40: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [2], True, dtype = torch.float32);  mul_199 = None
        add_49: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_46, mul_200);  add_46 = mul_200 = None
        pow_29: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg226_1, 3);  arg226_1 = None
        mul_201: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_40, -0.5);  sum_40 = None
        mul_202: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_201, pow_29);  mul_201 = pow_29 = None
        expand_15: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_202, _shape_param_230);  mul_202 = _shape_param_230 = None
        div_24: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_15, 512);  expand_15 = None
        pow_30: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg225_1, 1.0);  arg225_1 = None
        mul_203: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_30, 2.0);  pow_30 = None
        mul_204: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_24, mul_203);  div_24 = mul_203 = None
        add_50: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_49, mul_204);  add_49 = mul_204 = None
        convert_element_type_170: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16)
        convert_element_type_171: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.bfloat16);  arg227_1 = None
        mul_205: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_171, 1.1111111111111112);  convert_element_type_171 = None
        mul_206: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_170, mul_205);  convert_element_type_170 = mul_205 = None
        view_216: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_206, _shape_param_231);  mul_206 = _shape_param_231 = None
        permute_92: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_216, [1, 0])
        mm_94: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_92, arg228_1);  permute_92 = arg228_1 = None
        mm_95: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_216, arg229_1);  view_216 = arg229_1 = None
        view_217: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_95, _shape_param_232);  mm_95 = _shape_param_232 = None
        convert_element_type_172: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        view_218: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_217, _shape_param_233);  view_217 = _shape_param_233 = None
        permute_93: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        clone_36: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
        view_219: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_234);  clone_36 = _shape_param_234 = None
        bmm_36: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg230_1, view_219);  arg230_1 = None
        bmm_37: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_219, arg231_1);  view_219 = arg231_1 = None
        view_220: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_36, _shape_param_235);  bmm_36 = _shape_param_235 = None
        view_221: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_37, _shape_param_236);  bmm_37 = _shape_param_236 = None
        convert_element_type_173: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg232_1, torch.bfloat16);  arg232_1 = None
        mul_207: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_173, 1.1111111111111112);  convert_element_type_173 = None
        mul_208: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_221, mul_207);  view_221 = mul_207 = None
        convert_element_type_174: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_208, torch.float32);  mul_208 = None
        convert_element_type_175: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.float32);  arg233_1 = None
        sub_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_175, arg234_1);  convert_element_type_175 = arg234_1 = None
        exp_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        div_25: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_10, arg235_1);  exp_10 = arg235_1 = None
        mul_209: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_174, div_25);  convert_element_type_174 = None
        sum_41: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [-1], True)
        neg_9: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_9: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_9, sum_41, mul_209);  neg_9 = sum_41 = mul_209 = None
        convert_element_type_176: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        view_222: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_176, _shape_param_237);  convert_element_type_176 = _shape_param_237 = None
        view_223: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_222, _shape_param_238);  view_222 = _shape_param_238 = None
        view_224: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_223, _shape_param_239);  _shape_param_239 = None
        convert_element_type_177: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_223, torch.float32);  view_223 = None
        add_51: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_40, convert_element_type_177);  add_40 = convert_element_type_177 = None
        bmm_38: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg236_1, view_224);  arg236_1 = None
        bmm_39: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_224, arg237_1);  view_224 = arg237_1 = None
        view_225: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_38, _shape_param_240);  bmm_38 = _shape_param_240 = None
        view_226: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_39, _shape_param_241);  bmm_39 = _shape_param_241 = None
        permute_94: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_225, [0, 1, 3, 2]);  view_225 = None
        permute_95: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
        clone_37: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
        view_227: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_37, _shape_param_242);  clone_37 = _shape_param_242 = None
        view_228: "bf16[8192, 512]" = torch.ops.aten.view.default(view_227, _shape_param_243);  view_227 = _shape_param_243 = None
        permute_96: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_228, [1, 0])
        mm_96: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_96, arg238_1);  permute_96 = None
        mm_97: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_228, arg239_1);  view_228 = arg239_1 = None
        view_229: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_97, _shape_param_244);  mm_97 = _shape_param_244 = None
        convert_element_type_178: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_229, torch.float32);  view_229 = None
        convert_element_type_179: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        permute_97: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_94, [0, 2, 1, 3]);  permute_94 = None
        view_230: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_97, _shape_param_245);  permute_97 = _shape_param_245 = None
        clone_38: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_230, memory_format = torch.contiguous_format);  view_230 = None
        view_231: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_38, _shape_param_246);  clone_38 = _shape_param_246 = None
        permute_98: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_231, [1, 0])
        mm_98: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_98, arg238_1);  permute_98 = None
        mm_99: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_231, arg240_1);  view_231 = arg240_1 = None
        view_232: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_99, _shape_param_247);  mm_99 = _shape_param_247 = None
        convert_element_type_180: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_232, torch.float32);  view_232 = None
        add_52: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_178, convert_element_type_180);  convert_element_type_178 = convert_element_type_180 = None
        convert_element_type_181: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_98, torch.float32);  mm_98 = None
        permute_99: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_39: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
        view_233: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_39, _shape_param_248);  clone_39 = _shape_param_248 = None
        view_234: "bf16[8192, 512]" = torch.ops.aten.view.default(view_233, _shape_param_249);  view_233 = _shape_param_249 = None
        permute_100: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_234, [1, 0])
        mm_100: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_100, arg238_1);  permute_100 = arg238_1 = None
        mm_101: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_234, arg241_1);  view_234 = arg241_1 = None
        view_235: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_101, _shape_param_250);  mm_101 = _shape_param_250 = None
        convert_element_type_182: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_235, torch.float32);  view_235 = None
        add_53: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_52, convert_element_type_182);  add_52 = convert_element_type_182 = None
        convert_element_type_183: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_100, torch.float32);  mm_100 = None
        mul_210: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_53, arg242_1);  arg242_1 = None
        mul_211: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg243_1, arg244_1)
        mul_212: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_53, mul_211);  add_53 = mul_211 = None
        sum_42: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_212, [0, 1], True, dtype = torch.float32);  mul_212 = None
        view_236: "f32[512]" = torch.ops.aten.view.default(sum_42, _shape_param_251);  sum_42 = _shape_param_251 = None
        mul_213: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_210, arg243_1)
        mul_214: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_210, arg244_1);  mul_210 = None
        sum_43: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True, dtype = torch.float32);  mul_213 = None
        add_54: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_50, mul_214);  add_50 = mul_214 = None
        pow_31: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg244_1, 3);  arg244_1 = None
        mul_215: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_43, -0.5);  sum_43 = None
        mul_216: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_215, pow_31);  mul_215 = pow_31 = None
        expand_16: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_216, _shape_param_252);  mul_216 = _shape_param_252 = None
        div_26: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_16, 512);  expand_16 = None
        pow_32: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg243_1, 1.0);  arg243_1 = None
        mul_217: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_32, 2.0);  pow_32 = None
        mul_218: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_26, mul_217);  div_26 = mul_217 = None
        add_55: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_54, mul_218);  add_54 = mul_218 = None
        convert_element_type_184: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16)
        convert_element_type_185: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg245_1, torch.bfloat16);  arg245_1 = None
        mul_219: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_185, 1.1111111111111112);  convert_element_type_185 = None
        mul_220: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_184, mul_219);  convert_element_type_184 = mul_219 = None
        view_237: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_220, _shape_param_253);  mul_220 = _shape_param_253 = None
        permute_101: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_237, [1, 0])
        mm_102: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_101, arg246_1);  permute_101 = arg246_1 = None
        mm_103: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_237, arg247_1);  view_237 = arg247_1 = None
        view_238: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_103, _shape_param_254);  mm_103 = _shape_param_254 = None
        convert_element_type_186: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        convert_element_type_187: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_238, torch.bfloat16);  view_238 = None
        convert_element_type_188: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg248_1, torch.bfloat16);  arg248_1 = None
        mul_221: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_188, 1.1111111111111112);  convert_element_type_188 = None
        mul_222: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_187, mul_221);  convert_element_type_187 = mul_221 = None
        where_8: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg249_1, full_2, mul_222);  arg249_1 = mul_222 = None
        view_239: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_8, _shape_param_255);  where_8 = _shape_param_255 = None
        permute_102: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_239, [1, 0])
        mm_104: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_102, arg250_1);  permute_102 = arg250_1 = None
        mm_105: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_239, arg251_1);  view_239 = arg251_1 = None
        view_240: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_105, _shape_param_256);  mm_105 = _shape_param_256 = None
        convert_element_type_189: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_240, torch.float32);  view_240 = None
        convert_element_type_190: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        mul_223: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_189, arg252_1);  arg252_1 = None
        mul_224: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg253_1, arg254_1)
        mul_225: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_189, mul_224);  convert_element_type_189 = mul_224 = None
        sum_44: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1], True, dtype = torch.float32);  mul_225 = None
        view_241: "f32[512]" = torch.ops.aten.view.default(sum_44, _shape_param_257);  sum_44 = _shape_param_257 = None
        mul_226: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_223, arg253_1)
        mul_227: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_223, arg254_1);  mul_223 = None
        sum_45: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True, dtype = torch.float32);  mul_226 = None
        add_56: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_55, mul_227);  add_55 = mul_227 = None
        pow_33: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg254_1, 3);  arg254_1 = None
        mul_228: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_45, -0.5);  sum_45 = None
        mul_229: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_228, pow_33);  mul_228 = pow_33 = None
        expand_17: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_229, _shape_param_258);  mul_229 = _shape_param_258 = None
        div_27: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_17, 512);  expand_17 = None
        pow_34: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg253_1, 1.0);  arg253_1 = None
        mul_230: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_34, 2.0);  pow_34 = None
        mul_231: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_27, mul_230);  div_27 = mul_230 = None
        add_57: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_56, mul_231);  add_56 = mul_231 = None
        convert_element_type_191: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_57, torch.bfloat16)
        convert_element_type_192: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg255_1, torch.bfloat16);  arg255_1 = None
        mul_232: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_192, 1.1111111111111112);  convert_element_type_192 = None
        mul_233: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_191, mul_232);  convert_element_type_191 = mul_232 = None
        view_242: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_233, _shape_param_259);  mul_233 = _shape_param_259 = None
        permute_103: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_242, [1, 0])
        mm_106: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_103, arg256_1);  permute_103 = arg256_1 = None
        mm_107: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_242, arg257_1);  view_242 = arg257_1 = None
        view_243: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_107, _shape_param_260);  mm_107 = _shape_param_260 = None
        convert_element_type_193: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        view_244: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_243, _shape_param_261);  view_243 = _shape_param_261 = None
        permute_104: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
        clone_40: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
        view_245: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_262);  clone_40 = _shape_param_262 = None
        bmm_40: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg258_1, view_245);  arg258_1 = None
        bmm_41: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_245, arg259_1);  view_245 = arg259_1 = None
        view_246: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_40, _shape_param_263);  bmm_40 = _shape_param_263 = None
        view_247: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_41, _shape_param_264);  bmm_41 = _shape_param_264 = None
        convert_element_type_194: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg260_1, torch.bfloat16);  arg260_1 = None
        mul_234: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_194, 1.1111111111111112);  convert_element_type_194 = None
        mul_235: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_247, mul_234);  view_247 = mul_234 = None
        convert_element_type_195: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_235, torch.float32);  mul_235 = None
        view_248: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg261_1, _shape_param_265);  arg261_1 = _shape_param_265 = None
        convert_element_type_196: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_248, torch.bfloat16);  view_248 = None
        convert_element_type_197: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_196, torch.float32);  convert_element_type_196 = None
        sub_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_197, arg262_1);  convert_element_type_197 = arg262_1 = None
        exp_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        div_28: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_11, arg263_1);  exp_11 = arg263_1 = None
        mul_236: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_195, div_28);  convert_element_type_195 = None
        sum_46: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [-1], True)
        neg_10: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_28);  div_28 = None
        fma_10: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_10, sum_46, mul_236);  neg_10 = sum_46 = mul_236 = None
        convert_element_type_198: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        view_249: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_198, _shape_param_266);  convert_element_type_198 = _shape_param_266 = None
        bmm_42: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg264_1, view_249);  arg264_1 = None
        bmm_43: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_249, arg265_1);  view_249 = arg265_1 = None
        view_250: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_42, _shape_param_267);  bmm_42 = _shape_param_267 = None
        view_251: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_43, _shape_param_268);  bmm_43 = _shape_param_268 = None
        permute_105: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_250, [0, 1, 3, 2]);  view_250 = None
        permute_106: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        clone_41: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
        view_252: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_41, _shape_param_269);  clone_41 = _shape_param_269 = None
        view_253: "bf16[8192, 512]" = torch.ops.aten.view.default(view_252, _shape_param_270);  view_252 = _shape_param_270 = None
        permute_107: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_253, [1, 0])
        mm_108: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_107, arg34_1);  permute_107 = None
        mm_109: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_253, arg266_1);  view_253 = arg266_1 = None
        view_254: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_109, _shape_param_271);  mm_109 = _shape_param_271 = None
        convert_element_type_199: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_254, torch.float32);  view_254 = None
        add_58: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_48, convert_element_type_199);  add_48 = convert_element_type_199 = None
        convert_element_type_200: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        permute_108: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_105, [0, 2, 1, 3]);  permute_105 = None
        view_255: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_108, _shape_param_272);  permute_108 = _shape_param_272 = None
        clone_42: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_255, memory_format = torch.contiguous_format);  view_255 = None
        view_256: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_42, _shape_param_273);  clone_42 = _shape_param_273 = None
        permute_109: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_256, [1, 0])
        mm_110: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_109, arg34_1);  permute_109 = arg34_1 = None
        mm_111: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_256, arg267_1);  view_256 = arg267_1 = None
        view_257: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_111, _shape_param_274);  mm_111 = _shape_param_274 = None
        convert_element_type_201: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_257, torch.float32);  view_257 = None
        add_59: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_58, convert_element_type_201);  add_58 = convert_element_type_201 = None
        convert_element_type_202: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_110, torch.float32);  mm_110 = None
        permute_110: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
        clone_43: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_258: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_43, _shape_param_275);  clone_43 = _shape_param_275 = None
        view_259: "bf16[8192, 512]" = torch.ops.aten.view.default(view_258, _shape_param_276);  view_258 = _shape_param_276 = None
        permute_111: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_259, [1, 0])
        mm_112: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_111, arg268_1);  permute_111 = arg268_1 = None
        mm_113: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_259, arg269_1);  view_259 = arg269_1 = None
        view_260: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_113, _shape_param_277);  mm_113 = _shape_param_277 = None
        convert_element_type_203: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_260, torch.float32);  view_260 = None
        convert_element_type_204: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None
        mul_237: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_203, arg270_1);  arg270_1 = None
        mul_238: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg271_1, arg272_1)
        mul_239: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_203, mul_238);  convert_element_type_203 = mul_238 = None
        sum_47: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1], True, dtype = torch.float32);  mul_239 = None
        view_261: "f32[512]" = torch.ops.aten.view.default(sum_47, _shape_param_278);  sum_47 = _shape_param_278 = None
        mul_240: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_237, arg271_1)
        mul_241: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_237, arg272_1);  mul_237 = None
        sum_48: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True, dtype = torch.float32);  mul_240 = None
        add_60: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_57, mul_241);  add_57 = mul_241 = None
        pow_35: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg272_1, 3);  arg272_1 = None
        mul_242: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_48, -0.5);  sum_48 = None
        mul_243: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_242, pow_35);  mul_242 = pow_35 = None
        expand_18: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_243, _shape_param_279);  mul_243 = _shape_param_279 = None
        div_29: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_18, 512);  expand_18 = None
        pow_36: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg271_1, 1.0);  arg271_1 = None
        mul_244: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_36, 2.0);  pow_36 = None
        mul_245: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_29, mul_244);  div_29 = mul_244 = None
        add_61: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_60, mul_245);  add_60 = mul_245 = None
        convert_element_type_205: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16)
        convert_element_type_206: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg273_1, torch.bfloat16);  arg273_1 = None
        mul_246: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_206, 1.1111111111111112);  convert_element_type_206 = None
        mul_247: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_205, mul_246);  convert_element_type_205 = mul_246 = None
        view_262: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_247, _shape_param_280);  mul_247 = _shape_param_280 = None
        permute_112: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_262, [1, 0])
        mm_114: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_112, arg274_1);  permute_112 = arg274_1 = None
        mm_115: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_262, arg275_1);  view_262 = arg275_1 = None
        view_263: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_115, _shape_param_281);  mm_115 = _shape_param_281 = None
        convert_element_type_207: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        view_264: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_263, _shape_param_282);  view_263 = _shape_param_282 = None
        permute_113: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
        clone_44: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
        view_265: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_44, _shape_param_283);  clone_44 = _shape_param_283 = None
        bmm_44: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg276_1, view_265);  arg276_1 = None
        bmm_45: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_265, arg277_1);  view_265 = arg277_1 = None
        view_266: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_44, _shape_param_284);  bmm_44 = _shape_param_284 = None
        view_267: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_45, _shape_param_285);  bmm_45 = _shape_param_285 = None
        convert_element_type_208: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg278_1, torch.bfloat16);  arg278_1 = None
        mul_248: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_208, 1.1111111111111112);  convert_element_type_208 = None
        mul_249: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_267, mul_248);  view_267 = mul_248 = None
        convert_element_type_209: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_249, torch.float32);  mul_249 = None
        unsqueeze_1: "i64[1, 1, 1024, 1]" = torch.ops.aten.unsqueeze.default(arg279_1, 3)
        unsqueeze_2: "i64[1, 1, 1, 1024]" = torch.ops.aten.unsqueeze.default(arg279_1, 2);  arg279_1 = None
        le: "b8[1, 1, 1024, 1024]" = torch.ops.aten.le.Tensor(unsqueeze_2, unsqueeze_1);  unsqueeze_2 = unsqueeze_1 = None
        expand_19: "b8[8, 1, 1024, 1024]" = torch.ops.aten.expand.default(le, _shape_param_286);  le = _shape_param_286 = None
        full_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: "f32[8, 1, 1024, 1024]" = torch.ops.aten.where.self(expand_19, full_1, full_3);  expand_19 = full_1 = full_3 = None
        view_268: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg280_1, _shape_param_287);  arg280_1 = _shape_param_287 = None
        permute_114: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(arg281_1, [2, 0, 1]);  arg281_1 = None
        unsqueeze_3: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_114, 0);  permute_114 = None
        add_62: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_3, where_9);  unsqueeze_3 = where_9 = None
        add_63: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_268, add_62);  view_268 = add_62 = None
        convert_element_type_210: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_63, torch.bfloat16);  add_63 = None
        convert_element_type_211: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_210, torch.float32);  convert_element_type_210 = None
        sub_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_211, arg282_1);  convert_element_type_211 = arg282_1 = None
        exp_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        div_30: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_12, arg283_1);  exp_12 = arg283_1 = None
        mul_250: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_209, div_30);  convert_element_type_209 = None
        sum_49: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [-1], True)
        neg_11: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_30);  div_30 = None
        fma_11: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_11, sum_49, mul_250);  neg_11 = sum_49 = mul_250 = None
        convert_element_type_212: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        view_269: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_212, _shape_param_288);  convert_element_type_212 = _shape_param_288 = None
        view_270: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_269, _shape_param_289);  view_269 = _shape_param_289 = None
        view_271: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_270, _shape_param_290);  _shape_param_290 = None
        convert_element_type_213: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_270, torch.float32);  view_270 = None
        add_64: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_51, convert_element_type_213);  add_51 = convert_element_type_213 = None
        sum_50: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_64, [0], True, dtype = torch.float32);  add_64 = None
        squeeze: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_50, 0);  sum_50 = None
        permute_115: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
        full_4: "b8[1024, 1024, 1]" = torch.ops.aten.full.default(_shape_param_291, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_291 = None
        full_5: "f32[32, 8]" = torch.ops.aten.full.default(_shape_param_292, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_292 = None
        _unsafe_masked_index_put_accumulate: "f32[32, 8]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_5, full_4, [arg284_1], permute_115);  arg284_1 = permute_115 = None
        bmm_46: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg285_1, view_271);  arg285_1 = None
        bmm_47: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_271, arg286_1);  view_271 = arg286_1 = None
        view_272: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_46, _shape_param_293);  bmm_46 = _shape_param_293 = None
        view_273: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_47, _shape_param_294);  bmm_47 = _shape_param_294 = None
        permute_116: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_272, [0, 1, 3, 2]);  view_272 = None
        permute_117: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        clone_45: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_274: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_45, _shape_param_295);  clone_45 = _shape_param_295 = None
        view_275: "bf16[8192, 512]" = torch.ops.aten.view.default(view_274, _shape_param_296);  view_274 = _shape_param_296 = None
        permute_118: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_275, [1, 0])
        mm_116: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_118, arg287_1);  permute_118 = None
        mm_117: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_275, arg288_1);  view_275 = arg288_1 = None
        view_276: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_117, _shape_param_297);  mm_117 = _shape_param_297 = None
        convert_element_type_214: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_276, torch.float32);  view_276 = None
        convert_element_type_215: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        permute_119: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_116, [0, 2, 1, 3]);  permute_116 = None
        view_277: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_119, _shape_param_298);  permute_119 = _shape_param_298 = None
        clone_46: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_277, memory_format = torch.contiguous_format);  view_277 = None
        view_278: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_46, _shape_param_299);  clone_46 = _shape_param_299 = None
        permute_120: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_278, [1, 0])
        mm_118: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_120, arg287_1);  permute_120 = None
        mm_119: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_278, arg289_1);  view_278 = arg289_1 = None
        view_279: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_119, _shape_param_300);  mm_119 = _shape_param_300 = None
        convert_element_type_216: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        add_65: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_214, convert_element_type_216);  convert_element_type_214 = convert_element_type_216 = None
        convert_element_type_217: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        permute_121: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
        clone_47: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
        view_280: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_47, _shape_param_301);  clone_47 = _shape_param_301 = None
        view_281: "bf16[8192, 512]" = torch.ops.aten.view.default(view_280, _shape_param_302);  view_280 = _shape_param_302 = None
        permute_122: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_281, [1, 0])
        mm_120: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_122, arg287_1);  permute_122 = arg287_1 = None
        mm_121: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_281, arg290_1);  view_281 = arg290_1 = None
        view_282: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_121, _shape_param_303);  mm_121 = _shape_param_303 = None
        convert_element_type_218: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_282, torch.float32);  view_282 = None
        add_66: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_65, convert_element_type_218);  add_65 = convert_element_type_218 = None
        convert_element_type_219: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        mul_251: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_66, arg291_1);  arg291_1 = None
        mul_252: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg292_1, arg293_1)
        mul_253: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_252, 1.1111111111111112);  mul_252 = None
        mul_254: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_253, arg294_1)
        mul_255: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_66, mul_254);  add_66 = mul_254 = None
        sum_51: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 1], True, dtype = torch.float32);  mul_255 = None
        view_283: "f32[512]" = torch.ops.aten.view.default(sum_51, _shape_param_304);  sum_51 = _shape_param_304 = None
        mul_256: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_251, mul_253)
        mul_257: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_251, arg294_1);  mul_251 = None
        sum_52: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True, dtype = torch.float32);  mul_256 = None
        add_67: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_61, mul_257);  add_61 = mul_257 = None
        pow_37: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg294_1, 3);  arg294_1 = None
        mul_258: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_52, -0.5);  sum_52 = None
        mul_259: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_258, pow_37);  mul_258 = pow_37 = None
        expand_20: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_259, _shape_param_305);  mul_259 = _shape_param_305 = None
        div_31: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_20, 512);  expand_20 = None
        pow_38: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_253, 1.0);  mul_253 = None
        mul_260: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_38, 2.0);  pow_38 = None
        mul_261: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_31, mul_260);  div_31 = mul_260 = None
        add_68: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_67, mul_261);  add_67 = mul_261 = None
        convert_element_type_220: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg292_1, torch.float32);  arg292_1 = None
        mul_262: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_220, 1.1111111111111112);  convert_element_type_220 = None
        mul_263: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_68, mul_262);  add_68 = mul_262 = None
        ge: "b8[8, 1024]" = torch.ops.aten.ge.Scalar(arg295_1, 0)
        lt: "b8[8, 1024]" = torch.ops.aten.lt.Scalar(arg295_1, 32128)
        bitwise_and: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_1: "b8[8, 1024]" = torch.ops.aten.ne.Scalar(arg295_1, -1)
        bitwise_and_1: "b8[8, 1024]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_1);  bitwise_and = ne_1 = None
        unsqueeze_4: "b8[8, 1024, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_6: "f32[32128, 512]" = torch.ops.aten.full.default(_shape_param_306, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_306 = None
        _unsafe_masked_index_put_accumulate_1: "f32[32128, 512]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_6, unsqueeze_4, [arg295_1], mul_263);  mul_263 = None
        add_69: "f32[32128, 512]" = torch.ops.aten.add.Tensor(convert_element_type_7, _unsafe_masked_index_put_accumulate_1);  convert_element_type_7 = _unsafe_masked_index_put_accumulate_1 = None
        convert_element_type_221: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg296_1, torch.float32);  arg296_1 = None
        mul_264: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_221, 1.1111111111111112);  convert_element_type_221 = None
        mul_265: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_59, mul_264);  add_59 = mul_264 = None
        mul_266: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_265, arg297_1);  arg297_1 = None
        mul_267: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg298_1, arg299_1)
        mul_268: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_265, mul_267);  mul_265 = mul_267 = None
        sum_53: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1], True, dtype = torch.float32);  mul_268 = None
        view_284: "f32[512]" = torch.ops.aten.view.default(sum_53, _shape_param_307);  sum_53 = _shape_param_307 = None
        mul_269: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_266, arg298_1)
        mul_270: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_266, arg299_1);  mul_266 = None
        sum_54: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True, dtype = torch.float32);  mul_269 = None
        pow_39: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg299_1, 3);  arg299_1 = None
        mul_271: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_54, -0.5);  sum_54 = None
        mul_272: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_271, pow_39);  mul_271 = pow_39 = None
        expand_21: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_272, _shape_param_308);  mul_272 = _shape_param_308 = None
        div_32: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_21, 512);  expand_21 = None
        pow_40: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg298_1, 1.0);  arg298_1 = None
        mul_273: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_40, 2.0);  pow_40 = None
        mul_274: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_32, mul_273);  div_32 = mul_273 = None
        add_70: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(mul_270, mul_274);  mul_270 = mul_274 = None
        convert_element_type_222: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16)
        convert_element_type_223: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg300_1, torch.bfloat16);  arg300_1 = None
        mul_275: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_223, 1.1111111111111112);  convert_element_type_223 = None
        mul_276: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_222, mul_275);  convert_element_type_222 = mul_275 = None
        view_285: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_276, _shape_param_309);  mul_276 = _shape_param_309 = None
        permute_123: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_122: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_123, arg301_1);  permute_123 = arg301_1 = None
        mm_123: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_285, arg302_1);  view_285 = arg302_1 = None
        view_286: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_123, _shape_param_310);  mm_123 = _shape_param_310 = None
        convert_element_type_224: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_122, torch.float32);  mm_122 = None
        convert_element_type_225: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_286, torch.bfloat16);  view_286 = None
        convert_element_type_226: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.bfloat16);  arg303_1 = None
        mul_277: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_226, 1.1111111111111112);  convert_element_type_226 = None
        mul_278: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_225, mul_277);  convert_element_type_225 = mul_277 = None
        where_10: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg304_1, full_2, mul_278);  arg304_1 = mul_278 = None
        view_287: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_10, _shape_param_311);  where_10 = _shape_param_311 = None
        permute_124: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_287, [1, 0])
        mm_124: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_124, arg305_1);  permute_124 = arg305_1 = None
        mm_125: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_287, arg306_1);  view_287 = arg306_1 = None
        view_288: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_125, _shape_param_312);  mm_125 = _shape_param_312 = None
        convert_element_type_227: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_288, torch.float32);  view_288 = None
        convert_element_type_228: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_124, torch.float32);  mm_124 = None
        mul_279: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_227, arg307_1);  arg307_1 = None
        mul_280: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg308_1, arg309_1)
        mul_281: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_227, mul_280);  convert_element_type_227 = mul_280 = None
        sum_55: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 1], True, dtype = torch.float32);  mul_281 = None
        view_289: "f32[512]" = torch.ops.aten.view.default(sum_55, _shape_param_313);  sum_55 = _shape_param_313 = None
        mul_282: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_279, arg308_1)
        mul_283: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_279, arg309_1);  mul_279 = None
        sum_56: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True, dtype = torch.float32);  mul_282 = None
        add_71: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_70, mul_283);  add_70 = mul_283 = None
        pow_41: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg309_1, 3);  arg309_1 = None
        mul_284: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_56, -0.5);  sum_56 = None
        mul_285: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_284, pow_41);  mul_284 = pow_41 = None
        expand_22: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_285, _shape_param_314);  mul_285 = _shape_param_314 = None
        div_33: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_22, 512);  expand_22 = None
        pow_42: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg308_1, 1.0);  arg308_1 = None
        mul_286: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_42, 2.0);  pow_42 = None
        mul_287: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_33, mul_286);  div_33 = mul_286 = None
        add_72: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_71, mul_287);  add_71 = mul_287 = None
        convert_element_type_229: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16)
        convert_element_type_230: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.bfloat16);  arg310_1 = None
        mul_288: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_230, 1.1111111111111112);  convert_element_type_230 = None
        mul_289: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_229, mul_288);  convert_element_type_229 = mul_288 = None
        view_290: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_289, _shape_param_315);  mul_289 = _shape_param_315 = None
        permute_125: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_290, [1, 0])
        mm_126: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_125, arg311_1);  permute_125 = arg311_1 = None
        mm_127: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_290, arg312_1);  view_290 = arg312_1 = None
        view_291: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_127, _shape_param_316);  mm_127 = _shape_param_316 = None
        convert_element_type_231: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        view_292: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_291, _shape_param_317);  view_291 = _shape_param_317 = None
        permute_126: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
        clone_48: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
        view_293: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_318);  clone_48 = _shape_param_318 = None
        bmm_48: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg313_1, view_293);  arg313_1 = None
        bmm_49: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_293, arg314_1);  view_293 = arg314_1 = None
        view_294: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_48, _shape_param_319);  bmm_48 = _shape_param_319 = None
        view_295: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_49, _shape_param_320);  bmm_49 = _shape_param_320 = None
        convert_element_type_232: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.bfloat16);  arg315_1 = None
        mul_290: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_232, 1.1111111111111112);  convert_element_type_232 = None
        mul_291: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_295, mul_290);  view_295 = mul_290 = None
        convert_element_type_233: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_291, torch.float32);  mul_291 = None
        convert_element_type_234: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg316_1, torch.float32);  arg316_1 = None
        sub_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_234, arg317_1);  convert_element_type_234 = arg317_1 = None
        exp_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        div_34: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_13, arg318_1);  exp_13 = arg318_1 = None
        mul_292: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_233, div_34);  convert_element_type_233 = None
        sum_57: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [-1], True)
        neg_12: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_34);  div_34 = None
        fma_12: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_12, sum_57, mul_292);  neg_12 = sum_57 = mul_292 = None
        convert_element_type_235: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None
        view_296: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_235, _shape_param_321);  convert_element_type_235 = _shape_param_321 = None
        view_297: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_296, _shape_param_322);  view_296 = _shape_param_322 = None
        view_298: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_297, _shape_param_323);  _shape_param_323 = None
        convert_element_type_236: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_297, torch.float32);  view_297 = None
        bmm_50: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg319_1, view_298);  arg319_1 = None
        bmm_51: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_298, arg320_1);  view_298 = arg320_1 = None
        view_299: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_50, _shape_param_324);  bmm_50 = _shape_param_324 = None
        view_300: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_51, _shape_param_325);  bmm_51 = _shape_param_325 = None
        permute_127: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_299, [0, 1, 3, 2]);  view_299 = None
        permute_128: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
        clone_49: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
        view_301: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_49, _shape_param_326);  clone_49 = _shape_param_326 = None
        view_302: "bf16[8192, 512]" = torch.ops.aten.view.default(view_301, _shape_param_327);  view_301 = _shape_param_327 = None
        permute_129: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_302, [1, 0])
        mm_128: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_129, arg321_1);  permute_129 = None
        mm_129: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_302, arg322_1);  view_302 = arg322_1 = None
        view_303: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_129, _shape_param_328);  mm_129 = _shape_param_328 = None
        convert_element_type_237: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_303, torch.float32);  view_303 = None
        convert_element_type_238: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        permute_130: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_127, [0, 2, 1, 3]);  permute_127 = None
        view_304: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_130, _shape_param_329);  permute_130 = _shape_param_329 = None
        clone_50: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_304, memory_format = torch.contiguous_format);  view_304 = None
        view_305: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_50, _shape_param_330);  clone_50 = _shape_param_330 = None
        permute_131: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_305, [1, 0])
        mm_130: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_131, arg321_1);  permute_131 = None
        mm_131: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_305, arg323_1);  view_305 = arg323_1 = None
        view_306: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_131, _shape_param_331);  mm_131 = _shape_param_331 = None
        convert_element_type_239: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_306, torch.float32);  view_306 = None
        add_73: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_237, convert_element_type_239);  convert_element_type_237 = convert_element_type_239 = None
        convert_element_type_240: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        permute_132: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_300, [0, 2, 1, 3]);  view_300 = None
        clone_51: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
        view_307: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_51, _shape_param_332);  clone_51 = _shape_param_332 = None
        view_308: "bf16[8192, 512]" = torch.ops.aten.view.default(view_307, _shape_param_333);  view_307 = _shape_param_333 = None
        permute_133: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_308, [1, 0])
        mm_132: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_133, arg321_1);  permute_133 = arg321_1 = None
        mm_133: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_308, arg324_1);  view_308 = arg324_1 = None
        view_309: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_133, _shape_param_334);  mm_133 = _shape_param_334 = None
        convert_element_type_241: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_309, torch.float32);  view_309 = None
        add_74: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_73, convert_element_type_241);  add_73 = convert_element_type_241 = None
        convert_element_type_242: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None
        mul_293: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_74, arg325_1);  arg325_1 = None
        mul_294: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg326_1, arg327_1)
        mul_295: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_74, mul_294);  add_74 = mul_294 = None
        sum_58: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1], True, dtype = torch.float32);  mul_295 = None
        view_310: "f32[512]" = torch.ops.aten.view.default(sum_58, _shape_param_335);  sum_58 = _shape_param_335 = None
        mul_296: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_293, arg326_1)
        mul_297: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_293, arg327_1);  mul_293 = None
        sum_59: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True, dtype = torch.float32);  mul_296 = None
        add_75: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_72, mul_297);  add_72 = mul_297 = None
        pow_43: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg327_1, 3);  arg327_1 = None
        mul_298: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_59, -0.5);  sum_59 = None
        mul_299: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_298, pow_43);  mul_298 = pow_43 = None
        expand_23: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_299, _shape_param_336);  mul_299 = _shape_param_336 = None
        div_35: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_23, 512);  expand_23 = None
        pow_44: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg326_1, 1.0);  arg326_1 = None
        mul_300: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_44, 2.0);  pow_44 = None
        mul_301: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_35, mul_300);  div_35 = mul_300 = None
        add_76: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_75, mul_301);  add_75 = mul_301 = None
        convert_element_type_243: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16)
        convert_element_type_244: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.bfloat16);  arg328_1 = None
        mul_302: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_244, 1.1111111111111112);  convert_element_type_244 = None
        mul_303: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_243, mul_302);  convert_element_type_243 = mul_302 = None
        view_311: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_303, _shape_param_337);  mul_303 = _shape_param_337 = None
        permute_134: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_134: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_134, arg329_1);  permute_134 = arg329_1 = None
        mm_135: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_311, arg330_1);  view_311 = arg330_1 = None
        view_312: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_135, _shape_param_338);  mm_135 = _shape_param_338 = None
        convert_element_type_245: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_134, torch.float32);  mm_134 = None
        convert_element_type_246: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_312, torch.bfloat16);  view_312 = None
        convert_element_type_247: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg331_1, torch.bfloat16);  arg331_1 = None
        mul_304: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_247, 1.1111111111111112);  convert_element_type_247 = None
        mul_305: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_246, mul_304);  convert_element_type_246 = mul_304 = None
        where_11: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg332_1, full_2, mul_305);  arg332_1 = mul_305 = None
        view_313: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_11, _shape_param_339);  where_11 = _shape_param_339 = None
        permute_135: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_313, [1, 0])
        mm_136: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_135, arg333_1);  permute_135 = arg333_1 = None
        mm_137: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_313, arg334_1);  view_313 = arg334_1 = None
        view_314: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_137, _shape_param_340);  mm_137 = _shape_param_340 = None
        convert_element_type_248: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_314, torch.float32);  view_314 = None
        convert_element_type_249: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        mul_306: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_248, arg335_1);  arg335_1 = None
        mul_307: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg336_1, arg337_1)
        mul_308: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_248, mul_307);  convert_element_type_248 = mul_307 = None
        sum_60: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1], True, dtype = torch.float32);  mul_308 = None
        view_315: "f32[512]" = torch.ops.aten.view.default(sum_60, _shape_param_341);  sum_60 = _shape_param_341 = None
        mul_309: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_306, arg336_1)
        mul_310: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_306, arg337_1);  mul_306 = None
        sum_61: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True, dtype = torch.float32);  mul_309 = None
        add_77: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_76, mul_310);  add_76 = mul_310 = None
        pow_45: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg337_1, 3);  arg337_1 = None
        mul_311: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_61, -0.5);  sum_61 = None
        mul_312: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_311, pow_45);  mul_311 = pow_45 = None
        expand_24: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_312, _shape_param_342);  mul_312 = _shape_param_342 = None
        div_36: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_24, 512);  expand_24 = None
        pow_46: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg336_1, 1.0);  arg336_1 = None
        mul_313: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_46, 2.0);  pow_46 = None
        mul_314: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_36, mul_313);  div_36 = mul_313 = None
        add_78: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_77, mul_314);  add_77 = mul_314 = None
        convert_element_type_250: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16)
        convert_element_type_251: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.bfloat16);  arg338_1 = None
        mul_315: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_251, 1.1111111111111112);  convert_element_type_251 = None
        mul_316: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_250, mul_315);  convert_element_type_250 = mul_315 = None
        view_316: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_316, _shape_param_343);  mul_316 = _shape_param_343 = None
        permute_136: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_316, [1, 0])
        mm_138: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_136, arg339_1);  permute_136 = arg339_1 = None
        mm_139: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_316, arg340_1);  view_316 = arg340_1 = None
        view_317: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_139, _shape_param_344);  mm_139 = _shape_param_344 = None
        convert_element_type_252: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        view_318: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_317, _shape_param_345);  view_317 = _shape_param_345 = None
        permute_137: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
        clone_52: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_319: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_346);  clone_52 = _shape_param_346 = None
        bmm_52: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg341_1, view_319);  arg341_1 = None
        bmm_53: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_319, arg342_1);  view_319 = arg342_1 = None
        view_320: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_52, _shape_param_347);  bmm_52 = _shape_param_347 = None
        view_321: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_53, _shape_param_348);  bmm_53 = _shape_param_348 = None
        convert_element_type_253: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.bfloat16);  arg343_1 = None
        mul_317: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_253, 1.1111111111111112);  convert_element_type_253 = None
        mul_318: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_321, mul_317);  view_321 = mul_317 = None
        convert_element_type_254: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_318, torch.float32);  mul_318 = None
        convert_element_type_255: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg344_1, torch.float32);  arg344_1 = None
        sub_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_255, arg345_1);  convert_element_type_255 = arg345_1 = None
        exp_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        div_37: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_14, arg346_1);  exp_14 = arg346_1 = None
        mul_319: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_254, div_37);  convert_element_type_254 = None
        sum_62: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [-1], True)
        neg_13: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_37);  div_37 = None
        fma_13: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_13, sum_62, mul_319);  neg_13 = sum_62 = mul_319 = None
        convert_element_type_256: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None
        view_322: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_256, _shape_param_349);  convert_element_type_256 = _shape_param_349 = None
        view_323: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_322, _shape_param_350);  view_322 = _shape_param_350 = None
        view_324: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_323, _shape_param_351);  _shape_param_351 = None
        convert_element_type_257: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_323, torch.float32);  view_323 = None
        add_79: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(convert_element_type_236, convert_element_type_257);  convert_element_type_236 = convert_element_type_257 = None
        bmm_54: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg347_1, view_324);  arg347_1 = None
        bmm_55: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_324, arg348_1);  view_324 = arg348_1 = None
        view_325: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_54, _shape_param_352);  bmm_54 = _shape_param_352 = None
        view_326: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_55, _shape_param_353);  bmm_55 = _shape_param_353 = None
        permute_138: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_325, [0, 1, 3, 2]);  view_325 = None
        permute_139: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
        clone_53: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
        view_327: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_53, _shape_param_354);  clone_53 = _shape_param_354 = None
        view_328: "bf16[8192, 512]" = torch.ops.aten.view.default(view_327, _shape_param_355);  view_327 = _shape_param_355 = None
        permute_140: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_328, [1, 0])
        mm_140: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_140, arg349_1);  permute_140 = None
        mm_141: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_328, arg350_1);  view_328 = arg350_1 = None
        view_329: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_141, _shape_param_356);  mm_141 = _shape_param_356 = None
        convert_element_type_258: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_329, torch.float32);  view_329 = None
        convert_element_type_259: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        permute_141: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_138, [0, 2, 1, 3]);  permute_138 = None
        view_330: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_141, _shape_param_357);  permute_141 = _shape_param_357 = None
        clone_54: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_330, memory_format = torch.contiguous_format);  view_330 = None
        view_331: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_54, _shape_param_358);  clone_54 = _shape_param_358 = None
        permute_142: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_331, [1, 0])
        mm_142: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_142, arg349_1);  permute_142 = None
        mm_143: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_331, arg351_1);  view_331 = arg351_1 = None
        view_332: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_143, _shape_param_359);  mm_143 = _shape_param_359 = None
        convert_element_type_260: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_332, torch.float32);  view_332 = None
        add_80: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_258, convert_element_type_260);  convert_element_type_258 = convert_element_type_260 = None
        convert_element_type_261: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        permute_143: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        clone_55: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_143, memory_format = torch.contiguous_format);  permute_143 = None
        view_333: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_55, _shape_param_360);  clone_55 = _shape_param_360 = None
        view_334: "bf16[8192, 512]" = torch.ops.aten.view.default(view_333, _shape_param_361);  view_333 = _shape_param_361 = None
        permute_144: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_334, [1, 0])
        mm_144: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_144, arg349_1);  permute_144 = arg349_1 = None
        mm_145: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_334, arg352_1);  view_334 = arg352_1 = None
        view_335: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_145, _shape_param_362);  mm_145 = _shape_param_362 = None
        convert_element_type_262: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_335, torch.float32);  view_335 = None
        add_81: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_80, convert_element_type_262);  add_80 = convert_element_type_262 = None
        convert_element_type_263: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        mul_320: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, arg353_1);  arg353_1 = None
        mul_321: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg354_1, arg355_1)
        mul_322: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_81, mul_321);  add_81 = mul_321 = None
        sum_63: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_322, [0, 1], True, dtype = torch.float32);  mul_322 = None
        view_336: "f32[512]" = torch.ops.aten.view.default(sum_63, _shape_param_363);  sum_63 = _shape_param_363 = None
        mul_323: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_320, arg354_1)
        mul_324: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_320, arg355_1);  mul_320 = None
        sum_64: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_323, [2], True, dtype = torch.float32);  mul_323 = None
        add_82: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_78, mul_324);  add_78 = mul_324 = None
        pow_47: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg355_1, 3);  arg355_1 = None
        mul_325: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_64, -0.5);  sum_64 = None
        mul_326: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_325, pow_47);  mul_325 = pow_47 = None
        expand_25: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_326, _shape_param_364);  mul_326 = _shape_param_364 = None
        div_38: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_25, 512);  expand_25 = None
        pow_48: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg354_1, 1.0);  arg354_1 = None
        mul_327: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_48, 2.0);  pow_48 = None
        mul_328: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_38, mul_327);  div_38 = mul_327 = None
        add_83: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_82, mul_328);  add_82 = mul_328 = None
        convert_element_type_264: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16)
        convert_element_type_265: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg356_1, torch.bfloat16);  arg356_1 = None
        mul_329: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_265, 1.1111111111111112);  convert_element_type_265 = None
        mul_330: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_264, mul_329);  convert_element_type_264 = mul_329 = None
        view_337: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_330, _shape_param_365);  mul_330 = _shape_param_365 = None
        permute_145: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_337, [1, 0])
        mm_146: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_145, arg357_1);  permute_145 = arg357_1 = None
        mm_147: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_337, arg358_1);  view_337 = arg358_1 = None
        view_338: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_147, _shape_param_366);  mm_147 = _shape_param_366 = None
        convert_element_type_266: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_146, torch.float32);  mm_146 = None
        convert_element_type_267: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_338, torch.bfloat16);  view_338 = None
        convert_element_type_268: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg359_1, torch.bfloat16);  arg359_1 = None
        mul_331: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_268, 1.1111111111111112);  convert_element_type_268 = None
        mul_332: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_267, mul_331);  convert_element_type_267 = mul_331 = None
        where_12: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg360_1, full_2, mul_332);  arg360_1 = mul_332 = None
        view_339: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_12, _shape_param_367);  where_12 = _shape_param_367 = None
        permute_146: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_148: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_146, arg361_1);  permute_146 = arg361_1 = None
        mm_149: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_339, arg362_1);  view_339 = arg362_1 = None
        view_340: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_149, _shape_param_368);  mm_149 = _shape_param_368 = None
        convert_element_type_269: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_340, torch.float32);  view_340 = None
        convert_element_type_270: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_148, torch.float32);  mm_148 = None
        mul_333: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_269, arg363_1);  arg363_1 = None
        mul_334: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg364_1, arg365_1)
        mul_335: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_269, mul_334);  convert_element_type_269 = mul_334 = None
        sum_65: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_335, [0, 1], True, dtype = torch.float32);  mul_335 = None
        view_341: "f32[512]" = torch.ops.aten.view.default(sum_65, _shape_param_369);  sum_65 = _shape_param_369 = None
        mul_336: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_333, arg364_1)
        mul_337: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_333, arg365_1);  mul_333 = None
        sum_66: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_336, [2], True, dtype = torch.float32);  mul_336 = None
        add_84: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_83, mul_337);  add_83 = mul_337 = None
        pow_49: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg365_1, 3);  arg365_1 = None
        mul_338: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_66, -0.5);  sum_66 = None
        mul_339: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_338, pow_49);  mul_338 = pow_49 = None
        expand_26: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_339, _shape_param_370);  mul_339 = _shape_param_370 = None
        div_39: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_26, 512);  expand_26 = None
        pow_50: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg364_1, 1.0);  arg364_1 = None
        mul_340: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_50, 2.0);  pow_50 = None
        mul_341: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_39, mul_340);  div_39 = mul_340 = None
        add_85: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_84, mul_341);  add_84 = mul_341 = None
        convert_element_type_271: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16)
        convert_element_type_272: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg366_1, torch.bfloat16);  arg366_1 = None
        mul_342: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_272, 1.1111111111111112);  convert_element_type_272 = None
        mul_343: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_271, mul_342);  convert_element_type_271 = mul_342 = None
        view_342: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_343, _shape_param_371);  mul_343 = _shape_param_371 = None
        permute_147: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_342, [1, 0])
        mm_150: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_147, arg367_1);  permute_147 = arg367_1 = None
        mm_151: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_342, arg368_1);  view_342 = arg368_1 = None
        view_343: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_151, _shape_param_372);  mm_151 = _shape_param_372 = None
        convert_element_type_273: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        view_344: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_343, _shape_param_373);  view_343 = _shape_param_373 = None
        permute_148: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        clone_56: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
        view_345: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_56, _shape_param_374);  clone_56 = _shape_param_374 = None
        bmm_56: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg369_1, view_345);  arg369_1 = None
        bmm_57: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_345, arg370_1);  view_345 = arg370_1 = None
        view_346: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_56, _shape_param_375);  bmm_56 = _shape_param_375 = None
        view_347: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_57, _shape_param_376);  bmm_57 = _shape_param_376 = None
        convert_element_type_274: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg371_1, torch.bfloat16);  arg371_1 = None
        mul_344: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_274, 1.1111111111111112);  convert_element_type_274 = None
        mul_345: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_347, mul_344);  view_347 = mul_344 = None
        convert_element_type_275: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_345, torch.float32);  mul_345 = None
        convert_element_type_276: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg372_1, torch.float32);  arg372_1 = None
        sub_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_276, arg373_1);  convert_element_type_276 = arg373_1 = None
        exp_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        div_40: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_15, arg374_1);  exp_15 = arg374_1 = None
        mul_346: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_275, div_40);  convert_element_type_275 = None
        sum_67: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [-1], True)
        neg_14: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_40);  div_40 = None
        fma_14: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_14, sum_67, mul_346);  neg_14 = sum_67 = mul_346 = None
        convert_element_type_277: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None
        view_348: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_277, _shape_param_377);  convert_element_type_277 = _shape_param_377 = None
        view_349: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_348, _shape_param_378);  view_348 = _shape_param_378 = None
        view_350: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_349, _shape_param_379);  _shape_param_379 = None
        convert_element_type_278: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        add_86: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_79, convert_element_type_278);  add_79 = convert_element_type_278 = None
        bmm_58: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg375_1, view_350);  arg375_1 = None
        bmm_59: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_350, arg376_1);  view_350 = arg376_1 = None
        view_351: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_58, _shape_param_380);  bmm_58 = _shape_param_380 = None
        view_352: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_59, _shape_param_381);  bmm_59 = _shape_param_381 = None
        permute_149: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_351, [0, 1, 3, 2]);  view_351 = None
        permute_150: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
        clone_57: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_353: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_57, _shape_param_382);  clone_57 = _shape_param_382 = None
        view_354: "bf16[8192, 512]" = torch.ops.aten.view.default(view_353, _shape_param_383);  view_353 = _shape_param_383 = None
        permute_151: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_354, [1, 0])
        mm_152: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_151, arg377_1);  permute_151 = None
        mm_153: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_354, arg378_1);  view_354 = arg378_1 = None
        view_355: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_153, _shape_param_384);  mm_153 = _shape_param_384 = None
        convert_element_type_279: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_355, torch.float32);  view_355 = None
        convert_element_type_280: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        permute_152: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_149, [0, 2, 1, 3]);  permute_149 = None
        view_356: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_152, _shape_param_385);  permute_152 = _shape_param_385 = None
        clone_58: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_356, memory_format = torch.contiguous_format);  view_356 = None
        view_357: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_58, _shape_param_386);  clone_58 = _shape_param_386 = None
        permute_153: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_357, [1, 0])
        mm_154: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_153, arg377_1);  permute_153 = None
        mm_155: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_357, arg379_1);  view_357 = arg379_1 = None
        view_358: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_155, _shape_param_387);  mm_155 = _shape_param_387 = None
        convert_element_type_281: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_358, torch.float32);  view_358 = None
        add_87: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_279, convert_element_type_281);  convert_element_type_279 = convert_element_type_281 = None
        convert_element_type_282: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        permute_154: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None
        clone_59: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_154, memory_format = torch.contiguous_format);  permute_154 = None
        view_359: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_59, _shape_param_388);  clone_59 = _shape_param_388 = None
        view_360: "bf16[8192, 512]" = torch.ops.aten.view.default(view_359, _shape_param_389);  view_359 = _shape_param_389 = None
        permute_155: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_360, [1, 0])
        mm_156: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_155, arg377_1);  permute_155 = arg377_1 = None
        mm_157: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_360, arg380_1);  view_360 = arg380_1 = None
        view_361: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_157, _shape_param_390);  mm_157 = _shape_param_390 = None
        convert_element_type_283: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_361, torch.float32);  view_361 = None
        add_88: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_87, convert_element_type_283);  add_87 = convert_element_type_283 = None
        convert_element_type_284: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        mul_347: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_88, arg381_1);  arg381_1 = None
        mul_348: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg382_1, arg383_1)
        mul_349: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_88, mul_348);  add_88 = mul_348 = None
        sum_68: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_349, [0, 1], True, dtype = torch.float32);  mul_349 = None
        view_362: "f32[512]" = torch.ops.aten.view.default(sum_68, _shape_param_391);  sum_68 = _shape_param_391 = None
        mul_350: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_347, arg382_1)
        mul_351: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_347, arg383_1);  mul_347 = None
        sum_69: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_350, [2], True, dtype = torch.float32);  mul_350 = None
        add_89: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_85, mul_351);  add_85 = mul_351 = None
        pow_51: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg383_1, 3);  arg383_1 = None
        mul_352: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_69, -0.5);  sum_69 = None
        mul_353: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_352, pow_51);  mul_352 = pow_51 = None
        expand_27: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_353, _shape_param_392);  mul_353 = _shape_param_392 = None
        div_41: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_27, 512);  expand_27 = None
        pow_52: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg382_1, 1.0);  arg382_1 = None
        mul_354: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_52, 2.0);  pow_52 = None
        mul_355: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_41, mul_354);  div_41 = mul_354 = None
        add_90: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_89, mul_355);  add_89 = mul_355 = None
        convert_element_type_285: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16)
        convert_element_type_286: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg384_1, torch.bfloat16);  arg384_1 = None
        mul_356: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_286, 1.1111111111111112);  convert_element_type_286 = None
        mul_357: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_285, mul_356);  convert_element_type_285 = mul_356 = None
        view_363: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_357, _shape_param_393);  mul_357 = _shape_param_393 = None
        permute_156: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_363, [1, 0])
        mm_158: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_156, arg385_1);  permute_156 = arg385_1 = None
        mm_159: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_363, arg386_1);  view_363 = arg386_1 = None
        view_364: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_159, _shape_param_394);  mm_159 = _shape_param_394 = None
        convert_element_type_287: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_158, torch.float32);  mm_158 = None
        convert_element_type_288: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_364, torch.bfloat16);  view_364 = None
        convert_element_type_289: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg387_1, torch.bfloat16);  arg387_1 = None
        mul_358: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_289, 1.1111111111111112);  convert_element_type_289 = None
        mul_359: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_288, mul_358);  convert_element_type_288 = mul_358 = None
        where_13: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg388_1, full_2, mul_359);  arg388_1 = mul_359 = None
        view_365: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_13, _shape_param_395);  where_13 = _shape_param_395 = None
        permute_157: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_365, [1, 0])
        mm_160: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_157, arg389_1);  permute_157 = arg389_1 = None
        mm_161: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_365, arg390_1);  view_365 = arg390_1 = None
        view_366: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_161, _shape_param_396);  mm_161 = _shape_param_396 = None
        convert_element_type_290: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_366, torch.float32);  view_366 = None
        convert_element_type_291: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        mul_360: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_290, arg391_1);  arg391_1 = None
        mul_361: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg392_1, arg393_1)
        mul_362: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_290, mul_361);  convert_element_type_290 = mul_361 = None
        sum_70: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 1], True, dtype = torch.float32);  mul_362 = None
        view_367: "f32[512]" = torch.ops.aten.view.default(sum_70, _shape_param_397);  sum_70 = _shape_param_397 = None
        mul_363: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_360, arg392_1)
        mul_364: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_360, arg393_1);  mul_360 = None
        sum_71: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_363, [2], True, dtype = torch.float32);  mul_363 = None
        add_91: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_90, mul_364);  add_90 = mul_364 = None
        pow_53: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg393_1, 3);  arg393_1 = None
        mul_365: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_71, -0.5);  sum_71 = None
        mul_366: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_365, pow_53);  mul_365 = pow_53 = None
        expand_28: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_366, _shape_param_398);  mul_366 = _shape_param_398 = None
        div_42: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_28, 512);  expand_28 = None
        pow_54: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg392_1, 1.0);  arg392_1 = None
        mul_367: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_54, 2.0);  pow_54 = None
        mul_368: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_42, mul_367);  div_42 = mul_367 = None
        add_92: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_91, mul_368);  add_91 = mul_368 = None
        convert_element_type_292: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_92, torch.bfloat16)
        convert_element_type_293: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg394_1, torch.bfloat16);  arg394_1 = None
        mul_369: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_293, 1.1111111111111112);  convert_element_type_293 = None
        mul_370: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_292, mul_369);  convert_element_type_292 = mul_369 = None
        view_368: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_370, _shape_param_399);  mul_370 = _shape_param_399 = None
        permute_158: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_368, [1, 0])
        mm_162: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_158, arg395_1);  permute_158 = arg395_1 = None
        mm_163: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_368, arg396_1);  view_368 = arg396_1 = None
        view_369: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_163, _shape_param_400);  mm_163 = _shape_param_400 = None
        convert_element_type_294: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        view_370: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_369, _shape_param_401);  view_369 = _shape_param_401 = None
        permute_159: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
        clone_60: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
        view_371: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_402);  clone_60 = _shape_param_402 = None
        bmm_60: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg397_1, view_371);  arg397_1 = None
        bmm_61: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_371, arg398_1);  view_371 = arg398_1 = None
        view_372: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_60, _shape_param_403);  bmm_60 = _shape_param_403 = None
        view_373: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_61, _shape_param_404);  bmm_61 = _shape_param_404 = None
        convert_element_type_295: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg399_1, torch.bfloat16);  arg399_1 = None
        mul_371: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_295, 1.1111111111111112);  convert_element_type_295 = None
        mul_372: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_373, mul_371);  view_373 = mul_371 = None
        convert_element_type_296: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_372, torch.float32);  mul_372 = None
        convert_element_type_297: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg400_1, torch.float32);  arg400_1 = None
        sub_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_297, arg401_1);  convert_element_type_297 = arg401_1 = None
        exp_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        div_43: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_16, arg402_1);  exp_16 = arg402_1 = None
        mul_373: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_296, div_43);  convert_element_type_296 = None
        sum_72: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_373, [-1], True)
        neg_15: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_43);  div_43 = None
        fma_15: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_15, sum_72, mul_373);  neg_15 = sum_72 = mul_373 = None
        convert_element_type_298: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None
        view_374: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_298, _shape_param_405);  convert_element_type_298 = _shape_param_405 = None
        view_375: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_374, _shape_param_406);  view_374 = _shape_param_406 = None
        view_376: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_375, _shape_param_407);  _shape_param_407 = None
        convert_element_type_299: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_375, torch.float32);  view_375 = None
        add_93: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_86, convert_element_type_299);  add_86 = convert_element_type_299 = None
        bmm_62: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg403_1, view_376);  arg403_1 = None
        bmm_63: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_376, arg404_1);  view_376 = arg404_1 = None
        view_377: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_62, _shape_param_408);  bmm_62 = _shape_param_408 = None
        view_378: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_63, _shape_param_409);  bmm_63 = _shape_param_409 = None
        permute_160: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_377, [0, 1, 3, 2]);  view_377 = None
        permute_161: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_372, [0, 2, 1, 3]);  view_372 = None
        clone_61: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
        view_379: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_61, _shape_param_410);  clone_61 = _shape_param_410 = None
        view_380: "bf16[8192, 512]" = torch.ops.aten.view.default(view_379, _shape_param_411);  view_379 = _shape_param_411 = None
        permute_162: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_164: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_162, arg405_1);  permute_162 = None
        mm_165: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_380, arg406_1);  view_380 = arg406_1 = None
        view_381: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_165, _shape_param_412);  mm_165 = _shape_param_412 = None
        convert_element_type_300: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_381, torch.float32);  view_381 = None
        convert_element_type_301: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        permute_163: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_160, [0, 2, 1, 3]);  permute_160 = None
        view_382: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_163, _shape_param_413);  permute_163 = _shape_param_413 = None
        clone_62: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_382, memory_format = torch.contiguous_format);  view_382 = None
        view_383: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_62, _shape_param_414);  clone_62 = _shape_param_414 = None
        permute_164: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_383, [1, 0])
        mm_166: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_164, arg405_1);  permute_164 = None
        mm_167: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_383, arg407_1);  view_383 = arg407_1 = None
        view_384: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_167, _shape_param_415);  mm_167 = _shape_param_415 = None
        convert_element_type_302: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_384, torch.float32);  view_384 = None
        add_94: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_300, convert_element_type_302);  convert_element_type_300 = convert_element_type_302 = None
        convert_element_type_303: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        permute_165: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        clone_63: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_165, memory_format = torch.contiguous_format);  permute_165 = None
        view_385: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_63, _shape_param_416);  clone_63 = _shape_param_416 = None
        view_386: "bf16[8192, 512]" = torch.ops.aten.view.default(view_385, _shape_param_417);  view_385 = _shape_param_417 = None
        permute_166: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_386, [1, 0])
        mm_168: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_166, arg405_1);  permute_166 = arg405_1 = None
        mm_169: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_386, arg408_1);  view_386 = arg408_1 = None
        view_387: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_169, _shape_param_418);  mm_169 = _shape_param_418 = None
        convert_element_type_304: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_387, torch.float32);  view_387 = None
        add_95: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_94, convert_element_type_304);  add_94 = convert_element_type_304 = None
        convert_element_type_305: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        mul_374: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_95, arg409_1);  arg409_1 = None
        mul_375: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg410_1, arg411_1)
        mul_376: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_95, mul_375);  add_95 = mul_375 = None
        sum_73: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1], True, dtype = torch.float32);  mul_376 = None
        view_388: "f32[512]" = torch.ops.aten.view.default(sum_73, _shape_param_419);  sum_73 = _shape_param_419 = None
        mul_377: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_374, arg410_1)
        mul_378: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_374, arg411_1);  mul_374 = None
        sum_74: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True, dtype = torch.float32);  mul_377 = None
        add_96: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_92, mul_378);  add_92 = mul_378 = None
        pow_55: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg411_1, 3);  arg411_1 = None
        mul_379: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_74, -0.5);  sum_74 = None
        mul_380: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_379, pow_55);  mul_379 = pow_55 = None
        expand_29: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_380, _shape_param_420);  mul_380 = _shape_param_420 = None
        div_44: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_29, 512);  expand_29 = None
        pow_56: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg410_1, 1.0);  arg410_1 = None
        mul_381: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_56, 2.0);  pow_56 = None
        mul_382: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_44, mul_381);  div_44 = mul_381 = None
        add_97: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_96, mul_382);  add_96 = mul_382 = None
        convert_element_type_306: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_97, torch.bfloat16)
        convert_element_type_307: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg412_1, torch.bfloat16);  arg412_1 = None
        mul_383: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_307, 1.1111111111111112);  convert_element_type_307 = None
        mul_384: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_306, mul_383);  convert_element_type_306 = mul_383 = None
        view_389: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_384, _shape_param_421);  mul_384 = _shape_param_421 = None
        permute_167: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_389, [1, 0])
        mm_170: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_167, arg413_1);  permute_167 = arg413_1 = None
        mm_171: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_389, arg414_1);  view_389 = arg414_1 = None
        view_390: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_171, _shape_param_422);  mm_171 = _shape_param_422 = None
        convert_element_type_308: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_170, torch.float32);  mm_170 = None
        convert_element_type_309: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_390, torch.bfloat16);  view_390 = None
        convert_element_type_310: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg415_1, torch.bfloat16);  arg415_1 = None
        mul_385: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_310, 1.1111111111111112);  convert_element_type_310 = None
        mul_386: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_309, mul_385);  convert_element_type_309 = mul_385 = None
        where_14: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg416_1, full_2, mul_386);  arg416_1 = mul_386 = None
        view_391: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_14, _shape_param_423);  where_14 = _shape_param_423 = None
        permute_168: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_391, [1, 0])
        mm_172: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_168, arg417_1);  permute_168 = arg417_1 = None
        mm_173: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_391, arg418_1);  view_391 = arg418_1 = None
        view_392: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_173, _shape_param_424);  mm_173 = _shape_param_424 = None
        convert_element_type_311: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_392, torch.float32);  view_392 = None
        convert_element_type_312: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_172, torch.float32);  mm_172 = None
        mul_387: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_311, arg419_1);  arg419_1 = None
        mul_388: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg420_1, arg421_1)
        mul_389: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_311, mul_388);  convert_element_type_311 = mul_388 = None
        sum_75: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_389, [0, 1], True, dtype = torch.float32);  mul_389 = None
        view_393: "f32[512]" = torch.ops.aten.view.default(sum_75, _shape_param_425);  sum_75 = _shape_param_425 = None
        mul_390: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_387, arg420_1)
        mul_391: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_387, arg421_1);  mul_387 = None
        sum_76: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True, dtype = torch.float32);  mul_390 = None
        add_98: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_97, mul_391);  add_97 = mul_391 = None
        pow_57: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg421_1, 3);  arg421_1 = None
        mul_392: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_76, -0.5);  sum_76 = None
        mul_393: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_392, pow_57);  mul_392 = pow_57 = None
        expand_30: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_393, _shape_param_426);  mul_393 = _shape_param_426 = None
        div_45: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_30, 512);  expand_30 = None
        pow_58: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg420_1, 1.0);  arg420_1 = None
        mul_394: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_58, 2.0);  pow_58 = None
        mul_395: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_45, mul_394);  div_45 = mul_394 = None
        add_99: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_98, mul_395);  add_98 = mul_395 = None
        convert_element_type_313: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16)
        convert_element_type_314: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg422_1, torch.bfloat16);  arg422_1 = None
        mul_396: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_314, 1.1111111111111112);  convert_element_type_314 = None
        mul_397: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_313, mul_396);  convert_element_type_313 = mul_396 = None
        view_394: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_397, _shape_param_427);  mul_397 = _shape_param_427 = None
        permute_169: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_394, [1, 0])
        mm_174: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_169, arg423_1);  permute_169 = arg423_1 = None
        mm_175: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_394, arg424_1);  view_394 = arg424_1 = None
        view_395: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_175, _shape_param_428);  mm_175 = _shape_param_428 = None
        convert_element_type_315: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        view_396: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_395, _shape_param_429);  view_395 = _shape_param_429 = None
        permute_170: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None
        clone_64: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_397: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_430);  clone_64 = _shape_param_430 = None
        bmm_64: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg425_1, view_397);  arg425_1 = None
        bmm_65: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_397, arg426_1);  view_397 = arg426_1 = None
        view_398: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_64, _shape_param_431);  bmm_64 = _shape_param_431 = None
        view_399: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_65, _shape_param_432);  bmm_65 = _shape_param_432 = None
        convert_element_type_316: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg427_1, torch.bfloat16);  arg427_1 = None
        mul_398: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_316, 1.1111111111111112);  convert_element_type_316 = None
        mul_399: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_399, mul_398);  view_399 = mul_398 = None
        convert_element_type_317: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_399, torch.float32);  mul_399 = None
        convert_element_type_318: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg428_1, torch.float32);  arg428_1 = None
        sub_19: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_318, arg429_1);  convert_element_type_318 = arg429_1 = None
        exp_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        div_46: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_17, arg430_1);  exp_17 = arg430_1 = None
        mul_400: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_317, div_46);  convert_element_type_317 = None
        sum_77: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_400, [-1], True)
        neg_16: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_46);  div_46 = None
        fma_16: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_16, sum_77, mul_400);  neg_16 = sum_77 = mul_400 = None
        convert_element_type_319: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None
        view_400: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_319, _shape_param_433);  convert_element_type_319 = _shape_param_433 = None
        view_401: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_400, _shape_param_434);  view_400 = _shape_param_434 = None
        view_402: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_401, _shape_param_435);  _shape_param_435 = None
        convert_element_type_320: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_401, torch.float32);  view_401 = None
        add_100: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_93, convert_element_type_320);  add_93 = convert_element_type_320 = None
        bmm_66: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg431_1, view_402);  arg431_1 = None
        bmm_67: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_402, arg432_1);  view_402 = arg432_1 = None
        view_403: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_66, _shape_param_436);  bmm_66 = _shape_param_436 = None
        view_404: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_67, _shape_param_437);  bmm_67 = _shape_param_437 = None
        permute_171: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_403, [0, 1, 3, 2]);  view_403 = None
        permute_172: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
        clone_65: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
        view_405: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_65, _shape_param_438);  clone_65 = _shape_param_438 = None
        view_406: "bf16[8192, 512]" = torch.ops.aten.view.default(view_405, _shape_param_439);  view_405 = _shape_param_439 = None
        permute_173: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_406, [1, 0])
        mm_176: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_173, arg433_1);  permute_173 = None
        mm_177: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_406, arg434_1);  view_406 = arg434_1 = None
        view_407: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_177, _shape_param_440);  mm_177 = _shape_param_440 = None
        convert_element_type_321: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_407, torch.float32);  view_407 = None
        convert_element_type_322: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        permute_174: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_171, [0, 2, 1, 3]);  permute_171 = None
        view_408: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_174, _shape_param_441);  permute_174 = _shape_param_441 = None
        clone_66: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_408, memory_format = torch.contiguous_format);  view_408 = None
        view_409: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_66, _shape_param_442);  clone_66 = _shape_param_442 = None
        permute_175: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_409, [1, 0])
        mm_178: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_175, arg433_1);  permute_175 = None
        mm_179: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_409, arg435_1);  view_409 = arg435_1 = None
        view_410: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_179, _shape_param_443);  mm_179 = _shape_param_443 = None
        convert_element_type_323: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_410, torch.float32);  view_410 = None
        add_101: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_321, convert_element_type_323);  convert_element_type_321 = convert_element_type_323 = None
        convert_element_type_324: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        permute_176: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
        clone_67: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_176, memory_format = torch.contiguous_format);  permute_176 = None
        view_411: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_67, _shape_param_444);  clone_67 = _shape_param_444 = None
        view_412: "bf16[8192, 512]" = torch.ops.aten.view.default(view_411, _shape_param_445);  view_411 = _shape_param_445 = None
        permute_177: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_412, [1, 0])
        mm_180: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_177, arg433_1);  permute_177 = arg433_1 = None
        mm_181: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_412, arg436_1);  view_412 = arg436_1 = None
        view_413: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_181, _shape_param_446);  mm_181 = _shape_param_446 = None
        convert_element_type_325: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_413, torch.float32);  view_413 = None
        add_102: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_101, convert_element_type_325);  add_101 = convert_element_type_325 = None
        convert_element_type_326: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None
        mul_401: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_102, arg437_1);  arg437_1 = None
        mul_402: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg438_1, arg439_1)
        mul_403: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_102, mul_402);  add_102 = mul_402 = None
        sum_78: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1], True, dtype = torch.float32);  mul_403 = None
        view_414: "f32[512]" = torch.ops.aten.view.default(sum_78, _shape_param_447);  sum_78 = _shape_param_447 = None
        mul_404: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_401, arg438_1)
        mul_405: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_401, arg439_1);  mul_401 = None
        sum_79: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True, dtype = torch.float32);  mul_404 = None
        add_103: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_99, mul_405);  add_99 = mul_405 = None
        pow_59: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg439_1, 3);  arg439_1 = None
        mul_406: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_79, -0.5);  sum_79 = None
        mul_407: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_406, pow_59);  mul_406 = pow_59 = None
        expand_31: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_407, _shape_param_448);  mul_407 = _shape_param_448 = None
        div_47: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_31, 512);  expand_31 = None
        pow_60: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg438_1, 1.0);  arg438_1 = None
        mul_408: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
        mul_409: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_47, mul_408);  div_47 = mul_408 = None
        add_104: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_103, mul_409);  add_103 = mul_409 = None
        convert_element_type_327: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16)
        convert_element_type_328: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg440_1, torch.bfloat16);  arg440_1 = None
        mul_410: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_328, 1.1111111111111112);  convert_element_type_328 = None
        mul_411: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_327, mul_410);  convert_element_type_327 = mul_410 = None
        view_415: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_411, _shape_param_449);  mul_411 = _shape_param_449 = None
        permute_178: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_415, [1, 0])
        mm_182: "bf16[512, 2048]" = torch.ops.aten.mm.default(permute_178, arg441_1);  permute_178 = arg441_1 = None
        mm_183: "bf16[8192, 2048]" = torch.ops.aten.mm.default(view_415, arg442_1);  view_415 = arg442_1 = None
        view_416: "bf16[8, 1024, 2048]" = torch.ops.aten.view.default(mm_183, _shape_param_450);  mm_183 = _shape_param_450 = None
        convert_element_type_329: "f32[512, 2048]" = torch.ops.prims.convert_element_type.default(mm_182, torch.float32);  mm_182 = None
        convert_element_type_330: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(view_416, torch.bfloat16);  view_416 = None
        convert_element_type_331: "bf16[8, 1024, 2048]" = torch.ops.prims.convert_element_type.default(arg443_1, torch.bfloat16);  arg443_1 = None
        mul_412: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_331, 1.1111111111111112);  convert_element_type_331 = None
        mul_413: "bf16[8, 1024, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_330, mul_412);  convert_element_type_330 = mul_412 = None
        where_15: "bf16[8, 1024, 2048]" = torch.ops.aten.where.self(arg444_1, full_2, mul_413);  arg444_1 = full_2 = mul_413 = None
        view_417: "bf16[8192, 2048]" = torch.ops.aten.view.default(where_15, _shape_param_451);  where_15 = _shape_param_451 = None
        permute_179: "bf16[2048, 8192]" = torch.ops.aten.permute.default(view_417, [1, 0])
        mm_184: "bf16[2048, 512]" = torch.ops.aten.mm.default(permute_179, arg445_1);  permute_179 = arg445_1 = None
        mm_185: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_417, arg446_1);  view_417 = arg446_1 = None
        view_418: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_185, _shape_param_452);  mm_185 = _shape_param_452 = None
        convert_element_type_332: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_418, torch.float32);  view_418 = None
        convert_element_type_333: "f32[2048, 512]" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        mul_414: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_332, arg447_1);  arg447_1 = None
        mul_415: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg448_1, arg449_1)
        mul_416: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_332, mul_415);  convert_element_type_332 = mul_415 = None
        sum_80: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1], True, dtype = torch.float32);  mul_416 = None
        view_419: "f32[512]" = torch.ops.aten.view.default(sum_80, _shape_param_453);  sum_80 = _shape_param_453 = None
        mul_417: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_414, arg448_1)
        mul_418: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_414, arg449_1);  mul_414 = None
        sum_81: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_417, [2], True, dtype = torch.float32);  mul_417 = None
        add_105: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_104, mul_418);  add_104 = mul_418 = None
        pow_61: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg449_1, 3);  arg449_1 = None
        mul_419: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_81, -0.5);  sum_81 = None
        mul_420: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_419, pow_61);  mul_419 = pow_61 = None
        expand_32: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_420, _shape_param_454);  mul_420 = _shape_param_454 = None
        div_48: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_32, 512);  expand_32 = None
        pow_62: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(arg448_1, 1.0);  arg448_1 = None
        mul_421: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_62, 2.0);  pow_62 = None
        mul_422: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_48, mul_421);  div_48 = mul_421 = None
        add_106: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_105, mul_422);  add_105 = mul_422 = None
        convert_element_type_334: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16)
        convert_element_type_335: "bf16[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg450_1, torch.bfloat16);  arg450_1 = None
        mul_423: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_335, 1.1111111111111112);  convert_element_type_335 = None
        mul_424: "bf16[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_334, mul_423);  convert_element_type_334 = mul_423 = None
        view_420: "bf16[8192, 512]" = torch.ops.aten.view.default(mul_424, _shape_param_455);  mul_424 = _shape_param_455 = None
        permute_180: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_420, [1, 0])
        mm_186: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_180, arg451_1);  permute_180 = arg451_1 = None
        mm_187: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_420, arg452_1);  view_420 = arg452_1 = None
        view_421: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_187, _shape_param_456);  mm_187 = _shape_param_456 = None
        convert_element_type_336: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        view_422: "bf16[8, 1024, 8, 64]" = torch.ops.aten.view.default(view_421, _shape_param_457);  view_421 = _shape_param_457 = None
        permute_181: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
        clone_68: "bf16[8, 8, 1024, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
        view_423: "bf16[64, 1024, 64]" = torch.ops.aten.view.default(clone_68, _shape_param_458);  clone_68 = _shape_param_458 = None
        bmm_68: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(arg453_1, view_423);  arg453_1 = None
        bmm_69: "bf16[64, 1024, 1024]" = torch.ops.aten.bmm.default(view_423, arg454_1);  view_423 = arg454_1 = None
        view_424: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_68, _shape_param_459);  bmm_68 = _shape_param_459 = None
        view_425: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(bmm_69, _shape_param_460);  bmm_69 = _shape_param_460 = None
        convert_element_type_337: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(arg455_1, torch.bfloat16);  arg455_1 = None
        mul_425: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_337, 1.1111111111111112);  convert_element_type_337 = None
        mul_426: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(view_425, mul_425);  view_425 = mul_425 = None
        convert_element_type_338: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(mul_426, torch.float32);  mul_426 = None
        full_7: "f32[8, 1, 1024, 1024]" = torch.ops.aten.full.default(_shape_param_461, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_461 = None
        view_426: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(arg456_1, _shape_param_462);  arg456_1 = _shape_param_462 = None
        permute_182: "f32[8, 1024, 1024]" = torch.ops.aten.permute.default(arg457_1, [2, 0, 1]);  arg457_1 = None
        unsqueeze_5: "f32[1, 8, 1024, 1024]" = torch.ops.aten.unsqueeze.default(permute_182, 0);  permute_182 = None
        add_107: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(unsqueeze_5, full_7);  unsqueeze_5 = full_7 = None
        add_108: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(view_426, add_107);  view_426 = add_107 = None
        convert_element_type_339: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16);  add_108 = None
        convert_element_type_340: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(convert_element_type_339, torch.float32);  convert_element_type_339 = None
        sub_20: "f32[8, 8, 1024, 1024]" = torch.ops.aten.sub.Tensor(convert_element_type_340, arg458_1);  convert_element_type_340 = arg458_1 = None
        exp_18: "f32[8, 8, 1024, 1024]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        div_49: "f32[8, 8, 1024, 1024]" = torch.ops.aten.div.Tensor(exp_18, arg459_1);  exp_18 = arg459_1 = None
        mul_427: "f32[8, 8, 1024, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_338, div_49);  convert_element_type_338 = None
        sum_82: "f32[8, 8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [-1], True)
        neg_17: "f32[8, 8, 1024, 1024]" = torch.ops.aten.neg.default(div_49);  div_49 = None
        fma_17: "f32[8, 8, 1024, 1024]" = torch.ops.prims.fma.default(neg_17, sum_82, mul_427);  neg_17 = sum_82 = mul_427 = None
        convert_element_type_341: "bf16[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None
        view_427: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(convert_element_type_341, _shape_param_463);  convert_element_type_341 = _shape_param_463 = None
        view_428: "bf16[8, 8, 1024, 1024]" = torch.ops.aten.view.default(view_427, _shape_param_464);  view_427 = _shape_param_464 = None
        view_429: "bf16[64, 1024, 1024]" = torch.ops.aten.view.default(view_428, _shape_param_465);  _shape_param_465 = None
        convert_element_type_342: "f32[8, 8, 1024, 1024]" = torch.ops.prims.convert_element_type.default(view_428, torch.float32);  view_428 = None
        add_109: "f32[8, 8, 1024, 1024]" = torch.ops.aten.add.Tensor(add_100, convert_element_type_342);  add_100 = convert_element_type_342 = None
        sum_83: "f32[1, 8, 1024, 1024]" = torch.ops.aten.sum.dim_IntList(add_109, [0], True, dtype = torch.float32);  add_109 = None
        squeeze_1: "f32[8, 1024, 1024]" = torch.ops.aten.squeeze.dim(sum_83, 0);  sum_83 = None
        permute_183: "f32[1024, 1024, 8]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
        _unsafe_masked_index_put_accumulate_2: "f32[32, 8]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_5, full_4, [arg460_1], permute_183);  full_5 = full_4 = arg460_1 = permute_183 = None
        bmm_70: "bf16[64, 64, 1024]" = torch.ops.aten.bmm.default(arg461_1, view_429);  arg461_1 = None
        bmm_71: "bf16[64, 1024, 64]" = torch.ops.aten.bmm.default(view_429, arg462_1);  view_429 = arg462_1 = None
        view_430: "bf16[8, 8, 64, 1024]" = torch.ops.aten.view.default(bmm_70, _shape_param_466);  bmm_70 = _shape_param_466 = None
        view_431: "bf16[8, 8, 1024, 64]" = torch.ops.aten.view.default(bmm_71, _shape_param_467);  bmm_71 = _shape_param_467 = None
        permute_184: "bf16[8, 8, 1024, 64]" = torch.ops.aten.permute.default(view_430, [0, 1, 3, 2]);  view_430 = None
        permute_185: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        clone_69: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_185, memory_format = torch.contiguous_format);  permute_185 = None
        view_432: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_69, _shape_param_468);  clone_69 = _shape_param_468 = None
        view_433: "bf16[8192, 512]" = torch.ops.aten.view.default(view_432, _shape_param_469);  view_432 = _shape_param_469 = None
        permute_186: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_433, [1, 0])
        mm_188: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_186, arg463_1);  permute_186 = None
        mm_189: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_433, arg464_1);  view_433 = arg464_1 = None
        view_434: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_189, _shape_param_470);  mm_189 = _shape_param_470 = None
        convert_element_type_343: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_434, torch.float32);  view_434 = None
        convert_element_type_344: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        permute_187: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(permute_184, [0, 2, 1, 3]);  permute_184 = None
        view_435: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(permute_187, _shape_param_471);  permute_187 = _shape_param_471 = None
        clone_70: "bf16[8, 1024, 512]" = torch.ops.aten.clone.default(view_435, memory_format = torch.contiguous_format);  view_435 = None
        view_436: "bf16[8192, 512]" = torch.ops.aten.view.default(clone_70, _shape_param_472);  clone_70 = _shape_param_472 = None
        permute_188: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_190: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_188, arg463_1);  permute_188 = None
        mm_191: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_436, arg465_1);  view_436 = arg465_1 = None
        view_437: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_191, _shape_param_473);  mm_191 = _shape_param_473 = None
        convert_element_type_345: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        add_110: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(convert_element_type_343, convert_element_type_345);  convert_element_type_343 = convert_element_type_345 = None
        convert_element_type_346: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        permute_189: "bf16[8, 1024, 8, 64]" = torch.ops.aten.permute.default(view_431, [0, 2, 1, 3]);  view_431 = None
        clone_71: "bf16[8, 1024, 8, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_438: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(clone_71, _shape_param_474);  clone_71 = _shape_param_474 = None
        view_439: "bf16[8192, 512]" = torch.ops.aten.view.default(view_438, _shape_param_475);  view_438 = _shape_param_475 = None
        permute_190: "bf16[512, 8192]" = torch.ops.aten.permute.default(view_439, [1, 0])
        mm_192: "bf16[512, 512]" = torch.ops.aten.mm.default(permute_190, arg463_1);  permute_190 = arg463_1 = None
        mm_193: "bf16[8192, 512]" = torch.ops.aten.mm.default(view_439, arg466_1);  view_439 = arg466_1 = None
        view_440: "bf16[8, 1024, 512]" = torch.ops.aten.view.default(mm_193, _shape_param_476);  mm_193 = _shape_param_476 = None
        convert_element_type_347: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(view_440, torch.float32);  view_440 = None
        add_111: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_110, convert_element_type_347);  add_110 = convert_element_type_347 = None
        convert_element_type_348: "f32[512, 512]" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        mul_428: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_111, arg467_1);  arg467_1 = None
        mul_429: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(arg468_1, arg293_1);  arg293_1 = None
        mul_430: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_429, 1.1111111111111112);  mul_429 = None
        mul_431: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_430, arg469_1)
        mul_432: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_111, mul_431);  add_111 = mul_431 = None
        sum_84: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1], True, dtype = torch.float32);  mul_432 = None
        view_441: "f32[512]" = torch.ops.aten.view.default(sum_84, _shape_param_477);  sum_84 = _shape_param_477 = None
        mul_433: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_428, mul_430)
        mul_434: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(mul_428, arg469_1);  mul_428 = None
        sum_85: "f32[8, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_433, [2], True, dtype = torch.float32);  mul_433 = None
        add_112: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_106, mul_434);  add_106 = mul_434 = None
        pow_63: "f32[8, 1024, 1]" = torch.ops.aten.pow.Tensor_Scalar(arg469_1, 3);  arg469_1 = None
        mul_435: "f32[8, 1024, 1]" = torch.ops.aten.mul.Scalar(sum_85, -0.5);  sum_85 = None
        mul_436: "f32[8, 1024, 1]" = torch.ops.aten.mul.Tensor(mul_435, pow_63);  mul_435 = pow_63 = None
        expand_33: "f32[8, 1024, 512]" = torch.ops.aten.expand.default(mul_436, _shape_param_478);  mul_436 = _shape_param_478 = None
        div_50: "f32[8, 1024, 512]" = torch.ops.aten.div.Scalar(expand_33, 512);  expand_33 = None
        pow_64: "f32[8, 1024, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_430, 1.0);  mul_430 = None
        mul_437: "f32[8, 1024, 512]" = torch.ops.aten.mul.Scalar(pow_64, 2.0);  pow_64 = None
        mul_438: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(div_50, mul_437);  div_50 = mul_437 = None
        add_113: "f32[8, 1024, 512]" = torch.ops.aten.add.Tensor(add_112, mul_438);  add_112 = mul_438 = None
        convert_element_type_349: "f32[8, 1024, 512]" = torch.ops.prims.convert_element_type.default(arg468_1, torch.float32);  arg468_1 = None
        mul_439: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_349, 1.1111111111111112);  convert_element_type_349 = None
        mul_440: "f32[8, 1024, 512]" = torch.ops.aten.mul.Tensor(add_113, mul_439);  add_113 = mul_439 = None
        _unsafe_masked_index_put_accumulate_3: "f32[32128, 512]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_6, unsqueeze_4, [arg295_1], mul_440);  full_6 = unsqueeze_4 = arg295_1 = mul_440 = None
        add_114: "f32[32128, 512]" = torch.ops.aten.add.Tensor(add_69, _unsafe_masked_index_put_accumulate_3);  add_69 = _unsafe_masked_index_put_accumulate_3 = None
        return (view_6, convert_element_type_11, convert_element_type_15, view_11, convert_element_type_18, convert_element_type_25, convert_element_type_27, convert_element_type_29, view_31, convert_element_type_32, convert_element_type_39, convert_element_type_41, convert_element_type_43, view_52, convert_element_type_46, convert_element_type_50, view_57, convert_element_type_53, convert_element_type_60, convert_element_type_62, convert_element_type_64, view_77, convert_element_type_67, convert_element_type_74, convert_element_type_76, convert_element_type_78, view_98, convert_element_type_81, convert_element_type_85, view_103, convert_element_type_88, convert_element_type_95, convert_element_type_97, convert_element_type_99, view_123, convert_element_type_102, convert_element_type_109, convert_element_type_111, convert_element_type_113, view_144, convert_element_type_116, convert_element_type_120, view_149, convert_element_type_123, convert_element_type_130, convert_element_type_132, convert_element_type_134, view_169, convert_element_type_137, convert_element_type_144, convert_element_type_146, convert_element_type_148, view_190, convert_element_type_151, convert_element_type_155, view_195, convert_element_type_158, convert_element_type_165, convert_element_type_167, convert_element_type_169, view_215, convert_element_type_172, convert_element_type_179, convert_element_type_181, convert_element_type_183, view_236, convert_element_type_186, convert_element_type_190, view_241, convert_element_type_193, convert_element_type_200, convert_element_type_202, convert_element_type_204, view_261, convert_element_type_207, _unsafe_masked_index_put_accumulate, convert_element_type_215, convert_element_type_217, convert_element_type_219, view_283, view_284, convert_element_type_224, convert_element_type_228, view_289, convert_element_type_231, convert_element_type_238, convert_element_type_240, convert_element_type_242, view_310, convert_element_type_245, convert_element_type_249, view_315, convert_element_type_252, convert_element_type_259, convert_element_type_261, convert_element_type_263, view_336, convert_element_type_266, convert_element_type_270, view_341, convert_element_type_273, convert_element_type_280, convert_element_type_282, convert_element_type_284, view_362, convert_element_type_287, convert_element_type_291, view_367, convert_element_type_294, convert_element_type_301, convert_element_type_303, convert_element_type_305, view_388, convert_element_type_308, convert_element_type_312, view_393, convert_element_type_315, convert_element_type_322, convert_element_type_324, convert_element_type_326, view_414, convert_element_type_329, convert_element_type_333, view_419, convert_element_type_336, _unsafe_masked_index_put_accumulate_2, convert_element_type_344, convert_element_type_346, convert_element_type_348, view_441, add_114)



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
