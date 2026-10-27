"""
Standalone repro captured via capture_hook.
Label: GPTNeoForCausalLM
Pattern hash: 6ec2590f1f73
Shape hash: 226e0bee
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[32, 129]", arg3_1: "bf16[32, 128, 50257]", arg4_1: "f32[4096, 1]", arg5_1: "f32[4096, 1]", arg6_1: "bf16[32, 128, 50257]", arg7_1: "bf16[4096, 2048]", arg8_1: "bf16[50257, 2048]", arg9_1: "f32[2048]", arg10_1: "f32[32, 128, 2048]", arg11_1: "f32[32, 128, 1]", arg12_1: "bf16[2048, 8192]", arg13_1: "bf16[4096, 8192]", arg14_1: "bf16[4096, 8192]", arg15_1: "bf16[8192, 2048]", arg16_1: "bf16[4096, 2048]", arg17_1: "f32[2048]", arg18_1: "f32[32, 128, 2048]", arg19_1: "f32[32, 128, 1]", arg20_1: "bf16[2048, 2048]", arg21_1: "bf16[4096, 2048]", arg22_1: "bf16[512, 128, 128]", arg23_1: "bf16[512, 128, 128]", arg24_1: "f32[32, 16, 128, 128]", arg25_1: "b8[1, 1, 2048, 2048]", arg26_1: "bf16[512, 128, 128]", arg27_1: "bf16[512, 128, 128]", arg28_1: "bf16[4096, 2048]", arg29_1: "bf16[2048, 2048]", arg30_1: "bf16[2048, 2048]", arg31_1: "bf16[2048, 2048]", arg32_1: "f32[2048]", arg33_1: "f32[32, 128, 2048]", arg34_1: "f32[32, 128, 1]", arg35_1: "bf16[2048, 8192]", arg36_1: "bf16[4096, 8192]", arg37_1: "bf16[4096, 8192]", arg38_1: "bf16[8192, 2048]", arg39_1: "bf16[4096, 2048]", arg40_1: "f32[2048]", arg41_1: "f32[32, 128, 2048]", arg42_1: "f32[32, 128, 1]", arg43_1: "bf16[2048, 2048]", arg44_1: "bf16[4096, 2048]", arg45_1: "bf16[512, 128, 128]", arg46_1: "bf16[512, 128, 128]", arg47_1: "f32[32, 16, 128, 128]", arg48_1: "b8[1, 1, 2048, 2048]", arg49_1: "bf16[512, 128, 128]", arg50_1: "bf16[512, 128, 128]", arg51_1: "bf16[4096, 2048]", arg52_1: "bf16[2048, 2048]", arg53_1: "bf16[2048, 2048]", arg54_1: "bf16[2048, 2048]", arg55_1: "f32[2048]", arg56_1: "f32[32, 128, 2048]", arg57_1: "f32[32, 128, 1]", arg58_1: "bf16[2048, 8192]", arg59_1: "bf16[4096, 8192]", arg60_1: "bf16[4096, 8192]", arg61_1: "bf16[8192, 2048]", arg62_1: "bf16[4096, 2048]", arg63_1: "f32[2048]", arg64_1: "f32[32, 128, 2048]", arg65_1: "f32[32, 128, 1]", arg66_1: "bf16[2048, 2048]", arg67_1: "bf16[4096, 2048]", arg68_1: "bf16[512, 128, 128]", arg69_1: "bf16[512, 128, 128]", arg70_1: "f32[32, 16, 128, 128]", arg71_1: "b8[1, 1, 2048, 2048]", arg72_1: "bf16[512, 128, 128]", arg73_1: "bf16[512, 128, 128]", arg74_1: "bf16[4096, 2048]", arg75_1: "bf16[2048, 2048]", arg76_1: "bf16[2048, 2048]", arg77_1: "bf16[2048, 2048]", arg78_1: "f32[2048]", arg79_1: "f32[32, 128, 2048]", arg80_1: "f32[32, 128, 1]", arg81_1: "bf16[2048, 8192]", arg82_1: "bf16[4096, 8192]", arg83_1: "bf16[4096, 8192]", arg84_1: "bf16[8192, 2048]", arg85_1: "bf16[4096, 2048]", arg86_1: "f32[2048]", arg87_1: "f32[32, 128, 2048]", arg88_1: "f32[32, 128, 1]", arg89_1: "bf16[2048, 2048]", arg90_1: "bf16[4096, 2048]", arg91_1: "bf16[512, 128, 128]", arg92_1: "bf16[512, 128, 128]", arg93_1: "f32[32, 16, 128, 128]", arg94_1: "b8[1, 1, 2048, 2048]", arg95_1: "bf16[512, 128, 128]", arg96_1: "bf16[512, 128, 128]", arg97_1: "bf16[4096, 2048]", arg98_1: "bf16[2048, 2048]", arg99_1: "bf16[2048, 2048]", arg100_1: "bf16[2048, 2048]", arg101_1: "f32[2048]", arg102_1: "f32[32, 128, 2048]", arg103_1: "f32[32, 128, 1]", arg104_1: "bf16[2048, 8192]", arg105_1: "bf16[4096, 8192]", arg106_1: "bf16[4096, 8192]", arg107_1: "bf16[8192, 2048]", arg108_1: "bf16[4096, 2048]", arg109_1: "f32[2048]", arg110_1: "f32[32, 128, 2048]", arg111_1: "f32[32, 128, 1]", arg112_1: "bf16[2048, 2048]", arg113_1: "bf16[4096, 2048]", arg114_1: "bf16[512, 128, 128]", arg115_1: "bf16[512, 128, 128]", arg116_1: "f32[32, 16, 128, 128]", arg117_1: "b8[1, 1, 2048, 2048]", arg118_1: "bf16[512, 128, 128]", arg119_1: "bf16[512, 128, 128]", arg120_1: "bf16[4096, 2048]", arg121_1: "bf16[2048, 2048]", arg122_1: "bf16[2048, 2048]", arg123_1: "bf16[2048, 2048]", arg124_1: "f32[2048]", arg125_1: "f32[32, 128, 2048]", arg126_1: "f32[32, 128, 1]", arg127_1: "bf16[2048, 8192]", arg128_1: "bf16[4096, 8192]", arg129_1: "bf16[4096, 8192]", arg130_1: "bf16[8192, 2048]", arg131_1: "bf16[4096, 2048]", arg132_1: "f32[2048]", arg133_1: "f32[32, 128, 2048]", arg134_1: "f32[32, 128, 1]", arg135_1: "bf16[2048, 2048]", arg136_1: "bf16[4096, 2048]", arg137_1: "bf16[512, 128, 128]", arg138_1: "bf16[512, 128, 128]", arg139_1: "f32[32, 16, 128, 128]", arg140_1: "b8[1, 1, 2048, 2048]", arg141_1: "bf16[512, 128, 128]", arg142_1: "bf16[512, 128, 128]", arg143_1: "bf16[4096, 2048]", arg144_1: "bf16[2048, 2048]", arg145_1: "bf16[2048, 2048]", arg146_1: "bf16[2048, 2048]", arg147_1: "f32[2048]", arg148_1: "f32[32, 128, 2048]", arg149_1: "f32[32, 128, 1]", arg150_1: "bf16[2048, 8192]", arg151_1: "bf16[4096, 8192]", arg152_1: "bf16[4096, 8192]", arg153_1: "bf16[8192, 2048]", arg154_1: "bf16[4096, 2048]", arg155_1: "f32[2048]", arg156_1: "f32[32, 128, 2048]", arg157_1: "f32[32, 128, 1]", arg158_1: "bf16[2048, 2048]", arg159_1: "bf16[4096, 2048]", arg160_1: "bf16[512, 128, 128]", arg161_1: "bf16[512, 128, 128]", arg162_1: "f32[32, 16, 128, 128]", arg163_1: "b8[1, 1, 2048, 2048]", arg164_1: "bf16[512, 128, 128]", arg165_1: "bf16[512, 128, 128]", arg166_1: "bf16[4096, 2048]", arg167_1: "bf16[2048, 2048]", arg168_1: "bf16[2048, 2048]", arg169_1: "bf16[2048, 2048]", arg170_1: "f32[2048]", arg171_1: "f32[32, 128, 2048]", arg172_1: "f32[32, 128, 1]", arg173_1: "bf16[2048, 8192]", arg174_1: "bf16[4096, 8192]", arg175_1: "bf16[4096, 8192]", arg176_1: "bf16[8192, 2048]", arg177_1: "bf16[4096, 2048]", arg178_1: "f32[2048]", arg179_1: "f32[32, 128, 2048]", arg180_1: "f32[32, 128, 1]", arg181_1: "bf16[2048, 2048]", arg182_1: "bf16[4096, 2048]", arg183_1: "bf16[512, 128, 128]", arg184_1: "bf16[512, 128, 128]", arg185_1: "f32[32, 16, 128, 128]", arg186_1: "b8[1, 1, 2048, 2048]", arg187_1: "bf16[512, 128, 128]", arg188_1: "bf16[512, 128, 128]", arg189_1: "bf16[4096, 2048]", arg190_1: "bf16[2048, 2048]", arg191_1: "bf16[2048, 2048]", arg192_1: "bf16[2048, 2048]", arg193_1: "f32[2048]", arg194_1: "f32[32, 128, 2048]", arg195_1: "f32[32, 128, 1]", arg196_1: "bf16[2048, 8192]", arg197_1: "bf16[4096, 8192]", arg198_1: "bf16[4096, 8192]", arg199_1: "bf16[8192, 2048]", arg200_1: "bf16[4096, 2048]", arg201_1: "f32[2048]", arg202_1: "f32[32, 128, 2048]", arg203_1: "f32[32, 128, 1]", arg204_1: "bf16[2048, 2048]", arg205_1: "bf16[4096, 2048]", arg206_1: "bf16[512, 128, 128]", arg207_1: "bf16[512, 128, 128]", arg208_1: "f32[32, 16, 128, 128]", arg209_1: "b8[1, 1, 2048, 2048]", arg210_1: "bf16[512, 128, 128]", arg211_1: "bf16[512, 128, 128]", arg212_1: "bf16[4096, 2048]", arg213_1: "bf16[2048, 2048]", arg214_1: "bf16[2048, 2048]", arg215_1: "bf16[2048, 2048]", arg216_1: "f32[2048]", arg217_1: "f32[32, 128, 2048]", arg218_1: "f32[32, 128, 1]", arg219_1: "bf16[2048, 8192]", arg220_1: "bf16[4096, 8192]", arg221_1: "bf16[4096, 8192]", arg222_1: "bf16[8192, 2048]", arg223_1: "bf16[4096, 2048]", arg224_1: "f32[2048]", arg225_1: "f32[32, 128, 2048]", arg226_1: "f32[32, 128, 1]", arg227_1: "bf16[2048, 2048]", arg228_1: "bf16[4096, 2048]", arg229_1: "bf16[512, 128, 128]", arg230_1: "bf16[512, 128, 128]", arg231_1: "f32[32, 16, 128, 128]", arg232_1: "b8[1, 1, 2048, 2048]", arg233_1: "bf16[512, 128, 128]", arg234_1: "bf16[512, 128, 128]", arg235_1: "bf16[4096, 2048]", arg236_1: "bf16[2048, 2048]", arg237_1: "bf16[2048, 2048]", arg238_1: "bf16[2048, 2048]", arg239_1: "f32[2048]", arg240_1: "f32[32, 128, 2048]", arg241_1: "f32[32, 128, 1]", arg242_1: "bf16[2048, 8192]", arg243_1: "bf16[4096, 8192]", arg244_1: "bf16[4096, 8192]", arg245_1: "bf16[8192, 2048]", arg246_1: "bf16[4096, 2048]", arg247_1: "f32[2048]", arg248_1: "f32[32, 128, 2048]", arg249_1: "f32[32, 128, 1]", arg250_1: "bf16[2048, 2048]", arg251_1: "bf16[4096, 2048]", arg252_1: "bf16[512, 128, 128]", arg253_1: "bf16[512, 128, 128]", arg254_1: "f32[32, 16, 128, 128]", arg255_1: "b8[1, 1, 2048, 2048]", arg256_1: "bf16[512, 128, 128]", arg257_1: "bf16[512, 128, 128]", arg258_1: "bf16[4096, 2048]", arg259_1: "bf16[2048, 2048]", arg260_1: "bf16[2048, 2048]", arg261_1: "bf16[2048, 2048]", arg262_1: "f32[2048]", arg263_1: "f32[32, 128, 2048]", arg264_1: "f32[32, 128, 1]", arg265_1: "bf16[2048, 8192]", arg266_1: "bf16[4096, 8192]", arg267_1: "bf16[4096, 8192]", arg268_1: "bf16[8192, 2048]", arg269_1: "bf16[4096, 2048]", arg270_1: "f32[2048]", arg271_1: "f32[32, 128, 2048]", arg272_1: "f32[32, 128, 1]", arg273_1: "bf16[2048, 2048]", arg274_1: "bf16[4096, 2048]", arg275_1: "bf16[512, 128, 128]", arg276_1: "bf16[512, 128, 128]", arg277_1: "f32[32, 16, 128, 128]", arg278_1: "b8[1, 1, 2048, 2048]", arg279_1: "bf16[512, 128, 128]", arg280_1: "bf16[512, 128, 128]", arg281_1: "bf16[4096, 2048]", arg282_1: "bf16[2048, 2048]", arg283_1: "bf16[2048, 2048]", arg284_1: "bf16[2048, 2048]", arg285_1: "f32[2048]", arg286_1: "f32[32, 128, 2048]", arg287_1: "f32[32, 128, 1]", arg288_1: "bf16[2048, 8192]", arg289_1: "bf16[4096, 8192]", arg290_1: "bf16[4096, 8192]", arg291_1: "bf16[8192, 2048]", arg292_1: "bf16[4096, 2048]", arg293_1: "f32[2048]", arg294_1: "f32[32, 128, 2048]", arg295_1: "f32[32, 128, 1]", arg296_1: "bf16[2048, 2048]", arg297_1: "bf16[4096, 2048]", arg298_1: "bf16[512, 128, 128]", arg299_1: "bf16[512, 128, 128]", arg300_1: "f32[32, 16, 128, 128]", arg301_1: "b8[1, 1, 2048, 2048]", arg302_1: "bf16[512, 128, 128]", arg303_1: "bf16[512, 128, 128]", arg304_1: "bf16[4096, 2048]", arg305_1: "bf16[2048, 2048]", arg306_1: "bf16[2048, 2048]", arg307_1: "bf16[2048, 2048]", arg308_1: "f32[2048]", arg309_1: "f32[32, 128, 2048]", arg310_1: "f32[32, 128, 1]", arg311_1: "bf16[2048, 8192]", arg312_1: "bf16[4096, 8192]", arg313_1: "bf16[4096, 8192]", arg314_1: "bf16[8192, 2048]", arg315_1: "bf16[4096, 2048]", arg316_1: "f32[2048]", arg317_1: "f32[32, 128, 2048]", arg318_1: "f32[32, 128, 1]", arg319_1: "bf16[2048, 2048]", arg320_1: "bf16[4096, 2048]", arg321_1: "bf16[512, 128, 128]", arg322_1: "bf16[512, 128, 128]", arg323_1: "f32[32, 16, 128, 128]", arg324_1: "b8[1, 1, 2048, 2048]", arg325_1: "bf16[512, 128, 128]", arg326_1: "bf16[512, 128, 128]", arg327_1: "bf16[4096, 2048]", arg328_1: "bf16[2048, 2048]", arg329_1: "bf16[2048, 2048]", arg330_1: "bf16[2048, 2048]", arg331_1: "f32[2048]", arg332_1: "f32[32, 128, 2048]", arg333_1: "f32[32, 128, 1]", arg334_1: "bf16[2048, 8192]", arg335_1: "bf16[4096, 8192]", arg336_1: "bf16[4096, 8192]", arg337_1: "bf16[8192, 2048]", arg338_1: "bf16[4096, 2048]", arg339_1: "f32[2048]", arg340_1: "f32[32, 128, 2048]", arg341_1: "f32[32, 128, 1]", arg342_1: "bf16[2048, 2048]", arg343_1: "bf16[4096, 2048]", arg344_1: "bf16[512, 128, 128]", arg345_1: "bf16[512, 128, 128]", arg346_1: "f32[32, 16, 128, 128]", arg347_1: "b8[1, 1, 2048, 2048]", arg348_1: "bf16[512, 128, 128]", arg349_1: "bf16[512, 128, 128]", arg350_1: "bf16[4096, 2048]", arg351_1: "bf16[2048, 2048]", arg352_1: "bf16[2048, 2048]", arg353_1: "bf16[2048, 2048]", arg354_1: "f32[2048]", arg355_1: "f32[32, 128, 2048]", arg356_1: "f32[32, 128, 1]", arg357_1: "bf16[2048, 8192]", arg358_1: "bf16[4096, 8192]", arg359_1: "bf16[4096, 8192]", arg360_1: "bf16[8192, 2048]", arg361_1: "bf16[4096, 2048]", arg362_1: "f32[2048]", arg363_1: "f32[32, 128, 2048]", arg364_1: "f32[32, 128, 1]", arg365_1: "bf16[2048, 2048]", arg366_1: "bf16[4096, 2048]", arg367_1: "bf16[512, 128, 128]", arg368_1: "bf16[512, 128, 128]", arg369_1: "f32[32, 16, 128, 128]", arg370_1: "b8[1, 1, 2048, 2048]", arg371_1: "bf16[512, 128, 128]", arg372_1: "bf16[512, 128, 128]", arg373_1: "bf16[4096, 2048]", arg374_1: "bf16[2048, 2048]", arg375_1: "bf16[2048, 2048]", arg376_1: "bf16[2048, 2048]", arg377_1: "f32[2048]", arg378_1: "f32[32, 128, 2048]", arg379_1: "f32[32, 128, 1]", arg380_1: "bf16[2048, 8192]", arg381_1: "bf16[4096, 8192]", arg382_1: "bf16[4096, 8192]", arg383_1: "bf16[8192, 2048]", arg384_1: "bf16[4096, 2048]", arg385_1: "f32[2048]", arg386_1: "f32[32, 128, 2048]", arg387_1: "f32[32, 128, 1]", arg388_1: "bf16[2048, 2048]", arg389_1: "bf16[4096, 2048]", arg390_1: "bf16[512, 128, 128]", arg391_1: "bf16[512, 128, 128]", arg392_1: "f32[32, 16, 128, 128]", arg393_1: "b8[1, 1, 2048, 2048]", arg394_1: "bf16[512, 128, 128]", arg395_1: "bf16[512, 128, 128]", arg396_1: "bf16[4096, 2048]", arg397_1: "bf16[2048, 2048]", arg398_1: "bf16[2048, 2048]", arg399_1: "bf16[2048, 2048]", arg400_1: "f32[2048]", arg401_1: "f32[32, 128, 2048]", arg402_1: "f32[32, 128, 1]", arg403_1: "bf16[2048, 8192]", arg404_1: "bf16[4096, 8192]", arg405_1: "bf16[4096, 8192]", arg406_1: "bf16[8192, 2048]", arg407_1: "bf16[4096, 2048]", arg408_1: "f32[2048]", arg409_1: "f32[32, 128, 2048]", arg410_1: "f32[32, 128, 1]", arg411_1: "bf16[2048, 2048]", arg412_1: "bf16[4096, 2048]", arg413_1: "bf16[512, 128, 128]", arg414_1: "bf16[512, 128, 128]", arg415_1: "f32[32, 16, 128, 128]", arg416_1: "b8[1, 1, 2048, 2048]", arg417_1: "bf16[512, 128, 128]", arg418_1: "bf16[512, 128, 128]", arg419_1: "bf16[4096, 2048]", arg420_1: "bf16[2048, 2048]", arg421_1: "bf16[2048, 2048]", arg422_1: "bf16[2048, 2048]", arg423_1: "f32[2048]", arg424_1: "f32[32, 128, 2048]", arg425_1: "f32[32, 128, 1]", arg426_1: "bf16[2048, 8192]", arg427_1: "bf16[4096, 8192]", arg428_1: "bf16[4096, 8192]", arg429_1: "bf16[8192, 2048]", arg430_1: "bf16[4096, 2048]", arg431_1: "f32[2048]", arg432_1: "f32[32, 128, 2048]", arg433_1: "f32[32, 128, 1]", arg434_1: "bf16[2048, 2048]", arg435_1: "bf16[4096, 2048]", arg436_1: "bf16[512, 128, 128]", arg437_1: "bf16[512, 128, 128]", arg438_1: "f32[32, 16, 128, 128]", arg439_1: "b8[1, 1, 2048, 2048]", arg440_1: "bf16[512, 128, 128]", arg441_1: "bf16[512, 128, 128]", arg442_1: "bf16[4096, 2048]", arg443_1: "bf16[2048, 2048]", arg444_1: "bf16[2048, 2048]", arg445_1: "bf16[2048, 2048]", arg446_1: "f32[2048]", arg447_1: "f32[32, 128, 2048]", arg448_1: "f32[32, 128, 1]", arg449_1: "bf16[2048, 8192]", arg450_1: "bf16[4096, 8192]", arg451_1: "bf16[4096, 8192]", arg452_1: "bf16[8192, 2048]", arg453_1: "bf16[4096, 2048]", arg454_1: "f32[2048]", arg455_1: "f32[32, 128, 2048]", arg456_1: "f32[32, 128, 1]", arg457_1: "bf16[2048, 2048]", arg458_1: "bf16[4096, 2048]", arg459_1: "bf16[512, 128, 128]", arg460_1: "bf16[512, 128, 128]", arg461_1: "f32[32, 16, 128, 128]", arg462_1: "b8[1, 1, 2048, 2048]", arg463_1: "bf16[512, 128, 128]", arg464_1: "bf16[512, 128, 128]", arg465_1: "bf16[4096, 2048]", arg466_1: "bf16[2048, 2048]", arg467_1: "bf16[2048, 2048]", arg468_1: "bf16[2048, 2048]", arg469_1: "f32[2048]", arg470_1: "f32[32, 128, 2048]", arg471_1: "f32[32, 128, 1]", arg472_1: "bf16[2048, 8192]", arg473_1: "bf16[4096, 8192]", arg474_1: "bf16[4096, 8192]", arg475_1: "bf16[8192, 2048]", arg476_1: "bf16[4096, 2048]", arg477_1: "f32[2048]", arg478_1: "f32[32, 128, 2048]", arg479_1: "f32[32, 128, 1]", arg480_1: "bf16[2048, 2048]", arg481_1: "bf16[4096, 2048]", arg482_1: "bf16[512, 128, 128]", arg483_1: "bf16[512, 128, 128]", arg484_1: "f32[32, 16, 128, 128]", arg485_1: "b8[1, 1, 2048, 2048]", arg486_1: "bf16[512, 128, 128]", arg487_1: "bf16[512, 128, 128]", arg488_1: "bf16[4096, 2048]", arg489_1: "bf16[2048, 2048]", arg490_1: "bf16[2048, 2048]", arg491_1: "bf16[2048, 2048]", arg492_1: "f32[2048]", arg493_1: "f32[32, 128, 2048]", arg494_1: "f32[32, 128, 1]", arg495_1: "bf16[2048, 8192]", arg496_1: "bf16[4096, 8192]", arg497_1: "bf16[4096, 8192]", arg498_1: "bf16[8192, 2048]", arg499_1: "bf16[4096, 2048]", arg500_1: "f32[2048]", arg501_1: "f32[32, 128, 2048]", arg502_1: "f32[32, 128, 1]", arg503_1: "bf16[2048, 2048]", arg504_1: "bf16[4096, 2048]", arg505_1: "bf16[512, 128, 128]", arg506_1: "bf16[512, 128, 128]", arg507_1: "f32[32, 16, 128, 128]", arg508_1: "b8[1, 1, 2048, 2048]", arg509_1: "bf16[512, 128, 128]", arg510_1: "bf16[512, 128, 128]", arg511_1: "bf16[4096, 2048]", arg512_1: "bf16[2048, 2048]", arg513_1: "bf16[2048, 2048]", arg514_1: "bf16[2048, 2048]", arg515_1: "f32[2048]", arg516_1: "f32[32, 128, 2048]", arg517_1: "f32[32, 128, 1]", arg518_1: "bf16[2048, 8192]", arg519_1: "bf16[4096, 8192]", arg520_1: "bf16[4096, 8192]", arg521_1: "bf16[8192, 2048]", arg522_1: "bf16[4096, 2048]", arg523_1: "f32[2048]", arg524_1: "f32[32, 128, 2048]", arg525_1: "f32[32, 128, 1]", arg526_1: "bf16[2048, 2048]", arg527_1: "bf16[4096, 2048]", arg528_1: "bf16[512, 128, 128]", arg529_1: "bf16[512, 128, 128]", arg530_1: "f32[32, 16, 128, 128]", arg531_1: "b8[1, 1, 2048, 2048]", arg532_1: "bf16[512, 128, 128]", arg533_1: "bf16[512, 128, 128]", arg534_1: "bf16[4096, 2048]", arg535_1: "bf16[2048, 2048]", arg536_1: "bf16[2048, 2048]", arg537_1: "bf16[2048, 2048]", arg538_1: "f32[2048]", arg539_1: "f32[32, 128, 2048]", arg540_1: "f32[32, 128, 1]", arg541_1: "bf16[2048, 8192]", arg542_1: "bf16[4096, 8192]", arg543_1: "bf16[4096, 8192]", arg544_1: "bf16[8192, 2048]", arg545_1: "bf16[4096, 2048]", arg546_1: "f32[2048]", arg547_1: "f32[32, 128, 2048]", arg548_1: "f32[32, 128, 1]", arg549_1: "bf16[2048, 2048]", arg550_1: "bf16[4096, 2048]", arg551_1: "bf16[512, 128, 128]", arg552_1: "bf16[512, 128, 128]", arg553_1: "i64[1, 128]", arg554_1: "i64[32, 128]", arg555_1: "bf16[512, 128, 128]", arg556_1: "b8[1, 1, 2048, 2048]", arg557_1: "f32[32, 16, 128, 1]", arg558_1: "f32[32, 16, 128, 1]", arg559_1: "bf16[512, 128, 128]", arg560_1: "bf16[512, 128, 128]", arg561_1: "bf16[4096, 2048]", arg562_1: "bf16[2048, 2048]", arg563_1: "bf16[2048, 2048]", arg564_1: "bf16[2048, 2048]", arg565_1: "f32[2048]", arg566_1: "f32[32, 128, 2048]", arg567_1: "f32[1, 128, 2048]", arg568_1: "f32[32, 128, 1]", arg569_1: "f32[32, 128, 1]", arg570_1: "i64[32, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "i64[32, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 1, 9223372036854775807);  arg2_1 = None
        clone: "i64[32, 128]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None
        view: "i64[4096]" = torch.ops.aten.view.default(clone, [-1]);  clone = None
        unsqueeze: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[4096, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[50257]" = torch.ops.prims.iota.default(50257, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 50257]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[4096, 50257]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[4096, 50257]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[4096, 50257]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[4096, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = None
        mul: "f32[4096, 50257]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "f32[32, 128, 50257]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        view_2: "f32[4096, 50257]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        sub: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(view_2, arg4_1);  view_2 = arg4_1 = None
        sub_1: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        exp: "f32[4096, 50257]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        mul_1: "f32[4096, 50257]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[4096, 50257]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        view_3: "f32[32, 128, 50257]" = torch.ops.aten.view.default(sub_2, _shape_param_3);  sub_2 = _shape_param_3 = None
        convert_element_type_1: "bf16[32, 128, 50257]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        add: "bf16[32, 128, 50257]" = torch.ops.aten.add.Tensor(arg6_1, convert_element_type_1);  arg6_1 = convert_element_type_1 = None
        view_4: "bf16[4096, 50257]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        permute: "bf16[50257, 4096]" = torch.ops.aten.permute.default(view_4, [1, 0])
        constant_pad_nd: "bf16[50264, 4096]" = torch.ops.aten.constant_pad_nd.default(permute, _shape_param_5);  permute = _shape_param_5 = None
        mm: "bf16[50264, 2048]" = torch.ops.aten.mm.default(constant_pad_nd, arg7_1);  constant_pad_nd = arg7_1 = None
        slice_2: "bf16[50257, 2048]" = torch.ops.aten.slice.Tensor(mm, 0, 0, -7);  mm = None
        constant_pad_nd_1: "bf16[4096, 50264]" = torch.ops.aten.constant_pad_nd.default(view_4, _shape_param_6);  view_4 = _shape_param_6 = None
        constant_pad_nd_2: "bf16[50264, 2048]" = torch.ops.aten.constant_pad_nd.default(arg8_1, _shape_param_7);  arg8_1 = _shape_param_7 = None
        mm_1: "bf16[4096, 2048]" = torch.ops.aten.mm.default(constant_pad_nd_1, constant_pad_nd_2);  constant_pad_nd_1 = constant_pad_nd_2 = None
        view_5: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_1, _shape_param_8);  mm_1 = _shape_param_8 = None
        convert_element_type_2: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_5, torch.float32);  view_5 = None
        convert_element_type_3: "f32[50257, 2048]" = torch.ops.prims.convert_element_type.default(slice_2, torch.float32);  slice_2 = None
        mul_2: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_2, arg9_1);  arg9_1 = None
        mul_3: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, 2048)
        sum_2: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, arg10_1);  mul_2 = None
        sum_3: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg10_1, sum_3);  sum_3 = None
        sub_3: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_3, sum_2);  mul_3 = sum_2 = None
        sub_4: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_3, mul_5);  sub_3 = mul_5 = None
        mul_6: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg11_1, sub_4);  arg11_1 = sub_4 = None
        mul_7: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_2, arg10_1);  arg10_1 = None
        sum_4: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_5: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_2, [0, 1]);  convert_element_type_2 = None
        convert_element_type_4: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16)
        view_6: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_4, _shape_param_9);  convert_element_type_4 = _shape_param_9 = None
        mm_2: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_6, arg12_1);  arg12_1 = None
        permute_1: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_6, [1, 0])
        mm_3: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_1, arg13_1);  permute_1 = arg13_1 = None
        sum_6: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_6, [0], True, dtype = torch.float32);  view_6 = None
        view_7: "f32[2048]" = torch.ops.aten.view.default(sum_6, _shape_param_10);  sum_6 = _shape_param_10 = None
        convert_element_type_5: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_7, torch.bfloat16);  view_7 = None
        view_8: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_2, _shape_param_11);  mm_2 = _shape_param_11 = None
        convert_element_type_6: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_8, torch.float32);  view_8 = None
        convert_element_type_7: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_8: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_5, torch.float32);  convert_element_type_5 = None
        view_9: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg14_1, _shape_param_12);  arg14_1 = _shape_param_12 = None
        mul_8: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_9, 0.5)
        mul_9: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_6, mul_8);  mul_8 = None
        convert_element_type_9: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32)
        pow_1: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_9, 3.0)
        mul_10: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_1: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_9, mul_10);  view_9 = mul_10 = None
        mul_11: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_1, 0.7978845608028654);  add_1 = None
        tanh: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_2: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_12: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_6, add_2);  convert_element_type_6 = add_2 = None
        convert_element_type_10: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_13: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_5: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_13);  mul_13 = None
        mul_14: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_9, sub_5);  mul_9 = sub_5 = None
        mul_15: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_14, 0.7978845608028654);  mul_14 = None
        convert_element_type_11: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16)
        mul_16: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_15, 0.044715);  mul_15 = None
        pow_2: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_9, 2.0);  convert_element_type_9 = None
        mul_17: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_18: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        convert_element_type_12: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        add_3: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_11, convert_element_type_12);  convert_element_type_11 = convert_element_type_12 = None
        mul_19: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 0.5);  convert_element_type_10 = None
        add_4: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_3, mul_19);  add_3 = mul_19 = None
        view_10: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_4, _shape_param_13);  add_4 = _shape_param_13 = None
        mm_4: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_10, arg15_1);  arg15_1 = None
        permute_2: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_10, [1, 0])
        mm_5: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_2, arg16_1);  permute_2 = arg16_1 = None
        sum_7: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_10, [0], True, dtype = torch.float32);  view_10 = None
        view_11: "f32[8192]" = torch.ops.aten.view.default(sum_7, _shape_param_14);  sum_7 = _shape_param_14 = None
        convert_element_type_13: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_11, torch.bfloat16);  view_11 = None
        view_12: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_4, _shape_param_15);  mm_4 = _shape_param_15 = None
        convert_element_type_14: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        convert_element_type_15: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_16: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_13, torch.float32);  convert_element_type_13 = None
        mul_20: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_14, arg17_1);  arg17_1 = None
        mul_21: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_20, 2048)
        sum_8: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [2], True)
        mul_22: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_20, arg18_1);  mul_20 = None
        sum_9: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_22, [2], True);  mul_22 = None
        mul_23: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg18_1, sum_9);  sum_9 = None
        sub_6: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_21, sum_8);  mul_21 = sum_8 = None
        sub_7: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_6, mul_23);  sub_6 = mul_23 = None
        mul_24: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg19_1, sub_7);  arg19_1 = sub_7 = None
        mul_25: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_14, arg18_1);  arg18_1 = None
        sum_10: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_25, [0, 1]);  mul_25 = None
        sum_11: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_14, [0, 1]);  convert_element_type_14 = None
        add_5: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(mul_6, mul_24);  mul_6 = mul_24 = None
        convert_element_type_17: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_5, torch.bfloat16)
        view_13: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_17, _shape_param_16);  convert_element_type_17 = _shape_param_16 = None
        mm_6: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_13, arg20_1);  arg20_1 = None
        permute_3: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_13, [1, 0])
        mm_7: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_3, arg21_1);  permute_3 = arg21_1 = None
        sum_12: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_13, [0], True, dtype = torch.float32);  view_13 = None
        view_14: "f32[2048]" = torch.ops.aten.view.default(sum_12, _shape_param_17);  sum_12 = _shape_param_17 = None
        convert_element_type_18: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_14, torch.bfloat16);  view_14 = None
        view_15: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_6, _shape_param_18);  mm_6 = _shape_param_18 = None
        convert_element_type_19: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_20: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_18, torch.float32);  convert_element_type_18 = None
        view_16: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_15, _shape_param_19);  view_15 = _shape_param_19 = None
        permute_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        clone_1: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
        view_17: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_20);  clone_1 = _shape_param_20 = None
        bmm: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg22_1, view_17);  arg22_1 = None
        bmm_1: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_17, arg23_1);  view_17 = arg23_1 = None
        view_18: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_21);  bmm = _shape_param_21 = None
        view_19: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_22);  bmm_1 = _shape_param_22 = None
        convert_element_type_21: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_26: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_21, arg24_1);  convert_element_type_21 = None
        sum_13: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_26, [-1], True)
        neg: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg24_1);  arg24_1 = None
        fma: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_13, mul_26);  neg = sum_13 = mul_26 = None
        convert_element_type_22: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        full_2: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_3: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg25_1, 2, 0, 128);  arg25_1 = None
        slice_4: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 128);  slice_3 = None
        where_3: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_4, convert_element_type_22, full_2);  slice_4 = convert_element_type_22 = None
        view_20: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_3, _shape_param_23);  where_3 = _shape_param_23 = None
        bmm_2: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg26_1, view_20);  arg26_1 = None
        bmm_3: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_20, arg27_1);  view_20 = arg27_1 = None
        view_21: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_24);  bmm_2 = _shape_param_24 = None
        view_22: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_3, _shape_param_25);  bmm_3 = _shape_param_25 = None
        convert_element_type_23: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_21, torch.float32);  view_21 = None
        permute_5: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_23, [0, 1, 3, 2]);  convert_element_type_23 = None
        convert_element_type_24: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_5, torch.bfloat16);  permute_5 = None
        convert_element_type_25: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_22, torch.bfloat16);  view_22 = None
        permute_6: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_2: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_6, memory_format = torch.contiguous_format);  permute_6 = None
        view_23: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_2, _shape_param_26);  clone_2 = _shape_param_26 = None
        permute_7: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_24, [0, 2, 1, 3]);  convert_element_type_24 = None
        view_24: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_7, _shape_param_27);  permute_7 = _shape_param_27 = None
        permute_8: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_25, [0, 2, 1, 3]);  convert_element_type_25 = None
        clone_3: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
        view_25: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_3, _shape_param_28);  clone_3 = _shape_param_28 = None
        view_26: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_23, _shape_param_29);  view_23 = _shape_param_29 = None
        permute_9: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_26, [1, 0])
        mm_8: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_9, arg28_1);  permute_9 = None
        mm_9: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_26, arg29_1);  view_26 = arg29_1 = None
        view_27: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_9, _shape_param_30);  mm_9 = _shape_param_30 = None
        convert_element_type_26: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_27, torch.float32);  view_27 = None
        convert_element_type_27: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        clone_4: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_24, memory_format = torch.contiguous_format);  view_24 = None
        view_28: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_4, _shape_param_31);  clone_4 = _shape_param_31 = None
        permute_10: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_28, [1, 0])
        mm_10: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_10, arg28_1);  permute_10 = None
        mm_11: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_28, arg30_1);  view_28 = arg30_1 = None
        view_29: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_11, _shape_param_32);  mm_11 = _shape_param_32 = None
        convert_element_type_28: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_29, torch.float32);  view_29 = None
        add_6: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_26, convert_element_type_28);  convert_element_type_26 = convert_element_type_28 = None
        convert_element_type_29: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        view_30: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_25, _shape_param_33);  view_25 = _shape_param_33 = None
        permute_11: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_30, [1, 0])
        mm_12: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_11, arg28_1);  permute_11 = arg28_1 = None
        mm_13: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_30, arg31_1);  view_30 = arg31_1 = None
        view_31: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_13, _shape_param_34);  mm_13 = _shape_param_34 = None
        convert_element_type_30: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_31, torch.float32);  view_31 = None
        add_7: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_6, convert_element_type_30);  add_6 = convert_element_type_30 = None
        convert_element_type_31: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        mul_27: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_7, arg32_1);  arg32_1 = None
        mul_28: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_27, 2048)
        sum_14: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True)
        mul_29: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_27, arg33_1);  mul_27 = None
        sum_15: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_29, [2], True);  mul_29 = None
        mul_30: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg33_1, sum_15);  sum_15 = None
        sub_8: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_28, sum_14);  mul_28 = sum_14 = None
        sub_9: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_8, mul_30);  sub_8 = mul_30 = None
        mul_31: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg34_1, sub_9);  arg34_1 = sub_9 = None
        mul_32: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_7, arg33_1);  arg33_1 = None
        sum_16: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_32, [0, 1]);  mul_32 = None
        sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_7, [0, 1]);  add_7 = None
        add_8: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_5, mul_31);  add_5 = mul_31 = None
        convert_element_type_32: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16)
        view_32: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_32, _shape_param_35);  convert_element_type_32 = _shape_param_35 = None
        mm_14: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_32, arg35_1);  arg35_1 = None
        permute_12: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_32, [1, 0])
        mm_15: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_12, arg36_1);  permute_12 = arg36_1 = None
        sum_18: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_32, [0], True, dtype = torch.float32);  view_32 = None
        view_33: "f32[2048]" = torch.ops.aten.view.default(sum_18, _shape_param_36);  sum_18 = _shape_param_36 = None
        convert_element_type_33: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_33, torch.bfloat16);  view_33 = None
        view_34: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_14, _shape_param_37);  mm_14 = _shape_param_37 = None
        convert_element_type_34: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        convert_element_type_35: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_36: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_33, torch.float32);  convert_element_type_33 = None
        view_35: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg37_1, _shape_param_38);  arg37_1 = _shape_param_38 = None
        mul_33: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_35, 0.5)
        mul_34: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_34, mul_33);  mul_33 = None
        convert_element_type_37: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_35, torch.float32)
        pow_3: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_37, 3.0)
        mul_35: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_9: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_35, mul_35);  view_35 = mul_35 = None
        mul_36: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh_1: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_36);  mul_36 = None
        add_10: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_37: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_34, add_10);  convert_element_type_34 = add_10 = None
        convert_element_type_38: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_37, torch.bfloat16);  mul_37 = None
        mul_38: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_10: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_38);  mul_38 = None
        mul_39: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_34, sub_10);  mul_34 = sub_10 = None
        mul_40: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_39, 0.7978845608028654);  mul_39 = None
        convert_element_type_39: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16)
        mul_41: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_40, 0.044715);  mul_40 = None
        pow_4: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_37, 2.0);  convert_element_type_37 = None
        mul_42: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_4, 3.0);  pow_4 = None
        mul_43: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_41, mul_42);  mul_41 = mul_42 = None
        convert_element_type_40: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_43, torch.bfloat16);  mul_43 = None
        add_11: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_39, convert_element_type_40);  convert_element_type_39 = convert_element_type_40 = None
        mul_44: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 0.5);  convert_element_type_38 = None
        add_12: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_11, mul_44);  add_11 = mul_44 = None
        view_36: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_12, _shape_param_39);  add_12 = _shape_param_39 = None
        mm_16: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_36, arg38_1);  arg38_1 = None
        permute_13: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_36, [1, 0])
        mm_17: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_13, arg39_1);  permute_13 = arg39_1 = None
        sum_19: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_36, [0], True, dtype = torch.float32);  view_36 = None
        view_37: "f32[8192]" = torch.ops.aten.view.default(sum_19, _shape_param_40);  sum_19 = _shape_param_40 = None
        convert_element_type_41: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_37, torch.bfloat16);  view_37 = None
        view_38: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_16, _shape_param_41);  mm_16 = _shape_param_41 = None
        convert_element_type_42: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_38, torch.float32);  view_38 = None
        convert_element_type_43: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_44: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_41, torch.float32);  convert_element_type_41 = None
        mul_45: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_42, arg40_1);  arg40_1 = None
        mul_46: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_45, 2048)
        sum_20: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_45, [2], True)
        mul_47: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_45, arg41_1);  mul_45 = None
        sum_21: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_47, [2], True);  mul_47 = None
        mul_48: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg41_1, sum_21);  sum_21 = None
        sub_11: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_46, sum_20);  mul_46 = sum_20 = None
        sub_12: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_11, mul_48);  sub_11 = mul_48 = None
        mul_49: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg42_1, sub_12);  arg42_1 = sub_12 = None
        mul_50: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_42, arg41_1);  arg41_1 = None
        sum_22: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_50, [0, 1]);  mul_50 = None
        sum_23: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_42, [0, 1]);  convert_element_type_42 = None
        add_13: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_8, mul_49);  add_8 = mul_49 = None
        convert_element_type_45: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_13, torch.bfloat16)
        view_39: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_45, _shape_param_42);  convert_element_type_45 = _shape_param_42 = None
        mm_18: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_39, arg43_1);  arg43_1 = None
        permute_14: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_39, [1, 0])
        mm_19: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_14, arg44_1);  permute_14 = arg44_1 = None
        sum_24: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_39, [0], True, dtype = torch.float32);  view_39 = None
        view_40: "f32[2048]" = torch.ops.aten.view.default(sum_24, _shape_param_43);  sum_24 = _shape_param_43 = None
        convert_element_type_46: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_40, torch.bfloat16);  view_40 = None
        view_41: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_18, _shape_param_44);  mm_18 = _shape_param_44 = None
        convert_element_type_47: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_48: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_46, torch.float32);  convert_element_type_46 = None
        view_42: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_41, _shape_param_45);  view_41 = _shape_param_45 = None
        permute_15: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_42, [0, 2, 1, 3]);  view_42 = None
        clone_5: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
        view_43: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_46);  clone_5 = _shape_param_46 = None
        bmm_4: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg45_1, view_43);  arg45_1 = None
        bmm_5: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_43, arg46_1);  view_43 = arg46_1 = None
        view_44: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_47);  bmm_4 = _shape_param_47 = None
        view_45: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_5, _shape_param_48);  bmm_5 = _shape_param_48 = None
        convert_element_type_49: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_45, torch.float32);  view_45 = None
        mul_51: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_49, arg47_1);  convert_element_type_49 = None
        sum_25: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_51, [-1], True)
        neg_1: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg47_1);  arg47_1 = None
        fma_1: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_1, sum_25, mul_51);  neg_1 = sum_25 = mul_51 = None
        convert_element_type_50: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        slice_5: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg48_1, 2, 0, 128);  arg48_1 = None
        slice_6: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_5, 3, 0, 128);  slice_5 = None
        where_4: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_6, convert_element_type_50, full_2);  slice_6 = convert_element_type_50 = None
        view_46: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_4, _shape_param_49);  where_4 = _shape_param_49 = None
        bmm_6: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg49_1, view_46);  arg49_1 = None
        bmm_7: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_46, arg50_1);  view_46 = arg50_1 = None
        view_47: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_50);  bmm_6 = _shape_param_50 = None
        view_48: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_7, _shape_param_51);  bmm_7 = _shape_param_51 = None
        convert_element_type_51: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_47, torch.float32);  view_47 = None
        permute_16: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_51, [0, 1, 3, 2]);  convert_element_type_51 = None
        convert_element_type_52: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_16, torch.bfloat16);  permute_16 = None
        convert_element_type_53: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_48, torch.bfloat16);  view_48 = None
        permute_17: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        clone_6: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_49: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_6, _shape_param_52);  clone_6 = _shape_param_52 = None
        permute_18: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_52, [0, 2, 1, 3]);  convert_element_type_52 = None
        view_50: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_18, _shape_param_53);  permute_18 = _shape_param_53 = None
        permute_19: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_53, [0, 2, 1, 3]);  convert_element_type_53 = None
        clone_7: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
        view_51: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_7, _shape_param_54);  clone_7 = _shape_param_54 = None
        view_52: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_49, _shape_param_55);  view_49 = _shape_param_55 = None
        permute_20: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_52, [1, 0])
        mm_20: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_20, arg51_1);  permute_20 = None
        mm_21: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_52, arg52_1);  view_52 = arg52_1 = None
        view_53: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_21, _shape_param_56);  mm_21 = _shape_param_56 = None
        convert_element_type_54: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_53, torch.float32);  view_53 = None
        convert_element_type_55: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        clone_8: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_50, memory_format = torch.contiguous_format);  view_50 = None
        view_54: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_8, _shape_param_57);  clone_8 = _shape_param_57 = None
        permute_21: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_54, [1, 0])
        mm_22: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_21, arg51_1);  permute_21 = None
        mm_23: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_54, arg53_1);  view_54 = arg53_1 = None
        view_55: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_23, _shape_param_58);  mm_23 = _shape_param_58 = None
        convert_element_type_56: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_55, torch.float32);  view_55 = None
        add_14: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_54, convert_element_type_56);  convert_element_type_54 = convert_element_type_56 = None
        convert_element_type_57: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        view_56: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_51, _shape_param_59);  view_51 = _shape_param_59 = None
        permute_22: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_56, [1, 0])
        mm_24: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_22, arg51_1);  permute_22 = arg51_1 = None
        mm_25: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_56, arg54_1);  view_56 = arg54_1 = None
        view_57: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_25, _shape_param_60);  mm_25 = _shape_param_60 = None
        convert_element_type_58: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_57, torch.float32);  view_57 = None
        add_15: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_14, convert_element_type_58);  add_14 = convert_element_type_58 = None
        convert_element_type_59: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        mul_52: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_15, arg55_1);  arg55_1 = None
        mul_53: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_52, 2048)
        sum_26: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_52, [2], True)
        mul_54: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_52, arg56_1);  mul_52 = None
        sum_27: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_54, [2], True);  mul_54 = None
        mul_55: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg56_1, sum_27);  sum_27 = None
        sub_13: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_53, sum_26);  mul_53 = sum_26 = None
        sub_14: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_13, mul_55);  sub_13 = mul_55 = None
        mul_56: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg57_1, sub_14);  arg57_1 = sub_14 = None
        mul_57: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_15, arg56_1);  arg56_1 = None
        sum_28: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_57, [0, 1]);  mul_57 = None
        sum_29: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_15, [0, 1]);  add_15 = None
        add_16: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_13, mul_56);  add_13 = mul_56 = None
        convert_element_type_60: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16)
        view_58: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_60, _shape_param_61);  convert_element_type_60 = _shape_param_61 = None
        mm_26: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_58, arg58_1);  arg58_1 = None
        permute_23: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_58, [1, 0])
        mm_27: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_23, arg59_1);  permute_23 = arg59_1 = None
        sum_30: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_58, [0], True, dtype = torch.float32);  view_58 = None
        view_59: "f32[2048]" = torch.ops.aten.view.default(sum_30, _shape_param_62);  sum_30 = _shape_param_62 = None
        convert_element_type_61: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_59, torch.bfloat16);  view_59 = None
        view_60: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_26, _shape_param_63);  mm_26 = _shape_param_63 = None
        convert_element_type_62: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_60, torch.float32);  view_60 = None
        convert_element_type_63: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_64: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_61, torch.float32);  convert_element_type_61 = None
        view_61: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg60_1, _shape_param_64);  arg60_1 = _shape_param_64 = None
        mul_58: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_61, 0.5)
        mul_59: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_62, mul_58);  mul_58 = None
        convert_element_type_65: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_61, torch.float32)
        pow_5: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_65, 3.0)
        mul_60: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_17: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_61, mul_60);  view_61 = mul_60 = None
        mul_61: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_2: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_61);  mul_61 = None
        add_18: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_62: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_62, add_18);  convert_element_type_62 = add_18 = None
        convert_element_type_66: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_62, torch.bfloat16);  mul_62 = None
        mul_63: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_15: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_63);  mul_63 = None
        mul_64: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_59, sub_15);  mul_59 = sub_15 = None
        mul_65: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_64, 0.7978845608028654);  mul_64 = None
        convert_element_type_67: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_65, torch.bfloat16)
        mul_66: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_65, 0.044715);  mul_65 = None
        pow_6: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_65, 2.0);  convert_element_type_65 = None
        mul_67: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_6, 3.0);  pow_6 = None
        mul_68: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_66, mul_67);  mul_66 = mul_67 = None
        convert_element_type_68: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_68, torch.bfloat16);  mul_68 = None
        add_19: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_67, convert_element_type_68);  convert_element_type_67 = convert_element_type_68 = None
        mul_69: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 0.5);  convert_element_type_66 = None
        add_20: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_19, mul_69);  add_19 = mul_69 = None
        view_62: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_20, _shape_param_65);  add_20 = _shape_param_65 = None
        mm_28: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_62, arg61_1);  arg61_1 = None
        permute_24: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_62, [1, 0])
        mm_29: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_24, arg62_1);  permute_24 = arg62_1 = None
        sum_31: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_62, [0], True, dtype = torch.float32);  view_62 = None
        view_63: "f32[8192]" = torch.ops.aten.view.default(sum_31, _shape_param_66);  sum_31 = _shape_param_66 = None
        convert_element_type_69: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_63, torch.bfloat16);  view_63 = None
        view_64: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_28, _shape_param_67);  mm_28 = _shape_param_67 = None
        convert_element_type_70: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_64, torch.float32);  view_64 = None
        convert_element_type_71: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_72: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_69, torch.float32);  convert_element_type_69 = None
        mul_70: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_70, arg63_1);  arg63_1 = None
        mul_71: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_70, 2048)
        sum_32: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_70, [2], True)
        mul_72: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_70, arg64_1);  mul_70 = None
        sum_33: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_72, [2], True);  mul_72 = None
        mul_73: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg64_1, sum_33);  sum_33 = None
        sub_16: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_71, sum_32);  mul_71 = sum_32 = None
        sub_17: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_16, mul_73);  sub_16 = mul_73 = None
        mul_74: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg65_1, sub_17);  arg65_1 = sub_17 = None
        mul_75: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_70, arg64_1);  arg64_1 = None
        sum_34: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_75, [0, 1]);  mul_75 = None
        sum_35: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_70, [0, 1]);  convert_element_type_70 = None
        add_21: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_16, mul_74);  add_16 = mul_74 = None
        convert_element_type_73: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16)
        view_65: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_73, _shape_param_68);  convert_element_type_73 = _shape_param_68 = None
        mm_30: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_65, arg66_1);  arg66_1 = None
        permute_25: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_65, [1, 0])
        mm_31: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_25, arg67_1);  permute_25 = arg67_1 = None
        sum_36: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_65, [0], True, dtype = torch.float32);  view_65 = None
        view_66: "f32[2048]" = torch.ops.aten.view.default(sum_36, _shape_param_69);  sum_36 = _shape_param_69 = None
        convert_element_type_74: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_66, torch.bfloat16);  view_66 = None
        view_67: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_30, _shape_param_70);  mm_30 = _shape_param_70 = None
        convert_element_type_75: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_76: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_74, torch.float32);  convert_element_type_74 = None
        view_68: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_67, _shape_param_71);  view_67 = _shape_param_71 = None
        permute_26: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
        clone_9: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
        view_69: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_9, _shape_param_72);  clone_9 = _shape_param_72 = None
        bmm_8: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg68_1, view_69);  arg68_1 = None
        bmm_9: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_69, arg69_1);  view_69 = arg69_1 = None
        view_70: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_73);  bmm_8 = _shape_param_73 = None
        view_71: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_9, _shape_param_74);  bmm_9 = _shape_param_74 = None
        convert_element_type_77: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_71, torch.float32);  view_71 = None
        mul_76: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_77, arg70_1);  convert_element_type_77 = None
        sum_37: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_76, [-1], True)
        neg_2: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg70_1);  arg70_1 = None
        fma_2: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_2, sum_37, mul_76);  neg_2 = sum_37 = mul_76 = None
        convert_element_type_78: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        slice_7: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg71_1, 2, 0, 128);  arg71_1 = None
        slice_8: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 128);  slice_7 = None
        where_5: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_8, convert_element_type_78, full_2);  slice_8 = convert_element_type_78 = None
        view_72: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_5, _shape_param_75);  where_5 = _shape_param_75 = None
        bmm_10: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg72_1, view_72);  arg72_1 = None
        bmm_11: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_72, arg73_1);  view_72 = arg73_1 = None
        view_73: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_76);  bmm_10 = _shape_param_76 = None
        view_74: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_11, _shape_param_77);  bmm_11 = _shape_param_77 = None
        convert_element_type_79: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        permute_27: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_79, [0, 1, 3, 2]);  convert_element_type_79 = None
        convert_element_type_80: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_27, torch.bfloat16);  permute_27 = None
        convert_element_type_81: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_74, torch.bfloat16);  view_74 = None
        permute_28: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_10: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
        view_75: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_10, _shape_param_78);  clone_10 = _shape_param_78 = None
        permute_29: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_80, [0, 2, 1, 3]);  convert_element_type_80 = None
        view_76: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_29, _shape_param_79);  permute_29 = _shape_param_79 = None
        permute_30: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_81, [0, 2, 1, 3]);  convert_element_type_81 = None
        clone_11: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_77: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_11, _shape_param_80);  clone_11 = _shape_param_80 = None
        view_78: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_75, _shape_param_81);  view_75 = _shape_param_81 = None
        permute_31: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_78, [1, 0])
        mm_32: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_31, arg74_1);  permute_31 = None
        mm_33: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_78, arg75_1);  view_78 = arg75_1 = None
        view_79: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_33, _shape_param_82);  mm_33 = _shape_param_82 = None
        convert_element_type_82: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_79, torch.float32);  view_79 = None
        convert_element_type_83: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_32, torch.float32);  mm_32 = None
        clone_12: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_76, memory_format = torch.contiguous_format);  view_76 = None
        view_80: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_12, _shape_param_83);  clone_12 = _shape_param_83 = None
        permute_32: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_80, [1, 0])
        mm_34: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_32, arg74_1);  permute_32 = None
        mm_35: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_80, arg76_1);  view_80 = arg76_1 = None
        view_81: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_35, _shape_param_84);  mm_35 = _shape_param_84 = None
        convert_element_type_84: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_81, torch.float32);  view_81 = None
        add_22: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_82, convert_element_type_84);  convert_element_type_82 = convert_element_type_84 = None
        convert_element_type_85: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        view_82: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_77, _shape_param_85);  view_77 = _shape_param_85 = None
        permute_33: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_82, [1, 0])
        mm_36: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_33, arg74_1);  permute_33 = arg74_1 = None
        mm_37: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_82, arg77_1);  view_82 = arg77_1 = None
        view_83: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_37, _shape_param_86);  mm_37 = _shape_param_86 = None
        convert_element_type_86: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_83, torch.float32);  view_83 = None
        add_23: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_22, convert_element_type_86);  add_22 = convert_element_type_86 = None
        convert_element_type_87: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        mul_77: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_23, arg78_1);  arg78_1 = None
        mul_78: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_77, 2048)
        sum_38: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_77, [2], True)
        mul_79: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_77, arg79_1);  mul_77 = None
        sum_39: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_79, [2], True);  mul_79 = None
        mul_80: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg79_1, sum_39);  sum_39 = None
        sub_18: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_78, sum_38);  mul_78 = sum_38 = None
        sub_19: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_18, mul_80);  sub_18 = mul_80 = None
        mul_81: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg80_1, sub_19);  arg80_1 = sub_19 = None
        mul_82: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_23, arg79_1);  arg79_1 = None
        sum_40: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_82, [0, 1]);  mul_82 = None
        sum_41: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_23, [0, 1]);  add_23 = None
        add_24: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_21, mul_81);  add_21 = mul_81 = None
        convert_element_type_88: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16)
        view_84: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_88, _shape_param_87);  convert_element_type_88 = _shape_param_87 = None
        mm_38: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_84, arg81_1);  arg81_1 = None
        permute_34: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_84, [1, 0])
        mm_39: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_34, arg82_1);  permute_34 = arg82_1 = None
        sum_42: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_84, [0], True, dtype = torch.float32);  view_84 = None
        view_85: "f32[2048]" = torch.ops.aten.view.default(sum_42, _shape_param_88);  sum_42 = _shape_param_88 = None
        convert_element_type_89: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_85, torch.bfloat16);  view_85 = None
        view_86: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_38, _shape_param_89);  mm_38 = _shape_param_89 = None
        convert_element_type_90: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_86, torch.float32);  view_86 = None
        convert_element_type_91: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_92: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_89, torch.float32);  convert_element_type_89 = None
        view_87: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg83_1, _shape_param_90);  arg83_1 = _shape_param_90 = None
        mul_83: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
        mul_84: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_90, mul_83);  mul_83 = None
        convert_element_type_93: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_87, torch.float32)
        pow_7: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 3.0)
        mul_85: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_25: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_87, mul_85);  view_87 = mul_85 = None
        mul_86: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_3: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
        add_26: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_87: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_90, add_26);  convert_element_type_90 = add_26 = None
        convert_element_type_94: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None
        mul_88: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_20: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_88);  mul_88 = None
        mul_89: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_84, sub_20);  mul_84 = sub_20 = None
        mul_90: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_89, 0.7978845608028654);  mul_89 = None
        convert_element_type_95: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_90, torch.bfloat16)
        mul_91: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_90, 0.044715);  mul_90 = None
        pow_8: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_93, 2.0);  convert_element_type_93 = None
        mul_92: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_8, 3.0);  pow_8 = None
        mul_93: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_91, mul_92);  mul_91 = mul_92 = None
        convert_element_type_96: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_93, torch.bfloat16);  mul_93 = None
        add_27: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_95, convert_element_type_96);  convert_element_type_95 = convert_element_type_96 = None
        mul_94: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 0.5);  convert_element_type_94 = None
        add_28: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_27, mul_94);  add_27 = mul_94 = None
        view_88: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_28, _shape_param_91);  add_28 = _shape_param_91 = None
        mm_40: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_88, arg84_1);  arg84_1 = None
        permute_35: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_88, [1, 0])
        mm_41: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_35, arg85_1);  permute_35 = arg85_1 = None
        sum_43: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_88, [0], True, dtype = torch.float32);  view_88 = None
        view_89: "f32[8192]" = torch.ops.aten.view.default(sum_43, _shape_param_92);  sum_43 = _shape_param_92 = None
        convert_element_type_97: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_89, torch.bfloat16);  view_89 = None
        view_90: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_40, _shape_param_93);  mm_40 = _shape_param_93 = None
        convert_element_type_98: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_90, torch.float32);  view_90 = None
        convert_element_type_99: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_100: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_97, torch.float32);  convert_element_type_97 = None
        mul_95: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_98, arg86_1);  arg86_1 = None
        mul_96: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_95, 2048)
        sum_44: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_95, [2], True)
        mul_97: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_95, arg87_1);  mul_95 = None
        sum_45: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_97, [2], True);  mul_97 = None
        mul_98: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg87_1, sum_45);  sum_45 = None
        sub_21: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_96, sum_44);  mul_96 = sum_44 = None
        sub_22: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_21, mul_98);  sub_21 = mul_98 = None
        mul_99: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg88_1, sub_22);  arg88_1 = sub_22 = None
        mul_100: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_98, arg87_1);  arg87_1 = None
        sum_46: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_100, [0, 1]);  mul_100 = None
        sum_47: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_98, [0, 1]);  convert_element_type_98 = None
        add_29: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_24, mul_99);  add_24 = mul_99 = None
        convert_element_type_101: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_29, torch.bfloat16)
        view_91: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_101, _shape_param_94);  convert_element_type_101 = _shape_param_94 = None
        mm_42: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_91, arg89_1);  arg89_1 = None
        permute_36: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_91, [1, 0])
        mm_43: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_36, arg90_1);  permute_36 = arg90_1 = None
        sum_48: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_91, [0], True, dtype = torch.float32);  view_91 = None
        view_92: "f32[2048]" = torch.ops.aten.view.default(sum_48, _shape_param_95);  sum_48 = _shape_param_95 = None
        convert_element_type_102: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_92, torch.bfloat16);  view_92 = None
        view_93: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_42, _shape_param_96);  mm_42 = _shape_param_96 = None
        convert_element_type_103: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_104: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_102, torch.float32);  convert_element_type_102 = None
        view_94: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_93, _shape_param_97);  view_93 = _shape_param_97 = None
        permute_37: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
        clone_13: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_95: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_13, _shape_param_98);  clone_13 = _shape_param_98 = None
        bmm_12: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg91_1, view_95);  arg91_1 = None
        bmm_13: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_95, arg92_1);  view_95 = arg92_1 = None
        view_96: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_99);  bmm_12 = _shape_param_99 = None
        view_97: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_13, _shape_param_100);  bmm_13 = _shape_param_100 = None
        convert_element_type_105: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None
        mul_101: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_105, arg93_1);  convert_element_type_105 = None
        sum_49: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [-1], True)
        neg_3: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg93_1);  arg93_1 = None
        fma_3: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_3, sum_49, mul_101);  neg_3 = sum_49 = mul_101 = None
        convert_element_type_106: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        slice_9: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg94_1, 2, 0, 128);  arg94_1 = None
        slice_10: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_9, 3, 0, 128);  slice_9 = None
        where_6: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_10, convert_element_type_106, full_2);  slice_10 = convert_element_type_106 = None
        view_98: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_6, _shape_param_101);  where_6 = _shape_param_101 = None
        bmm_14: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg95_1, view_98);  arg95_1 = None
        bmm_15: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_98, arg96_1);  view_98 = arg96_1 = None
        view_99: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_102);  bmm_14 = _shape_param_102 = None
        view_100: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_15, _shape_param_103);  bmm_15 = _shape_param_103 = None
        convert_element_type_107: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_99, torch.float32);  view_99 = None
        permute_38: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_107, [0, 1, 3, 2]);  convert_element_type_107 = None
        convert_element_type_108: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_38, torch.bfloat16);  permute_38 = None
        convert_element_type_109: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_100, torch.bfloat16);  view_100 = None
        permute_39: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        clone_14: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_39, memory_format = torch.contiguous_format);  permute_39 = None
        view_101: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_14, _shape_param_104);  clone_14 = _shape_param_104 = None
        permute_40: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_108, [0, 2, 1, 3]);  convert_element_type_108 = None
        view_102: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_40, _shape_param_105);  permute_40 = _shape_param_105 = None
        permute_41: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_109, [0, 2, 1, 3]);  convert_element_type_109 = None
        clone_15: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_103: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_15, _shape_param_106);  clone_15 = _shape_param_106 = None
        view_104: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_101, _shape_param_107);  view_101 = _shape_param_107 = None
        permute_42: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_104, [1, 0])
        mm_44: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_42, arg97_1);  permute_42 = None
        mm_45: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_104, arg98_1);  view_104 = arg98_1 = None
        view_105: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_45, _shape_param_108);  mm_45 = _shape_param_108 = None
        convert_element_type_110: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_105, torch.float32);  view_105 = None
        convert_element_type_111: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        clone_16: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_102, memory_format = torch.contiguous_format);  view_102 = None
        view_106: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_16, _shape_param_109);  clone_16 = _shape_param_109 = None
        permute_43: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_106, [1, 0])
        mm_46: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_43, arg97_1);  permute_43 = None
        mm_47: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_106, arg99_1);  view_106 = arg99_1 = None
        view_107: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_47, _shape_param_110);  mm_47 = _shape_param_110 = None
        convert_element_type_112: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        add_30: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_110, convert_element_type_112);  convert_element_type_110 = convert_element_type_112 = None
        convert_element_type_113: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        view_108: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_103, _shape_param_111);  view_103 = _shape_param_111 = None
        permute_44: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_108, [1, 0])
        mm_48: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_44, arg97_1);  permute_44 = arg97_1 = None
        mm_49: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_108, arg100_1);  view_108 = arg100_1 = None
        view_109: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_49, _shape_param_112);  mm_49 = _shape_param_112 = None
        convert_element_type_114: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_109, torch.float32);  view_109 = None
        add_31: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_30, convert_element_type_114);  add_30 = convert_element_type_114 = None
        convert_element_type_115: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_48, torch.float32);  mm_48 = None
        mul_102: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_31, arg101_1);  arg101_1 = None
        mul_103: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_102, 2048)
        sum_50: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_102, [2], True)
        mul_104: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_102, arg102_1);  mul_102 = None
        sum_51: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_104, [2], True);  mul_104 = None
        mul_105: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg102_1, sum_51);  sum_51 = None
        sub_23: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_103, sum_50);  mul_103 = sum_50 = None
        sub_24: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_23, mul_105);  sub_23 = mul_105 = None
        mul_106: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg103_1, sub_24);  arg103_1 = sub_24 = None
        mul_107: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_31, arg102_1);  arg102_1 = None
        sum_52: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_107, [0, 1]);  mul_107 = None
        sum_53: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_31, [0, 1]);  add_31 = None
        add_32: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_29, mul_106);  add_29 = mul_106 = None
        convert_element_type_116: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16)
        view_110: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_116, _shape_param_113);  convert_element_type_116 = _shape_param_113 = None
        mm_50: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_110, arg104_1);  arg104_1 = None
        permute_45: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_110, [1, 0])
        mm_51: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_45, arg105_1);  permute_45 = arg105_1 = None
        sum_54: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_110, [0], True, dtype = torch.float32);  view_110 = None
        view_111: "f32[2048]" = torch.ops.aten.view.default(sum_54, _shape_param_114);  sum_54 = _shape_param_114 = None
        convert_element_type_117: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_111, torch.bfloat16);  view_111 = None
        view_112: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_50, _shape_param_115);  mm_50 = _shape_param_115 = None
        convert_element_type_118: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_112, torch.float32);  view_112 = None
        convert_element_type_119: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_120: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_117, torch.float32);  convert_element_type_117 = None
        view_113: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg106_1, _shape_param_116);  arg106_1 = _shape_param_116 = None
        mul_108: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_113, 0.5)
        mul_109: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_118, mul_108);  mul_108 = None
        convert_element_type_121: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_113, torch.float32)
        pow_9: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_121, 3.0)
        mul_110: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_33: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_113, mul_110);  view_113 = mul_110 = None
        mul_111: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_4: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_111);  mul_111 = None
        add_34: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_112: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_118, add_34);  convert_element_type_118 = add_34 = None
        convert_element_type_122: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_112, torch.bfloat16);  mul_112 = None
        mul_113: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_25: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_113);  mul_113 = None
        mul_114: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_109, sub_25);  mul_109 = sub_25 = None
        mul_115: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_114, 0.7978845608028654);  mul_114 = None
        convert_element_type_123: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_115, torch.bfloat16)
        mul_116: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_115, 0.044715);  mul_115 = None
        pow_10: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_121, 2.0);  convert_element_type_121 = None
        mul_117: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_10, 3.0);  pow_10 = None
        mul_118: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_116, mul_117);  mul_116 = mul_117 = None
        convert_element_type_124: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_118, torch.bfloat16);  mul_118 = None
        add_35: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_123, convert_element_type_124);  convert_element_type_123 = convert_element_type_124 = None
        mul_119: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_122, 0.5);  convert_element_type_122 = None
        add_36: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_35, mul_119);  add_35 = mul_119 = None
        view_114: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_36, _shape_param_117);  add_36 = _shape_param_117 = None
        mm_52: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_114, arg107_1);  arg107_1 = None
        permute_46: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_114, [1, 0])
        mm_53: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_46, arg108_1);  permute_46 = arg108_1 = None
        sum_55: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_114, [0], True, dtype = torch.float32);  view_114 = None
        view_115: "f32[8192]" = torch.ops.aten.view.default(sum_55, _shape_param_118);  sum_55 = _shape_param_118 = None
        convert_element_type_125: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_115, torch.bfloat16);  view_115 = None
        view_116: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_52, _shape_param_119);  mm_52 = _shape_param_119 = None
        convert_element_type_126: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_116, torch.float32);  view_116 = None
        convert_element_type_127: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_128: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_125, torch.float32);  convert_element_type_125 = None
        mul_120: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_126, arg109_1);  arg109_1 = None
        mul_121: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, 2048)
        sum_56: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_120, [2], True)
        mul_122: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, arg110_1);  mul_120 = None
        sum_57: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True);  mul_122 = None
        mul_123: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg110_1, sum_57);  sum_57 = None
        sub_26: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_121, sum_56);  mul_121 = sum_56 = None
        sub_27: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_26, mul_123);  sub_26 = mul_123 = None
        mul_124: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg111_1, sub_27);  arg111_1 = sub_27 = None
        mul_125: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_126, arg110_1);  arg110_1 = None
        sum_58: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_125, [0, 1]);  mul_125 = None
        sum_59: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_126, [0, 1]);  convert_element_type_126 = None
        add_37: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_32, mul_124);  add_32 = mul_124 = None
        convert_element_type_129: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_37, torch.bfloat16)
        view_117: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_129, _shape_param_120);  convert_element_type_129 = _shape_param_120 = None
        mm_54: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_117, arg112_1);  arg112_1 = None
        permute_47: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_117, [1, 0])
        mm_55: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_47, arg113_1);  permute_47 = arg113_1 = None
        sum_60: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_117, [0], True, dtype = torch.float32);  view_117 = None
        view_118: "f32[2048]" = torch.ops.aten.view.default(sum_60, _shape_param_121);  sum_60 = _shape_param_121 = None
        convert_element_type_130: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_118, torch.bfloat16);  view_118 = None
        view_119: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_54, _shape_param_122);  mm_54 = _shape_param_122 = None
        convert_element_type_131: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_132: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_130, torch.float32);  convert_element_type_130 = None
        view_120: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_119, _shape_param_123);  view_119 = _shape_param_123 = None
        permute_48: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        clone_17: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
        view_121: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_17, _shape_param_124);  clone_17 = _shape_param_124 = None
        bmm_16: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg114_1, view_121);  arg114_1 = None
        bmm_17: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_121, arg115_1);  view_121 = arg115_1 = None
        view_122: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_125);  bmm_16 = _shape_param_125 = None
        view_123: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_17, _shape_param_126);  bmm_17 = _shape_param_126 = None
        convert_element_type_133: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_123, torch.float32);  view_123 = None
        mul_126: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_133, arg116_1);  convert_element_type_133 = None
        sum_61: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_126, [-1], True)
        neg_4: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg116_1);  arg116_1 = None
        fma_4: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_4, sum_61, mul_126);  neg_4 = sum_61 = mul_126 = None
        convert_element_type_134: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        slice_11: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg117_1, 2, 0, 128);  arg117_1 = None
        slice_12: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 128);  slice_11 = None
        where_7: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_12, convert_element_type_134, full_2);  slice_12 = convert_element_type_134 = None
        view_124: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_7, _shape_param_127);  where_7 = _shape_param_127 = None
        bmm_18: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg118_1, view_124);  arg118_1 = None
        bmm_19: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_124, arg119_1);  view_124 = arg119_1 = None
        view_125: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_128);  bmm_18 = _shape_param_128 = None
        view_126: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_19, _shape_param_129);  bmm_19 = _shape_param_129 = None
        convert_element_type_135: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None
        permute_49: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_135, [0, 1, 3, 2]);  convert_element_type_135 = None
        convert_element_type_136: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_49, torch.bfloat16);  permute_49 = None
        convert_element_type_137: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_126, torch.bfloat16);  view_126 = None
        permute_50: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        clone_18: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_127: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_18, _shape_param_130);  clone_18 = _shape_param_130 = None
        permute_51: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_136, [0, 2, 1, 3]);  convert_element_type_136 = None
        view_128: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_51, _shape_param_131);  permute_51 = _shape_param_131 = None
        permute_52: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_137, [0, 2, 1, 3]);  convert_element_type_137 = None
        clone_19: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
        view_129: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_19, _shape_param_132);  clone_19 = _shape_param_132 = None
        view_130: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_127, _shape_param_133);  view_127 = _shape_param_133 = None
        permute_53: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_130, [1, 0])
        mm_56: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_53, arg120_1);  permute_53 = None
        mm_57: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_130, arg121_1);  view_130 = arg121_1 = None
        view_131: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_57, _shape_param_134);  mm_57 = _shape_param_134 = None
        convert_element_type_138: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_131, torch.float32);  view_131 = None
        convert_element_type_139: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        clone_20: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_128, memory_format = torch.contiguous_format);  view_128 = None
        view_132: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_20, _shape_param_135);  clone_20 = _shape_param_135 = None
        permute_54: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_132, [1, 0])
        mm_58: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_54, arg120_1);  permute_54 = None
        mm_59: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_132, arg122_1);  view_132 = arg122_1 = None
        view_133: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_59, _shape_param_136);  mm_59 = _shape_param_136 = None
        convert_element_type_140: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_133, torch.float32);  view_133 = None
        add_38: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_138, convert_element_type_140);  convert_element_type_138 = convert_element_type_140 = None
        convert_element_type_141: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        view_134: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_129, _shape_param_137);  view_129 = _shape_param_137 = None
        permute_55: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_134, [1, 0])
        mm_60: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_55, arg120_1);  permute_55 = arg120_1 = None
        mm_61: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_134, arg123_1);  view_134 = arg123_1 = None
        view_135: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_61, _shape_param_138);  mm_61 = _shape_param_138 = None
        convert_element_type_142: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_135, torch.float32);  view_135 = None
        add_39: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_38, convert_element_type_142);  add_38 = convert_element_type_142 = None
        convert_element_type_143: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        mul_127: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_39, arg124_1);  arg124_1 = None
        mul_128: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_127, 2048)
        sum_62: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_127, [2], True)
        mul_129: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_127, arg125_1);  mul_127 = None
        sum_63: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True);  mul_129 = None
        mul_130: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg125_1, sum_63);  sum_63 = None
        sub_28: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_128, sum_62);  mul_128 = sum_62 = None
        sub_29: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_28, mul_130);  sub_28 = mul_130 = None
        mul_131: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg126_1, sub_29);  arg126_1 = sub_29 = None
        mul_132: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_39, arg125_1);  arg125_1 = None
        sum_64: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_132, [0, 1]);  mul_132 = None
        sum_65: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_39, [0, 1]);  add_39 = None
        add_40: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_37, mul_131);  add_37 = mul_131 = None
        convert_element_type_144: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16)
        view_136: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_144, _shape_param_139);  convert_element_type_144 = _shape_param_139 = None
        mm_62: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_136, arg127_1);  arg127_1 = None
        permute_56: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_136, [1, 0])
        mm_63: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_56, arg128_1);  permute_56 = arg128_1 = None
        sum_66: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_136, [0], True, dtype = torch.float32);  view_136 = None
        view_137: "f32[2048]" = torch.ops.aten.view.default(sum_66, _shape_param_140);  sum_66 = _shape_param_140 = None
        convert_element_type_145: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_137, torch.bfloat16);  view_137 = None
        view_138: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_62, _shape_param_141);  mm_62 = _shape_param_141 = None
        convert_element_type_146: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_138, torch.float32);  view_138 = None
        convert_element_type_147: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_148: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_145, torch.float32);  convert_element_type_145 = None
        view_139: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg129_1, _shape_param_142);  arg129_1 = _shape_param_142 = None
        mul_133: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_139, 0.5)
        mul_134: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_146, mul_133);  mul_133 = None
        convert_element_type_149: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_139, torch.float32)
        pow_11: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_149, 3.0)
        mul_135: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_41: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_139, mul_135);  view_139 = mul_135 = None
        mul_136: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_5: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_136);  mul_136 = None
        add_42: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_137: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_146, add_42);  convert_element_type_146 = add_42 = None
        convert_element_type_150: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_137, torch.bfloat16);  mul_137 = None
        mul_138: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_30: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_138);  mul_138 = None
        mul_139: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_134, sub_30);  mul_134 = sub_30 = None
        mul_140: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_139, 0.7978845608028654);  mul_139 = None
        convert_element_type_151: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_140, torch.bfloat16)
        mul_141: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_140, 0.044715);  mul_140 = None
        pow_12: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_149, 2.0);  convert_element_type_149 = None
        mul_142: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_12, 3.0);  pow_12 = None
        mul_143: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_141, mul_142);  mul_141 = mul_142 = None
        convert_element_type_152: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None
        add_43: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_151, convert_element_type_152);  convert_element_type_151 = convert_element_type_152 = None
        mul_144: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_150, 0.5);  convert_element_type_150 = None
        add_44: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_43, mul_144);  add_43 = mul_144 = None
        view_140: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_44, _shape_param_143);  add_44 = _shape_param_143 = None
        mm_64: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_140, arg130_1);  arg130_1 = None
        permute_57: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_140, [1, 0])
        mm_65: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_57, arg131_1);  permute_57 = arg131_1 = None
        sum_67: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_140, [0], True, dtype = torch.float32);  view_140 = None
        view_141: "f32[8192]" = torch.ops.aten.view.default(sum_67, _shape_param_144);  sum_67 = _shape_param_144 = None
        convert_element_type_153: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_141, torch.bfloat16);  view_141 = None
        view_142: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_64, _shape_param_145);  mm_64 = _shape_param_145 = None
        convert_element_type_154: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_142, torch.float32);  view_142 = None
        convert_element_type_155: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_156: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_153, torch.float32);  convert_element_type_153 = None
        mul_145: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_154, arg132_1);  arg132_1 = None
        mul_146: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_145, 2048)
        sum_68: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True)
        mul_147: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_145, arg133_1);  mul_145 = None
        sum_69: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_147, [2], True);  mul_147 = None
        mul_148: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg133_1, sum_69);  sum_69 = None
        sub_31: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_146, sum_68);  mul_146 = sum_68 = None
        sub_32: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_31, mul_148);  sub_31 = mul_148 = None
        mul_149: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg134_1, sub_32);  arg134_1 = sub_32 = None
        mul_150: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_154, arg133_1);  arg133_1 = None
        sum_70: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_150, [0, 1]);  mul_150 = None
        sum_71: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_154, [0, 1]);  convert_element_type_154 = None
        add_45: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_40, mul_149);  add_40 = mul_149 = None
        convert_element_type_157: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16)
        view_143: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_157, _shape_param_146);  convert_element_type_157 = _shape_param_146 = None
        mm_66: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_143, arg135_1);  arg135_1 = None
        permute_58: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_143, [1, 0])
        mm_67: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_58, arg136_1);  permute_58 = arg136_1 = None
        sum_72: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_143, [0], True, dtype = torch.float32);  view_143 = None
        view_144: "f32[2048]" = torch.ops.aten.view.default(sum_72, _shape_param_147);  sum_72 = _shape_param_147 = None
        convert_element_type_158: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_144, torch.bfloat16);  view_144 = None
        view_145: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_66, _shape_param_148);  mm_66 = _shape_param_148 = None
        convert_element_type_159: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_160: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_158, torch.float32);  convert_element_type_158 = None
        view_146: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_145, _shape_param_149);  view_145 = _shape_param_149 = None
        permute_59: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
        clone_21: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
        view_147: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_21, _shape_param_150);  clone_21 = _shape_param_150 = None
        bmm_20: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg137_1, view_147);  arg137_1 = None
        bmm_21: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_147, arg138_1);  view_147 = arg138_1 = None
        view_148: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_151);  bmm_20 = _shape_param_151 = None
        view_149: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_21, _shape_param_152);  bmm_21 = _shape_param_152 = None
        convert_element_type_161: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_149, torch.float32);  view_149 = None
        mul_151: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_161, arg139_1);  convert_element_type_161 = None
        sum_73: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [-1], True)
        neg_5: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg139_1);  arg139_1 = None
        fma_5: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_5, sum_73, mul_151);  neg_5 = sum_73 = mul_151 = None
        convert_element_type_162: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        slice_13: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg140_1, 2, 0, 128);  arg140_1 = None
        slice_14: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_13, 3, 0, 128);  slice_13 = None
        where_8: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_14, convert_element_type_162, full_2);  slice_14 = convert_element_type_162 = None
        view_150: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_8, _shape_param_153);  where_8 = _shape_param_153 = None
        bmm_22: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg141_1, view_150);  arg141_1 = None
        bmm_23: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_150, arg142_1);  view_150 = arg142_1 = None
        view_151: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_154);  bmm_22 = _shape_param_154 = None
        view_152: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_23, _shape_param_155);  bmm_23 = _shape_param_155 = None
        convert_element_type_163: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        permute_60: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_163, [0, 1, 3, 2]);  convert_element_type_163 = None
        convert_element_type_164: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_60, torch.bfloat16);  permute_60 = None
        convert_element_type_165: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_152, torch.bfloat16);  view_152 = None
        permute_61: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
        clone_22: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
        view_153: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_22, _shape_param_156);  clone_22 = _shape_param_156 = None
        permute_62: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_164, [0, 2, 1, 3]);  convert_element_type_164 = None
        view_154: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_62, _shape_param_157);  permute_62 = _shape_param_157 = None
        permute_63: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_165, [0, 2, 1, 3]);  convert_element_type_165 = None
        clone_23: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
        view_155: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_23, _shape_param_158);  clone_23 = _shape_param_158 = None
        view_156: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_153, _shape_param_159);  view_153 = _shape_param_159 = None
        permute_64: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_156, [1, 0])
        mm_68: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_64, arg143_1);  permute_64 = None
        mm_69: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_156, arg144_1);  view_156 = arg144_1 = None
        view_157: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_69, _shape_param_160);  mm_69 = _shape_param_160 = None
        convert_element_type_166: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_157, torch.float32);  view_157 = None
        convert_element_type_167: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        clone_24: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_154, memory_format = torch.contiguous_format);  view_154 = None
        view_158: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_24, _shape_param_161);  clone_24 = _shape_param_161 = None
        permute_65: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_158, [1, 0])
        mm_70: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_65, arg143_1);  permute_65 = None
        mm_71: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_158, arg145_1);  view_158 = arg145_1 = None
        view_159: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_71, _shape_param_162);  mm_71 = _shape_param_162 = None
        convert_element_type_168: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_159, torch.float32);  view_159 = None
        add_46: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_166, convert_element_type_168);  convert_element_type_166 = convert_element_type_168 = None
        convert_element_type_169: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        view_160: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_155, _shape_param_163);  view_155 = _shape_param_163 = None
        permute_66: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_160, [1, 0])
        mm_72: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_66, arg143_1);  permute_66 = arg143_1 = None
        mm_73: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_160, arg146_1);  view_160 = arg146_1 = None
        view_161: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_73, _shape_param_164);  mm_73 = _shape_param_164 = None
        convert_element_type_170: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_161, torch.float32);  view_161 = None
        add_47: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_46, convert_element_type_170);  add_46 = convert_element_type_170 = None
        convert_element_type_171: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        mul_152: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_47, arg147_1);  arg147_1 = None
        mul_153: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, 2048)
        sum_74: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True)
        mul_154: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, arg148_1);  mul_152 = None
        sum_75: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_154, [2], True);  mul_154 = None
        mul_155: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg148_1, sum_75);  sum_75 = None
        sub_33: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_153, sum_74);  mul_153 = sum_74 = None
        sub_34: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_33, mul_155);  sub_33 = mul_155 = None
        mul_156: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg149_1, sub_34);  arg149_1 = sub_34 = None
        mul_157: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_47, arg148_1);  arg148_1 = None
        sum_76: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_157, [0, 1]);  mul_157 = None
        sum_77: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_47, [0, 1]);  add_47 = None
        add_48: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_45, mul_156);  add_45 = mul_156 = None
        convert_element_type_172: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16)
        view_162: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_172, _shape_param_165);  convert_element_type_172 = _shape_param_165 = None
        mm_74: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_162, arg150_1);  arg150_1 = None
        permute_67: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_162, [1, 0])
        mm_75: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_67, arg151_1);  permute_67 = arg151_1 = None
        sum_78: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_162, [0], True, dtype = torch.float32);  view_162 = None
        view_163: "f32[2048]" = torch.ops.aten.view.default(sum_78, _shape_param_166);  sum_78 = _shape_param_166 = None
        convert_element_type_173: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_163, torch.bfloat16);  view_163 = None
        view_164: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_74, _shape_param_167);  mm_74 = _shape_param_167 = None
        convert_element_type_174: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_164, torch.float32);  view_164 = None
        convert_element_type_175: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_176: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_173, torch.float32);  convert_element_type_173 = None
        view_165: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg152_1, _shape_param_168);  arg152_1 = _shape_param_168 = None
        mul_158: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_165, 0.5)
        mul_159: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_174, mul_158);  mul_158 = None
        convert_element_type_177: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_165, torch.float32)
        pow_13: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_177, 3.0)
        mul_160: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_49: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_165, mul_160);  view_165 = mul_160 = None
        mul_161: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_6: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_161);  mul_161 = None
        add_50: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_162: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_174, add_50);  convert_element_type_174 = add_50 = None
        convert_element_type_178: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_162, torch.bfloat16);  mul_162 = None
        mul_163: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_35: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_163);  mul_163 = None
        mul_164: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_159, sub_35);  mul_159 = sub_35 = None
        mul_165: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_164, 0.7978845608028654);  mul_164 = None
        convert_element_type_179: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_165, torch.bfloat16)
        mul_166: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_165, 0.044715);  mul_165 = None
        pow_14: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_177, 2.0);  convert_element_type_177 = None
        mul_167: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_168: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_166, mul_167);  mul_166 = mul_167 = None
        convert_element_type_180: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        add_51: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_179, convert_element_type_180);  convert_element_type_179 = convert_element_type_180 = None
        mul_169: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_178, 0.5);  convert_element_type_178 = None
        add_52: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_51, mul_169);  add_51 = mul_169 = None
        view_166: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_52, _shape_param_169);  add_52 = _shape_param_169 = None
        mm_76: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_166, arg153_1);  arg153_1 = None
        permute_68: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_166, [1, 0])
        mm_77: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_68, arg154_1);  permute_68 = arg154_1 = None
        sum_79: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_166, [0], True, dtype = torch.float32);  view_166 = None
        view_167: "f32[8192]" = torch.ops.aten.view.default(sum_79, _shape_param_170);  sum_79 = _shape_param_170 = None
        convert_element_type_181: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_167, torch.bfloat16);  view_167 = None
        view_168: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_76, _shape_param_171);  mm_76 = _shape_param_171 = None
        convert_element_type_182: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_168, torch.float32);  view_168 = None
        convert_element_type_183: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_184: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_181, torch.float32);  convert_element_type_181 = None
        mul_170: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_182, arg155_1);  arg155_1 = None
        mul_171: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, 2048)
        sum_80: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True)
        mul_172: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, arg156_1);  mul_170 = None
        sum_81: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
        mul_173: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg156_1, sum_81);  sum_81 = None
        sub_36: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_171, sum_80);  mul_171 = sum_80 = None
        sub_37: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_36, mul_173);  sub_36 = mul_173 = None
        mul_174: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg157_1, sub_37);  arg157_1 = sub_37 = None
        mul_175: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_182, arg156_1);  arg156_1 = None
        sum_82: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_175, [0, 1]);  mul_175 = None
        sum_83: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_182, [0, 1]);  convert_element_type_182 = None
        add_53: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_48, mul_174);  add_48 = mul_174 = None
        convert_element_type_185: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_53, torch.bfloat16)
        view_169: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_185, _shape_param_172);  convert_element_type_185 = _shape_param_172 = None
        mm_78: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_169, arg158_1);  arg158_1 = None
        permute_69: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_169, [1, 0])
        mm_79: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_69, arg159_1);  permute_69 = arg159_1 = None
        sum_84: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_169, [0], True, dtype = torch.float32);  view_169 = None
        view_170: "f32[2048]" = torch.ops.aten.view.default(sum_84, _shape_param_173);  sum_84 = _shape_param_173 = None
        convert_element_type_186: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_170, torch.bfloat16);  view_170 = None
        view_171: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_78, _shape_param_174);  mm_78 = _shape_param_174 = None
        convert_element_type_187: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_188: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_186, torch.float32);  convert_element_type_186 = None
        view_172: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_171, _shape_param_175);  view_171 = _shape_param_175 = None
        permute_70: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        clone_25: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_173: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_25, _shape_param_176);  clone_25 = _shape_param_176 = None
        bmm_24: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg160_1, view_173);  arg160_1 = None
        bmm_25: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_173, arg161_1);  view_173 = arg161_1 = None
        view_174: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_177);  bmm_24 = _shape_param_177 = None
        view_175: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_25, _shape_param_178);  bmm_25 = _shape_param_178 = None
        convert_element_type_189: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_175, torch.float32);  view_175 = None
        mul_176: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_189, arg162_1);  convert_element_type_189 = None
        sum_85: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_176, [-1], True)
        neg_6: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg162_1);  arg162_1 = None
        fma_6: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_6, sum_85, mul_176);  neg_6 = sum_85 = mul_176 = None
        convert_element_type_190: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        slice_15: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg163_1, 2, 0, 128);  arg163_1 = None
        slice_16: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 128);  slice_15 = None
        where_9: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_16, convert_element_type_190, full_2);  slice_16 = convert_element_type_190 = None
        view_176: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_9, _shape_param_179);  where_9 = _shape_param_179 = None
        bmm_26: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg164_1, view_176);  arg164_1 = None
        bmm_27: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_176, arg165_1);  view_176 = arg165_1 = None
        view_177: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_180);  bmm_26 = _shape_param_180 = None
        view_178: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_27, _shape_param_181);  bmm_27 = _shape_param_181 = None
        convert_element_type_191: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        permute_71: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_191, [0, 1, 3, 2]);  convert_element_type_191 = None
        convert_element_type_192: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_71, torch.bfloat16);  permute_71 = None
        convert_element_type_193: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_178, torch.bfloat16);  view_178 = None
        permute_72: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_174, [0, 2, 1, 3]);  view_174 = None
        clone_26: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_72, memory_format = torch.contiguous_format);  permute_72 = None
        view_179: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_26, _shape_param_182);  clone_26 = _shape_param_182 = None
        permute_73: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_192, [0, 2, 1, 3]);  convert_element_type_192 = None
        view_180: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_73, _shape_param_183);  permute_73 = _shape_param_183 = None
        permute_74: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_193, [0, 2, 1, 3]);  convert_element_type_193 = None
        clone_27: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_74, memory_format = torch.contiguous_format);  permute_74 = None
        view_181: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_27, _shape_param_184);  clone_27 = _shape_param_184 = None
        view_182: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_179, _shape_param_185);  view_179 = _shape_param_185 = None
        permute_75: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_182, [1, 0])
        mm_80: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_75, arg166_1);  permute_75 = None
        mm_81: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_182, arg167_1);  view_182 = arg167_1 = None
        view_183: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_81, _shape_param_186);  mm_81 = _shape_param_186 = None
        convert_element_type_194: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_183, torch.float32);  view_183 = None
        convert_element_type_195: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        clone_28: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_180, memory_format = torch.contiguous_format);  view_180 = None
        view_184: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_28, _shape_param_187);  clone_28 = _shape_param_187 = None
        permute_76: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_184, [1, 0])
        mm_82: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_76, arg166_1);  permute_76 = None
        mm_83: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_184, arg168_1);  view_184 = arg168_1 = None
        view_185: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_83, _shape_param_188);  mm_83 = _shape_param_188 = None
        convert_element_type_196: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_185, torch.float32);  view_185 = None
        add_54: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_194, convert_element_type_196);  convert_element_type_194 = convert_element_type_196 = None
        convert_element_type_197: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        view_186: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_181, _shape_param_189);  view_181 = _shape_param_189 = None
        permute_77: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_186, [1, 0])
        mm_84: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_77, arg166_1);  permute_77 = arg166_1 = None
        mm_85: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_186, arg169_1);  view_186 = arg169_1 = None
        view_187: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_85, _shape_param_190);  mm_85 = _shape_param_190 = None
        convert_element_type_198: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_187, torch.float32);  view_187 = None
        add_55: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_54, convert_element_type_198);  add_54 = convert_element_type_198 = None
        convert_element_type_199: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        mul_177: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_55, arg170_1);  arg170_1 = None
        mul_178: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_177, 2048)
        sum_86: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_177, [2], True)
        mul_179: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_177, arg171_1);  mul_177 = None
        sum_87: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_179, [2], True);  mul_179 = None
        mul_180: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg171_1, sum_87);  sum_87 = None
        sub_38: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_178, sum_86);  mul_178 = sum_86 = None
        sub_39: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_38, mul_180);  sub_38 = mul_180 = None
        mul_181: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg172_1, sub_39);  arg172_1 = sub_39 = None
        mul_182: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_55, arg171_1);  arg171_1 = None
        sum_88: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 1]);  mul_182 = None
        sum_89: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_55, [0, 1]);  add_55 = None
        add_56: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_53, mul_181);  add_53 = mul_181 = None
        convert_element_type_200: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16)
        view_188: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_200, _shape_param_191);  convert_element_type_200 = _shape_param_191 = None
        mm_86: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_188, arg173_1);  arg173_1 = None
        permute_78: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_188, [1, 0])
        mm_87: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_78, arg174_1);  permute_78 = arg174_1 = None
        sum_90: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True, dtype = torch.float32);  view_188 = None
        view_189: "f32[2048]" = torch.ops.aten.view.default(sum_90, _shape_param_192);  sum_90 = _shape_param_192 = None
        convert_element_type_201: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_189, torch.bfloat16);  view_189 = None
        view_190: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_86, _shape_param_193);  mm_86 = _shape_param_193 = None
        convert_element_type_202: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_190, torch.float32);  view_190 = None
        convert_element_type_203: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_204: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_201, torch.float32);  convert_element_type_201 = None
        view_191: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg175_1, _shape_param_194);  arg175_1 = _shape_param_194 = None
        mul_183: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_191, 0.5)
        mul_184: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_202, mul_183);  mul_183 = None
        convert_element_type_205: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_191, torch.float32)
        pow_15: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_205, 3.0)
        mul_185: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_57: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_191, mul_185);  view_191 = mul_185 = None
        mul_186: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_7: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_186);  mul_186 = None
        add_58: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_187: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_202, add_58);  convert_element_type_202 = add_58 = None
        convert_element_type_206: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_187, torch.bfloat16);  mul_187 = None
        mul_188: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_40: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_188);  mul_188 = None
        mul_189: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_184, sub_40);  mul_184 = sub_40 = None
        mul_190: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_189, 0.7978845608028654);  mul_189 = None
        convert_element_type_207: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_190, torch.bfloat16)
        mul_191: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_190, 0.044715);  mul_190 = None
        pow_16: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_205, 2.0);  convert_element_type_205 = None
        mul_192: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_193: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None
        convert_element_type_208: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_193, torch.bfloat16);  mul_193 = None
        add_59: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_207, convert_element_type_208);  convert_element_type_207 = convert_element_type_208 = None
        mul_194: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_206, 0.5);  convert_element_type_206 = None
        add_60: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_59, mul_194);  add_59 = mul_194 = None
        view_192: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_60, _shape_param_195);  add_60 = _shape_param_195 = None
        mm_88: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_192, arg176_1);  arg176_1 = None
        permute_79: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_192, [1, 0])
        mm_89: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_79, arg177_1);  permute_79 = arg177_1 = None
        sum_91: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_192, [0], True, dtype = torch.float32);  view_192 = None
        view_193: "f32[8192]" = torch.ops.aten.view.default(sum_91, _shape_param_196);  sum_91 = _shape_param_196 = None
        convert_element_type_209: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_193, torch.bfloat16);  view_193 = None
        view_194: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_88, _shape_param_197);  mm_88 = _shape_param_197 = None
        convert_element_type_210: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_194, torch.float32);  view_194 = None
        convert_element_type_211: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_212: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_209, torch.float32);  convert_element_type_209 = None
        mul_195: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_210, arg178_1);  arg178_1 = None
        mul_196: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_195, 2048)
        sum_92: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True)
        mul_197: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_195, arg179_1);  mul_195 = None
        sum_93: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_197, [2], True);  mul_197 = None
        mul_198: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg179_1, sum_93);  sum_93 = None
        sub_41: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_196, sum_92);  mul_196 = sum_92 = None
        sub_42: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_41, mul_198);  sub_41 = mul_198 = None
        mul_199: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg180_1, sub_42);  arg180_1 = sub_42 = None
        mul_200: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_210, arg179_1);  arg179_1 = None
        sum_94: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1]);  mul_200 = None
        sum_95: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_210, [0, 1]);  convert_element_type_210 = None
        add_61: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_56, mul_199);  add_56 = mul_199 = None
        convert_element_type_213: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16)
        view_195: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_213, _shape_param_198);  convert_element_type_213 = _shape_param_198 = None
        mm_90: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_195, arg181_1);  arg181_1 = None
        permute_80: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_195, [1, 0])
        mm_91: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_80, arg182_1);  permute_80 = arg182_1 = None
        sum_96: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True, dtype = torch.float32);  view_195 = None
        view_196: "f32[2048]" = torch.ops.aten.view.default(sum_96, _shape_param_199);  sum_96 = _shape_param_199 = None
        convert_element_type_214: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_196, torch.bfloat16);  view_196 = None
        view_197: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_90, _shape_param_200);  mm_90 = _shape_param_200 = None
        convert_element_type_215: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_216: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_214, torch.float32);  convert_element_type_214 = None
        view_198: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_197, _shape_param_201);  view_197 = _shape_param_201 = None
        permute_81: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_198, [0, 2, 1, 3]);  view_198 = None
        clone_29: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
        view_199: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_29, _shape_param_202);  clone_29 = _shape_param_202 = None
        bmm_28: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg183_1, view_199);  arg183_1 = None
        bmm_29: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_199, arg184_1);  view_199 = arg184_1 = None
        view_200: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_203);  bmm_28 = _shape_param_203 = None
        view_201: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_29, _shape_param_204);  bmm_29 = _shape_param_204 = None
        convert_element_type_217: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_201, torch.float32);  view_201 = None
        mul_201: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_217, arg185_1);  convert_element_type_217 = None
        sum_97: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [-1], True)
        neg_7: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg185_1);  arg185_1 = None
        fma_7: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_7, sum_97, mul_201);  neg_7 = sum_97 = mul_201 = None
        convert_element_type_218: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        slice_17: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg186_1, 2, 0, 128);  arg186_1 = None
        slice_18: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_17, 3, 0, 128);  slice_17 = None
        where_10: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_18, convert_element_type_218, full_2);  slice_18 = convert_element_type_218 = None
        view_202: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_10, _shape_param_205);  where_10 = _shape_param_205 = None
        bmm_30: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg187_1, view_202);  arg187_1 = None
        bmm_31: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_202, arg188_1);  view_202 = arg188_1 = None
        view_203: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_206);  bmm_30 = _shape_param_206 = None
        view_204: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_31, _shape_param_207);  bmm_31 = _shape_param_207 = None
        convert_element_type_219: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_203, torch.float32);  view_203 = None
        permute_82: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_219, [0, 1, 3, 2]);  convert_element_type_219 = None
        convert_element_type_220: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_82, torch.bfloat16);  permute_82 = None
        convert_element_type_221: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_204, torch.bfloat16);  view_204 = None
        permute_83: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
        clone_30: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_205: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_30, _shape_param_208);  clone_30 = _shape_param_208 = None
        permute_84: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_220, [0, 2, 1, 3]);  convert_element_type_220 = None
        view_206: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_84, _shape_param_209);  permute_84 = _shape_param_209 = None
        permute_85: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_221, [0, 2, 1, 3]);  convert_element_type_221 = None
        clone_31: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
        view_207: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_31, _shape_param_210);  clone_31 = _shape_param_210 = None
        view_208: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_205, _shape_param_211);  view_205 = _shape_param_211 = None
        permute_86: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_208, [1, 0])
        mm_92: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_86, arg189_1);  permute_86 = None
        mm_93: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_208, arg190_1);  view_208 = arg190_1 = None
        view_209: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_93, _shape_param_212);  mm_93 = _shape_param_212 = None
        convert_element_type_222: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        convert_element_type_223: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        clone_32: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_206, memory_format = torch.contiguous_format);  view_206 = None
        view_210: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_32, _shape_param_213);  clone_32 = _shape_param_213 = None
        permute_87: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_210, [1, 0])
        mm_94: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_87, arg189_1);  permute_87 = None
        mm_95: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_210, arg191_1);  view_210 = arg191_1 = None
        view_211: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_95, _shape_param_214);  mm_95 = _shape_param_214 = None
        convert_element_type_224: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_211, torch.float32);  view_211 = None
        add_62: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_222, convert_element_type_224);  convert_element_type_222 = convert_element_type_224 = None
        convert_element_type_225: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        view_212: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_207, _shape_param_215);  view_207 = _shape_param_215 = None
        permute_88: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_212, [1, 0])
        mm_96: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_88, arg189_1);  permute_88 = arg189_1 = None
        mm_97: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_212, arg192_1);  view_212 = arg192_1 = None
        view_213: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_97, _shape_param_216);  mm_97 = _shape_param_216 = None
        convert_element_type_226: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_213, torch.float32);  view_213 = None
        add_63: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_62, convert_element_type_226);  add_62 = convert_element_type_226 = None
        convert_element_type_227: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        mul_202: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_63, arg193_1);  arg193_1 = None
        mul_203: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_202, 2048)
        sum_98: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True)
        mul_204: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_202, arg194_1);  mul_202 = None
        sum_99: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [2], True);  mul_204 = None
        mul_205: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg194_1, sum_99);  sum_99 = None
        sub_43: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_203, sum_98);  mul_203 = sum_98 = None
        sub_44: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_43, mul_205);  sub_43 = mul_205 = None
        mul_206: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg195_1, sub_44);  arg195_1 = sub_44 = None
        mul_207: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_63, arg194_1);  arg194_1 = None
        sum_100: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_207, [0, 1]);  mul_207 = None
        sum_101: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_63, [0, 1]);  add_63 = None
        add_64: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_61, mul_206);  add_61 = mul_206 = None
        convert_element_type_228: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16)
        view_214: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_228, _shape_param_217);  convert_element_type_228 = _shape_param_217 = None
        mm_98: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_214, arg196_1);  arg196_1 = None
        permute_89: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_214, [1, 0])
        mm_99: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_89, arg197_1);  permute_89 = arg197_1 = None
        sum_102: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True, dtype = torch.float32);  view_214 = None
        view_215: "f32[2048]" = torch.ops.aten.view.default(sum_102, _shape_param_218);  sum_102 = _shape_param_218 = None
        convert_element_type_229: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_215, torch.bfloat16);  view_215 = None
        view_216: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_98, _shape_param_219);  mm_98 = _shape_param_219 = None
        convert_element_type_230: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_216, torch.float32);  view_216 = None
        convert_element_type_231: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_232: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_229, torch.float32);  convert_element_type_229 = None
        view_217: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg198_1, _shape_param_220);  arg198_1 = _shape_param_220 = None
        mul_208: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
        mul_209: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_230, mul_208);  mul_208 = None
        convert_element_type_233: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_217, torch.float32)
        pow_17: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 3.0)
        mul_210: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_65: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_217, mul_210);  view_217 = mul_210 = None
        mul_211: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_8: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_211);  mul_211 = None
        add_66: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_212: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_230, add_66);  convert_element_type_230 = add_66 = None
        convert_element_type_234: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_212, torch.bfloat16);  mul_212 = None
        mul_213: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_45: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_213);  mul_213 = None
        mul_214: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_209, sub_45);  mul_209 = sub_45 = None
        mul_215: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_214, 0.7978845608028654);  mul_214 = None
        convert_element_type_235: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_215, torch.bfloat16)
        mul_216: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_215, 0.044715);  mul_215 = None
        pow_18: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_233, 2.0);  convert_element_type_233 = None
        mul_217: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_218: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_216, mul_217);  mul_216 = mul_217 = None
        convert_element_type_236: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_218, torch.bfloat16);  mul_218 = None
        add_67: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_235, convert_element_type_236);  convert_element_type_235 = convert_element_type_236 = None
        mul_219: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_234, 0.5);  convert_element_type_234 = None
        add_68: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_67, mul_219);  add_67 = mul_219 = None
        view_218: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_68, _shape_param_221);  add_68 = _shape_param_221 = None
        mm_100: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_218, arg199_1);  arg199_1 = None
        permute_90: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_218, [1, 0])
        mm_101: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_90, arg200_1);  permute_90 = arg200_1 = None
        sum_103: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True, dtype = torch.float32);  view_218 = None
        view_219: "f32[8192]" = torch.ops.aten.view.default(sum_103, _shape_param_222);  sum_103 = _shape_param_222 = None
        convert_element_type_237: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_219, torch.bfloat16);  view_219 = None
        view_220: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_100, _shape_param_223);  mm_100 = _shape_param_223 = None
        convert_element_type_238: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_220, torch.float32);  view_220 = None
        convert_element_type_239: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_240: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_237, torch.float32);  convert_element_type_237 = None
        mul_220: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_238, arg201_1);  arg201_1 = None
        mul_221: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_220, 2048)
        sum_104: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [2], True)
        mul_222: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_220, arg202_1);  mul_220 = None
        sum_105: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
        mul_223: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg202_1, sum_105);  sum_105 = None
        sub_46: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_221, sum_104);  mul_221 = sum_104 = None
        sub_47: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_46, mul_223);  sub_46 = mul_223 = None
        mul_224: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg203_1, sub_47);  arg203_1 = sub_47 = None
        mul_225: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_238, arg202_1);  arg202_1 = None
        sum_106: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_225, [0, 1]);  mul_225 = None
        sum_107: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_238, [0, 1]);  convert_element_type_238 = None
        add_69: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_64, mul_224);  add_64 = mul_224 = None
        convert_element_type_241: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_69, torch.bfloat16)
        view_221: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_241, _shape_param_224);  convert_element_type_241 = _shape_param_224 = None
        mm_102: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_221, arg204_1);  arg204_1 = None
        permute_91: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_221, [1, 0])
        mm_103: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_91, arg205_1);  permute_91 = arg205_1 = None
        sum_108: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True, dtype = torch.float32);  view_221 = None
        view_222: "f32[2048]" = torch.ops.aten.view.default(sum_108, _shape_param_225);  sum_108 = _shape_param_225 = None
        convert_element_type_242: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_222, torch.bfloat16);  view_222 = None
        view_223: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_102, _shape_param_226);  mm_102 = _shape_param_226 = None
        convert_element_type_243: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_244: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_242, torch.float32);  convert_element_type_242 = None
        view_224: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_223, _shape_param_227);  view_223 = _shape_param_227 = None
        permute_92: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_224, [0, 2, 1, 3]);  view_224 = None
        clone_33: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
        view_225: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_33, _shape_param_228);  clone_33 = _shape_param_228 = None
        bmm_32: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg206_1, view_225);  arg206_1 = None
        bmm_33: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_225, arg207_1);  view_225 = arg207_1 = None
        view_226: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_229);  bmm_32 = _shape_param_229 = None
        view_227: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_33, _shape_param_230);  bmm_33 = _shape_param_230 = None
        convert_element_type_245: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_227, torch.float32);  view_227 = None
        mul_226: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_245, arg208_1);  convert_element_type_245 = None
        sum_109: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_226, [-1], True)
        neg_8: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg208_1);  arg208_1 = None
        fma_8: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_8, sum_109, mul_226);  neg_8 = sum_109 = mul_226 = None
        convert_element_type_246: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        slice_19: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg209_1, 2, 0, 128);  arg209_1 = None
        slice_20: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 128);  slice_19 = None
        where_11: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_20, convert_element_type_246, full_2);  slice_20 = convert_element_type_246 = None
        view_228: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_11, _shape_param_231);  where_11 = _shape_param_231 = None
        bmm_34: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg210_1, view_228);  arg210_1 = None
        bmm_35: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_228, arg211_1);  view_228 = arg211_1 = None
        view_229: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_232);  bmm_34 = _shape_param_232 = None
        view_230: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_35, _shape_param_233);  bmm_35 = _shape_param_233 = None
        convert_element_type_247: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_229, torch.float32);  view_229 = None
        permute_93: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_247, [0, 1, 3, 2]);  convert_element_type_247 = None
        convert_element_type_248: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_93, torch.bfloat16);  permute_93 = None
        convert_element_type_249: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_230, torch.bfloat16);  view_230 = None
        permute_94: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
        clone_34: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None
        view_231: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_34, _shape_param_234);  clone_34 = _shape_param_234 = None
        permute_95: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_248, [0, 2, 1, 3]);  convert_element_type_248 = None
        view_232: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_95, _shape_param_235);  permute_95 = _shape_param_235 = None
        permute_96: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_249, [0, 2, 1, 3]);  convert_element_type_249 = None
        clone_35: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
        view_233: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_35, _shape_param_236);  clone_35 = _shape_param_236 = None
        view_234: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_231, _shape_param_237);  view_231 = _shape_param_237 = None
        permute_97: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_234, [1, 0])
        mm_104: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_97, arg212_1);  permute_97 = None
        mm_105: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_234, arg213_1);  view_234 = arg213_1 = None
        view_235: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_105, _shape_param_238);  mm_105 = _shape_param_238 = None
        convert_element_type_250: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_235, torch.float32);  view_235 = None
        convert_element_type_251: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        clone_36: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_232, memory_format = torch.contiguous_format);  view_232 = None
        view_236: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_36, _shape_param_239);  clone_36 = _shape_param_239 = None
        permute_98: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_236, [1, 0])
        mm_106: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_98, arg212_1);  permute_98 = None
        mm_107: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_236, arg214_1);  view_236 = arg214_1 = None
        view_237: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_107, _shape_param_240);  mm_107 = _shape_param_240 = None
        convert_element_type_252: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_237, torch.float32);  view_237 = None
        add_70: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_250, convert_element_type_252);  convert_element_type_250 = convert_element_type_252 = None
        convert_element_type_253: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        view_238: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_233, _shape_param_241);  view_233 = _shape_param_241 = None
        permute_99: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_238, [1, 0])
        mm_108: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_99, arg212_1);  permute_99 = arg212_1 = None
        mm_109: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_238, arg215_1);  view_238 = arg215_1 = None
        view_239: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_109, _shape_param_242);  mm_109 = _shape_param_242 = None
        convert_element_type_254: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        add_71: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_70, convert_element_type_254);  add_70 = convert_element_type_254 = None
        convert_element_type_255: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        mul_227: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_71, arg216_1);  arg216_1 = None
        mul_228: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_227, 2048)
        sum_110: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
        mul_229: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_227, arg217_1);  mul_227 = None
        sum_111: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
        mul_230: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg217_1, sum_111);  sum_111 = None
        sub_48: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_228, sum_110);  mul_228 = sum_110 = None
        sub_49: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_48, mul_230);  sub_48 = mul_230 = None
        mul_231: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg218_1, sub_49);  arg218_1 = sub_49 = None
        mul_232: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_71, arg217_1);  arg217_1 = None
        sum_112: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
        sum_113: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_71, [0, 1]);  add_71 = None
        add_72: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_69, mul_231);  add_69 = mul_231 = None
        convert_element_type_256: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16)
        view_240: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_256, _shape_param_243);  convert_element_type_256 = _shape_param_243 = None
        mm_110: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_240, arg219_1);  arg219_1 = None
        permute_100: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_240, [1, 0])
        mm_111: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_100, arg220_1);  permute_100 = arg220_1 = None
        sum_114: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True, dtype = torch.float32);  view_240 = None
        view_241: "f32[2048]" = torch.ops.aten.view.default(sum_114, _shape_param_244);  sum_114 = _shape_param_244 = None
        convert_element_type_257: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_241, torch.bfloat16);  view_241 = None
        view_242: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_110, _shape_param_245);  mm_110 = _shape_param_245 = None
        convert_element_type_258: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        convert_element_type_259: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_260: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_257, torch.float32);  convert_element_type_257 = None
        view_243: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg221_1, _shape_param_246);  arg221_1 = _shape_param_246 = None
        mul_233: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_243, 0.5)
        mul_234: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_258, mul_233);  mul_233 = None
        convert_element_type_261: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_243, torch.float32)
        pow_19: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_261, 3.0)
        mul_235: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_73: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_243, mul_235);  view_243 = mul_235 = None
        mul_236: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_9: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_236);  mul_236 = None
        add_74: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_237: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_258, add_74);  convert_element_type_258 = add_74 = None
        convert_element_type_262: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_237, torch.bfloat16);  mul_237 = None
        mul_238: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_50: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_238);  mul_238 = None
        mul_239: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_234, sub_50);  mul_234 = sub_50 = None
        mul_240: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_239, 0.7978845608028654);  mul_239 = None
        convert_element_type_263: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_240, torch.bfloat16)
        mul_241: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_240, 0.044715);  mul_240 = None
        pow_20: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_261, 2.0);  convert_element_type_261 = None
        mul_242: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_243: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
        convert_element_type_264: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_243, torch.bfloat16);  mul_243 = None
        add_75: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_263, convert_element_type_264);  convert_element_type_263 = convert_element_type_264 = None
        mul_244: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_262, 0.5);  convert_element_type_262 = None
        add_76: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_75, mul_244);  add_75 = mul_244 = None
        view_244: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_76, _shape_param_247);  add_76 = _shape_param_247 = None
        mm_112: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_244, arg222_1);  arg222_1 = None
        permute_101: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_244, [1, 0])
        mm_113: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_101, arg223_1);  permute_101 = arg223_1 = None
        sum_115: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_244, [0], True, dtype = torch.float32);  view_244 = None
        view_245: "f32[8192]" = torch.ops.aten.view.default(sum_115, _shape_param_248);  sum_115 = _shape_param_248 = None
        convert_element_type_265: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_245, torch.bfloat16);  view_245 = None
        view_246: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_112, _shape_param_249);  mm_112 = _shape_param_249 = None
        convert_element_type_266: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_246, torch.float32);  view_246 = None
        convert_element_type_267: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_268: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_265, torch.float32);  convert_element_type_265 = None
        mul_245: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_266, arg224_1);  arg224_1 = None
        mul_246: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_245, 2048)
        sum_116: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True)
        mul_247: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_245, arg225_1);  mul_245 = None
        sum_117: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_247, [2], True);  mul_247 = None
        mul_248: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg225_1, sum_117);  sum_117 = None
        sub_51: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_246, sum_116);  mul_246 = sum_116 = None
        sub_52: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_51, mul_248);  sub_51 = mul_248 = None
        mul_249: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg226_1, sub_52);  arg226_1 = sub_52 = None
        mul_250: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_266, arg225_1);  arg225_1 = None
        sum_118: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 1]);  mul_250 = None
        sum_119: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_266, [0, 1]);  convert_element_type_266 = None
        add_77: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_72, mul_249);  add_72 = mul_249 = None
        convert_element_type_269: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16)
        view_247: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_269, _shape_param_250);  convert_element_type_269 = _shape_param_250 = None
        mm_114: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_247, arg227_1);  arg227_1 = None
        permute_102: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_247, [1, 0])
        mm_115: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_102, arg228_1);  permute_102 = arg228_1 = None
        sum_120: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_247, [0], True, dtype = torch.float32);  view_247 = None
        view_248: "f32[2048]" = torch.ops.aten.view.default(sum_120, _shape_param_251);  sum_120 = _shape_param_251 = None
        convert_element_type_270: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_248, torch.bfloat16);  view_248 = None
        view_249: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_114, _shape_param_252);  mm_114 = _shape_param_252 = None
        convert_element_type_271: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_272: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_270, torch.float32);  convert_element_type_270 = None
        view_250: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_249, _shape_param_253);  view_249 = _shape_param_253 = None
        permute_103: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        clone_37: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
        view_251: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_37, _shape_param_254);  clone_37 = _shape_param_254 = None
        bmm_36: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg229_1, view_251);  arg229_1 = None
        bmm_37: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_251, arg230_1);  view_251 = arg230_1 = None
        view_252: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_255);  bmm_36 = _shape_param_255 = None
        view_253: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_37, _shape_param_256);  bmm_37 = _shape_param_256 = None
        convert_element_type_273: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_253, torch.float32);  view_253 = None
        mul_251: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_273, arg231_1);  convert_element_type_273 = None
        sum_121: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [-1], True)
        neg_9: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg231_1);  arg231_1 = None
        fma_9: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_9, sum_121, mul_251);  neg_9 = sum_121 = mul_251 = None
        convert_element_type_274: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        slice_21: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg232_1, 2, 0, 128);  arg232_1 = None
        slice_22: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_21, 3, 0, 128);  slice_21 = None
        where_12: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_22, convert_element_type_274, full_2);  slice_22 = convert_element_type_274 = None
        view_254: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_12, _shape_param_257);  where_12 = _shape_param_257 = None
        bmm_38: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg233_1, view_254);  arg233_1 = None
        bmm_39: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_254, arg234_1);  view_254 = arg234_1 = None
        view_255: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_258);  bmm_38 = _shape_param_258 = None
        view_256: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_39, _shape_param_259);  bmm_39 = _shape_param_259 = None
        convert_element_type_275: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_255, torch.float32);  view_255 = None
        permute_104: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_275, [0, 1, 3, 2]);  convert_element_type_275 = None
        convert_element_type_276: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_104, torch.bfloat16);  permute_104 = None
        convert_element_type_277: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_256, torch.bfloat16);  view_256 = None
        permute_105: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
        clone_38: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
        view_257: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_38, _shape_param_260);  clone_38 = _shape_param_260 = None
        permute_106: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_276, [0, 2, 1, 3]);  convert_element_type_276 = None
        view_258: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_106, _shape_param_261);  permute_106 = _shape_param_261 = None
        permute_107: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_277, [0, 2, 1, 3]);  convert_element_type_277 = None
        clone_39: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
        view_259: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_39, _shape_param_262);  clone_39 = _shape_param_262 = None
        view_260: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_257, _shape_param_263);  view_257 = _shape_param_263 = None
        permute_108: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_260, [1, 0])
        mm_116: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_108, arg235_1);  permute_108 = None
        mm_117: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_260, arg236_1);  view_260 = arg236_1 = None
        view_261: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_117, _shape_param_264);  mm_117 = _shape_param_264 = None
        convert_element_type_278: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        convert_element_type_279: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        clone_40: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_258, memory_format = torch.contiguous_format);  view_258 = None
        view_262: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_40, _shape_param_265);  clone_40 = _shape_param_265 = None
        permute_109: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_262, [1, 0])
        mm_118: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_109, arg235_1);  permute_109 = None
        mm_119: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_262, arg237_1);  view_262 = arg237_1 = None
        view_263: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_119, _shape_param_266);  mm_119 = _shape_param_266 = None
        convert_element_type_280: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        add_78: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_278, convert_element_type_280);  convert_element_type_278 = convert_element_type_280 = None
        convert_element_type_281: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        view_264: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_259, _shape_param_267);  view_259 = _shape_param_267 = None
        permute_110: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_264, [1, 0])
        mm_120: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_110, arg235_1);  permute_110 = arg235_1 = None
        mm_121: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_264, arg238_1);  view_264 = arg238_1 = None
        view_265: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_121, _shape_param_268);  mm_121 = _shape_param_268 = None
        convert_element_type_282: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        add_79: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_78, convert_element_type_282);  add_78 = convert_element_type_282 = None
        convert_element_type_283: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        mul_252: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_79, arg239_1);  arg239_1 = None
        mul_253: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_252, 2048)
        sum_122: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True)
        mul_254: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_252, arg240_1);  mul_252 = None
        sum_123: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_254, [2], True);  mul_254 = None
        mul_255: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg240_1, sum_123);  sum_123 = None
        sub_53: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_253, sum_122);  mul_253 = sum_122 = None
        sub_54: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_53, mul_255);  sub_53 = mul_255 = None
        mul_256: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg241_1, sub_54);  arg241_1 = sub_54 = None
        mul_257: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_79, arg240_1);  arg240_1 = None
        sum_124: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_257, [0, 1]);  mul_257 = None
        sum_125: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_79, [0, 1]);  add_79 = None
        add_80: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_77, mul_256);  add_77 = mul_256 = None
        convert_element_type_284: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16)
        view_266: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_284, _shape_param_269);  convert_element_type_284 = _shape_param_269 = None
        mm_122: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_266, arg242_1);  arg242_1 = None
        permute_111: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_266, [1, 0])
        mm_123: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_111, arg243_1);  permute_111 = arg243_1 = None
        sum_126: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True, dtype = torch.float32);  view_266 = None
        view_267: "f32[2048]" = torch.ops.aten.view.default(sum_126, _shape_param_270);  sum_126 = _shape_param_270 = None
        convert_element_type_285: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_267, torch.bfloat16);  view_267 = None
        view_268: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_122, _shape_param_271);  mm_122 = _shape_param_271 = None
        convert_element_type_286: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_268, torch.float32);  view_268 = None
        convert_element_type_287: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_288: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_285, torch.float32);  convert_element_type_285 = None
        view_269: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg244_1, _shape_param_272);  arg244_1 = _shape_param_272 = None
        mul_258: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_269, 0.5)
        mul_259: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_286, mul_258);  mul_258 = None
        convert_element_type_289: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_269, torch.float32)
        pow_21: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_289, 3.0)
        mul_260: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_81: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_269, mul_260);  view_269 = mul_260 = None
        mul_261: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_10: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_261);  mul_261 = None
        add_82: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_262: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_286, add_82);  convert_element_type_286 = add_82 = None
        convert_element_type_290: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_262, torch.bfloat16);  mul_262 = None
        mul_263: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_55: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_263);  mul_263 = None
        mul_264: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_259, sub_55);  mul_259 = sub_55 = None
        mul_265: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_264, 0.7978845608028654);  mul_264 = None
        convert_element_type_291: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_265, torch.bfloat16)
        mul_266: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_265, 0.044715);  mul_265 = None
        pow_22: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_289, 2.0);  convert_element_type_289 = None
        mul_267: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_268: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_266, mul_267);  mul_266 = mul_267 = None
        convert_element_type_292: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_268, torch.bfloat16);  mul_268 = None
        add_83: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_291, convert_element_type_292);  convert_element_type_291 = convert_element_type_292 = None
        mul_269: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_290, 0.5);  convert_element_type_290 = None
        add_84: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_83, mul_269);  add_83 = mul_269 = None
        view_270: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_84, _shape_param_273);  add_84 = _shape_param_273 = None
        mm_124: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_270, arg245_1);  arg245_1 = None
        permute_112: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_270, [1, 0])
        mm_125: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_112, arg246_1);  permute_112 = arg246_1 = None
        sum_127: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True, dtype = torch.float32);  view_270 = None
        view_271: "f32[8192]" = torch.ops.aten.view.default(sum_127, _shape_param_274);  sum_127 = _shape_param_274 = None
        convert_element_type_293: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_271, torch.bfloat16);  view_271 = None
        view_272: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_124, _shape_param_275);  mm_124 = _shape_param_275 = None
        convert_element_type_294: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_272, torch.float32);  view_272 = None
        convert_element_type_295: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_296: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_293, torch.float32);  convert_element_type_293 = None
        mul_270: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_294, arg247_1);  arg247_1 = None
        mul_271: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_270, 2048)
        sum_128: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
        mul_272: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_270, arg248_1);  mul_270 = None
        sum_129: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
        mul_273: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg248_1, sum_129);  sum_129 = None
        sub_56: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_271, sum_128);  mul_271 = sum_128 = None
        sub_57: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_56, mul_273);  sub_56 = mul_273 = None
        mul_274: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg249_1, sub_57);  arg249_1 = sub_57 = None
        mul_275: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_294, arg248_1);  arg248_1 = None
        sum_130: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
        sum_131: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_294, [0, 1]);  convert_element_type_294 = None
        add_85: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_80, mul_274);  add_80 = mul_274 = None
        convert_element_type_297: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_85, torch.bfloat16)
        view_273: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_297, _shape_param_276);  convert_element_type_297 = _shape_param_276 = None
        mm_126: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_273, arg250_1);  arg250_1 = None
        permute_113: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_273, [1, 0])
        mm_127: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_113, arg251_1);  permute_113 = arg251_1 = None
        sum_132: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True, dtype = torch.float32);  view_273 = None
        view_274: "f32[2048]" = torch.ops.aten.view.default(sum_132, _shape_param_277);  sum_132 = _shape_param_277 = None
        convert_element_type_298: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_274, torch.bfloat16);  view_274 = None
        view_275: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_126, _shape_param_278);  mm_126 = _shape_param_278 = None
        convert_element_type_299: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_300: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_298, torch.float32);  convert_element_type_298 = None
        view_276: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_275, _shape_param_279);  view_275 = _shape_param_279 = None
        permute_114: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_276, [0, 2, 1, 3]);  view_276 = None
        clone_41: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
        view_277: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_41, _shape_param_280);  clone_41 = _shape_param_280 = None
        bmm_40: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg252_1, view_277);  arg252_1 = None
        bmm_41: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_277, arg253_1);  view_277 = arg253_1 = None
        view_278: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_281);  bmm_40 = _shape_param_281 = None
        view_279: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_41, _shape_param_282);  bmm_41 = _shape_param_282 = None
        convert_element_type_301: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_279, torch.float32);  view_279 = None
        mul_276: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_301, arg254_1);  convert_element_type_301 = None
        sum_133: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
        neg_10: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg254_1);  arg254_1 = None
        fma_10: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_10, sum_133, mul_276);  neg_10 = sum_133 = mul_276 = None
        convert_element_type_302: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        slice_23: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg255_1, 2, 0, 128);  arg255_1 = None
        slice_24: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 128);  slice_23 = None
        where_13: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_24, convert_element_type_302, full_2);  slice_24 = convert_element_type_302 = None
        view_280: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_13, _shape_param_283);  where_13 = _shape_param_283 = None
        bmm_42: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg256_1, view_280);  arg256_1 = None
        bmm_43: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_280, arg257_1);  view_280 = arg257_1 = None
        view_281: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_284);  bmm_42 = _shape_param_284 = None
        view_282: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_43, _shape_param_285);  bmm_43 = _shape_param_285 = None
        convert_element_type_303: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_281, torch.float32);  view_281 = None
        permute_115: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_303, [0, 1, 3, 2]);  convert_element_type_303 = None
        convert_element_type_304: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_115, torch.bfloat16);  permute_115 = None
        convert_element_type_305: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_282, torch.bfloat16);  view_282 = None
        permute_116: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
        clone_42: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_116, memory_format = torch.contiguous_format);  permute_116 = None
        view_283: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_42, _shape_param_286);  clone_42 = _shape_param_286 = None
        permute_117: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_304, [0, 2, 1, 3]);  convert_element_type_304 = None
        view_284: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_117, _shape_param_287);  permute_117 = _shape_param_287 = None
        permute_118: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_305, [0, 2, 1, 3]);  convert_element_type_305 = None
        clone_43: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_118, memory_format = torch.contiguous_format);  permute_118 = None
        view_285: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_43, _shape_param_288);  clone_43 = _shape_param_288 = None
        view_286: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_283, _shape_param_289);  view_283 = _shape_param_289 = None
        permute_119: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_286, [1, 0])
        mm_128: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_119, arg258_1);  permute_119 = None
        mm_129: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_286, arg259_1);  view_286 = arg259_1 = None
        view_287: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_129, _shape_param_290);  mm_129 = _shape_param_290 = None
        convert_element_type_306: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_287, torch.float32);  view_287 = None
        convert_element_type_307: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        clone_44: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_284, memory_format = torch.contiguous_format);  view_284 = None
        view_288: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_44, _shape_param_291);  clone_44 = _shape_param_291 = None
        permute_120: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_288, [1, 0])
        mm_130: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_120, arg258_1);  permute_120 = None
        mm_131: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_288, arg260_1);  view_288 = arg260_1 = None
        view_289: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_131, _shape_param_292);  mm_131 = _shape_param_292 = None
        convert_element_type_308: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_289, torch.float32);  view_289 = None
        add_86: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_306, convert_element_type_308);  convert_element_type_306 = convert_element_type_308 = None
        convert_element_type_309: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        view_290: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_285, _shape_param_293);  view_285 = _shape_param_293 = None
        permute_121: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_290, [1, 0])
        mm_132: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_121, arg258_1);  permute_121 = arg258_1 = None
        mm_133: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_290, arg261_1);  view_290 = arg261_1 = None
        view_291: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_133, _shape_param_294);  mm_133 = _shape_param_294 = None
        convert_element_type_310: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_291, torch.float32);  view_291 = None
        add_87: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_86, convert_element_type_310);  add_86 = convert_element_type_310 = None
        convert_element_type_311: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None
        mul_277: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_87, arg262_1);  arg262_1 = None
        mul_278: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_277, 2048)
        sum_134: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True)
        mul_279: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_277, arg263_1);  mul_277 = None
        sum_135: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_279, [2], True);  mul_279 = None
        mul_280: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg263_1, sum_135);  sum_135 = None
        sub_58: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_278, sum_134);  mul_278 = sum_134 = None
        sub_59: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_58, mul_280);  sub_58 = mul_280 = None
        mul_281: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg264_1, sub_59);  arg264_1 = sub_59 = None
        mul_282: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_87, arg263_1);  arg263_1 = None
        sum_136: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_282, [0, 1]);  mul_282 = None
        sum_137: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_87, [0, 1]);  add_87 = None
        add_88: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_85, mul_281);  add_85 = mul_281 = None
        convert_element_type_312: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16)
        view_292: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_312, _shape_param_295);  convert_element_type_312 = _shape_param_295 = None
        mm_134: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_292, arg265_1);  arg265_1 = None
        permute_122: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_292, [1, 0])
        mm_135: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_122, arg266_1);  permute_122 = arg266_1 = None
        sum_138: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_292, [0], True, dtype = torch.float32);  view_292 = None
        view_293: "f32[2048]" = torch.ops.aten.view.default(sum_138, _shape_param_296);  sum_138 = _shape_param_296 = None
        convert_element_type_313: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_293, torch.bfloat16);  view_293 = None
        view_294: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_134, _shape_param_297);  mm_134 = _shape_param_297 = None
        convert_element_type_314: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_294, torch.float32);  view_294 = None
        convert_element_type_315: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_316: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_313, torch.float32);  convert_element_type_313 = None
        view_295: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg267_1, _shape_param_298);  arg267_1 = _shape_param_298 = None
        mul_283: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_295, 0.5)
        mul_284: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_314, mul_283);  mul_283 = None
        convert_element_type_317: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_295, torch.float32)
        pow_23: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_317, 3.0)
        mul_285: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_89: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_295, mul_285);  view_295 = mul_285 = None
        mul_286: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_11: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_286);  mul_286 = None
        add_90: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_287: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_314, add_90);  convert_element_type_314 = add_90 = None
        convert_element_type_318: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_287, torch.bfloat16);  mul_287 = None
        mul_288: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_60: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_288);  mul_288 = None
        mul_289: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_284, sub_60);  mul_284 = sub_60 = None
        mul_290: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_289, 0.7978845608028654);  mul_289 = None
        convert_element_type_319: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_290, torch.bfloat16)
        mul_291: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_290, 0.044715);  mul_290 = None
        pow_24: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_317, 2.0);  convert_element_type_317 = None
        mul_292: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_293: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
        convert_element_type_320: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_293, torch.bfloat16);  mul_293 = None
        add_91: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_319, convert_element_type_320);  convert_element_type_319 = convert_element_type_320 = None
        mul_294: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_318, 0.5);  convert_element_type_318 = None
        add_92: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_91, mul_294);  add_91 = mul_294 = None
        view_296: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_92, _shape_param_299);  add_92 = _shape_param_299 = None
        mm_136: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_296, arg268_1);  arg268_1 = None
        permute_123: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_296, [1, 0])
        mm_137: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_123, arg269_1);  permute_123 = arg269_1 = None
        sum_139: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True, dtype = torch.float32);  view_296 = None
        view_297: "f32[8192]" = torch.ops.aten.view.default(sum_139, _shape_param_300);  sum_139 = _shape_param_300 = None
        convert_element_type_321: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_297, torch.bfloat16);  view_297 = None
        view_298: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_136, _shape_param_301);  mm_136 = _shape_param_301 = None
        convert_element_type_322: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_298, torch.float32);  view_298 = None
        convert_element_type_323: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_324: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_321, torch.float32);  convert_element_type_321 = None
        mul_295: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_322, arg270_1);  arg270_1 = None
        mul_296: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_295, 2048)
        sum_140: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_295, [2], True)
        mul_297: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_295, arg271_1);  mul_295 = None
        sum_141: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True);  mul_297 = None
        mul_298: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg271_1, sum_141);  sum_141 = None
        sub_61: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_296, sum_140);  mul_296 = sum_140 = None
        sub_62: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_61, mul_298);  sub_61 = mul_298 = None
        mul_299: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg272_1, sub_62);  arg272_1 = sub_62 = None
        mul_300: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_322, arg271_1);  arg271_1 = None
        sum_142: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1]);  mul_300 = None
        sum_143: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_322, [0, 1]);  convert_element_type_322 = None
        add_93: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_88, mul_299);  add_88 = mul_299 = None
        convert_element_type_325: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_93, torch.bfloat16)
        view_299: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_325, _shape_param_302);  convert_element_type_325 = _shape_param_302 = None
        mm_138: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_299, arg273_1);  arg273_1 = None
        permute_124: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_299, [1, 0])
        mm_139: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_124, arg274_1);  permute_124 = arg274_1 = None
        sum_144: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True, dtype = torch.float32);  view_299 = None
        view_300: "f32[2048]" = torch.ops.aten.view.default(sum_144, _shape_param_303);  sum_144 = _shape_param_303 = None
        convert_element_type_326: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_300, torch.bfloat16);  view_300 = None
        view_301: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_138, _shape_param_304);  mm_138 = _shape_param_304 = None
        convert_element_type_327: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_328: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_326, torch.float32);  convert_element_type_326 = None
        view_302: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_301, _shape_param_305);  view_301 = _shape_param_305 = None
        permute_125: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
        clone_45: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_303: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_45, _shape_param_306);  clone_45 = _shape_param_306 = None
        bmm_44: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg275_1, view_303);  arg275_1 = None
        bmm_45: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_303, arg276_1);  view_303 = arg276_1 = None
        view_304: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_307);  bmm_44 = _shape_param_307 = None
        view_305: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_45, _shape_param_308);  bmm_45 = _shape_param_308 = None
        convert_element_type_329: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None
        mul_301: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_329, arg277_1);  convert_element_type_329 = None
        sum_145: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [-1], True)
        neg_11: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg277_1);  arg277_1 = None
        fma_11: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_11, sum_145, mul_301);  neg_11 = sum_145 = mul_301 = None
        convert_element_type_330: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        slice_25: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg278_1, 2, 0, 128);  arg278_1 = None
        slice_26: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_25, 3, 0, 128);  slice_25 = None
        where_14: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_26, convert_element_type_330, full_2);  slice_26 = convert_element_type_330 = None
        view_306: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_14, _shape_param_309);  where_14 = _shape_param_309 = None
        bmm_46: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg279_1, view_306);  arg279_1 = None
        bmm_47: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_306, arg280_1);  view_306 = arg280_1 = None
        view_307: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_310);  bmm_46 = _shape_param_310 = None
        view_308: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_47, _shape_param_311);  bmm_47 = _shape_param_311 = None
        convert_element_type_331: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_307, torch.float32);  view_307 = None
        permute_126: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_331, [0, 1, 3, 2]);  convert_element_type_331 = None
        convert_element_type_332: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_126, torch.bfloat16);  permute_126 = None
        convert_element_type_333: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_308, torch.bfloat16);  view_308 = None
        permute_127: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
        clone_46: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_309: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_46, _shape_param_312);  clone_46 = _shape_param_312 = None
        permute_128: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_332, [0, 2, 1, 3]);  convert_element_type_332 = None
        view_310: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_128, _shape_param_313);  permute_128 = _shape_param_313 = None
        permute_129: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_333, [0, 2, 1, 3]);  convert_element_type_333 = None
        clone_47: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_311: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_47, _shape_param_314);  clone_47 = _shape_param_314 = None
        view_312: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_309, _shape_param_315);  view_309 = _shape_param_315 = None
        permute_130: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_312, [1, 0])
        mm_140: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_130, arg281_1);  permute_130 = None
        mm_141: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_312, arg282_1);  view_312 = arg282_1 = None
        view_313: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_141, _shape_param_316);  mm_141 = _shape_param_316 = None
        convert_element_type_334: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_313, torch.float32);  view_313 = None
        convert_element_type_335: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        clone_48: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_310, memory_format = torch.contiguous_format);  view_310 = None
        view_314: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_48, _shape_param_317);  clone_48 = _shape_param_317 = None
        permute_131: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_314, [1, 0])
        mm_142: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_131, arg281_1);  permute_131 = None
        mm_143: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_314, arg283_1);  view_314 = arg283_1 = None
        view_315: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_143, _shape_param_318);  mm_143 = _shape_param_318 = None
        convert_element_type_336: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_315, torch.float32);  view_315 = None
        add_94: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_334, convert_element_type_336);  convert_element_type_334 = convert_element_type_336 = None
        convert_element_type_337: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        view_316: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_311, _shape_param_319);  view_311 = _shape_param_319 = None
        permute_132: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_316, [1, 0])
        mm_144: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_132, arg281_1);  permute_132 = arg281_1 = None
        mm_145: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_316, arg284_1);  view_316 = arg284_1 = None
        view_317: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_145, _shape_param_320);  mm_145 = _shape_param_320 = None
        convert_element_type_338: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_317, torch.float32);  view_317 = None
        add_95: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_94, convert_element_type_338);  add_94 = convert_element_type_338 = None
        convert_element_type_339: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        mul_302: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_95, arg285_1);  arg285_1 = None
        mul_303: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_302, 2048)
        sum_146: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True)
        mul_304: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_302, arg286_1);  mul_302 = None
        sum_147: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True);  mul_304 = None
        mul_305: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg286_1, sum_147);  sum_147 = None
        sub_63: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_303, sum_146);  mul_303 = sum_146 = None
        sub_64: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_63, mul_305);  sub_63 = mul_305 = None
        mul_306: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg287_1, sub_64);  arg287_1 = sub_64 = None
        mul_307: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_95, arg286_1);  arg286_1 = None
        sum_148: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_307, [0, 1]);  mul_307 = None
        sum_149: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_95, [0, 1]);  add_95 = None
        add_96: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_93, mul_306);  add_93 = mul_306 = None
        convert_element_type_340: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16)
        view_318: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_340, _shape_param_321);  convert_element_type_340 = _shape_param_321 = None
        mm_146: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_318, arg288_1);  arg288_1 = None
        permute_133: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_318, [1, 0])
        mm_147: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_133, arg289_1);  permute_133 = arg289_1 = None
        sum_150: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True, dtype = torch.float32);  view_318 = None
        view_319: "f32[2048]" = torch.ops.aten.view.default(sum_150, _shape_param_322);  sum_150 = _shape_param_322 = None
        convert_element_type_341: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_319, torch.bfloat16);  view_319 = None
        view_320: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_146, _shape_param_323);  mm_146 = _shape_param_323 = None
        convert_element_type_342: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_320, torch.float32);  view_320 = None
        convert_element_type_343: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_344: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_341, torch.float32);  convert_element_type_341 = None
        view_321: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg290_1, _shape_param_324);  arg290_1 = _shape_param_324 = None
        mul_308: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_321, 0.5)
        mul_309: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_342, mul_308);  mul_308 = None
        convert_element_type_345: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_321, torch.float32)
        pow_25: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_345, 3.0)
        mul_310: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_97: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_321, mul_310);  view_321 = mul_310 = None
        mul_311: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_12: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_311);  mul_311 = None
        add_98: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_312: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_342, add_98);  convert_element_type_342 = add_98 = None
        convert_element_type_346: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None
        mul_313: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_65: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_313);  mul_313 = None
        mul_314: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_309, sub_65);  mul_309 = sub_65 = None
        mul_315: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_314, 0.7978845608028654);  mul_314 = None
        convert_element_type_347: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_315, torch.bfloat16)
        mul_316: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_315, 0.044715);  mul_315 = None
        pow_26: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_345, 2.0);  convert_element_type_345 = None
        mul_317: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
        mul_318: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
        convert_element_type_348: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_318, torch.bfloat16);  mul_318 = None
        add_99: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_347, convert_element_type_348);  convert_element_type_347 = convert_element_type_348 = None
        mul_319: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_346, 0.5);  convert_element_type_346 = None
        add_100: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_99, mul_319);  add_99 = mul_319 = None
        view_322: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_100, _shape_param_325);  add_100 = _shape_param_325 = None
        mm_148: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_322, arg291_1);  arg291_1 = None
        permute_134: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_322, [1, 0])
        mm_149: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_134, arg292_1);  permute_134 = arg292_1 = None
        sum_151: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True, dtype = torch.float32);  view_322 = None
        view_323: "f32[8192]" = torch.ops.aten.view.default(sum_151, _shape_param_326);  sum_151 = _shape_param_326 = None
        convert_element_type_349: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_323, torch.bfloat16);  view_323 = None
        view_324: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_148, _shape_param_327);  mm_148 = _shape_param_327 = None
        convert_element_type_350: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_324, torch.float32);  view_324 = None
        convert_element_type_351: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_352: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_349, torch.float32);  convert_element_type_349 = None
        mul_320: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_350, arg293_1);  arg293_1 = None
        mul_321: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_320, 2048)
        sum_152: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True)
        mul_322: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_320, arg294_1);  mul_320 = None
        sum_153: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [2], True);  mul_322 = None
        mul_323: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg294_1, sum_153);  sum_153 = None
        sub_66: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_321, sum_152);  mul_321 = sum_152 = None
        sub_67: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_66, mul_323);  sub_66 = mul_323 = None
        mul_324: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg295_1, sub_67);  arg295_1 = sub_67 = None
        mul_325: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_350, arg294_1);  arg294_1 = None
        sum_154: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_325, [0, 1]);  mul_325 = None
        sum_155: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_350, [0, 1]);  convert_element_type_350 = None
        add_101: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_96, mul_324);  add_96 = mul_324 = None
        convert_element_type_353: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_101, torch.bfloat16)
        view_325: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_353, _shape_param_328);  convert_element_type_353 = _shape_param_328 = None
        mm_150: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_325, arg296_1);  arg296_1 = None
        permute_135: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_325, [1, 0])
        mm_151: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_135, arg297_1);  permute_135 = arg297_1 = None
        sum_156: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True, dtype = torch.float32);  view_325 = None
        view_326: "f32[2048]" = torch.ops.aten.view.default(sum_156, _shape_param_329);  sum_156 = _shape_param_329 = None
        convert_element_type_354: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_326, torch.bfloat16);  view_326 = None
        view_327: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_150, _shape_param_330);  mm_150 = _shape_param_330 = None
        convert_element_type_355: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        convert_element_type_356: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_354, torch.float32);  convert_element_type_354 = None
        view_328: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_327, _shape_param_331);  view_327 = _shape_param_331 = None
        permute_136: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_328, [0, 2, 1, 3]);  view_328 = None
        clone_49: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
        view_329: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_49, _shape_param_332);  clone_49 = _shape_param_332 = None
        bmm_48: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg298_1, view_329);  arg298_1 = None
        bmm_49: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_329, arg299_1);  view_329 = arg299_1 = None
        view_330: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_48, _shape_param_333);  bmm_48 = _shape_param_333 = None
        view_331: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_49, _shape_param_334);  bmm_49 = _shape_param_334 = None
        convert_element_type_357: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_331, torch.float32);  view_331 = None
        mul_326: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_357, arg300_1);  convert_element_type_357 = None
        sum_157: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [-1], True)
        neg_12: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg300_1);  arg300_1 = None
        fma_12: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_12, sum_157, mul_326);  neg_12 = sum_157 = mul_326 = None
        convert_element_type_358: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None
        slice_27: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg301_1, 2, 0, 128);  arg301_1 = None
        slice_28: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_27, 3, 0, 128);  slice_27 = None
        where_15: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_28, convert_element_type_358, full_2);  slice_28 = convert_element_type_358 = None
        view_332: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_15, _shape_param_335);  where_15 = _shape_param_335 = None
        bmm_50: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg302_1, view_332);  arg302_1 = None
        bmm_51: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_332, arg303_1);  view_332 = arg303_1 = None
        view_333: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_50, _shape_param_336);  bmm_50 = _shape_param_336 = None
        view_334: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_51, _shape_param_337);  bmm_51 = _shape_param_337 = None
        convert_element_type_359: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_333, torch.float32);  view_333 = None
        permute_137: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_359, [0, 1, 3, 2]);  convert_element_type_359 = None
        convert_element_type_360: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_137, torch.bfloat16);  permute_137 = None
        convert_element_type_361: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_334, torch.bfloat16);  view_334 = None
        permute_138: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        clone_50: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_138, memory_format = torch.contiguous_format);  permute_138 = None
        view_335: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_50, _shape_param_338);  clone_50 = _shape_param_338 = None
        permute_139: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_360, [0, 2, 1, 3]);  convert_element_type_360 = None
        view_336: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_139, _shape_param_339);  permute_139 = _shape_param_339 = None
        permute_140: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_361, [0, 2, 1, 3]);  convert_element_type_361 = None
        clone_51: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_140, memory_format = torch.contiguous_format);  permute_140 = None
        view_337: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_51, _shape_param_340);  clone_51 = _shape_param_340 = None
        view_338: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_335, _shape_param_341);  view_335 = _shape_param_341 = None
        permute_141: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_338, [1, 0])
        mm_152: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_141, arg304_1);  permute_141 = None
        mm_153: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_338, arg305_1);  view_338 = arg305_1 = None
        view_339: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_153, _shape_param_342);  mm_153 = _shape_param_342 = None
        convert_element_type_362: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        convert_element_type_363: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        clone_52: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_336, memory_format = torch.contiguous_format);  view_336 = None
        view_340: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_52, _shape_param_343);  clone_52 = _shape_param_343 = None
        permute_142: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_340, [1, 0])
        mm_154: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_142, arg304_1);  permute_142 = None
        mm_155: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_340, arg306_1);  view_340 = arg306_1 = None
        view_341: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_155, _shape_param_344);  mm_155 = _shape_param_344 = None
        convert_element_type_364: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_341, torch.float32);  view_341 = None
        add_102: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_362, convert_element_type_364);  convert_element_type_362 = convert_element_type_364 = None
        convert_element_type_365: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        view_342: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_337, _shape_param_345);  view_337 = _shape_param_345 = None
        permute_143: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_342, [1, 0])
        mm_156: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_143, arg304_1);  permute_143 = arg304_1 = None
        mm_157: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_342, arg307_1);  view_342 = arg307_1 = None
        view_343: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_157, _shape_param_346);  mm_157 = _shape_param_346 = None
        convert_element_type_366: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_343, torch.float32);  view_343 = None
        add_103: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_102, convert_element_type_366);  add_102 = convert_element_type_366 = None
        convert_element_type_367: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        mul_327: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_103, arg308_1);  arg308_1 = None
        mul_328: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_327, 2048)
        sum_158: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True)
        mul_329: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_327, arg309_1);  mul_327 = None
        sum_159: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True);  mul_329 = None
        mul_330: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg309_1, sum_159);  sum_159 = None
        sub_68: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_328, sum_158);  mul_328 = sum_158 = None
        sub_69: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_68, mul_330);  sub_68 = mul_330 = None
        mul_331: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg310_1, sub_69);  arg310_1 = sub_69 = None
        mul_332: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_103, arg309_1);  arg309_1 = None
        sum_160: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1]);  mul_332 = None
        sum_161: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_103, [0, 1]);  add_103 = None
        add_104: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_101, mul_331);  add_101 = mul_331 = None
        convert_element_type_368: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16)
        view_344: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_368, _shape_param_347);  convert_element_type_368 = _shape_param_347 = None
        mm_158: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_344, arg311_1);  arg311_1 = None
        permute_144: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_344, [1, 0])
        mm_159: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_144, arg312_1);  permute_144 = arg312_1 = None
        sum_162: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True, dtype = torch.float32);  view_344 = None
        view_345: "f32[2048]" = torch.ops.aten.view.default(sum_162, _shape_param_348);  sum_162 = _shape_param_348 = None
        convert_element_type_369: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_345, torch.bfloat16);  view_345 = None
        view_346: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_158, _shape_param_349);  mm_158 = _shape_param_349 = None
        convert_element_type_370: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_346, torch.float32);  view_346 = None
        convert_element_type_371: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_372: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_369, torch.float32);  convert_element_type_369 = None
        view_347: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg313_1, _shape_param_350);  arg313_1 = _shape_param_350 = None
        mul_333: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_347, 0.5)
        mul_334: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_370, mul_333);  mul_333 = None
        convert_element_type_373: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_347, torch.float32)
        pow_27: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_373, 3.0)
        mul_335: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_105: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_347, mul_335);  view_347 = mul_335 = None
        mul_336: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_105, 0.7978845608028654);  add_105 = None
        tanh_13: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_336);  mul_336 = None
        add_106: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_337: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_370, add_106);  convert_element_type_370 = add_106 = None
        convert_element_type_374: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_337, torch.bfloat16);  mul_337 = None
        mul_338: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_70: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_338);  mul_338 = None
        mul_339: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_334, sub_70);  mul_334 = sub_70 = None
        mul_340: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_339, 0.7978845608028654);  mul_339 = None
        convert_element_type_375: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_340, torch.bfloat16)
        mul_341: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_340, 0.044715);  mul_340 = None
        pow_28: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_373, 2.0);  convert_element_type_373 = None
        mul_342: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_28, 3.0);  pow_28 = None
        mul_343: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_341, mul_342);  mul_341 = mul_342 = None
        convert_element_type_376: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_343, torch.bfloat16);  mul_343 = None
        add_107: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_375, convert_element_type_376);  convert_element_type_375 = convert_element_type_376 = None
        mul_344: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_374, 0.5);  convert_element_type_374 = None
        add_108: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_107, mul_344);  add_107 = mul_344 = None
        view_348: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_108, _shape_param_351);  add_108 = _shape_param_351 = None
        mm_160: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_348, arg314_1);  arg314_1 = None
        permute_145: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_348, [1, 0])
        mm_161: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_145, arg315_1);  permute_145 = arg315_1 = None
        sum_163: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_348, [0], True, dtype = torch.float32);  view_348 = None
        view_349: "f32[8192]" = torch.ops.aten.view.default(sum_163, _shape_param_352);  sum_163 = _shape_param_352 = None
        convert_element_type_377: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_349, torch.bfloat16);  view_349 = None
        view_350: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_160, _shape_param_353);  mm_160 = _shape_param_353 = None
        convert_element_type_378: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_350, torch.float32);  view_350 = None
        convert_element_type_379: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None
        convert_element_type_380: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_377, torch.float32);  convert_element_type_377 = None
        mul_345: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_378, arg316_1);  arg316_1 = None
        mul_346: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_345, 2048)
        sum_164: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [2], True)
        mul_347: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_345, arg317_1);  mul_345 = None
        sum_165: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True);  mul_347 = None
        mul_348: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg317_1, sum_165);  sum_165 = None
        sub_71: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_346, sum_164);  mul_346 = sum_164 = None
        sub_72: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_71, mul_348);  sub_71 = mul_348 = None
        mul_349: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg318_1, sub_72);  arg318_1 = sub_72 = None
        mul_350: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_378, arg317_1);  arg317_1 = None
        sum_166: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1]);  mul_350 = None
        sum_167: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_378, [0, 1]);  convert_element_type_378 = None
        add_109: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_104, mul_349);  add_104 = mul_349 = None
        convert_element_type_381: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16)
        view_351: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_381, _shape_param_354);  convert_element_type_381 = _shape_param_354 = None
        mm_162: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_351, arg319_1);  arg319_1 = None
        permute_146: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_351, [1, 0])
        mm_163: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_146, arg320_1);  permute_146 = arg320_1 = None
        sum_168: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_351, [0], True, dtype = torch.float32);  view_351 = None
        view_352: "f32[2048]" = torch.ops.aten.view.default(sum_168, _shape_param_355);  sum_168 = _shape_param_355 = None
        convert_element_type_382: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_352, torch.bfloat16);  view_352 = None
        view_353: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_162, _shape_param_356);  mm_162 = _shape_param_356 = None
        convert_element_type_383: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        convert_element_type_384: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_382, torch.float32);  convert_element_type_382 = None
        view_354: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_353, _shape_param_357);  view_353 = _shape_param_357 = None
        permute_147: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_354, [0, 2, 1, 3]);  view_354 = None
        clone_53: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_355: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_53, _shape_param_358);  clone_53 = _shape_param_358 = None
        bmm_52: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg321_1, view_355);  arg321_1 = None
        bmm_53: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_355, arg322_1);  view_355 = arg322_1 = None
        view_356: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_52, _shape_param_359);  bmm_52 = _shape_param_359 = None
        view_357: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_53, _shape_param_360);  bmm_53 = _shape_param_360 = None
        convert_element_type_385: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_357, torch.float32);  view_357 = None
        mul_351: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_385, arg323_1);  convert_element_type_385 = None
        sum_169: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_351, [-1], True)
        neg_13: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg323_1);  arg323_1 = None
        fma_13: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_13, sum_169, mul_351);  neg_13 = sum_169 = mul_351 = None
        convert_element_type_386: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None
        slice_29: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg324_1, 2, 0, 128);  arg324_1 = None
        slice_30: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_29, 3, 0, 128);  slice_29 = None
        where_16: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_30, convert_element_type_386, full_2);  slice_30 = convert_element_type_386 = None
        view_358: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_16, _shape_param_361);  where_16 = _shape_param_361 = None
        bmm_54: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg325_1, view_358);  arg325_1 = None
        bmm_55: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_358, arg326_1);  view_358 = arg326_1 = None
        view_359: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_54, _shape_param_362);  bmm_54 = _shape_param_362 = None
        view_360: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_55, _shape_param_363);  bmm_55 = _shape_param_363 = None
        convert_element_type_387: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_359, torch.float32);  view_359 = None
        permute_148: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_387, [0, 1, 3, 2]);  convert_element_type_387 = None
        convert_element_type_388: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_148, torch.bfloat16);  permute_148 = None
        convert_element_type_389: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_360, torch.bfloat16);  view_360 = None
        permute_149: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
        clone_54: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        view_361: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_54, _shape_param_364);  clone_54 = _shape_param_364 = None
        permute_150: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_388, [0, 2, 1, 3]);  convert_element_type_388 = None
        view_362: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_150, _shape_param_365);  permute_150 = _shape_param_365 = None
        permute_151: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_389, [0, 2, 1, 3]);  convert_element_type_389 = None
        clone_55: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None
        view_363: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_55, _shape_param_366);  clone_55 = _shape_param_366 = None
        view_364: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_361, _shape_param_367);  view_361 = _shape_param_367 = None
        permute_152: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_164: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_152, arg327_1);  permute_152 = None
        mm_165: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_364, arg328_1);  view_364 = arg328_1 = None
        view_365: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_165, _shape_param_368);  mm_165 = _shape_param_368 = None
        convert_element_type_390: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_365, torch.float32);  view_365 = None
        convert_element_type_391: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        clone_56: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_362, memory_format = torch.contiguous_format);  view_362 = None
        view_366: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_56, _shape_param_369);  clone_56 = _shape_param_369 = None
        permute_153: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_366, [1, 0])
        mm_166: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_153, arg327_1);  permute_153 = None
        mm_167: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_366, arg329_1);  view_366 = arg329_1 = None
        view_367: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_167, _shape_param_370);  mm_167 = _shape_param_370 = None
        convert_element_type_392: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_367, torch.float32);  view_367 = None
        add_110: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_390, convert_element_type_392);  convert_element_type_390 = convert_element_type_392 = None
        convert_element_type_393: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        view_368: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_363, _shape_param_371);  view_363 = _shape_param_371 = None
        permute_154: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_368, [1, 0])
        mm_168: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_154, arg327_1);  permute_154 = arg327_1 = None
        mm_169: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_368, arg330_1);  view_368 = arg330_1 = None
        view_369: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_169, _shape_param_372);  mm_169 = _shape_param_372 = None
        convert_element_type_394: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_369, torch.float32);  view_369 = None
        add_111: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_110, convert_element_type_394);  add_110 = convert_element_type_394 = None
        convert_element_type_395: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        mul_352: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_111, arg331_1);  arg331_1 = None
        mul_353: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_352, 2048)
        sum_170: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_352, [2], True)
        mul_354: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_352, arg332_1);  mul_352 = None
        sum_171: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True);  mul_354 = None
        mul_355: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg332_1, sum_171);  sum_171 = None
        sub_73: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_353, sum_170);  mul_353 = sum_170 = None
        sub_74: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_73, mul_355);  sub_73 = mul_355 = None
        mul_356: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg333_1, sub_74);  arg333_1 = sub_74 = None
        mul_357: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_111, arg332_1);  arg332_1 = None
        sum_172: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 1]);  mul_357 = None
        sum_173: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_111, [0, 1]);  add_111 = None
        add_112: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_109, mul_356);  add_109 = mul_356 = None
        convert_element_type_396: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16)
        view_370: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_396, _shape_param_373);  convert_element_type_396 = _shape_param_373 = None
        mm_170: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_370, arg334_1);  arg334_1 = None
        permute_155: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_370, [1, 0])
        mm_171: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_155, arg335_1);  permute_155 = arg335_1 = None
        sum_174: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_370, [0], True, dtype = torch.float32);  view_370 = None
        view_371: "f32[2048]" = torch.ops.aten.view.default(sum_174, _shape_param_374);  sum_174 = _shape_param_374 = None
        convert_element_type_397: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_371, torch.bfloat16);  view_371 = None
        view_372: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_170, _shape_param_375);  mm_170 = _shape_param_375 = None
        convert_element_type_398: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_372, torch.float32);  view_372 = None
        convert_element_type_399: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        convert_element_type_400: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_397, torch.float32);  convert_element_type_397 = None
        view_373: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg336_1, _shape_param_376);  arg336_1 = _shape_param_376 = None
        mul_358: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_373, 0.5)
        mul_359: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_398, mul_358);  mul_358 = None
        convert_element_type_401: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_373, torch.float32)
        pow_29: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_401, 3.0)
        mul_360: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_113: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_373, mul_360);  view_373 = mul_360 = None
        mul_361: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        tanh_14: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_361);  mul_361 = None
        add_114: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_362: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_398, add_114);  convert_element_type_398 = add_114 = None
        convert_element_type_402: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_362, torch.bfloat16);  mul_362 = None
        mul_363: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_75: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_363);  mul_363 = None
        mul_364: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_359, sub_75);  mul_359 = sub_75 = None
        mul_365: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_364, 0.7978845608028654);  mul_364 = None
        convert_element_type_403: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_365, torch.bfloat16)
        mul_366: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_365, 0.044715);  mul_365 = None
        pow_30: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_401, 2.0);  convert_element_type_401 = None
        mul_367: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
        mul_368: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
        convert_element_type_404: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_368, torch.bfloat16);  mul_368 = None
        add_115: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_403, convert_element_type_404);  convert_element_type_403 = convert_element_type_404 = None
        mul_369: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_402, 0.5);  convert_element_type_402 = None
        add_116: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_115, mul_369);  add_115 = mul_369 = None
        view_374: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_116, _shape_param_377);  add_116 = _shape_param_377 = None
        mm_172: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_374, arg337_1);  arg337_1 = None
        permute_156: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_374, [1, 0])
        mm_173: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_156, arg338_1);  permute_156 = arg338_1 = None
        sum_175: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_374, [0], True, dtype = torch.float32);  view_374 = None
        view_375: "f32[8192]" = torch.ops.aten.view.default(sum_175, _shape_param_378);  sum_175 = _shape_param_378 = None
        convert_element_type_405: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_375, torch.bfloat16);  view_375 = None
        view_376: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_172, _shape_param_379);  mm_172 = _shape_param_379 = None
        convert_element_type_406: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_376, torch.float32);  view_376 = None
        convert_element_type_407: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        convert_element_type_408: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_405, torch.float32);  convert_element_type_405 = None
        mul_370: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_406, arg339_1);  arg339_1 = None
        mul_371: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_370, 2048)
        sum_176: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True)
        mul_372: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_370, arg340_1);  mul_370 = None
        sum_177: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
        mul_373: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg340_1, sum_177);  sum_177 = None
        sub_76: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_371, sum_176);  mul_371 = sum_176 = None
        sub_77: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_76, mul_373);  sub_76 = mul_373 = None
        mul_374: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg341_1, sub_77);  arg341_1 = sub_77 = None
        mul_375: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_406, arg340_1);  arg340_1 = None
        sum_178: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
        sum_179: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_406, [0, 1]);  convert_element_type_406 = None
        add_117: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_112, mul_374);  add_112 = mul_374 = None
        convert_element_type_409: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_117, torch.bfloat16)
        view_377: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_409, _shape_param_380);  convert_element_type_409 = _shape_param_380 = None
        mm_174: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_377, arg342_1);  arg342_1 = None
        permute_157: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_377, [1, 0])
        mm_175: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_157, arg343_1);  permute_157 = arg343_1 = None
        sum_180: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True, dtype = torch.float32);  view_377 = None
        view_378: "f32[2048]" = torch.ops.aten.view.default(sum_180, _shape_param_381);  sum_180 = _shape_param_381 = None
        convert_element_type_410: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_378, torch.bfloat16);  view_378 = None
        view_379: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_174, _shape_param_382);  mm_174 = _shape_param_382 = None
        convert_element_type_411: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        convert_element_type_412: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_410, torch.float32);  convert_element_type_410 = None
        view_380: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_379, _shape_param_383);  view_379 = _shape_param_383 = None
        permute_158: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
        clone_57: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
        view_381: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_57, _shape_param_384);  clone_57 = _shape_param_384 = None
        bmm_56: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg344_1, view_381);  arg344_1 = None
        bmm_57: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_381, arg345_1);  view_381 = arg345_1 = None
        view_382: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_56, _shape_param_385);  bmm_56 = _shape_param_385 = None
        view_383: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_57, _shape_param_386);  bmm_57 = _shape_param_386 = None
        convert_element_type_413: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_383, torch.float32);  view_383 = None
        mul_376: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_413, arg346_1);  convert_element_type_413 = None
        sum_181: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [-1], True)
        neg_14: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg346_1);  arg346_1 = None
        fma_14: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_14, sum_181, mul_376);  neg_14 = sum_181 = mul_376 = None
        convert_element_type_414: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None
        slice_31: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg347_1, 2, 0, 128);  arg347_1 = None
        slice_32: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_31, 3, 0, 128);  slice_31 = None
        where_17: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_32, convert_element_type_414, full_2);  slice_32 = convert_element_type_414 = None
        view_384: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_17, _shape_param_387);  where_17 = _shape_param_387 = None
        bmm_58: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg348_1, view_384);  arg348_1 = None
        bmm_59: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_384, arg349_1);  view_384 = arg349_1 = None
        view_385: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_58, _shape_param_388);  bmm_58 = _shape_param_388 = None
        view_386: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_59, _shape_param_389);  bmm_59 = _shape_param_389 = None
        convert_element_type_415: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_385, torch.float32);  view_385 = None
        permute_159: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_415, [0, 1, 3, 2]);  convert_element_type_415 = None
        convert_element_type_416: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_159, torch.bfloat16);  permute_159 = None
        convert_element_type_417: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_386, torch.bfloat16);  view_386 = None
        permute_160: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
        clone_58: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
        view_387: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_58, _shape_param_390);  clone_58 = _shape_param_390 = None
        permute_161: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_416, [0, 2, 1, 3]);  convert_element_type_416 = None
        view_388: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_161, _shape_param_391);  permute_161 = _shape_param_391 = None
        permute_162: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_417, [0, 2, 1, 3]);  convert_element_type_417 = None
        clone_59: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
        view_389: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_59, _shape_param_392);  clone_59 = _shape_param_392 = None
        view_390: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_387, _shape_param_393);  view_387 = _shape_param_393 = None
        permute_163: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_390, [1, 0])
        mm_176: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_163, arg350_1);  permute_163 = None
        mm_177: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_390, arg351_1);  view_390 = arg351_1 = None
        view_391: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_177, _shape_param_394);  mm_177 = _shape_param_394 = None
        convert_element_type_418: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_391, torch.float32);  view_391 = None
        convert_element_type_419: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        clone_60: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_388, memory_format = torch.contiguous_format);  view_388 = None
        view_392: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_60, _shape_param_395);  clone_60 = _shape_param_395 = None
        permute_164: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_392, [1, 0])
        mm_178: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_164, arg350_1);  permute_164 = None
        mm_179: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_392, arg352_1);  view_392 = arg352_1 = None
        view_393: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_179, _shape_param_396);  mm_179 = _shape_param_396 = None
        convert_element_type_420: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        add_118: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_418, convert_element_type_420);  convert_element_type_418 = convert_element_type_420 = None
        convert_element_type_421: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        view_394: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_389, _shape_param_397);  view_389 = _shape_param_397 = None
        permute_165: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_394, [1, 0])
        mm_180: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_165, arg350_1);  permute_165 = arg350_1 = None
        mm_181: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_394, arg353_1);  view_394 = arg353_1 = None
        view_395: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_181, _shape_param_398);  mm_181 = _shape_param_398 = None
        convert_element_type_422: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_395, torch.float32);  view_395 = None
        add_119: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_118, convert_element_type_422);  add_118 = convert_element_type_422 = None
        convert_element_type_423: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None
        mul_377: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_119, arg354_1);  arg354_1 = None
        mul_378: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_377, 2048)
        sum_182: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True)
        mul_379: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_377, arg355_1);  mul_377 = None
        sum_183: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_379, [2], True);  mul_379 = None
        mul_380: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg355_1, sum_183);  sum_183 = None
        sub_78: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_378, sum_182);  mul_378 = sum_182 = None
        sub_79: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_78, mul_380);  sub_78 = mul_380 = None
        mul_381: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg356_1, sub_79);  arg356_1 = sub_79 = None
        mul_382: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_119, arg355_1);  arg355_1 = None
        sum_184: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 1]);  mul_382 = None
        sum_185: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_119, [0, 1]);  add_119 = None
        add_120: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_117, mul_381);  add_117 = mul_381 = None
        convert_element_type_424: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16)
        view_396: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_424, _shape_param_399);  convert_element_type_424 = _shape_param_399 = None
        mm_182: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_396, arg357_1);  arg357_1 = None
        permute_166: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_396, [1, 0])
        mm_183: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_166, arg358_1);  permute_166 = arg358_1 = None
        sum_186: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_396, [0], True, dtype = torch.float32);  view_396 = None
        view_397: "f32[2048]" = torch.ops.aten.view.default(sum_186, _shape_param_400);  sum_186 = _shape_param_400 = None
        convert_element_type_425: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_397, torch.bfloat16);  view_397 = None
        view_398: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_182, _shape_param_401);  mm_182 = _shape_param_401 = None
        convert_element_type_426: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        convert_element_type_427: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_428: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_425, torch.float32);  convert_element_type_425 = None
        view_399: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg359_1, _shape_param_402);  arg359_1 = _shape_param_402 = None
        mul_383: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_399, 0.5)
        mul_384: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_426, mul_383);  mul_383 = None
        convert_element_type_429: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_399, torch.float32)
        pow_31: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_429, 3.0)
        mul_385: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_31, 0.044715);  pow_31 = None
        add_121: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_399, mul_385);  view_399 = mul_385 = None
        mul_386: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_121, 0.7978845608028654);  add_121 = None
        tanh_15: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_386);  mul_386 = None
        add_122: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_387: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_426, add_122);  convert_element_type_426 = add_122 = None
        convert_element_type_430: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_387, torch.bfloat16);  mul_387 = None
        mul_388: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_80: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_388);  mul_388 = None
        mul_389: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_384, sub_80);  mul_384 = sub_80 = None
        mul_390: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_389, 0.7978845608028654);  mul_389 = None
        convert_element_type_431: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_390, torch.bfloat16)
        mul_391: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_390, 0.044715);  mul_390 = None
        pow_32: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_429, 2.0);  convert_element_type_429 = None
        mul_392: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
        mul_393: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
        convert_element_type_432: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_393, torch.bfloat16);  mul_393 = None
        add_123: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_431, convert_element_type_432);  convert_element_type_431 = convert_element_type_432 = None
        mul_394: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_430, 0.5);  convert_element_type_430 = None
        add_124: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_123, mul_394);  add_123 = mul_394 = None
        view_400: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_124, _shape_param_403);  add_124 = _shape_param_403 = None
        mm_184: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_400, arg360_1);  arg360_1 = None
        permute_167: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_400, [1, 0])
        mm_185: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_167, arg361_1);  permute_167 = arg361_1 = None
        sum_187: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_400, [0], True, dtype = torch.float32);  view_400 = None
        view_401: "f32[8192]" = torch.ops.aten.view.default(sum_187, _shape_param_404);  sum_187 = _shape_param_404 = None
        convert_element_type_433: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_401, torch.bfloat16);  view_401 = None
        view_402: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_184, _shape_param_405);  mm_184 = _shape_param_405 = None
        convert_element_type_434: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_402, torch.float32);  view_402 = None
        convert_element_type_435: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None
        convert_element_type_436: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_433, torch.float32);  convert_element_type_433 = None
        mul_395: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_434, arg362_1);  arg362_1 = None
        mul_396: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_395, 2048)
        sum_188: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
        mul_397: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_395, arg363_1);  mul_395 = None
        sum_189: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
        mul_398: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg363_1, sum_189);  sum_189 = None
        sub_81: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_396, sum_188);  mul_396 = sum_188 = None
        sub_82: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_81, mul_398);  sub_81 = mul_398 = None
        mul_399: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg364_1, sub_82);  arg364_1 = sub_82 = None
        mul_400: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_434, arg363_1);  arg363_1 = None
        sum_190: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
        sum_191: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_434, [0, 1]);  convert_element_type_434 = None
        add_125: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_120, mul_399);  add_120 = mul_399 = None
        convert_element_type_437: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_125, torch.bfloat16)
        view_403: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_437, _shape_param_406);  convert_element_type_437 = _shape_param_406 = None
        mm_186: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_403, arg365_1);  arg365_1 = None
        permute_168: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_403, [1, 0])
        mm_187: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_168, arg366_1);  permute_168 = arg366_1 = None
        sum_192: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True, dtype = torch.float32);  view_403 = None
        view_404: "f32[2048]" = torch.ops.aten.view.default(sum_192, _shape_param_407);  sum_192 = _shape_param_407 = None
        convert_element_type_438: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_404, torch.bfloat16);  view_404 = None
        view_405: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_186, _shape_param_408);  mm_186 = _shape_param_408 = None
        convert_element_type_439: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None
        convert_element_type_440: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_438, torch.float32);  convert_element_type_438 = None
        view_406: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_405, _shape_param_409);  view_405 = _shape_param_409 = None
        permute_169: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        clone_61: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_407: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_61, _shape_param_410);  clone_61 = _shape_param_410 = None
        bmm_60: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg367_1, view_407);  arg367_1 = None
        bmm_61: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_407, arg368_1);  view_407 = arg368_1 = None
        view_408: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_60, _shape_param_411);  bmm_60 = _shape_param_411 = None
        view_409: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_61, _shape_param_412);  bmm_61 = _shape_param_412 = None
        convert_element_type_441: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_409, torch.float32);  view_409 = None
        mul_401: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_441, arg369_1);  convert_element_type_441 = None
        sum_193: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_401, [-1], True)
        neg_15: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg369_1);  arg369_1 = None
        fma_15: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_15, sum_193, mul_401);  neg_15 = sum_193 = mul_401 = None
        convert_element_type_442: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None
        slice_33: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg370_1, 2, 0, 128);  arg370_1 = None
        slice_34: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_33, 3, 0, 128);  slice_33 = None
        where_18: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_34, convert_element_type_442, full_2);  slice_34 = convert_element_type_442 = None
        view_410: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_18, _shape_param_413);  where_18 = _shape_param_413 = None
        bmm_62: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg371_1, view_410);  arg371_1 = None
        bmm_63: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_410, arg372_1);  view_410 = arg372_1 = None
        view_411: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_62, _shape_param_414);  bmm_62 = _shape_param_414 = None
        view_412: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_63, _shape_param_415);  bmm_63 = _shape_param_415 = None
        convert_element_type_443: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_411, torch.float32);  view_411 = None
        permute_170: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_443, [0, 1, 3, 2]);  convert_element_type_443 = None
        convert_element_type_444: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_170, torch.bfloat16);  permute_170 = None
        convert_element_type_445: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_412, torch.bfloat16);  view_412 = None
        permute_171: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None
        clone_62: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
        view_413: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_62, _shape_param_416);  clone_62 = _shape_param_416 = None
        permute_172: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_444, [0, 2, 1, 3]);  convert_element_type_444 = None
        view_414: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_172, _shape_param_417);  permute_172 = _shape_param_417 = None
        permute_173: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_445, [0, 2, 1, 3]);  convert_element_type_445 = None
        clone_63: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
        view_415: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_63, _shape_param_418);  clone_63 = _shape_param_418 = None
        view_416: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_413, _shape_param_419);  view_413 = _shape_param_419 = None
        permute_174: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_416, [1, 0])
        mm_188: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_174, arg373_1);  permute_174 = None
        mm_189: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_416, arg374_1);  view_416 = arg374_1 = None
        view_417: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_189, _shape_param_420);  mm_189 = _shape_param_420 = None
        convert_element_type_446: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_417, torch.float32);  view_417 = None
        convert_element_type_447: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        clone_64: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_414, memory_format = torch.contiguous_format);  view_414 = None
        view_418: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_64, _shape_param_421);  clone_64 = _shape_param_421 = None
        permute_175: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_418, [1, 0])
        mm_190: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_175, arg373_1);  permute_175 = None
        mm_191: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_418, arg375_1);  view_418 = arg375_1 = None
        view_419: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_191, _shape_param_422);  mm_191 = _shape_param_422 = None
        convert_element_type_448: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        add_126: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_446, convert_element_type_448);  convert_element_type_446 = convert_element_type_448 = None
        convert_element_type_449: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        view_420: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_415, _shape_param_423);  view_415 = _shape_param_423 = None
        permute_176: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_420, [1, 0])
        mm_192: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_176, arg373_1);  permute_176 = arg373_1 = None
        mm_193: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_420, arg376_1);  view_420 = arg376_1 = None
        view_421: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_193, _shape_param_424);  mm_193 = _shape_param_424 = None
        convert_element_type_450: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_421, torch.float32);  view_421 = None
        add_127: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_126, convert_element_type_450);  add_126 = convert_element_type_450 = None
        convert_element_type_451: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        mul_402: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_127, arg377_1);  arg377_1 = None
        mul_403: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_402, 2048)
        sum_194: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
        mul_404: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_402, arg378_1);  mul_402 = None
        sum_195: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        mul_405: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg378_1, sum_195);  sum_195 = None
        sub_83: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_403, sum_194);  mul_403 = sum_194 = None
        sub_84: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_83, mul_405);  sub_83 = mul_405 = None
        mul_406: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg379_1, sub_84);  arg379_1 = sub_84 = None
        mul_407: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_127, arg378_1);  arg378_1 = None
        sum_196: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        sum_197: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_127, [0, 1]);  add_127 = None
        add_128: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_125, mul_406);  add_125 = mul_406 = None
        convert_element_type_452: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16)
        view_422: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_452, _shape_param_425);  convert_element_type_452 = _shape_param_425 = None
        mm_194: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_422, arg380_1);  arg380_1 = None
        permute_177: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_422, [1, 0])
        mm_195: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_177, arg381_1);  permute_177 = arg381_1 = None
        sum_198: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_422, [0], True, dtype = torch.float32);  view_422 = None
        view_423: "f32[2048]" = torch.ops.aten.view.default(sum_198, _shape_param_426);  sum_198 = _shape_param_426 = None
        convert_element_type_453: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_423, torch.bfloat16);  view_423 = None
        view_424: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_194, _shape_param_427);  mm_194 = _shape_param_427 = None
        convert_element_type_454: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_424, torch.float32);  view_424 = None
        convert_element_type_455: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        convert_element_type_456: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_453, torch.float32);  convert_element_type_453 = None
        view_425: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg382_1, _shape_param_428);  arg382_1 = _shape_param_428 = None
        mul_408: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_425, 0.5)
        mul_409: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_454, mul_408);  mul_408 = None
        convert_element_type_457: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_425, torch.float32)
        pow_33: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_457, 3.0)
        mul_410: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_129: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_425, mul_410);  view_425 = mul_410 = None
        mul_411: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_129, 0.7978845608028654);  add_129 = None
        tanh_16: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_411);  mul_411 = None
        add_130: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_412: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_454, add_130);  convert_element_type_454 = add_130 = None
        convert_element_type_458: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_412, torch.bfloat16);  mul_412 = None
        mul_413: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_85: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_413);  mul_413 = None
        mul_414: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_409, sub_85);  mul_409 = sub_85 = None
        mul_415: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_414, 0.7978845608028654);  mul_414 = None
        convert_element_type_459: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_415, torch.bfloat16)
        mul_416: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_415, 0.044715);  mul_415 = None
        pow_34: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_457, 2.0);  convert_element_type_457 = None
        mul_417: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
        mul_418: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_416, mul_417);  mul_416 = mul_417 = None
        convert_element_type_460: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_418, torch.bfloat16);  mul_418 = None
        add_131: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_459, convert_element_type_460);  convert_element_type_459 = convert_element_type_460 = None
        mul_419: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_458, 0.5);  convert_element_type_458 = None
        add_132: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_131, mul_419);  add_131 = mul_419 = None
        view_426: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_132, _shape_param_429);  add_132 = _shape_param_429 = None
        mm_196: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_426, arg383_1);  arg383_1 = None
        permute_178: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_426, [1, 0])
        mm_197: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_178, arg384_1);  permute_178 = arg384_1 = None
        sum_199: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_426, [0], True, dtype = torch.float32);  view_426 = None
        view_427: "f32[8192]" = torch.ops.aten.view.default(sum_199, _shape_param_430);  sum_199 = _shape_param_430 = None
        convert_element_type_461: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_427, torch.bfloat16);  view_427 = None
        view_428: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_196, _shape_param_431);  mm_196 = _shape_param_431 = None
        convert_element_type_462: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_428, torch.float32);  view_428 = None
        convert_element_type_463: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None
        convert_element_type_464: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_461, torch.float32);  convert_element_type_461 = None
        mul_420: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_462, arg385_1);  arg385_1 = None
        mul_421: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_420, 2048)
        sum_200: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_420, [2], True)
        mul_422: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_420, arg386_1);  mul_420 = None
        sum_201: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_422, [2], True);  mul_422 = None
        mul_423: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg386_1, sum_201);  sum_201 = None
        sub_86: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_421, sum_200);  mul_421 = sum_200 = None
        sub_87: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_86, mul_423);  sub_86 = mul_423 = None
        mul_424: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg387_1, sub_87);  arg387_1 = sub_87 = None
        mul_425: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_462, arg386_1);  arg386_1 = None
        sum_202: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_425, [0, 1]);  mul_425 = None
        sum_203: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_462, [0, 1]);  convert_element_type_462 = None
        add_133: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_128, mul_424);  add_128 = mul_424 = None
        convert_element_type_465: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16)
        view_429: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_465, _shape_param_432);  convert_element_type_465 = _shape_param_432 = None
        mm_198: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_429, arg388_1);  arg388_1 = None
        permute_179: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_429, [1, 0])
        mm_199: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_179, arg389_1);  permute_179 = arg389_1 = None
        sum_204: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_429, [0], True, dtype = torch.float32);  view_429 = None
        view_430: "f32[2048]" = torch.ops.aten.view.default(sum_204, _shape_param_433);  sum_204 = _shape_param_433 = None
        convert_element_type_466: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_430, torch.bfloat16);  view_430 = None
        view_431: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_198, _shape_param_434);  mm_198 = _shape_param_434 = None
        convert_element_type_467: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None
        convert_element_type_468: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_466, torch.float32);  convert_element_type_466 = None
        view_432: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_431, _shape_param_435);  view_431 = _shape_param_435 = None
        permute_180: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None
        clone_65: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
        view_433: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_65, _shape_param_436);  clone_65 = _shape_param_436 = None
        bmm_64: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg390_1, view_433);  arg390_1 = None
        bmm_65: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_433, arg391_1);  view_433 = arg391_1 = None
        view_434: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_64, _shape_param_437);  bmm_64 = _shape_param_437 = None
        view_435: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_65, _shape_param_438);  bmm_65 = _shape_param_438 = None
        convert_element_type_469: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_435, torch.float32);  view_435 = None
        mul_426: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_469, arg392_1);  convert_element_type_469 = None
        sum_205: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [-1], True)
        neg_16: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg392_1);  arg392_1 = None
        fma_16: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_16, sum_205, mul_426);  neg_16 = sum_205 = mul_426 = None
        convert_element_type_470: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None
        slice_35: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg393_1, 2, 0, 128);  arg393_1 = None
        slice_36: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 128);  slice_35 = None
        where_19: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_36, convert_element_type_470, full_2);  slice_36 = convert_element_type_470 = None
        view_436: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_19, _shape_param_439);  where_19 = _shape_param_439 = None
        bmm_66: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg394_1, view_436);  arg394_1 = None
        bmm_67: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_436, arg395_1);  view_436 = arg395_1 = None
        view_437: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_66, _shape_param_440);  bmm_66 = _shape_param_440 = None
        view_438: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_67, _shape_param_441);  bmm_67 = _shape_param_441 = None
        convert_element_type_471: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        permute_181: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_471, [0, 1, 3, 2]);  convert_element_type_471 = None
        convert_element_type_472: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_181, torch.bfloat16);  permute_181 = None
        convert_element_type_473: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_438, torch.bfloat16);  view_438 = None
        permute_182: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
        clone_66: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
        view_439: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_66, _shape_param_442);  clone_66 = _shape_param_442 = None
        permute_183: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_472, [0, 2, 1, 3]);  convert_element_type_472 = None
        view_440: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_183, _shape_param_443);  permute_183 = _shape_param_443 = None
        permute_184: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_473, [0, 2, 1, 3]);  convert_element_type_473 = None
        clone_67: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
        view_441: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_67, _shape_param_444);  clone_67 = _shape_param_444 = None
        view_442: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_439, _shape_param_445);  view_439 = _shape_param_445 = None
        permute_185: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_442, [1, 0])
        mm_200: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_185, arg396_1);  permute_185 = None
        mm_201: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_442, arg397_1);  view_442 = arg397_1 = None
        view_443: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_201, _shape_param_446);  mm_201 = _shape_param_446 = None
        convert_element_type_474: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_443, torch.float32);  view_443 = None
        convert_element_type_475: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        clone_68: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_440, memory_format = torch.contiguous_format);  view_440 = None
        view_444: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_68, _shape_param_447);  clone_68 = _shape_param_447 = None
        permute_186: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_444, [1, 0])
        mm_202: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_186, arg396_1);  permute_186 = None
        mm_203: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_444, arg398_1);  view_444 = arg398_1 = None
        view_445: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_203, _shape_param_448);  mm_203 = _shape_param_448 = None
        convert_element_type_476: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_445, torch.float32);  view_445 = None
        add_134: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_474, convert_element_type_476);  convert_element_type_474 = convert_element_type_476 = None
        convert_element_type_477: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_202, torch.float32);  mm_202 = None
        view_446: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_441, _shape_param_449);  view_441 = _shape_param_449 = None
        permute_187: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_446, [1, 0])
        mm_204: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_187, arg396_1);  permute_187 = arg396_1 = None
        mm_205: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_446, arg399_1);  view_446 = arg399_1 = None
        view_447: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_205, _shape_param_450);  mm_205 = _shape_param_450 = None
        convert_element_type_478: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_447, torch.float32);  view_447 = None
        add_135: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_134, convert_element_type_478);  add_134 = convert_element_type_478 = None
        convert_element_type_479: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_204, torch.float32);  mm_204 = None
        mul_427: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_135, arg400_1);  arg400_1 = None
        mul_428: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_427, 2048)
        sum_206: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True)
        mul_429: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_427, arg401_1);  mul_427 = None
        sum_207: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_429, [2], True);  mul_429 = None
        mul_430: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg401_1, sum_207);  sum_207 = None
        sub_88: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_428, sum_206);  mul_428 = sum_206 = None
        sub_89: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_88, mul_430);  sub_88 = mul_430 = None
        mul_431: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg402_1, sub_89);  arg402_1 = sub_89 = None
        mul_432: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_135, arg401_1);  arg401_1 = None
        sum_208: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1]);  mul_432 = None
        sum_209: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_135, [0, 1]);  add_135 = None
        add_136: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_133, mul_431);  add_133 = mul_431 = None
        convert_element_type_480: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16)
        view_448: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_480, _shape_param_451);  convert_element_type_480 = _shape_param_451 = None
        mm_206: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_448, arg403_1);  arg403_1 = None
        permute_188: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_448, [1, 0])
        mm_207: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_188, arg404_1);  permute_188 = arg404_1 = None
        sum_210: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True, dtype = torch.float32);  view_448 = None
        view_449: "f32[2048]" = torch.ops.aten.view.default(sum_210, _shape_param_452);  sum_210 = _shape_param_452 = None
        convert_element_type_481: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_449, torch.bfloat16);  view_449 = None
        view_450: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_206, _shape_param_453);  mm_206 = _shape_param_453 = None
        convert_element_type_482: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_450, torch.float32);  view_450 = None
        convert_element_type_483: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        convert_element_type_484: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_481, torch.float32);  convert_element_type_481 = None
        view_451: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg405_1, _shape_param_454);  arg405_1 = _shape_param_454 = None
        mul_433: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_451, 0.5)
        mul_434: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_482, mul_433);  mul_433 = None
        convert_element_type_485: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_451, torch.float32)
        pow_35: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_485, 3.0)
        mul_435: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_35, 0.044715);  pow_35 = None
        add_137: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_451, mul_435);  view_451 = mul_435 = None
        mul_436: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_137, 0.7978845608028654);  add_137 = None
        tanh_17: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_436);  mul_436 = None
        add_138: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_437: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_482, add_138);  convert_element_type_482 = add_138 = None
        convert_element_type_486: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_437, torch.bfloat16);  mul_437 = None
        mul_438: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_90: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_438);  mul_438 = None
        mul_439: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_434, sub_90);  mul_434 = sub_90 = None
        mul_440: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_439, 0.7978845608028654);  mul_439 = None
        convert_element_type_487: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_440, torch.bfloat16)
        mul_441: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_440, 0.044715);  mul_440 = None
        pow_36: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_485, 2.0);  convert_element_type_485 = None
        mul_442: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_443: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_441, mul_442);  mul_441 = mul_442 = None
        convert_element_type_488: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_443, torch.bfloat16);  mul_443 = None
        add_139: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_487, convert_element_type_488);  convert_element_type_487 = convert_element_type_488 = None
        mul_444: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_486, 0.5);  convert_element_type_486 = None
        add_140: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_139, mul_444);  add_139 = mul_444 = None
        view_452: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_140, _shape_param_455);  add_140 = _shape_param_455 = None
        mm_208: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_452, arg406_1);  arg406_1 = None
        permute_189: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_452, [1, 0])
        mm_209: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_189, arg407_1);  permute_189 = arg407_1 = None
        sum_211: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_452, [0], True, dtype = torch.float32);  view_452 = None
        view_453: "f32[8192]" = torch.ops.aten.view.default(sum_211, _shape_param_456);  sum_211 = _shape_param_456 = None
        convert_element_type_489: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_453, torch.bfloat16);  view_453 = None
        view_454: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_208, _shape_param_457);  mm_208 = _shape_param_457 = None
        convert_element_type_490: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_454, torch.float32);  view_454 = None
        convert_element_type_491: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_209, torch.float32);  mm_209 = None
        convert_element_type_492: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_489, torch.float32);  convert_element_type_489 = None
        mul_445: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_490, arg408_1);  arg408_1 = None
        mul_446: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_445, 2048)
        sum_212: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_445, [2], True)
        mul_447: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_445, arg409_1);  mul_445 = None
        sum_213: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True);  mul_447 = None
        mul_448: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg409_1, sum_213);  sum_213 = None
        sub_91: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_446, sum_212);  mul_446 = sum_212 = None
        sub_92: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_91, mul_448);  sub_91 = mul_448 = None
        mul_449: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg410_1, sub_92);  arg410_1 = sub_92 = None
        mul_450: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_490, arg409_1);  arg409_1 = None
        sum_214: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_450, [0, 1]);  mul_450 = None
        sum_215: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_490, [0, 1]);  convert_element_type_490 = None
        add_141: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_136, mul_449);  add_136 = mul_449 = None
        convert_element_type_493: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_141, torch.bfloat16)
        view_455: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_493, _shape_param_458);  convert_element_type_493 = _shape_param_458 = None
        mm_210: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_455, arg411_1);  arg411_1 = None
        permute_190: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_455, [1, 0])
        mm_211: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_190, arg412_1);  permute_190 = arg412_1 = None
        sum_216: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_455, [0], True, dtype = torch.float32);  view_455 = None
        view_456: "f32[2048]" = torch.ops.aten.view.default(sum_216, _shape_param_459);  sum_216 = _shape_param_459 = None
        convert_element_type_494: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_456, torch.bfloat16);  view_456 = None
        view_457: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_210, _shape_param_460);  mm_210 = _shape_param_460 = None
        convert_element_type_495: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_211, torch.float32);  mm_211 = None
        convert_element_type_496: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_494, torch.float32);  convert_element_type_494 = None
        view_458: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_457, _shape_param_461);  view_457 = _shape_param_461 = None
        permute_191: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
        clone_69: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
        view_459: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_69, _shape_param_462);  clone_69 = _shape_param_462 = None
        bmm_68: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg413_1, view_459);  arg413_1 = None
        bmm_69: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_459, arg414_1);  view_459 = arg414_1 = None
        view_460: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_68, _shape_param_463);  bmm_68 = _shape_param_463 = None
        view_461: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_69, _shape_param_464);  bmm_69 = _shape_param_464 = None
        convert_element_type_497: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_461, torch.float32);  view_461 = None
        mul_451: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_497, arg415_1);  convert_element_type_497 = None
        sum_217: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [-1], True)
        neg_17: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg415_1);  arg415_1 = None
        fma_17: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_17, sum_217, mul_451);  neg_17 = sum_217 = mul_451 = None
        convert_element_type_498: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None
        slice_37: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg416_1, 2, 0, 128);  arg416_1 = None
        slice_38: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_37, 3, 0, 128);  slice_37 = None
        where_20: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_38, convert_element_type_498, full_2);  slice_38 = convert_element_type_498 = None
        view_462: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_20, _shape_param_465);  where_20 = _shape_param_465 = None
        bmm_70: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg417_1, view_462);  arg417_1 = None
        bmm_71: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_462, arg418_1);  view_462 = arg418_1 = None
        view_463: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_70, _shape_param_466);  bmm_70 = _shape_param_466 = None
        view_464: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_71, _shape_param_467);  bmm_71 = _shape_param_467 = None
        convert_element_type_499: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_463, torch.float32);  view_463 = None
        permute_192: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_499, [0, 1, 3, 2]);  convert_element_type_499 = None
        convert_element_type_500: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_192, torch.bfloat16);  permute_192 = None
        convert_element_type_501: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_464, torch.bfloat16);  view_464 = None
        permute_193: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None
        clone_70: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
        view_465: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_70, _shape_param_468);  clone_70 = _shape_param_468 = None
        permute_194: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_500, [0, 2, 1, 3]);  convert_element_type_500 = None
        view_466: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_194, _shape_param_469);  permute_194 = _shape_param_469 = None
        permute_195: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_501, [0, 2, 1, 3]);  convert_element_type_501 = None
        clone_71: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
        view_467: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_71, _shape_param_470);  clone_71 = _shape_param_470 = None
        view_468: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_465, _shape_param_471);  view_465 = _shape_param_471 = None
        permute_196: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_468, [1, 0])
        mm_212: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_196, arg419_1);  permute_196 = None
        mm_213: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_468, arg420_1);  view_468 = arg420_1 = None
        view_469: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_213, _shape_param_472);  mm_213 = _shape_param_472 = None
        convert_element_type_502: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_469, torch.float32);  view_469 = None
        convert_element_type_503: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_212, torch.float32);  mm_212 = None
        clone_72: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_466, memory_format = torch.contiguous_format);  view_466 = None
        view_470: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_72, _shape_param_473);  clone_72 = _shape_param_473 = None
        permute_197: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_470, [1, 0])
        mm_214: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_197, arg419_1);  permute_197 = None
        mm_215: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_470, arg421_1);  view_470 = arg421_1 = None
        view_471: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_215, _shape_param_474);  mm_215 = _shape_param_474 = None
        convert_element_type_504: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_471, torch.float32);  view_471 = None
        add_142: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_502, convert_element_type_504);  convert_element_type_502 = convert_element_type_504 = None
        convert_element_type_505: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_214, torch.float32);  mm_214 = None
        view_472: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_467, _shape_param_475);  view_467 = _shape_param_475 = None
        permute_198: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_472, [1, 0])
        mm_216: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_198, arg419_1);  permute_198 = arg419_1 = None
        mm_217: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_472, arg422_1);  view_472 = arg422_1 = None
        view_473: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_217, _shape_param_476);  mm_217 = _shape_param_476 = None
        convert_element_type_506: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_473, torch.float32);  view_473 = None
        add_143: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_142, convert_element_type_506);  add_142 = convert_element_type_506 = None
        convert_element_type_507: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_216, torch.float32);  mm_216 = None
        mul_452: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_143, arg423_1);  arg423_1 = None
        mul_453: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_452, 2048)
        sum_218: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_452, [2], True)
        mul_454: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_452, arg424_1);  mul_452 = None
        sum_219: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True);  mul_454 = None
        mul_455: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg424_1, sum_219);  sum_219 = None
        sub_93: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_453, sum_218);  mul_453 = sum_218 = None
        sub_94: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_93, mul_455);  sub_93 = mul_455 = None
        mul_456: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg425_1, sub_94);  arg425_1 = sub_94 = None
        mul_457: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_143, arg424_1);  arg424_1 = None
        sum_220: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 1]);  mul_457 = None
        sum_221: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_143, [0, 1]);  add_143 = None
        add_144: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_141, mul_456);  add_141 = mul_456 = None
        convert_element_type_508: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16)
        view_474: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_508, _shape_param_477);  convert_element_type_508 = _shape_param_477 = None
        mm_218: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_474, arg426_1);  arg426_1 = None
        permute_199: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_474, [1, 0])
        mm_219: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_199, arg427_1);  permute_199 = arg427_1 = None
        sum_222: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_474, [0], True, dtype = torch.float32);  view_474 = None
        view_475: "f32[2048]" = torch.ops.aten.view.default(sum_222, _shape_param_478);  sum_222 = _shape_param_478 = None
        convert_element_type_509: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_475, torch.bfloat16);  view_475 = None
        view_476: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_218, _shape_param_479);  mm_218 = _shape_param_479 = None
        convert_element_type_510: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_476, torch.float32);  view_476 = None
        convert_element_type_511: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_219, torch.float32);  mm_219 = None
        convert_element_type_512: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_509, torch.float32);  convert_element_type_509 = None
        view_477: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg428_1, _shape_param_480);  arg428_1 = _shape_param_480 = None
        mul_458: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_477, 0.5)
        mul_459: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_510, mul_458);  mul_458 = None
        convert_element_type_513: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_477, torch.float32)
        pow_37: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_513, 3.0)
        mul_460: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_145: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_477, mul_460);  view_477 = mul_460 = None
        mul_461: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_145, 0.7978845608028654);  add_145 = None
        tanh_18: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_461);  mul_461 = None
        add_146: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_462: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_510, add_146);  convert_element_type_510 = add_146 = None
        convert_element_type_514: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_462, torch.bfloat16);  mul_462 = None
        mul_463: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_95: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_463);  mul_463 = None
        mul_464: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_459, sub_95);  mul_459 = sub_95 = None
        mul_465: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_464, 0.7978845608028654);  mul_464 = None
        convert_element_type_515: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_465, torch.bfloat16)
        mul_466: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_465, 0.044715);  mul_465 = None
        pow_38: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_513, 2.0);  convert_element_type_513 = None
        mul_467: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_468: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
        convert_element_type_516: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_468, torch.bfloat16);  mul_468 = None
        add_147: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_515, convert_element_type_516);  convert_element_type_515 = convert_element_type_516 = None
        mul_469: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_514, 0.5);  convert_element_type_514 = None
        add_148: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_147, mul_469);  add_147 = mul_469 = None
        view_478: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_148, _shape_param_481);  add_148 = _shape_param_481 = None
        mm_220: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_478, arg429_1);  arg429_1 = None
        permute_200: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_478, [1, 0])
        mm_221: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_200, arg430_1);  permute_200 = arg430_1 = None
        sum_223: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True, dtype = torch.float32);  view_478 = None
        view_479: "f32[8192]" = torch.ops.aten.view.default(sum_223, _shape_param_482);  sum_223 = _shape_param_482 = None
        convert_element_type_517: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_479, torch.bfloat16);  view_479 = None
        view_480: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_220, _shape_param_483);  mm_220 = _shape_param_483 = None
        convert_element_type_518: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_480, torch.float32);  view_480 = None
        convert_element_type_519: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_221, torch.float32);  mm_221 = None
        convert_element_type_520: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_517, torch.float32);  convert_element_type_517 = None
        mul_470: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_518, arg431_1);  arg431_1 = None
        mul_471: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_470, 2048)
        sum_224: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [2], True)
        mul_472: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_470, arg432_1);  mul_470 = None
        sum_225: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
        mul_473: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg432_1, sum_225);  sum_225 = None
        sub_96: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_471, sum_224);  mul_471 = sum_224 = None
        sub_97: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_96, mul_473);  sub_96 = mul_473 = None
        mul_474: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg433_1, sub_97);  arg433_1 = sub_97 = None
        mul_475: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_518, arg432_1);  arg432_1 = None
        sum_226: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
        sum_227: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_518, [0, 1]);  convert_element_type_518 = None
        add_149: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_144, mul_474);  add_144 = mul_474 = None
        convert_element_type_521: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16)
        view_481: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_521, _shape_param_484);  convert_element_type_521 = _shape_param_484 = None
        mm_222: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_481, arg434_1);  arg434_1 = None
        permute_201: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_481, [1, 0])
        mm_223: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_201, arg435_1);  permute_201 = arg435_1 = None
        sum_228: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True, dtype = torch.float32);  view_481 = None
        view_482: "f32[2048]" = torch.ops.aten.view.default(sum_228, _shape_param_485);  sum_228 = _shape_param_485 = None
        convert_element_type_522: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_482, torch.bfloat16);  view_482 = None
        view_483: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_222, _shape_param_486);  mm_222 = _shape_param_486 = None
        convert_element_type_523: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_223, torch.float32);  mm_223 = None
        convert_element_type_524: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_522, torch.float32);  convert_element_type_522 = None
        view_484: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_483, _shape_param_487);  view_483 = _shape_param_487 = None
        permute_202: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
        clone_73: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
        view_485: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_73, _shape_param_488);  clone_73 = _shape_param_488 = None
        bmm_72: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg436_1, view_485);  arg436_1 = None
        bmm_73: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_485, arg437_1);  view_485 = arg437_1 = None
        view_486: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_72, _shape_param_489);  bmm_72 = _shape_param_489 = None
        view_487: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_73, _shape_param_490);  bmm_73 = _shape_param_490 = None
        convert_element_type_525: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_487, torch.float32);  view_487 = None
        mul_476: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_525, arg438_1);  convert_element_type_525 = None
        sum_229: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_476, [-1], True)
        neg_18: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg438_1);  arg438_1 = None
        fma_18: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_18, sum_229, mul_476);  neg_18 = sum_229 = mul_476 = None
        convert_element_type_526: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None
        slice_39: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg439_1, 2, 0, 128);  arg439_1 = None
        slice_40: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_39, 3, 0, 128);  slice_39 = None
        where_21: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_40, convert_element_type_526, full_2);  slice_40 = convert_element_type_526 = None
        view_488: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_21, _shape_param_491);  where_21 = _shape_param_491 = None
        bmm_74: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg440_1, view_488);  arg440_1 = None
        bmm_75: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_488, arg441_1);  view_488 = arg441_1 = None
        view_489: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_74, _shape_param_492);  bmm_74 = _shape_param_492 = None
        view_490: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_75, _shape_param_493);  bmm_75 = _shape_param_493 = None
        convert_element_type_527: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_489, torch.float32);  view_489 = None
        permute_203: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_527, [0, 1, 3, 2]);  convert_element_type_527 = None
        convert_element_type_528: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_203, torch.bfloat16);  permute_203 = None
        convert_element_type_529: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_490, torch.bfloat16);  view_490 = None
        permute_204: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        clone_74: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
        view_491: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_74, _shape_param_494);  clone_74 = _shape_param_494 = None
        permute_205: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_528, [0, 2, 1, 3]);  convert_element_type_528 = None
        view_492: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_205, _shape_param_495);  permute_205 = _shape_param_495 = None
        permute_206: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_529, [0, 2, 1, 3]);  convert_element_type_529 = None
        clone_75: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_206, memory_format = torch.contiguous_format);  permute_206 = None
        view_493: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_75, _shape_param_496);  clone_75 = _shape_param_496 = None
        view_494: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_491, _shape_param_497);  view_491 = _shape_param_497 = None
        permute_207: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_224: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_207, arg442_1);  permute_207 = None
        mm_225: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_494, arg443_1);  view_494 = arg443_1 = None
        view_495: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_225, _shape_param_498);  mm_225 = _shape_param_498 = None
        convert_element_type_530: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_495, torch.float32);  view_495 = None
        convert_element_type_531: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_224, torch.float32);  mm_224 = None
        clone_76: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_492, memory_format = torch.contiguous_format);  view_492 = None
        view_496: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_76, _shape_param_499);  clone_76 = _shape_param_499 = None
        permute_208: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_496, [1, 0])
        mm_226: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_208, arg442_1);  permute_208 = None
        mm_227: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_496, arg444_1);  view_496 = arg444_1 = None
        view_497: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_227, _shape_param_500);  mm_227 = _shape_param_500 = None
        convert_element_type_532: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_497, torch.float32);  view_497 = None
        add_150: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_530, convert_element_type_532);  convert_element_type_530 = convert_element_type_532 = None
        convert_element_type_533: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_226, torch.float32);  mm_226 = None
        view_498: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_493, _shape_param_501);  view_493 = _shape_param_501 = None
        permute_209: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_498, [1, 0])
        mm_228: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_209, arg442_1);  permute_209 = arg442_1 = None
        mm_229: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_498, arg445_1);  view_498 = arg445_1 = None
        view_499: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_229, _shape_param_502);  mm_229 = _shape_param_502 = None
        convert_element_type_534: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_499, torch.float32);  view_499 = None
        add_151: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_150, convert_element_type_534);  add_150 = convert_element_type_534 = None
        convert_element_type_535: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_228, torch.float32);  mm_228 = None
        mul_477: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_151, arg446_1);  arg446_1 = None
        mul_478: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_477, 2048)
        sum_230: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_477, [2], True)
        mul_479: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_477, arg447_1);  mul_477 = None
        sum_231: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True);  mul_479 = None
        mul_480: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg447_1, sum_231);  sum_231 = None
        sub_98: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_478, sum_230);  mul_478 = sum_230 = None
        sub_99: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_98, mul_480);  sub_98 = mul_480 = None
        mul_481: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg448_1, sub_99);  arg448_1 = sub_99 = None
        mul_482: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_151, arg447_1);  arg447_1 = None
        sum_232: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 1]);  mul_482 = None
        sum_233: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_151, [0, 1]);  add_151 = None
        add_152: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_149, mul_481);  add_149 = mul_481 = None
        convert_element_type_536: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16)
        view_500: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_536, _shape_param_503);  convert_element_type_536 = _shape_param_503 = None
        mm_230: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_500, arg449_1);  arg449_1 = None
        permute_210: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_500, [1, 0])
        mm_231: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_210, arg450_1);  permute_210 = arg450_1 = None
        sum_234: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True, dtype = torch.float32);  view_500 = None
        view_501: "f32[2048]" = torch.ops.aten.view.default(sum_234, _shape_param_504);  sum_234 = _shape_param_504 = None
        convert_element_type_537: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_501, torch.bfloat16);  view_501 = None
        view_502: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_230, _shape_param_505);  mm_230 = _shape_param_505 = None
        convert_element_type_538: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_502, torch.float32);  view_502 = None
        convert_element_type_539: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_231, torch.float32);  mm_231 = None
        convert_element_type_540: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_537, torch.float32);  convert_element_type_537 = None
        view_503: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg451_1, _shape_param_506);  arg451_1 = _shape_param_506 = None
        mul_483: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
        mul_484: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_538, mul_483);  mul_483 = None
        convert_element_type_541: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_503, torch.float32)
        pow_39: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_541, 3.0)
        mul_485: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_39, 0.044715);  pow_39 = None
        add_153: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_503, mul_485);  view_503 = mul_485 = None
        mul_486: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_153, 0.7978845608028654);  add_153 = None
        tanh_19: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_486);  mul_486 = None
        add_154: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_487: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_538, add_154);  convert_element_type_538 = add_154 = None
        convert_element_type_542: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_487, torch.bfloat16);  mul_487 = None
        mul_488: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_100: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_488);  mul_488 = None
        mul_489: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_484, sub_100);  mul_484 = sub_100 = None
        mul_490: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_489, 0.7978845608028654);  mul_489 = None
        convert_element_type_543: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_490, torch.bfloat16)
        mul_491: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_490, 0.044715);  mul_490 = None
        pow_40: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_541, 2.0);  convert_element_type_541 = None
        mul_492: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_493: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
        convert_element_type_544: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_493, torch.bfloat16);  mul_493 = None
        add_155: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_543, convert_element_type_544);  convert_element_type_543 = convert_element_type_544 = None
        mul_494: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_542, 0.5);  convert_element_type_542 = None
        add_156: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_155, mul_494);  add_155 = mul_494 = None
        view_504: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_156, _shape_param_507);  add_156 = _shape_param_507 = None
        mm_232: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_504, arg452_1);  arg452_1 = None
        permute_211: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_504, [1, 0])
        mm_233: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_211, arg453_1);  permute_211 = arg453_1 = None
        sum_235: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_504, [0], True, dtype = torch.float32);  view_504 = None
        view_505: "f32[8192]" = torch.ops.aten.view.default(sum_235, _shape_param_508);  sum_235 = _shape_param_508 = None
        convert_element_type_545: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_505, torch.bfloat16);  view_505 = None
        view_506: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_232, _shape_param_509);  mm_232 = _shape_param_509 = None
        convert_element_type_546: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_506, torch.float32);  view_506 = None
        convert_element_type_547: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_233, torch.float32);  mm_233 = None
        convert_element_type_548: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_545, torch.float32);  convert_element_type_545 = None
        mul_495: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_546, arg454_1);  arg454_1 = None
        mul_496: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_495, 2048)
        sum_236: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
        mul_497: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_495, arg455_1);  mul_495 = None
        sum_237: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
        mul_498: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg455_1, sum_237);  sum_237 = None
        sub_101: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_496, sum_236);  mul_496 = sum_236 = None
        sub_102: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_101, mul_498);  sub_101 = mul_498 = None
        mul_499: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg456_1, sub_102);  arg456_1 = sub_102 = None
        mul_500: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_546, arg455_1);  arg455_1 = None
        sum_238: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
        sum_239: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_546, [0, 1]);  convert_element_type_546 = None
        add_157: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_152, mul_499);  add_152 = mul_499 = None
        convert_element_type_549: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_157, torch.bfloat16)
        view_507: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_549, _shape_param_510);  convert_element_type_549 = _shape_param_510 = None
        mm_234: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_507, arg457_1);  arg457_1 = None
        permute_212: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_507, [1, 0])
        mm_235: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_212, arg458_1);  permute_212 = arg458_1 = None
        sum_240: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True, dtype = torch.float32);  view_507 = None
        view_508: "f32[2048]" = torch.ops.aten.view.default(sum_240, _shape_param_511);  sum_240 = _shape_param_511 = None
        convert_element_type_550: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_508, torch.bfloat16);  view_508 = None
        view_509: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_234, _shape_param_512);  mm_234 = _shape_param_512 = None
        convert_element_type_551: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None
        convert_element_type_552: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_550, torch.float32);  convert_element_type_550 = None
        view_510: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_509, _shape_param_513);  view_509 = _shape_param_513 = None
        permute_213: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_510, [0, 2, 1, 3]);  view_510 = None
        clone_77: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
        view_511: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_77, _shape_param_514);  clone_77 = _shape_param_514 = None
        bmm_76: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg459_1, view_511);  arg459_1 = None
        bmm_77: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_511, arg460_1);  view_511 = arg460_1 = None
        view_512: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_76, _shape_param_515);  bmm_76 = _shape_param_515 = None
        view_513: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_77, _shape_param_516);  bmm_77 = _shape_param_516 = None
        convert_element_type_553: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_513, torch.float32);  view_513 = None
        mul_501: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_553, arg461_1);  convert_element_type_553 = None
        sum_241: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_501, [-1], True)
        neg_19: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg461_1);  arg461_1 = None
        fma_19: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_19, sum_241, mul_501);  neg_19 = sum_241 = mul_501 = None
        convert_element_type_554: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None
        slice_41: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg462_1, 2, 0, 128);  arg462_1 = None
        slice_42: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_41, 3, 0, 128);  slice_41 = None
        where_22: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_42, convert_element_type_554, full_2);  slice_42 = convert_element_type_554 = None
        view_514: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_22, _shape_param_517);  where_22 = _shape_param_517 = None
        bmm_78: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg463_1, view_514);  arg463_1 = None
        bmm_79: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_514, arg464_1);  view_514 = arg464_1 = None
        view_515: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_78, _shape_param_518);  bmm_78 = _shape_param_518 = None
        view_516: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_79, _shape_param_519);  bmm_79 = _shape_param_519 = None
        convert_element_type_555: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_515, torch.float32);  view_515 = None
        permute_214: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_555, [0, 1, 3, 2]);  convert_element_type_555 = None
        convert_element_type_556: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_214, torch.bfloat16);  permute_214 = None
        convert_element_type_557: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_516, torch.bfloat16);  view_516 = None
        permute_215: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_78: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
        view_517: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_78, _shape_param_520);  clone_78 = _shape_param_520 = None
        permute_216: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_556, [0, 2, 1, 3]);  convert_element_type_556 = None
        view_518: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_216, _shape_param_521);  permute_216 = _shape_param_521 = None
        permute_217: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_557, [0, 2, 1, 3]);  convert_element_type_557 = None
        clone_79: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
        view_519: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_79, _shape_param_522);  clone_79 = _shape_param_522 = None
        view_520: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_517, _shape_param_523);  view_517 = _shape_param_523 = None
        permute_218: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_520, [1, 0])
        mm_236: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_218, arg465_1);  permute_218 = None
        mm_237: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_520, arg466_1);  view_520 = arg466_1 = None
        view_521: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_237, _shape_param_524);  mm_237 = _shape_param_524 = None
        convert_element_type_558: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_521, torch.float32);  view_521 = None
        convert_element_type_559: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_236, torch.float32);  mm_236 = None
        clone_80: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_518, memory_format = torch.contiguous_format);  view_518 = None
        view_522: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_80, _shape_param_525);  clone_80 = _shape_param_525 = None
        permute_219: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_522, [1, 0])
        mm_238: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_219, arg465_1);  permute_219 = None
        mm_239: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_522, arg467_1);  view_522 = arg467_1 = None
        view_523: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_239, _shape_param_526);  mm_239 = _shape_param_526 = None
        convert_element_type_560: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_523, torch.float32);  view_523 = None
        add_158: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_558, convert_element_type_560);  convert_element_type_558 = convert_element_type_560 = None
        convert_element_type_561: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_238, torch.float32);  mm_238 = None
        view_524: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_519, _shape_param_527);  view_519 = _shape_param_527 = None
        permute_220: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_524, [1, 0])
        mm_240: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_220, arg465_1);  permute_220 = arg465_1 = None
        mm_241: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_524, arg468_1);  view_524 = arg468_1 = None
        view_525: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_241, _shape_param_528);  mm_241 = _shape_param_528 = None
        convert_element_type_562: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_525, torch.float32);  view_525 = None
        add_159: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_158, convert_element_type_562);  add_158 = convert_element_type_562 = None
        convert_element_type_563: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_240, torch.float32);  mm_240 = None
        mul_502: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_159, arg469_1);  arg469_1 = None
        mul_503: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_502, 2048)
        sum_242: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_502, [2], True)
        mul_504: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_502, arg470_1);  mul_502 = None
        sum_243: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_504, [2], True);  mul_504 = None
        mul_505: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg470_1, sum_243);  sum_243 = None
        sub_103: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_503, sum_242);  mul_503 = sum_242 = None
        sub_104: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_103, mul_505);  sub_103 = mul_505 = None
        mul_506: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg471_1, sub_104);  arg471_1 = sub_104 = None
        mul_507: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_159, arg470_1);  arg470_1 = None
        sum_244: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 1]);  mul_507 = None
        sum_245: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_159, [0, 1]);  add_159 = None
        add_160: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_157, mul_506);  add_157 = mul_506 = None
        convert_element_type_564: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16)
        view_526: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_564, _shape_param_529);  convert_element_type_564 = _shape_param_529 = None
        mm_242: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_526, arg472_1);  arg472_1 = None
        permute_221: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_526, [1, 0])
        mm_243: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_221, arg473_1);  permute_221 = arg473_1 = None
        sum_246: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_526, [0], True, dtype = torch.float32);  view_526 = None
        view_527: "f32[2048]" = torch.ops.aten.view.default(sum_246, _shape_param_530);  sum_246 = _shape_param_530 = None
        convert_element_type_565: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_527, torch.bfloat16);  view_527 = None
        view_528: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_242, _shape_param_531);  mm_242 = _shape_param_531 = None
        convert_element_type_566: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_528, torch.float32);  view_528 = None
        convert_element_type_567: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_243, torch.float32);  mm_243 = None
        convert_element_type_568: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_565, torch.float32);  convert_element_type_565 = None
        view_529: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg474_1, _shape_param_532);  arg474_1 = _shape_param_532 = None
        mul_508: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_509: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_566, mul_508);  mul_508 = None
        convert_element_type_569: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_529, torch.float32)
        pow_41: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_569, 3.0)
        mul_510: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_161: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_529, mul_510);  view_529 = mul_510 = None
        mul_511: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_20: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_511);  mul_511 = None
        add_162: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_512: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_566, add_162);  convert_element_type_566 = add_162 = None
        convert_element_type_570: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_512, torch.bfloat16);  mul_512 = None
        mul_513: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_105: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_513);  mul_513 = None
        mul_514: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_509, sub_105);  mul_509 = sub_105 = None
        mul_515: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_514, 0.7978845608028654);  mul_514 = None
        convert_element_type_571: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_515, torch.bfloat16)
        mul_516: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_515, 0.044715);  mul_515 = None
        pow_42: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_569, 2.0);  convert_element_type_569 = None
        mul_517: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_518: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_516, mul_517);  mul_516 = mul_517 = None
        convert_element_type_572: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_518, torch.bfloat16);  mul_518 = None
        add_163: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_571, convert_element_type_572);  convert_element_type_571 = convert_element_type_572 = None
        mul_519: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_570, 0.5);  convert_element_type_570 = None
        add_164: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_163, mul_519);  add_163 = mul_519 = None
        view_530: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_164, _shape_param_533);  add_164 = _shape_param_533 = None
        mm_244: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_530, arg475_1);  arg475_1 = None
        permute_222: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_530, [1, 0])
        mm_245: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_222, arg476_1);  permute_222 = arg476_1 = None
        sum_247: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_530, [0], True, dtype = torch.float32);  view_530 = None
        view_531: "f32[8192]" = torch.ops.aten.view.default(sum_247, _shape_param_534);  sum_247 = _shape_param_534 = None
        convert_element_type_573: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_531, torch.bfloat16);  view_531 = None
        view_532: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_244, _shape_param_535);  mm_244 = _shape_param_535 = None
        convert_element_type_574: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_532, torch.float32);  view_532 = None
        convert_element_type_575: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_245, torch.float32);  mm_245 = None
        convert_element_type_576: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_573, torch.float32);  convert_element_type_573 = None
        mul_520: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_574, arg477_1);  arg477_1 = None
        mul_521: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_520, 2048)
        sum_248: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_520, [2], True)
        mul_522: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_520, arg478_1);  mul_520 = None
        sum_249: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_522, [2], True);  mul_522 = None
        mul_523: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg478_1, sum_249);  sum_249 = None
        sub_106: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_521, sum_248);  mul_521 = sum_248 = None
        sub_107: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_106, mul_523);  sub_106 = mul_523 = None
        mul_524: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg479_1, sub_107);  arg479_1 = sub_107 = None
        mul_525: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_574, arg478_1);  arg478_1 = None
        sum_250: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_525, [0, 1]);  mul_525 = None
        sum_251: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_574, [0, 1]);  convert_element_type_574 = None
        add_165: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_160, mul_524);  add_160 = mul_524 = None
        convert_element_type_577: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_165, torch.bfloat16)
        view_533: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_577, _shape_param_536);  convert_element_type_577 = _shape_param_536 = None
        mm_246: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_533, arg480_1);  arg480_1 = None
        permute_223: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_533, [1, 0])
        mm_247: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_223, arg481_1);  permute_223 = arg481_1 = None
        sum_252: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_533, [0], True, dtype = torch.float32);  view_533 = None
        view_534: "f32[2048]" = torch.ops.aten.view.default(sum_252, _shape_param_537);  sum_252 = _shape_param_537 = None
        convert_element_type_578: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_534, torch.bfloat16);  view_534 = None
        view_535: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_246, _shape_param_538);  mm_246 = _shape_param_538 = None
        convert_element_type_579: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_247, torch.float32);  mm_247 = None
        convert_element_type_580: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_578, torch.float32);  convert_element_type_578 = None
        view_536: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_535, _shape_param_539);  view_535 = _shape_param_539 = None
        permute_224: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_536, [0, 2, 1, 3]);  view_536 = None
        clone_81: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
        view_537: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_81, _shape_param_540);  clone_81 = _shape_param_540 = None
        bmm_80: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg482_1, view_537);  arg482_1 = None
        bmm_81: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_537, arg483_1);  view_537 = arg483_1 = None
        view_538: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_80, _shape_param_541);  bmm_80 = _shape_param_541 = None
        view_539: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_81, _shape_param_542);  bmm_81 = _shape_param_542 = None
        convert_element_type_581: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_539, torch.float32);  view_539 = None
        mul_526: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_581, arg484_1);  convert_element_type_581 = None
        sum_253: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_526, [-1], True)
        neg_20: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg484_1);  arg484_1 = None
        fma_20: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_20, sum_253, mul_526);  neg_20 = sum_253 = mul_526 = None
        convert_element_type_582: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None
        slice_43: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg485_1, 2, 0, 128);  arg485_1 = None
        slice_44: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_43, 3, 0, 128);  slice_43 = None
        where_23: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_44, convert_element_type_582, full_2);  slice_44 = convert_element_type_582 = None
        view_540: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_23, _shape_param_543);  where_23 = _shape_param_543 = None
        bmm_82: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg486_1, view_540);  arg486_1 = None
        bmm_83: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_540, arg487_1);  view_540 = arg487_1 = None
        view_541: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_82, _shape_param_544);  bmm_82 = _shape_param_544 = None
        view_542: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_83, _shape_param_545);  bmm_83 = _shape_param_545 = None
        convert_element_type_583: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_541, torch.float32);  view_541 = None
        permute_225: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_583, [0, 1, 3, 2]);  convert_element_type_583 = None
        convert_element_type_584: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_225, torch.bfloat16);  permute_225 = None
        convert_element_type_585: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_542, torch.bfloat16);  view_542 = None
        permute_226: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None
        clone_82: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
        view_543: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_82, _shape_param_546);  clone_82 = _shape_param_546 = None
        permute_227: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_584, [0, 2, 1, 3]);  convert_element_type_584 = None
        view_544: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_227, _shape_param_547);  permute_227 = _shape_param_547 = None
        permute_228: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_585, [0, 2, 1, 3]);  convert_element_type_585 = None
        clone_83: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
        view_545: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_83, _shape_param_548);  clone_83 = _shape_param_548 = None
        view_546: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_543, _shape_param_549);  view_543 = _shape_param_549 = None
        permute_229: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_546, [1, 0])
        mm_248: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_229, arg488_1);  permute_229 = None
        mm_249: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_546, arg489_1);  view_546 = arg489_1 = None
        view_547: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_249, _shape_param_550);  mm_249 = _shape_param_550 = None
        convert_element_type_586: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_547, torch.float32);  view_547 = None
        convert_element_type_587: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_248, torch.float32);  mm_248 = None
        clone_84: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_544, memory_format = torch.contiguous_format);  view_544 = None
        view_548: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_84, _shape_param_551);  clone_84 = _shape_param_551 = None
        permute_230: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_250: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_230, arg488_1);  permute_230 = None
        mm_251: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_548, arg490_1);  view_548 = arg490_1 = None
        view_549: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_251, _shape_param_552);  mm_251 = _shape_param_552 = None
        convert_element_type_588: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_549, torch.float32);  view_549 = None
        add_166: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_586, convert_element_type_588);  convert_element_type_586 = convert_element_type_588 = None
        convert_element_type_589: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_250, torch.float32);  mm_250 = None
        view_550: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_545, _shape_param_553);  view_545 = _shape_param_553 = None
        permute_231: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_550, [1, 0])
        mm_252: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_231, arg488_1);  permute_231 = arg488_1 = None
        mm_253: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_550, arg491_1);  view_550 = arg491_1 = None
        view_551: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_253, _shape_param_554);  mm_253 = _shape_param_554 = None
        convert_element_type_590: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_551, torch.float32);  view_551 = None
        add_167: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_166, convert_element_type_590);  add_166 = convert_element_type_590 = None
        convert_element_type_591: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_252, torch.float32);  mm_252 = None
        mul_527: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_167, arg492_1);  arg492_1 = None
        mul_528: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_527, 2048)
        sum_254: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True)
        mul_529: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_527, arg493_1);  mul_527 = None
        sum_255: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [2], True);  mul_529 = None
        mul_530: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg493_1, sum_255);  sum_255 = None
        sub_108: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_528, sum_254);  mul_528 = sum_254 = None
        sub_109: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_108, mul_530);  sub_108 = mul_530 = None
        mul_531: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg494_1, sub_109);  arg494_1 = sub_109 = None
        mul_532: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_167, arg493_1);  arg493_1 = None
        sum_256: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 1]);  mul_532 = None
        sum_257: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
        add_168: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_165, mul_531);  add_165 = mul_531 = None
        convert_element_type_592: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_168, torch.bfloat16)
        view_552: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_592, _shape_param_555);  convert_element_type_592 = _shape_param_555 = None
        mm_254: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_552, arg495_1);  arg495_1 = None
        permute_232: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_552, [1, 0])
        mm_255: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_232, arg496_1);  permute_232 = arg496_1 = None
        sum_258: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True, dtype = torch.float32);  view_552 = None
        view_553: "f32[2048]" = torch.ops.aten.view.default(sum_258, _shape_param_556);  sum_258 = _shape_param_556 = None
        convert_element_type_593: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_553, torch.bfloat16);  view_553 = None
        view_554: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_254, _shape_param_557);  mm_254 = _shape_param_557 = None
        convert_element_type_594: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_554, torch.float32);  view_554 = None
        convert_element_type_595: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_255, torch.float32);  mm_255 = None
        convert_element_type_596: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_593, torch.float32);  convert_element_type_593 = None
        view_555: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg497_1, _shape_param_558);  arg497_1 = _shape_param_558 = None
        mul_533: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_555, 0.5)
        mul_534: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_594, mul_533);  mul_533 = None
        convert_element_type_597: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_555, torch.float32)
        pow_43: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_597, 3.0)
        mul_535: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_43, 0.044715);  pow_43 = None
        add_169: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_555, mul_535);  view_555 = mul_535 = None
        mul_536: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_169, 0.7978845608028654);  add_169 = None
        tanh_21: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_536);  mul_536 = None
        add_170: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_537: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_594, add_170);  convert_element_type_594 = add_170 = None
        convert_element_type_598: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_537, torch.bfloat16);  mul_537 = None
        mul_538: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_110: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_538);  mul_538 = None
        mul_539: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_534, sub_110);  mul_534 = sub_110 = None
        mul_540: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_539, 0.7978845608028654);  mul_539 = None
        convert_element_type_599: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_540, torch.bfloat16)
        mul_541: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_540, 0.044715);  mul_540 = None
        pow_44: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_597, 2.0);  convert_element_type_597 = None
        mul_542: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_543: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_541, mul_542);  mul_541 = mul_542 = None
        convert_element_type_600: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_543, torch.bfloat16);  mul_543 = None
        add_171: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_599, convert_element_type_600);  convert_element_type_599 = convert_element_type_600 = None
        mul_544: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_598, 0.5);  convert_element_type_598 = None
        add_172: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_171, mul_544);  add_171 = mul_544 = None
        view_556: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_172, _shape_param_559);  add_172 = _shape_param_559 = None
        mm_256: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_556, arg498_1);  arg498_1 = None
        permute_233: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_556, [1, 0])
        mm_257: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_233, arg499_1);  permute_233 = arg499_1 = None
        sum_259: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True, dtype = torch.float32);  view_556 = None
        view_557: "f32[8192]" = torch.ops.aten.view.default(sum_259, _shape_param_560);  sum_259 = _shape_param_560 = None
        convert_element_type_601: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_557, torch.bfloat16);  view_557 = None
        view_558: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_256, _shape_param_561);  mm_256 = _shape_param_561 = None
        convert_element_type_602: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_558, torch.float32);  view_558 = None
        convert_element_type_603: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_257, torch.float32);  mm_257 = None
        convert_element_type_604: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_601, torch.float32);  convert_element_type_601 = None
        mul_545: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_602, arg500_1);  arg500_1 = None
        mul_546: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_545, 2048)
        sum_260: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_545, [2], True)
        mul_547: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_545, arg501_1);  mul_545 = None
        sum_261: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_547, [2], True);  mul_547 = None
        mul_548: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg501_1, sum_261);  sum_261 = None
        sub_111: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_546, sum_260);  mul_546 = sum_260 = None
        sub_112: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_111, mul_548);  sub_111 = mul_548 = None
        mul_549: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg502_1, sub_112);  arg502_1 = sub_112 = None
        mul_550: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_602, arg501_1);  arg501_1 = None
        sum_262: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 1]);  mul_550 = None
        sum_263: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_602, [0, 1]);  convert_element_type_602 = None
        add_173: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_168, mul_549);  add_168 = mul_549 = None
        convert_element_type_605: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_173, torch.bfloat16)
        view_559: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_605, _shape_param_562);  convert_element_type_605 = _shape_param_562 = None
        mm_258: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_559, arg503_1);  arg503_1 = None
        permute_234: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_559, [1, 0])
        mm_259: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_234, arg504_1);  permute_234 = arg504_1 = None
        sum_264: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_559, [0], True, dtype = torch.float32);  view_559 = None
        view_560: "f32[2048]" = torch.ops.aten.view.default(sum_264, _shape_param_563);  sum_264 = _shape_param_563 = None
        convert_element_type_606: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_560, torch.bfloat16);  view_560 = None
        view_561: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_258, _shape_param_564);  mm_258 = _shape_param_564 = None
        convert_element_type_607: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_259, torch.float32);  mm_259 = None
        convert_element_type_608: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_606, torch.float32);  convert_element_type_606 = None
        view_562: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_561, _shape_param_565);  view_561 = _shape_param_565 = None
        permute_235: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None
        clone_85: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
        view_563: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_85, _shape_param_566);  clone_85 = _shape_param_566 = None
        bmm_84: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg505_1, view_563);  arg505_1 = None
        bmm_85: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_563, arg506_1);  view_563 = arg506_1 = None
        view_564: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_84, _shape_param_567);  bmm_84 = _shape_param_567 = None
        view_565: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_85, _shape_param_568);  bmm_85 = _shape_param_568 = None
        convert_element_type_609: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_565, torch.float32);  view_565 = None
        mul_551: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_609, arg507_1);  convert_element_type_609 = None
        sum_265: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [-1], True)
        neg_21: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg507_1);  arg507_1 = None
        fma_21: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_21, sum_265, mul_551);  neg_21 = sum_265 = mul_551 = None
        convert_element_type_610: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None
        slice_45: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg508_1, 2, 0, 128);  arg508_1 = None
        slice_46: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_45, 3, 0, 128);  slice_45 = None
        where_24: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_46, convert_element_type_610, full_2);  slice_46 = convert_element_type_610 = None
        view_566: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_24, _shape_param_569);  where_24 = _shape_param_569 = None
        bmm_86: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg509_1, view_566);  arg509_1 = None
        bmm_87: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_566, arg510_1);  view_566 = arg510_1 = None
        view_567: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_86, _shape_param_570);  bmm_86 = _shape_param_570 = None
        view_568: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_87, _shape_param_571);  bmm_87 = _shape_param_571 = None
        convert_element_type_611: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_567, torch.float32);  view_567 = None
        permute_236: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_611, [0, 1, 3, 2]);  convert_element_type_611 = None
        convert_element_type_612: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_236, torch.bfloat16);  permute_236 = None
        convert_element_type_613: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_568, torch.bfloat16);  view_568 = None
        permute_237: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_564, [0, 2, 1, 3]);  view_564 = None
        clone_86: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_569: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_86, _shape_param_572);  clone_86 = _shape_param_572 = None
        permute_238: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_612, [0, 2, 1, 3]);  convert_element_type_612 = None
        view_570: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_238, _shape_param_573);  permute_238 = _shape_param_573 = None
        permute_239: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_613, [0, 2, 1, 3]);  convert_element_type_613 = None
        clone_87: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
        view_571: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_87, _shape_param_574);  clone_87 = _shape_param_574 = None
        view_572: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_569, _shape_param_575);  view_569 = _shape_param_575 = None
        permute_240: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_572, [1, 0])
        mm_260: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_240, arg511_1);  permute_240 = None
        mm_261: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_572, arg512_1);  view_572 = arg512_1 = None
        view_573: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_261, _shape_param_576);  mm_261 = _shape_param_576 = None
        convert_element_type_614: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_573, torch.float32);  view_573 = None
        convert_element_type_615: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_260, torch.float32);  mm_260 = None
        clone_88: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_570, memory_format = torch.contiguous_format);  view_570 = None
        view_574: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_88, _shape_param_577);  clone_88 = _shape_param_577 = None
        permute_241: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_574, [1, 0])
        mm_262: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_241, arg511_1);  permute_241 = None
        mm_263: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_574, arg513_1);  view_574 = arg513_1 = None
        view_575: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_263, _shape_param_578);  mm_263 = _shape_param_578 = None
        convert_element_type_616: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_575, torch.float32);  view_575 = None
        add_174: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_614, convert_element_type_616);  convert_element_type_614 = convert_element_type_616 = None
        convert_element_type_617: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_262, torch.float32);  mm_262 = None
        view_576: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_571, _shape_param_579);  view_571 = _shape_param_579 = None
        permute_242: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_576, [1, 0])
        mm_264: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_242, arg511_1);  permute_242 = arg511_1 = None
        mm_265: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_576, arg514_1);  view_576 = arg514_1 = None
        view_577: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_265, _shape_param_580);  mm_265 = _shape_param_580 = None
        convert_element_type_618: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_577, torch.float32);  view_577 = None
        add_175: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_174, convert_element_type_618);  add_174 = convert_element_type_618 = None
        convert_element_type_619: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_264, torch.float32);  mm_264 = None
        mul_552: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_175, arg515_1);  arg515_1 = None
        mul_553: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_552, 2048)
        sum_266: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_552, [2], True)
        mul_554: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_552, arg516_1);  mul_552 = None
        sum_267: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True);  mul_554 = None
        mul_555: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg516_1, sum_267);  sum_267 = None
        sub_113: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_553, sum_266);  mul_553 = sum_266 = None
        sub_114: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_113, mul_555);  sub_113 = mul_555 = None
        mul_556: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg517_1, sub_114);  arg517_1 = sub_114 = None
        mul_557: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_175, arg516_1);  arg516_1 = None
        sum_268: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 1]);  mul_557 = None
        sum_269: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_175, [0, 1]);  add_175 = None
        add_176: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_173, mul_556);  add_173 = mul_556 = None
        convert_element_type_620: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16)
        view_578: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_620, _shape_param_581);  convert_element_type_620 = _shape_param_581 = None
        mm_266: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_578, arg518_1);  arg518_1 = None
        permute_243: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_578, [1, 0])
        mm_267: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_243, arg519_1);  permute_243 = arg519_1 = None
        sum_270: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_578, [0], True, dtype = torch.float32);  view_578 = None
        view_579: "f32[2048]" = torch.ops.aten.view.default(sum_270, _shape_param_582);  sum_270 = _shape_param_582 = None
        convert_element_type_621: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_579, torch.bfloat16);  view_579 = None
        view_580: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_266, _shape_param_583);  mm_266 = _shape_param_583 = None
        convert_element_type_622: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_580, torch.float32);  view_580 = None
        convert_element_type_623: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_267, torch.float32);  mm_267 = None
        convert_element_type_624: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_621, torch.float32);  convert_element_type_621 = None
        view_581: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg520_1, _shape_param_584);  arg520_1 = _shape_param_584 = None
        mul_558: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_581, 0.5)
        mul_559: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_622, mul_558);  mul_558 = None
        convert_element_type_625: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_581, torch.float32)
        pow_45: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_625, 3.0)
        mul_560: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_177: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_581, mul_560);  view_581 = mul_560 = None
        mul_561: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_177, 0.7978845608028654);  add_177 = None
        tanh_22: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_561);  mul_561 = None
        add_178: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_562: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_622, add_178);  convert_element_type_622 = add_178 = None
        convert_element_type_626: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_562, torch.bfloat16);  mul_562 = None
        mul_563: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_115: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_563);  mul_563 = None
        mul_564: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_559, sub_115);  mul_559 = sub_115 = None
        mul_565: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_564, 0.7978845608028654);  mul_564 = None
        convert_element_type_627: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_565, torch.bfloat16)
        mul_566: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_565, 0.044715);  mul_565 = None
        pow_46: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_625, 2.0);  convert_element_type_625 = None
        mul_567: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_568: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_566, mul_567);  mul_566 = mul_567 = None
        convert_element_type_628: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_568, torch.bfloat16);  mul_568 = None
        add_179: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_627, convert_element_type_628);  convert_element_type_627 = convert_element_type_628 = None
        mul_569: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_626, 0.5);  convert_element_type_626 = None
        add_180: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_179, mul_569);  add_179 = mul_569 = None
        view_582: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_180, _shape_param_585);  add_180 = _shape_param_585 = None
        mm_268: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_582, arg521_1);  arg521_1 = None
        permute_244: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_582, [1, 0])
        mm_269: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_244, arg522_1);  permute_244 = arg522_1 = None
        sum_271: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True, dtype = torch.float32);  view_582 = None
        view_583: "f32[8192]" = torch.ops.aten.view.default(sum_271, _shape_param_586);  sum_271 = _shape_param_586 = None
        convert_element_type_629: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_583, torch.bfloat16);  view_583 = None
        view_584: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_268, _shape_param_587);  mm_268 = _shape_param_587 = None
        convert_element_type_630: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_584, torch.float32);  view_584 = None
        convert_element_type_631: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_269, torch.float32);  mm_269 = None
        convert_element_type_632: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_629, torch.float32);  convert_element_type_629 = None
        mul_570: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_630, arg523_1);  arg523_1 = None
        mul_571: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_570, 2048)
        sum_272: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [2], True)
        mul_572: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_570, arg524_1);  mul_570 = None
        sum_273: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [2], True);  mul_572 = None
        mul_573: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg524_1, sum_273);  sum_273 = None
        sub_116: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_571, sum_272);  mul_571 = sum_272 = None
        sub_117: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_116, mul_573);  sub_116 = mul_573 = None
        mul_574: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg525_1, sub_117);  arg525_1 = sub_117 = None
        mul_575: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_630, arg524_1);  arg524_1 = None
        sum_274: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 1]);  mul_575 = None
        sum_275: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_630, [0, 1]);  convert_element_type_630 = None
        add_181: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_176, mul_574);  add_176 = mul_574 = None
        convert_element_type_633: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_181, torch.bfloat16)
        view_585: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_633, _shape_param_588);  convert_element_type_633 = _shape_param_588 = None
        mm_270: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_585, arg526_1);  arg526_1 = None
        permute_245: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_585, [1, 0])
        mm_271: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_245, arg527_1);  permute_245 = arg527_1 = None
        sum_276: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True, dtype = torch.float32);  view_585 = None
        view_586: "f32[2048]" = torch.ops.aten.view.default(sum_276, _shape_param_589);  sum_276 = _shape_param_589 = None
        convert_element_type_634: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_586, torch.bfloat16);  view_586 = None
        view_587: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_270, _shape_param_590);  mm_270 = _shape_param_590 = None
        convert_element_type_635: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_271, torch.float32);  mm_271 = None
        convert_element_type_636: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_634, torch.float32);  convert_element_type_634 = None
        view_588: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_587, _shape_param_591);  view_587 = _shape_param_591 = None
        permute_246: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
        clone_89: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
        view_589: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_89, _shape_param_592);  clone_89 = _shape_param_592 = None
        bmm_88: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg528_1, view_589);  arg528_1 = None
        bmm_89: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_589, arg529_1);  view_589 = arg529_1 = None
        view_590: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_88, _shape_param_593);  bmm_88 = _shape_param_593 = None
        view_591: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_89, _shape_param_594);  bmm_89 = _shape_param_594 = None
        convert_element_type_637: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_591, torch.float32);  view_591 = None
        mul_576: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_637, arg530_1);  convert_element_type_637 = None
        sum_277: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_576, [-1], True)
        neg_22: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(arg530_1);  arg530_1 = None
        fma_22: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_22, sum_277, mul_576);  neg_22 = sum_277 = mul_576 = None
        convert_element_type_638: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None
        slice_47: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg531_1, 2, 0, 128);  arg531_1 = None
        slice_48: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 128);  slice_47 = None
        where_25: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, convert_element_type_638, full_2);  slice_48 = convert_element_type_638 = None
        view_592: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_25, _shape_param_595);  where_25 = _shape_param_595 = None
        bmm_90: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg532_1, view_592);  arg532_1 = None
        bmm_91: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_592, arg533_1);  view_592 = arg533_1 = None
        view_593: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_90, _shape_param_596);  bmm_90 = _shape_param_596 = None
        view_594: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_91, _shape_param_597);  bmm_91 = _shape_param_597 = None
        convert_element_type_639: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_593, torch.float32);  view_593 = None
        permute_247: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_639, [0, 1, 3, 2]);  convert_element_type_639 = None
        convert_element_type_640: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_247, torch.bfloat16);  permute_247 = None
        convert_element_type_641: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_594, torch.bfloat16);  view_594 = None
        permute_248: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_590, [0, 2, 1, 3]);  view_590 = None
        clone_90: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_248, memory_format = torch.contiguous_format);  permute_248 = None
        view_595: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_90, _shape_param_598);  clone_90 = _shape_param_598 = None
        permute_249: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_640, [0, 2, 1, 3]);  convert_element_type_640 = None
        view_596: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_249, _shape_param_599);  permute_249 = _shape_param_599 = None
        permute_250: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_641, [0, 2, 1, 3]);  convert_element_type_641 = None
        clone_91: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_597: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_91, _shape_param_600);  clone_91 = _shape_param_600 = None
        view_598: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_595, _shape_param_601);  view_595 = _shape_param_601 = None
        permute_251: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_598, [1, 0])
        mm_272: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_251, arg534_1);  permute_251 = None
        mm_273: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_598, arg535_1);  view_598 = arg535_1 = None
        view_599: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_273, _shape_param_602);  mm_273 = _shape_param_602 = None
        convert_element_type_642: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_599, torch.float32);  view_599 = None
        convert_element_type_643: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_272, torch.float32);  mm_272 = None
        clone_92: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_596, memory_format = torch.contiguous_format);  view_596 = None
        view_600: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_92, _shape_param_603);  clone_92 = _shape_param_603 = None
        permute_252: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_274: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_252, arg534_1);  permute_252 = None
        mm_275: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_600, arg536_1);  view_600 = arg536_1 = None
        view_601: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_275, _shape_param_604);  mm_275 = _shape_param_604 = None
        convert_element_type_644: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_601, torch.float32);  view_601 = None
        add_182: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_642, convert_element_type_644);  convert_element_type_642 = convert_element_type_644 = None
        convert_element_type_645: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_274, torch.float32);  mm_274 = None
        view_602: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_597, _shape_param_605);  view_597 = _shape_param_605 = None
        permute_253: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_602, [1, 0])
        mm_276: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_253, arg534_1);  permute_253 = arg534_1 = None
        mm_277: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_602, arg537_1);  view_602 = arg537_1 = None
        view_603: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_277, _shape_param_606);  mm_277 = _shape_param_606 = None
        convert_element_type_646: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_603, torch.float32);  view_603 = None
        add_183: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_182, convert_element_type_646);  add_182 = convert_element_type_646 = None
        convert_element_type_647: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_276, torch.float32);  mm_276 = None
        mul_577: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_183, arg538_1);  arg538_1 = None
        mul_578: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_577, 2048)
        sum_278: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_577, [2], True)
        mul_579: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_577, arg539_1);  mul_577 = None
        sum_279: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True);  mul_579 = None
        mul_580: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg539_1, sum_279);  sum_279 = None
        sub_118: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_578, sum_278);  mul_578 = sum_278 = None
        sub_119: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_118, mul_580);  sub_118 = mul_580 = None
        mul_581: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg540_1, sub_119);  arg540_1 = sub_119 = None
        mul_582: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_183, arg539_1);  arg539_1 = None
        sum_280: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_582, [0, 1]);  mul_582 = None
        sum_281: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_183, [0, 1]);  add_183 = None
        add_184: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_181, mul_581);  add_181 = mul_581 = None
        convert_element_type_648: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16)
        view_604: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_648, _shape_param_607);  convert_element_type_648 = _shape_param_607 = None
        mm_278: "bf16[4096, 8192]" = torch.ops.aten.mm.default(view_604, arg541_1);  arg541_1 = None
        permute_254: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_604, [1, 0])
        mm_279: "bf16[2048, 8192]" = torch.ops.aten.mm.default(permute_254, arg542_1);  permute_254 = arg542_1 = None
        sum_282: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_604, [0], True, dtype = torch.float32);  view_604 = None
        view_605: "f32[2048]" = torch.ops.aten.view.default(sum_282, _shape_param_608);  sum_282 = _shape_param_608 = None
        convert_element_type_649: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_605, torch.bfloat16);  view_605 = None
        view_606: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(mm_278, _shape_param_609);  mm_278 = _shape_param_609 = None
        convert_element_type_650: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_606, torch.float32);  view_606 = None
        convert_element_type_651: "f32[2048, 8192]" = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None
        convert_element_type_652: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_649, torch.float32);  convert_element_type_649 = None
        view_607: "bf16[32, 128, 8192]" = torch.ops.aten.view.default(arg543_1, _shape_param_610);  arg543_1 = _shape_param_610 = None
        mul_583: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(view_607, 0.5)
        mul_584: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_650, mul_583);  mul_583 = None
        convert_element_type_653: "f32[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(view_607, torch.float32)
        pow_47: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_653, 3.0)
        mul_585: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_47, 0.044715);  pow_47 = None
        add_185: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(view_607, mul_585);  view_607 = mul_585 = None
        mul_586: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(add_185, 0.7978845608028654);  add_185 = None
        tanh_23: "f32[32, 128, 8192]" = torch.ops.aten.tanh.default(mul_586);  mul_586 = None
        add_186: "f32[32, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_587: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_650, add_186);  convert_element_type_650 = add_186 = None
        convert_element_type_654: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_587, torch.bfloat16);  mul_587 = None
        mul_588: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_120: "f32[32, 128, 8192]" = torch.ops.aten.sub.Tensor(1, mul_588);  mul_588 = None
        mul_589: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_584, sub_120);  mul_584 = sub_120 = None
        mul_590: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_589, 0.7978845608028654);  mul_589 = None
        convert_element_type_655: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_590, torch.bfloat16)
        mul_591: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_590, 0.044715);  mul_590 = None
        pow_48: "f32[32, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_653, 2.0);  convert_element_type_653 = None
        mul_592: "f32[32, 128, 8192]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_593: "f32[32, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_591, mul_592);  mul_591 = mul_592 = None
        convert_element_type_656: "bf16[32, 128, 8192]" = torch.ops.prims.convert_element_type.default(mul_593, torch.bfloat16);  mul_593 = None
        add_187: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(convert_element_type_655, convert_element_type_656);  convert_element_type_655 = convert_element_type_656 = None
        mul_594: "bf16[32, 128, 8192]" = torch.ops.aten.mul.Tensor(convert_element_type_654, 0.5);  convert_element_type_654 = None
        add_188: "bf16[32, 128, 8192]" = torch.ops.aten.add.Tensor(add_187, mul_594);  add_187 = mul_594 = None
        view_608: "bf16[4096, 8192]" = torch.ops.aten.view.default(add_188, _shape_param_611);  add_188 = _shape_param_611 = None
        mm_280: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_608, arg544_1);  arg544_1 = None
        permute_255: "bf16[8192, 4096]" = torch.ops.aten.permute.default(view_608, [1, 0])
        mm_281: "bf16[8192, 2048]" = torch.ops.aten.mm.default(permute_255, arg545_1);  permute_255 = arg545_1 = None
        sum_283: "f32[1, 8192]" = torch.ops.aten.sum.dim_IntList(view_608, [0], True, dtype = torch.float32);  view_608 = None
        view_609: "f32[8192]" = torch.ops.aten.view.default(sum_283, _shape_param_612);  sum_283 = _shape_param_612 = None
        convert_element_type_657: "bf16[8192]" = torch.ops.prims.convert_element_type.default(view_609, torch.bfloat16);  view_609 = None
        view_610: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_280, _shape_param_613);  mm_280 = _shape_param_613 = None
        convert_element_type_658: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_610, torch.float32);  view_610 = None
        convert_element_type_659: "f32[8192, 2048]" = torch.ops.prims.convert_element_type.default(mm_281, torch.float32);  mm_281 = None
        convert_element_type_660: "f32[8192]" = torch.ops.prims.convert_element_type.default(convert_element_type_657, torch.float32);  convert_element_type_657 = None
        mul_595: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_658, arg546_1);  arg546_1 = None
        mul_596: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_595, 2048)
        sum_284: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [2], True)
        mul_597: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_595, arg547_1);  mul_595 = None
        sum_285: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_597, [2], True);  mul_597 = None
        mul_598: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg547_1, sum_285);  sum_285 = None
        sub_121: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_596, sum_284);  mul_596 = sum_284 = None
        sub_122: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_121, mul_598);  sub_121 = mul_598 = None
        mul_599: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(arg548_1, sub_122);  arg548_1 = sub_122 = None
        mul_600: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(convert_element_type_658, arg547_1);  arg547_1 = None
        sum_286: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 1]);  mul_600 = None
        sum_287: "f32[2048]" = torch.ops.aten.sum.dim_IntList(convert_element_type_658, [0, 1]);  convert_element_type_658 = None
        add_189: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_184, mul_599);  add_184 = mul_599 = None
        convert_element_type_661: "bf16[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(add_189, torch.bfloat16)
        view_611: "bf16[4096, 2048]" = torch.ops.aten.view.default(convert_element_type_661, _shape_param_614);  convert_element_type_661 = _shape_param_614 = None
        mm_282: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_611, arg549_1);  arg549_1 = None
        permute_256: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_611, [1, 0])
        mm_283: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_256, arg550_1);  permute_256 = arg550_1 = None
        sum_288: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True, dtype = torch.float32);  view_611 = None
        view_612: "f32[2048]" = torch.ops.aten.view.default(sum_288, _shape_param_615);  sum_288 = _shape_param_615 = None
        convert_element_type_662: "bf16[2048]" = torch.ops.prims.convert_element_type.default(view_612, torch.bfloat16);  view_612 = None
        view_613: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_282, _shape_param_616);  mm_282 = _shape_param_616 = None
        convert_element_type_663: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_283, torch.float32);  mm_283 = None
        convert_element_type_664: "f32[2048]" = torch.ops.prims.convert_element_type.default(convert_element_type_662, torch.float32);  convert_element_type_662 = None
        view_614: "bf16[32, 128, 16, 128]" = torch.ops.aten.view.default(view_613, _shape_param_617);  view_613 = _shape_param_617 = None
        permute_257: "bf16[32, 16, 128, 128]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        clone_93: "bf16[32, 16, 128, 128]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
        view_615: "bf16[512, 128, 128]" = torch.ops.aten.view.default(clone_93, _shape_param_618);  clone_93 = _shape_param_618 = None
        bmm_92: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg551_1, view_615);  arg551_1 = None
        bmm_93: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_615, arg552_1);  view_615 = arg552_1 = None
        view_616: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_92, _shape_param_619);  bmm_92 = _shape_param_619 = None
        view_617: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_93, _shape_param_620);  bmm_93 = _shape_param_620 = None
        convert_element_type_665: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_617, torch.float32);  view_617 = None
        iota_1: "i64[32]" = torch.ops.prims.iota.default(32, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1: "i64[32, 1]" = torch.ops.aten.unsqueeze.default(iota_1, 1);  iota_1 = None
        unsqueeze_2: "i64[32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 2);  unsqueeze_1 = None
        unsqueeze_3: "i64[32, 1, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 3);  unsqueeze_2 = None
        unsqueeze_4: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(arg553_1, 1)
        unsqueeze_5: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3)
        unsqueeze_6: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        full_3: "b8[]" = torch.ops.aten.full.default([], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_6, unsqueeze_5)
        bitwise_and: "b8[1, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(full_3, le);  full_3 = le = None
        index: "i64[32, 1, 128, 1]" = torch.ops.aten.index.Tensor(arg554_1, [unsqueeze_3, unsqueeze_5]);  unsqueeze_5 = None
        index_1: "i64[32, 1, 1, 128]" = torch.ops.aten.index.Tensor(arg554_1, [unsqueeze_3, unsqueeze_6]);  arg554_1 = unsqueeze_3 = unsqueeze_6 = None
        eq_1: "b8[32, 1, 128, 128]" = torch.ops.aten.eq.Tensor(index, index_1);  index = index_1 = None
        bitwise_and_1: "b8[32, 1, 128, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, eq_1);  bitwise_and = eq_1 = None
        expand_1: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(bitwise_and_1, _shape_param_621);  bitwise_and_1 = _shape_param_621 = None
        full_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand_1, full_1, full_4);  expand_1 = full_1 = full_4 = None
        view_618: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(arg555_1, _shape_param_622);  arg555_1 = _shape_param_622 = None
        slice_49: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg556_1, 2, 0, 128);  arg556_1 = None
        slice_50: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_49, 3, 0, 128);  slice_49 = None
        full_5: "bf16[]" = torch.ops.aten.full.default([], -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_50, view_618, full_5);  view_618 = full_5 = None
        add_190: "f32[32, 16, 128, 128]" = torch.ops.aten.add.Tensor(where_27, where_26);  where_27 = where_26 = None
        sub_123: "f32[32, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add_190, arg557_1);  add_190 = arg557_1 = None
        exp_1: "f32[32, 16, 128, 128]" = torch.ops.aten.exp.default(sub_123);  sub_123 = None
        div_1: "f32[32, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, arg558_1);  exp_1 = arg558_1 = None
        mul_601: "f32[32, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_665, div_1);  convert_element_type_665 = None
        sum_289: "f32[32, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [-1], True)
        neg_23: "f32[32, 16, 128, 128]" = torch.ops.aten.neg.default(div_1);  div_1 = None
        fma_23: "f32[32, 16, 128, 128]" = torch.ops.prims.fma.default(neg_23, sum_289, mul_601);  neg_23 = sum_289 = mul_601 = None
        convert_element_type_666: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None
        where_28: "bf16[32, 16, 128, 128]" = torch.ops.aten.where.self(slice_50, convert_element_type_666, full_2);  slice_50 = convert_element_type_666 = full_2 = None
        view_619: "bf16[512, 128, 128]" = torch.ops.aten.view.default(where_28, _shape_param_623);  where_28 = _shape_param_623 = None
        bmm_94: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(arg559_1, view_619);  arg559_1 = None
        bmm_95: "bf16[512, 128, 128]" = torch.ops.aten.bmm.default(view_619, arg560_1);  view_619 = arg560_1 = None
        view_620: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_94, _shape_param_624);  bmm_94 = _shape_param_624 = None
        view_621: "bf16[32, 16, 128, 128]" = torch.ops.aten.view.default(bmm_95, _shape_param_625);  bmm_95 = _shape_param_625 = None
        convert_element_type_667: "f32[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_620, torch.float32);  view_620 = None
        permute_258: "f32[32, 16, 128, 128]" = torch.ops.aten.permute.default(convert_element_type_667, [0, 1, 3, 2]);  convert_element_type_667 = None
        convert_element_type_668: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(permute_258, torch.bfloat16);  permute_258 = None
        convert_element_type_669: "bf16[32, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_621, torch.bfloat16);  view_621 = None
        permute_259: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(view_616, [0, 2, 1, 3]);  view_616 = None
        clone_94: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
        view_622: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_94, _shape_param_626);  clone_94 = _shape_param_626 = None
        permute_260: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_668, [0, 2, 1, 3]);  convert_element_type_668 = None
        view_623: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(permute_260, _shape_param_627);  permute_260 = _shape_param_627 = None
        permute_261: "bf16[32, 128, 16, 128]" = torch.ops.aten.permute.default(convert_element_type_669, [0, 2, 1, 3]);  convert_element_type_669 = None
        clone_95: "bf16[32, 128, 16, 128]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
        view_624: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(clone_95, _shape_param_628);  clone_95 = _shape_param_628 = None
        view_625: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_622, _shape_param_629);  view_622 = _shape_param_629 = None
        permute_262: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_284: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_262, arg561_1);  permute_262 = None
        mm_285: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_625, arg562_1);  view_625 = arg562_1 = None
        view_626: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_285, _shape_param_630);  mm_285 = _shape_param_630 = None
        convert_element_type_670: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_626, torch.float32);  view_626 = None
        convert_element_type_671: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_284, torch.float32);  mm_284 = None
        clone_96: "bf16[32, 128, 2048]" = torch.ops.aten.clone.default(view_623, memory_format = torch.contiguous_format);  view_623 = None
        view_627: "bf16[4096, 2048]" = torch.ops.aten.view.default(clone_96, _shape_param_631);  clone_96 = _shape_param_631 = None
        permute_263: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_627, [1, 0])
        mm_286: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_263, arg561_1);  permute_263 = None
        mm_287: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_627, arg563_1);  view_627 = arg563_1 = None
        view_628: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_287, _shape_param_632);  mm_287 = _shape_param_632 = None
        convert_element_type_672: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_628, torch.float32);  view_628 = None
        add_191: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_670, convert_element_type_672);  convert_element_type_670 = convert_element_type_672 = None
        convert_element_type_673: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_286, torch.float32);  mm_286 = None
        view_629: "bf16[4096, 2048]" = torch.ops.aten.view.default(view_624, _shape_param_633);  view_624 = _shape_param_633 = None
        permute_264: "bf16[2048, 4096]" = torch.ops.aten.permute.default(view_629, [1, 0])
        mm_288: "bf16[2048, 2048]" = torch.ops.aten.mm.default(permute_264, arg561_1);  permute_264 = arg561_1 = None
        mm_289: "bf16[4096, 2048]" = torch.ops.aten.mm.default(view_629, arg564_1);  view_629 = arg564_1 = None
        view_630: "bf16[32, 128, 2048]" = torch.ops.aten.view.default(mm_289, _shape_param_634);  mm_289 = _shape_param_634 = None
        convert_element_type_674: "f32[32, 128, 2048]" = torch.ops.prims.convert_element_type.default(view_630, torch.float32);  view_630 = None
        add_192: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_191, convert_element_type_674);  add_191 = convert_element_type_674 = None
        convert_element_type_675: "f32[2048, 2048]" = torch.ops.prims.convert_element_type.default(mm_288, torch.float32);  mm_288 = None
        mul_602: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_192, arg565_1);  arg565_1 = None
        mul_603: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_602, 2048)
        sum_290: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True)
        add_193: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(arg566_1, arg567_1);  arg566_1 = arg567_1 = None
        sub_124: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(add_193, arg568_1);  add_193 = arg568_1 = None
        mul_604: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_124, arg569_1);  sub_124 = None
        mul_605: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_602, mul_604);  mul_602 = None
        sum_291: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_605, [2], True);  mul_605 = None
        mul_606: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_604, sum_291);  sum_291 = None
        sub_125: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(mul_603, sum_290);  mul_603 = sum_290 = None
        sub_126: "f32[32, 128, 2048]" = torch.ops.aten.sub.Tensor(sub_125, mul_606);  sub_125 = mul_606 = None
        div_2: "f32[32, 128, 1]" = torch.ops.aten.div.Tensor(arg569_1, 2048);  arg569_1 = None
        mul_607: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(div_2, sub_126);  div_2 = sub_126 = None
        mul_608: "f32[32, 128, 2048]" = torch.ops.aten.mul.Tensor(add_192, mul_604);  mul_604 = None
        sum_292: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 1]);  mul_608 = None
        sum_293: "f32[2048]" = torch.ops.aten.sum.dim_IntList(add_192, [0, 1]);  add_192 = None
        add_194: "f32[32, 128, 2048]" = torch.ops.aten.add.Tensor(add_189, mul_607);  add_189 = mul_607 = None
        sum_294: "f32[1, 128, 2048]" = torch.ops.aten.sum.dim_IntList(add_194, [0], True, dtype = torch.float32)
        full_6: "b8[1, 128, 1]" = torch.ops.aten.full.default(_shape_param_635, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_635 = None
        full_7: "f32[2048, 2048]" = torch.ops.aten.full.default(_shape_param_636, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_636 = None
        _unsafe_masked_index_put_accumulate: "f32[2048, 2048]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_7, full_6, [arg553_1], sum_294);  full_7 = full_6 = arg553_1 = sum_294 = None
        ge: "b8[32, 128]" = torch.ops.aten.ge.Scalar(arg570_1, 0)
        lt: "b8[32, 128]" = torch.ops.aten.lt.Scalar(arg570_1, 50257)
        bitwise_and_2: "b8[32, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_1: "b8[32, 128]" = torch.ops.aten.ne.Scalar(arg570_1, -1)
        bitwise_and_3: "b8[32, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_2, ne_1);  bitwise_and_2 = ne_1 = None
        unsqueeze_7: "b8[32, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_3, -1);  bitwise_and_3 = None
        full_8: "f32[50257, 2048]" = torch.ops.aten.full.default(_shape_param_637, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_637 = None
        _unsafe_masked_index_put_accumulate_1: "f32[50257, 2048]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_8, unsqueeze_7, [arg570_1], add_194);  full_8 = unsqueeze_7 = arg570_1 = add_194 = None
        add_195: "f32[50257, 2048]" = torch.ops.aten.add.Tensor(convert_element_type_3, _unsafe_masked_index_put_accumulate_1);  convert_element_type_3 = _unsafe_masked_index_put_accumulate_1 = None
        return (sum_4, sum_5, convert_element_type_7, convert_element_type_8, convert_element_type_15, convert_element_type_16, sum_10, sum_11, convert_element_type_19, convert_element_type_20, convert_element_type_27, convert_element_type_29, convert_element_type_31, sum_16, sum_17, convert_element_type_35, convert_element_type_36, convert_element_type_43, convert_element_type_44, sum_22, sum_23, convert_element_type_47, convert_element_type_48, convert_element_type_55, convert_element_type_57, convert_element_type_59, sum_28, sum_29, convert_element_type_63, convert_element_type_64, convert_element_type_71, convert_element_type_72, sum_34, sum_35, convert_element_type_75, convert_element_type_76, convert_element_type_83, convert_element_type_85, convert_element_type_87, sum_40, sum_41, convert_element_type_91, convert_element_type_92, convert_element_type_99, convert_element_type_100, sum_46, sum_47, convert_element_type_103, convert_element_type_104, convert_element_type_111, convert_element_type_113, convert_element_type_115, sum_52, sum_53, convert_element_type_119, convert_element_type_120, convert_element_type_127, convert_element_type_128, sum_58, sum_59, convert_element_type_131, convert_element_type_132, convert_element_type_139, convert_element_type_141, convert_element_type_143, sum_64, sum_65, convert_element_type_147, convert_element_type_148, convert_element_type_155, convert_element_type_156, sum_70, sum_71, convert_element_type_159, convert_element_type_160, convert_element_type_167, convert_element_type_169, convert_element_type_171, sum_76, sum_77, convert_element_type_175, convert_element_type_176, convert_element_type_183, convert_element_type_184, sum_82, sum_83, convert_element_type_187, convert_element_type_188, convert_element_type_195, convert_element_type_197, convert_element_type_199, sum_88, sum_89, convert_element_type_203, convert_element_type_204, convert_element_type_211, convert_element_type_212, sum_94, sum_95, convert_element_type_215, convert_element_type_216, convert_element_type_223, convert_element_type_225, convert_element_type_227, sum_100, sum_101, convert_element_type_231, convert_element_type_232, convert_element_type_239, convert_element_type_240, sum_106, sum_107, convert_element_type_243, convert_element_type_244, convert_element_type_251, convert_element_type_253, convert_element_type_255, sum_112, sum_113, convert_element_type_259, convert_element_type_260, convert_element_type_267, convert_element_type_268, sum_118, sum_119, convert_element_type_271, convert_element_type_272, convert_element_type_279, convert_element_type_281, convert_element_type_283, sum_124, sum_125, convert_element_type_287, convert_element_type_288, convert_element_type_295, convert_element_type_296, sum_130, sum_131, convert_element_type_299, convert_element_type_300, convert_element_type_307, convert_element_type_309, convert_element_type_311, sum_136, sum_137, convert_element_type_315, convert_element_type_316, convert_element_type_323, convert_element_type_324, sum_142, sum_143, convert_element_type_327, convert_element_type_328, convert_element_type_335, convert_element_type_337, convert_element_type_339, sum_148, sum_149, convert_element_type_343, convert_element_type_344, convert_element_type_351, convert_element_type_352, sum_154, sum_155, convert_element_type_355, convert_element_type_356, convert_element_type_363, convert_element_type_365, convert_element_type_367, sum_160, sum_161, convert_element_type_371, convert_element_type_372, convert_element_type_379, convert_element_type_380, sum_166, sum_167, convert_element_type_383, convert_element_type_384, convert_element_type_391, convert_element_type_393, convert_element_type_395, sum_172, sum_173, convert_element_type_399, convert_element_type_400, convert_element_type_407, convert_element_type_408, sum_178, sum_179, convert_element_type_411, convert_element_type_412, convert_element_type_419, convert_element_type_421, convert_element_type_423, sum_184, sum_185, convert_element_type_427, convert_element_type_428, convert_element_type_435, convert_element_type_436, sum_190, sum_191, convert_element_type_439, convert_element_type_440, convert_element_type_447, convert_element_type_449, convert_element_type_451, sum_196, sum_197, convert_element_type_455, convert_element_type_456, convert_element_type_463, convert_element_type_464, sum_202, sum_203, convert_element_type_467, convert_element_type_468, convert_element_type_475, convert_element_type_477, convert_element_type_479, sum_208, sum_209, convert_element_type_483, convert_element_type_484, convert_element_type_491, convert_element_type_492, sum_214, sum_215, convert_element_type_495, convert_element_type_496, convert_element_type_503, convert_element_type_505, convert_element_type_507, sum_220, sum_221, convert_element_type_511, convert_element_type_512, convert_element_type_519, convert_element_type_520, sum_226, sum_227, convert_element_type_523, convert_element_type_524, convert_element_type_531, convert_element_type_533, convert_element_type_535, sum_232, sum_233, convert_element_type_539, convert_element_type_540, convert_element_type_547, convert_element_type_548, sum_238, sum_239, convert_element_type_551, convert_element_type_552, convert_element_type_559, convert_element_type_561, convert_element_type_563, sum_244, sum_245, convert_element_type_567, convert_element_type_568, convert_element_type_575, convert_element_type_576, sum_250, sum_251, convert_element_type_579, convert_element_type_580, convert_element_type_587, convert_element_type_589, convert_element_type_591, sum_256, sum_257, convert_element_type_595, convert_element_type_596, convert_element_type_603, convert_element_type_604, sum_262, sum_263, convert_element_type_607, convert_element_type_608, convert_element_type_615, convert_element_type_617, convert_element_type_619, sum_268, sum_269, convert_element_type_623, convert_element_type_624, convert_element_type_631, convert_element_type_632, sum_274, sum_275, convert_element_type_635, convert_element_type_636, convert_element_type_643, convert_element_type_645, convert_element_type_647, sum_280, sum_281, convert_element_type_651, convert_element_type_652, convert_element_type_659, convert_element_type_660, sum_286, sum_287, convert_element_type_663, convert_element_type_664, convert_element_type_671, convert_element_type_673, convert_element_type_675, sum_292, sum_293, _unsafe_masked_index_put_accumulate, add_195)



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
