"""
Standalone repro captured via capture_hook.
Label: GPTJForCausalLM
Pattern hash: 06520c3adb35
Shape hash: 4cd3ae57
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
    def forward(self, arg0_1: "f32[]", arg1_1: "f32[]", arg2_1: "i64[1, 129]", arg3_1: "bf16[1, 128, 50400]", arg4_1: "f32[128, 1]", arg5_1: "f32[128, 1]", arg6_1: "bf16[1, 128, 50400]", arg7_1: "bf16[50400, 4096]", arg8_1: "bf16[128, 4096]", arg9_1: "f32[4096]", arg10_1: "f32[1, 128, 4096]", arg11_1: "f32[1, 128, 1]", arg12_1: "bf16[4096, 16384]", arg13_1: "bf16[128, 16384]", arg14_1: "bf16[128, 16384]", arg15_1: "bf16[16384, 4096]", arg16_1: "bf16[128, 4096]", arg17_1: "bf16[128, 4096]", arg18_1: "bf16[4096, 4096]", arg19_1: "bf16[16, 128, 128]", arg20_1: "bf16[16, 256, 128]", arg21_1: "f32[1, 16, 128, 128]", arg22_1: "bf16[16, 256, 128]", arg23_1: "bf16[16, 128, 256]", arg24_1: "bf16[1, 128, 1, 32, 1]", arg25_1: "bf16[1, 128, 1, 32, 1]", arg26_1: "bf16[4096, 4096]", arg27_1: "bf16[4096, 4096]", arg28_1: "bf16[4096, 4096]", arg29_1: "f32[4096]", arg30_1: "f32[1, 128, 4096]", arg31_1: "f32[1, 128, 1]", arg32_1: "bf16[4096, 16384]", arg33_1: "bf16[128, 16384]", arg34_1: "bf16[128, 16384]", arg35_1: "bf16[16384, 4096]", arg36_1: "bf16[128, 4096]", arg37_1: "bf16[128, 4096]", arg38_1: "bf16[4096, 4096]", arg39_1: "bf16[16, 128, 128]", arg40_1: "bf16[16, 256, 128]", arg41_1: "f32[1, 16, 128, 128]", arg42_1: "bf16[16, 256, 128]", arg43_1: "bf16[16, 128, 256]", arg44_1: "bf16[1, 128, 1, 32, 1]", arg45_1: "bf16[1, 128, 1, 32, 1]", arg46_1: "bf16[4096, 4096]", arg47_1: "bf16[4096, 4096]", arg48_1: "bf16[4096, 4096]", arg49_1: "f32[4096]", arg50_1: "f32[1, 128, 4096]", arg51_1: "f32[1, 128, 1]", arg52_1: "bf16[4096, 16384]", arg53_1: "bf16[128, 16384]", arg54_1: "bf16[128, 16384]", arg55_1: "bf16[16384, 4096]", arg56_1: "bf16[128, 4096]", arg57_1: "bf16[128, 4096]", arg58_1: "bf16[4096, 4096]", arg59_1: "bf16[16, 128, 128]", arg60_1: "bf16[16, 256, 128]", arg61_1: "f32[1, 16, 128, 128]", arg62_1: "bf16[16, 256, 128]", arg63_1: "bf16[16, 128, 256]", arg64_1: "bf16[1, 128, 1, 32, 1]", arg65_1: "bf16[1, 128, 1, 32, 1]", arg66_1: "bf16[4096, 4096]", arg67_1: "bf16[4096, 4096]", arg68_1: "bf16[4096, 4096]", arg69_1: "f32[4096]", arg70_1: "f32[1, 128, 4096]", arg71_1: "f32[1, 128, 1]", arg72_1: "bf16[4096, 16384]", arg73_1: "bf16[128, 16384]", arg74_1: "bf16[128, 16384]", arg75_1: "bf16[16384, 4096]", arg76_1: "bf16[128, 4096]", arg77_1: "bf16[128, 4096]", arg78_1: "bf16[4096, 4096]", arg79_1: "bf16[16, 128, 128]", arg80_1: "bf16[16, 256, 128]", arg81_1: "f32[1, 16, 128, 128]", arg82_1: "bf16[16, 256, 128]", arg83_1: "bf16[16, 128, 256]", arg84_1: "bf16[1, 128, 1, 32, 1]", arg85_1: "bf16[1, 128, 1, 32, 1]", arg86_1: "bf16[4096, 4096]", arg87_1: "bf16[4096, 4096]", arg88_1: "bf16[4096, 4096]", arg89_1: "f32[4096]", arg90_1: "f32[1, 128, 4096]", arg91_1: "f32[1, 128, 1]", arg92_1: "bf16[4096, 16384]", arg93_1: "bf16[128, 16384]", arg94_1: "bf16[128, 16384]", arg95_1: "bf16[16384, 4096]", arg96_1: "bf16[128, 4096]", arg97_1: "bf16[128, 4096]", arg98_1: "bf16[4096, 4096]", arg99_1: "bf16[16, 128, 128]", arg100_1: "bf16[16, 256, 128]", arg101_1: "f32[1, 16, 128, 128]", arg102_1: "bf16[16, 256, 128]", arg103_1: "bf16[16, 128, 256]", arg104_1: "bf16[1, 128, 1, 32, 1]", arg105_1: "bf16[1, 128, 1, 32, 1]", arg106_1: "bf16[4096, 4096]", arg107_1: "bf16[4096, 4096]", arg108_1: "bf16[4096, 4096]", arg109_1: "f32[4096]", arg110_1: "f32[1, 128, 4096]", arg111_1: "f32[1, 128, 1]", arg112_1: "bf16[4096, 16384]", arg113_1: "bf16[128, 16384]", arg114_1: "bf16[128, 16384]", arg115_1: "bf16[16384, 4096]", arg116_1: "bf16[128, 4096]", arg117_1: "bf16[128, 4096]", arg118_1: "bf16[4096, 4096]", arg119_1: "bf16[16, 128, 128]", arg120_1: "bf16[16, 256, 128]", arg121_1: "f32[1, 16, 128, 128]", arg122_1: "bf16[16, 256, 128]", arg123_1: "bf16[16, 128, 256]", arg124_1: "bf16[1, 128, 1, 32, 1]", arg125_1: "bf16[1, 128, 1, 32, 1]", arg126_1: "bf16[4096, 4096]", arg127_1: "bf16[4096, 4096]", arg128_1: "bf16[4096, 4096]", arg129_1: "f32[4096]", arg130_1: "f32[1, 128, 4096]", arg131_1: "f32[1, 128, 1]", arg132_1: "bf16[4096, 16384]", arg133_1: "bf16[128, 16384]", arg134_1: "bf16[128, 16384]", arg135_1: "bf16[16384, 4096]", arg136_1: "bf16[128, 4096]", arg137_1: "bf16[128, 4096]", arg138_1: "bf16[4096, 4096]", arg139_1: "bf16[16, 128, 128]", arg140_1: "bf16[16, 256, 128]", arg141_1: "f32[1, 16, 128, 128]", arg142_1: "bf16[16, 256, 128]", arg143_1: "bf16[16, 128, 256]", arg144_1: "bf16[1, 128, 1, 32, 1]", arg145_1: "bf16[1, 128, 1, 32, 1]", arg146_1: "bf16[4096, 4096]", arg147_1: "bf16[4096, 4096]", arg148_1: "bf16[4096, 4096]", arg149_1: "f32[4096]", arg150_1: "f32[1, 128, 4096]", arg151_1: "f32[1, 128, 1]", arg152_1: "bf16[4096, 16384]", arg153_1: "bf16[128, 16384]", arg154_1: "bf16[128, 16384]", arg155_1: "bf16[16384, 4096]", arg156_1: "bf16[128, 4096]", arg157_1: "bf16[128, 4096]", arg158_1: "bf16[4096, 4096]", arg159_1: "bf16[16, 128, 128]", arg160_1: "bf16[16, 256, 128]", arg161_1: "f32[1, 16, 128, 128]", arg162_1: "bf16[16, 256, 128]", arg163_1: "bf16[16, 128, 256]", arg164_1: "bf16[1, 128, 1, 32, 1]", arg165_1: "bf16[1, 128, 1, 32, 1]", arg166_1: "bf16[4096, 4096]", arg167_1: "bf16[4096, 4096]", arg168_1: "bf16[4096, 4096]", arg169_1: "f32[4096]", arg170_1: "f32[1, 128, 4096]", arg171_1: "f32[1, 128, 1]", arg172_1: "bf16[4096, 16384]", arg173_1: "bf16[128, 16384]", arg174_1: "bf16[128, 16384]", arg175_1: "bf16[16384, 4096]", arg176_1: "bf16[128, 4096]", arg177_1: "bf16[128, 4096]", arg178_1: "bf16[4096, 4096]", arg179_1: "bf16[16, 128, 128]", arg180_1: "bf16[16, 256, 128]", arg181_1: "f32[1, 16, 128, 128]", arg182_1: "bf16[16, 256, 128]", arg183_1: "bf16[16, 128, 256]", arg184_1: "bf16[1, 128, 1, 32, 1]", arg185_1: "bf16[1, 128, 1, 32, 1]", arg186_1: "bf16[4096, 4096]", arg187_1: "bf16[4096, 4096]", arg188_1: "bf16[4096, 4096]", arg189_1: "f32[4096]", arg190_1: "f32[1, 128, 4096]", arg191_1: "f32[1, 128, 1]", arg192_1: "bf16[4096, 16384]", arg193_1: "bf16[128, 16384]", arg194_1: "bf16[128, 16384]", arg195_1: "bf16[16384, 4096]", arg196_1: "bf16[128, 4096]", arg197_1: "bf16[128, 4096]", arg198_1: "bf16[4096, 4096]", arg199_1: "bf16[16, 128, 128]", arg200_1: "bf16[16, 256, 128]", arg201_1: "f32[1, 16, 128, 128]", arg202_1: "bf16[16, 256, 128]", arg203_1: "bf16[16, 128, 256]", arg204_1: "bf16[1, 128, 1, 32, 1]", arg205_1: "bf16[1, 128, 1, 32, 1]", arg206_1: "bf16[4096, 4096]", arg207_1: "bf16[4096, 4096]", arg208_1: "bf16[4096, 4096]", arg209_1: "f32[4096]", arg210_1: "f32[1, 128, 4096]", arg211_1: "f32[1, 128, 1]", arg212_1: "bf16[4096, 16384]", arg213_1: "bf16[128, 16384]", arg214_1: "bf16[128, 16384]", arg215_1: "bf16[16384, 4096]", arg216_1: "bf16[128, 4096]", arg217_1: "bf16[128, 4096]", arg218_1: "bf16[4096, 4096]", arg219_1: "bf16[16, 128, 128]", arg220_1: "bf16[16, 256, 128]", arg221_1: "f32[1, 16, 128, 128]", arg222_1: "bf16[16, 256, 128]", arg223_1: "bf16[16, 128, 256]", arg224_1: "bf16[1, 128, 1, 32, 1]", arg225_1: "bf16[1, 128, 1, 32, 1]", arg226_1: "bf16[4096, 4096]", arg227_1: "bf16[4096, 4096]", arg228_1: "bf16[4096, 4096]", arg229_1: "f32[4096]", arg230_1: "f32[1, 128, 4096]", arg231_1: "f32[1, 128, 1]", arg232_1: "bf16[4096, 16384]", arg233_1: "bf16[128, 16384]", arg234_1: "bf16[128, 16384]", arg235_1: "bf16[16384, 4096]", arg236_1: "bf16[128, 4096]", arg237_1: "bf16[128, 4096]", arg238_1: "bf16[4096, 4096]", arg239_1: "bf16[16, 128, 128]", arg240_1: "bf16[16, 256, 128]", arg241_1: "f32[1, 16, 128, 128]", arg242_1: "bf16[16, 256, 128]", arg243_1: "bf16[16, 128, 256]", arg244_1: "bf16[1, 128, 1, 32, 1]", arg245_1: "bf16[1, 128, 1, 32, 1]", arg246_1: "bf16[4096, 4096]", arg247_1: "bf16[4096, 4096]", arg248_1: "bf16[4096, 4096]", arg249_1: "f32[4096]", arg250_1: "f32[1, 128, 4096]", arg251_1: "f32[1, 128, 1]", arg252_1: "bf16[4096, 16384]", arg253_1: "bf16[128, 16384]", arg254_1: "bf16[128, 16384]", arg255_1: "bf16[16384, 4096]", arg256_1: "bf16[128, 4096]", arg257_1: "bf16[128, 4096]", arg258_1: "bf16[4096, 4096]", arg259_1: "bf16[16, 128, 128]", arg260_1: "bf16[16, 256, 128]", arg261_1: "f32[1, 16, 128, 128]", arg262_1: "bf16[16, 256, 128]", arg263_1: "bf16[16, 128, 256]", arg264_1: "bf16[1, 128, 1, 32, 1]", arg265_1: "bf16[1, 128, 1, 32, 1]", arg266_1: "bf16[4096, 4096]", arg267_1: "bf16[4096, 4096]", arg268_1: "bf16[4096, 4096]", arg269_1: "f32[4096]", arg270_1: "f32[1, 128, 4096]", arg271_1: "f32[1, 128, 1]", arg272_1: "bf16[4096, 16384]", arg273_1: "bf16[128, 16384]", arg274_1: "bf16[128, 16384]", arg275_1: "bf16[16384, 4096]", arg276_1: "bf16[128, 4096]", arg277_1: "bf16[128, 4096]", arg278_1: "bf16[4096, 4096]", arg279_1: "bf16[16, 128, 128]", arg280_1: "bf16[16, 256, 128]", arg281_1: "f32[1, 16, 128, 128]", arg282_1: "bf16[16, 256, 128]", arg283_1: "bf16[16, 128, 256]", arg284_1: "bf16[1, 128, 1, 32, 1]", arg285_1: "bf16[1, 128, 1, 32, 1]", arg286_1: "bf16[4096, 4096]", arg287_1: "bf16[4096, 4096]", arg288_1: "bf16[4096, 4096]", arg289_1: "f32[4096]", arg290_1: "f32[1, 128, 4096]", arg291_1: "f32[1, 128, 1]", arg292_1: "bf16[4096, 16384]", arg293_1: "bf16[128, 16384]", arg294_1: "bf16[128, 16384]", arg295_1: "bf16[16384, 4096]", arg296_1: "bf16[128, 4096]", arg297_1: "bf16[128, 4096]", arg298_1: "bf16[4096, 4096]", arg299_1: "bf16[16, 128, 128]", arg300_1: "bf16[16, 256, 128]", arg301_1: "f32[1, 16, 128, 128]", arg302_1: "bf16[16, 256, 128]", arg303_1: "bf16[16, 128, 256]", arg304_1: "bf16[1, 128, 1, 32, 1]", arg305_1: "bf16[1, 128, 1, 32, 1]", arg306_1: "bf16[4096, 4096]", arg307_1: "bf16[4096, 4096]", arg308_1: "bf16[4096, 4096]", arg309_1: "f32[4096]", arg310_1: "f32[1, 128, 4096]", arg311_1: "f32[1, 128, 1]", arg312_1: "bf16[4096, 16384]", arg313_1: "bf16[128, 16384]", arg314_1: "bf16[128, 16384]", arg315_1: "bf16[16384, 4096]", arg316_1: "bf16[128, 4096]", arg317_1: "bf16[128, 4096]", arg318_1: "bf16[4096, 4096]", arg319_1: "bf16[16, 128, 128]", arg320_1: "bf16[16, 256, 128]", arg321_1: "f32[1, 16, 128, 128]", arg322_1: "bf16[16, 256, 128]", arg323_1: "bf16[16, 128, 256]", arg324_1: "bf16[1, 128, 1, 32, 1]", arg325_1: "bf16[1, 128, 1, 32, 1]", arg326_1: "bf16[4096, 4096]", arg327_1: "bf16[4096, 4096]", arg328_1: "bf16[4096, 4096]", arg329_1: "f32[4096]", arg330_1: "f32[1, 128, 4096]", arg331_1: "f32[1, 128, 1]", arg332_1: "bf16[4096, 16384]", arg333_1: "bf16[128, 16384]", arg334_1: "bf16[128, 16384]", arg335_1: "bf16[16384, 4096]", arg336_1: "bf16[128, 4096]", arg337_1: "bf16[128, 4096]", arg338_1: "bf16[4096, 4096]", arg339_1: "bf16[16, 128, 128]", arg340_1: "bf16[16, 256, 128]", arg341_1: "f32[1, 16, 128, 128]", arg342_1: "bf16[16, 256, 128]", arg343_1: "bf16[16, 128, 256]", arg344_1: "bf16[1, 128, 1, 32, 1]", arg345_1: "bf16[1, 128, 1, 32, 1]", arg346_1: "bf16[4096, 4096]", arg347_1: "bf16[4096, 4096]", arg348_1: "bf16[4096, 4096]", arg349_1: "f32[4096]", arg350_1: "f32[1, 128, 4096]", arg351_1: "f32[1, 128, 1]", arg352_1: "bf16[4096, 16384]", arg353_1: "bf16[128, 16384]", arg354_1: "bf16[128, 16384]", arg355_1: "bf16[16384, 4096]", arg356_1: "bf16[128, 4096]", arg357_1: "bf16[128, 4096]", arg358_1: "bf16[4096, 4096]", arg359_1: "bf16[16, 128, 128]", arg360_1: "bf16[16, 256, 128]", arg361_1: "f32[1, 16, 128, 128]", arg362_1: "bf16[16, 256, 128]", arg363_1: "bf16[16, 128, 256]", arg364_1: "bf16[1, 128, 1, 32, 1]", arg365_1: "bf16[1, 128, 1, 32, 1]", arg366_1: "bf16[4096, 4096]", arg367_1: "bf16[4096, 4096]", arg368_1: "bf16[4096, 4096]", arg369_1: "f32[4096]", arg370_1: "f32[1, 128, 4096]", arg371_1: "f32[1, 128, 1]", arg372_1: "bf16[4096, 16384]", arg373_1: "bf16[128, 16384]", arg374_1: "bf16[128, 16384]", arg375_1: "bf16[16384, 4096]", arg376_1: "bf16[128, 4096]", arg377_1: "bf16[128, 4096]", arg378_1: "bf16[4096, 4096]", arg379_1: "bf16[16, 128, 128]", arg380_1: "bf16[16, 256, 128]", arg381_1: "f32[1, 16, 128, 128]", arg382_1: "bf16[16, 256, 128]", arg383_1: "bf16[16, 128, 256]", arg384_1: "bf16[1, 128, 1, 32, 1]", arg385_1: "bf16[1, 128, 1, 32, 1]", arg386_1: "bf16[4096, 4096]", arg387_1: "bf16[4096, 4096]", arg388_1: "bf16[4096, 4096]", arg389_1: "f32[4096]", arg390_1: "f32[1, 128, 4096]", arg391_1: "f32[1, 128, 1]", arg392_1: "bf16[4096, 16384]", arg393_1: "bf16[128, 16384]", arg394_1: "bf16[128, 16384]", arg395_1: "bf16[16384, 4096]", arg396_1: "bf16[128, 4096]", arg397_1: "bf16[128, 4096]", arg398_1: "bf16[4096, 4096]", arg399_1: "bf16[16, 128, 128]", arg400_1: "bf16[16, 256, 128]", arg401_1: "f32[1, 16, 128, 128]", arg402_1: "bf16[16, 256, 128]", arg403_1: "bf16[16, 128, 256]", arg404_1: "bf16[1, 128, 1, 32, 1]", arg405_1: "bf16[1, 128, 1, 32, 1]", arg406_1: "bf16[4096, 4096]", arg407_1: "bf16[4096, 4096]", arg408_1: "bf16[4096, 4096]", arg409_1: "f32[4096]", arg410_1: "f32[1, 128, 4096]", arg411_1: "f32[1, 128, 1]", arg412_1: "bf16[4096, 16384]", arg413_1: "bf16[128, 16384]", arg414_1: "bf16[128, 16384]", arg415_1: "bf16[16384, 4096]", arg416_1: "bf16[128, 4096]", arg417_1: "bf16[128, 4096]", arg418_1: "bf16[4096, 4096]", arg419_1: "bf16[16, 128, 128]", arg420_1: "bf16[16, 256, 128]", arg421_1: "f32[1, 16, 128, 128]", arg422_1: "bf16[16, 256, 128]", arg423_1: "bf16[16, 128, 256]", arg424_1: "bf16[1, 128, 1, 32, 1]", arg425_1: "bf16[1, 128, 1, 32, 1]", arg426_1: "bf16[4096, 4096]", arg427_1: "bf16[4096, 4096]", arg428_1: "bf16[4096, 4096]", arg429_1: "f32[4096]", arg430_1: "f32[1, 128, 4096]", arg431_1: "f32[1, 128, 1]", arg432_1: "bf16[4096, 16384]", arg433_1: "bf16[128, 16384]", arg434_1: "bf16[128, 16384]", arg435_1: "bf16[16384, 4096]", arg436_1: "bf16[128, 4096]", arg437_1: "bf16[128, 4096]", arg438_1: "bf16[4096, 4096]", arg439_1: "bf16[16, 128, 128]", arg440_1: "bf16[16, 256, 128]", arg441_1: "f32[1, 16, 128, 128]", arg442_1: "bf16[16, 256, 128]", arg443_1: "bf16[16, 128, 256]", arg444_1: "bf16[1, 128, 1, 32, 1]", arg445_1: "bf16[1, 128, 1, 32, 1]", arg446_1: "bf16[4096, 4096]", arg447_1: "bf16[4096, 4096]", arg448_1: "bf16[4096, 4096]", arg449_1: "f32[4096]", arg450_1: "f32[1, 128, 4096]", arg451_1: "f32[1, 128, 1]", arg452_1: "bf16[4096, 16384]", arg453_1: "bf16[128, 16384]", arg454_1: "bf16[128, 16384]", arg455_1: "bf16[16384, 4096]", arg456_1: "bf16[128, 4096]", arg457_1: "bf16[128, 4096]", arg458_1: "bf16[4096, 4096]", arg459_1: "bf16[16, 128, 128]", arg460_1: "bf16[16, 256, 128]", arg461_1: "f32[1, 16, 128, 128]", arg462_1: "bf16[16, 256, 128]", arg463_1: "bf16[16, 128, 256]", arg464_1: "bf16[1, 128, 1, 32, 1]", arg465_1: "bf16[1, 128, 1, 32, 1]", arg466_1: "bf16[4096, 4096]", arg467_1: "bf16[4096, 4096]", arg468_1: "bf16[4096, 4096]", arg469_1: "f32[4096]", arg470_1: "f32[1, 128, 4096]", arg471_1: "f32[1, 128, 1]", arg472_1: "bf16[4096, 16384]", arg473_1: "bf16[128, 16384]", arg474_1: "bf16[128, 16384]", arg475_1: "bf16[16384, 4096]", arg476_1: "bf16[128, 4096]", arg477_1: "bf16[128, 4096]", arg478_1: "bf16[4096, 4096]", arg479_1: "bf16[16, 128, 128]", arg480_1: "bf16[16, 256, 128]", arg481_1: "f32[1, 16, 128, 128]", arg482_1: "bf16[16, 256, 128]", arg483_1: "bf16[16, 128, 256]", arg484_1: "bf16[1, 128, 1, 32, 1]", arg485_1: "bf16[1, 128, 1, 32, 1]", arg486_1: "bf16[4096, 4096]", arg487_1: "bf16[4096, 4096]", arg488_1: "bf16[4096, 4096]", arg489_1: "f32[4096]", arg490_1: "f32[1, 128, 4096]", arg491_1: "f32[1, 128, 1]", arg492_1: "bf16[4096, 16384]", arg493_1: "bf16[128, 16384]", arg494_1: "bf16[128, 16384]", arg495_1: "bf16[16384, 4096]", arg496_1: "bf16[128, 4096]", arg497_1: "bf16[128, 4096]", arg498_1: "bf16[4096, 4096]", arg499_1: "bf16[16, 128, 128]", arg500_1: "bf16[16, 256, 128]", arg501_1: "f32[1, 16, 128, 128]", arg502_1: "bf16[16, 256, 128]", arg503_1: "bf16[16, 128, 256]", arg504_1: "bf16[1, 128, 1, 32, 1]", arg505_1: "bf16[1, 128, 1, 32, 1]", arg506_1: "bf16[4096, 4096]", arg507_1: "bf16[4096, 4096]", arg508_1: "bf16[4096, 4096]", arg509_1: "f32[4096]", arg510_1: "f32[1, 128, 4096]", arg511_1: "f32[1, 128, 1]", arg512_1: "bf16[4096, 16384]", arg513_1: "bf16[128, 16384]", arg514_1: "bf16[128, 16384]", arg515_1: "bf16[16384, 4096]", arg516_1: "bf16[128, 4096]", arg517_1: "bf16[128, 4096]", arg518_1: "bf16[4096, 4096]", arg519_1: "bf16[16, 128, 128]", arg520_1: "bf16[16, 256, 128]", arg521_1: "f32[1, 16, 128, 128]", arg522_1: "bf16[16, 256, 128]", arg523_1: "bf16[16, 128, 256]", arg524_1: "bf16[1, 128, 1, 32, 1]", arg525_1: "bf16[1, 128, 1, 32, 1]", arg526_1: "bf16[4096, 4096]", arg527_1: "bf16[4096, 4096]", arg528_1: "bf16[4096, 4096]", arg529_1: "f32[4096]", arg530_1: "f32[1, 128, 4096]", arg531_1: "f32[1, 128, 1]", arg532_1: "bf16[4096, 16384]", arg533_1: "bf16[128, 16384]", arg534_1: "bf16[128, 16384]", arg535_1: "bf16[16384, 4096]", arg536_1: "bf16[128, 4096]", arg537_1: "bf16[128, 4096]", arg538_1: "bf16[4096, 4096]", arg539_1: "bf16[16, 128, 128]", arg540_1: "bf16[16, 256, 128]", arg541_1: "f32[1, 16, 128, 128]", arg542_1: "bf16[16, 256, 128]", arg543_1: "bf16[16, 128, 256]", arg544_1: "bf16[1, 128, 1, 32, 1]", arg545_1: "bf16[1, 128, 1, 32, 1]", arg546_1: "bf16[4096, 4096]", arg547_1: "bf16[4096, 4096]", arg548_1: "bf16[4096, 4096]", arg549_1: "f32[4096]", arg550_1: "bf16[1, 128, 4096]", arg551_1: "f32[1, 128, 4096]", arg552_1: "f32[1, 128, 1]", arg553_1: "f32[1, 128, 1]", arg554_1: "bf16[4096, 16384]", arg555_1: "bf16[128, 16384]", arg556_1: "bf16[128, 16384]", arg557_1: "bf16[16384, 4096]", arg558_1: "bf16[128, 4096]", arg559_1: "bf16[128, 4096]", arg560_1: "bf16[4096, 4096]", arg561_1: "bf16[16, 128, 128]", arg562_1: "bf16[16, 256, 128]", arg563_1: "f32[1, 16, 128, 128]", arg564_1: "bf16[16, 256, 128]", arg565_1: "bf16[16, 128, 256]", arg566_1: "bf16[1, 128, 1, 32, 1]", arg567_1: "bf16[1, 128, 1, 32, 1]", arg568_1: "bf16[4096, 4096]", arg569_1: "bf16[4096, 4096]", arg570_1: "bf16[4096, 4096]", arg571_1: "f32[4096]", arg572_1: "f32[1, 128, 1]", arg573_1: "f32[1, 128, 1]", arg574_1: "i64[1, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825, _shape_param_826, _shape_param_827, _shape_param_828, _shape_param_829, _shape_param_830, _shape_param_831, _shape_param_832, _shape_param_833, _shape_param_834, _shape_param_835, _shape_param_836, _shape_param_837, _shape_param_838, _shape_param_839, _shape_param_840, _shape_param_841, _shape_param_842, _shape_param_843, _shape_param_844, _shape_param_845, _shape_param_846, _shape_param_847, _shape_param_848, _shape_param_849):
        # No stacktrace found for following nodes
        div: "f32[]" = torch.ops.aten.div.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg2_1, 1, 1, 9223372036854775807);  arg2_1 = None
        view: "i64[128]" = torch.ops.aten.view.default(slice_1, [-1]);  slice_1 = None
        unsqueeze: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(view, 1);  view = None
        ne: "b8[128, 1]" = torch.ops.aten.ne.Scalar(unsqueeze, -100)
        full: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[128, 1]" = torch.ops.aten.where.self(ne, unsqueeze, full);  unsqueeze = full = None
        iota: "i64[50400]" = torch.ops.prims.iota.default(50400, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_1: "i64[1, 50400]" = torch.ops.aten.view.default(iota, _shape_param_0);  iota = _shape_param_0 = None
        expand: "i64[128, 50400]" = torch.ops.aten.expand.default(where, _shape_param_1);  where = _shape_param_1 = None
        eq: "b8[128, 50400]" = torch.ops.aten.eq.Tensor(expand, view_1);  expand = view_1 = None
        scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        where_1: "f32[128, 50400]" = torch.ops.aten.where.self(eq, scalar_tensor_1, scalar_tensor);  eq = scalar_tensor_1 = scalar_tensor = None
        full_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[128, 1]" = torch.ops.aten.where.self(ne, div, full_1);  ne = div = full_1 = None
        mul: "f32[128, 50400]" = torch.ops.aten.mul.Tensor(where_1, where_2);  where_1 = where_2 = None
        convert_element_type: "f32[1, 128, 50400]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        view_2: "f32[128, 50400]" = torch.ops.aten.view.default(convert_element_type, _shape_param_2);  convert_element_type = _shape_param_2 = None
        sub: "f32[128, 50400]" = torch.ops.aten.sub.Tensor(view_2, arg4_1);  view_2 = arg4_1 = None
        sub_1: "f32[128, 50400]" = torch.ops.aten.sub.Tensor(sub, arg5_1);  sub = arg5_1 = None
        exp: "f32[128, 50400]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(mul, [1], True)
        mul_1: "f32[128, 50400]" = torch.ops.aten.mul.Tensor(exp, sum_1);  exp = sum_1 = None
        sub_2: "f32[128, 50400]" = torch.ops.aten.sub.Tensor(mul, mul_1);  mul = mul_1 = None
        view_3: "f32[1, 128, 50400]" = torch.ops.aten.view.default(sub_2, _shape_param_3);  sub_2 = _shape_param_3 = None
        convert_element_type_1: "bf16[1, 128, 50400]" = torch.ops.prims.convert_element_type.default(view_3, torch.bfloat16);  view_3 = None
        add: "bf16[1, 128, 50400]" = torch.ops.aten.add.Tensor(arg6_1, convert_element_type_1);  arg6_1 = convert_element_type_1 = None
        view_4: "bf16[128, 50400]" = torch.ops.aten.view.default(add, _shape_param_4);  add = _shape_param_4 = None
        mm: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_4, arg7_1);  arg7_1 = None
        permute: "bf16[50400, 128]" = torch.ops.aten.permute.default(view_4, [1, 0])
        mm_1: "bf16[50400, 4096]" = torch.ops.aten.mm.default(permute, arg8_1);  permute = arg8_1 = None
        sum_2: "f32[1, 50400]" = torch.ops.aten.sum.dim_IntList(view_4, [0], True, dtype = torch.float32);  view_4 = None
        view_5: "f32[50400]" = torch.ops.aten.view.default(sum_2, _shape_param_5);  sum_2 = _shape_param_5 = None
        convert_element_type_2: "bf16[50400]" = torch.ops.prims.convert_element_type.default(view_5, torch.bfloat16);  view_5 = None
        view_6: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm, _shape_param_6);  mm = _shape_param_6 = None
        convert_element_type_3: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_6, torch.float32);  view_6 = None
        convert_element_type_4: "f32[50400, 4096]" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_5: "f32[50400]" = torch.ops.prims.convert_element_type.default(convert_element_type_2, torch.float32);  convert_element_type_2 = None
        mul_2: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg9_1);  arg9_1 = None
        mul_3: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_2, 4096)
        sum_3: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_2, [2], True)
        mul_4: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_2, arg10_1);  mul_2 = None
        sum_4: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_4, [2], True);  mul_4 = None
        mul_5: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg10_1, sum_4);  sum_4 = None
        sub_3: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_3, sum_3);  mul_3 = sum_3 = None
        sub_4: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_3, mul_5);  sub_3 = mul_5 = None
        mul_6: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg11_1, sub_4);  arg11_1 = sub_4 = None
        mul_7: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(convert_element_type_3, arg10_1);  arg10_1 = None
        sum_5: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_7, [0, 1]);  mul_7 = None
        sum_6: "f32[4096]" = torch.ops.aten.sum.dim_IntList(convert_element_type_3, [0, 1]);  convert_element_type_3 = None
        convert_element_type_6: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(mul_6, torch.bfloat16)
        view_7: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_6, _shape_param_7);  convert_element_type_6 = _shape_param_7 = None
        mm_2: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_7, arg12_1);  arg12_1 = None
        permute_1: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_7, [1, 0])
        mm_3: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_1, arg13_1);  arg13_1 = None
        sum_7: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_7, [0], True, dtype = torch.float32)
        view_8: "f32[4096]" = torch.ops.aten.view.default(sum_7, _shape_param_8);  sum_7 = _shape_param_8 = None
        convert_element_type_7: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_8, torch.bfloat16);  view_8 = None
        view_9: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_2, _shape_param_9);  mm_2 = _shape_param_9 = None
        convert_element_type_8: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_9, torch.float32);  view_9 = None
        convert_element_type_9: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_10: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_7, torch.float32);  convert_element_type_7 = None
        view_10: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg14_1, _shape_param_10);  arg14_1 = _shape_param_10 = None
        mul_8: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        mul_9: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_8, mul_8);  mul_8 = None
        convert_element_type_11: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_10, torch.float32)
        pow_1: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_11, 3.0)
        mul_10: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_1: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_10, mul_10);  view_10 = mul_10 = None
        mul_11: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_1, 0.7978845608028654);  add_1 = None
        tanh: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_11);  mul_11 = None
        add_2: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_12: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_8, add_2);  convert_element_type_8 = add_2 = None
        convert_element_type_12: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_12, torch.bfloat16);  mul_12 = None
        mul_13: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_5: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_13);  mul_13 = None
        mul_14: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_9, sub_5);  mul_9 = sub_5 = None
        mul_15: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_14, 0.7978845608028654);  mul_14 = None
        convert_element_type_13: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16)
        mul_16: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_15, 0.044715);  mul_15 = None
        pow_2: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_11, 2.0);  convert_element_type_11 = None
        mul_17: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_2, 3.0);  pow_2 = None
        mul_18: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_16, mul_17);  mul_16 = mul_17 = None
        convert_element_type_14: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_18, torch.bfloat16);  mul_18 = None
        add_3: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_13, convert_element_type_14);  convert_element_type_13 = convert_element_type_14 = None
        mul_19: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 0.5);  convert_element_type_12 = None
        add_4: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_3, mul_19);  add_3 = mul_19 = None
        view_11: "bf16[128, 16384]" = torch.ops.aten.view.default(add_4, _shape_param_11);  add_4 = _shape_param_11 = None
        mm_4: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_11, arg15_1);  arg15_1 = None
        permute_2: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_11, [1, 0])
        mm_5: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_2, arg16_1);  permute_2 = None
        sum_8: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_11, [0], True, dtype = torch.float32);  view_11 = None
        view_12: "f32[16384]" = torch.ops.aten.view.default(sum_8, _shape_param_12);  sum_8 = _shape_param_12 = None
        convert_element_type_15: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_12, torch.bfloat16);  view_12 = None
        view_13: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_4, _shape_param_13);  mm_4 = _shape_param_13 = None
        convert_element_type_16: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_13, torch.float32);  view_13 = None
        convert_element_type_17: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_18: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_15, torch.float32);  convert_element_type_15 = None
        mm_6: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_1, arg17_1);  permute_1 = arg17_1 = None
        mm_7: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_7, arg18_1);  view_7 = arg18_1 = None
        view_14: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_7, _shape_param_14);  mm_7 = _shape_param_14 = None
        convert_element_type_19: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_6, torch.float32);  mm_6 = None
        view_15: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_14, _shape_param_15);  view_14 = _shape_param_15 = None
        permute_3: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_15, [0, 2, 1, 3]);  view_15 = None
        view_16: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_3, _shape_param_16);  permute_3 = _shape_param_16 = None
        bmm: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg19_1, view_16);  arg19_1 = None
        bmm_1: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_16, arg20_1);  view_16 = arg20_1 = None
        view_17: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm, _shape_param_17);  bmm = _shape_param_17 = None
        view_18: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_1, _shape_param_18);  bmm_1 = _shape_param_18 = None
        convert_element_type_20: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_18, torch.float32);  view_18 = None
        mul_20: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_20, arg21_1);  convert_element_type_20 = None
        sum_9: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_20, [-1], True)
        neg: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg21_1);  arg21_1 = None
        fma: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg, sum_9, mul_20);  neg = sum_9 = mul_20 = None
        convert_element_type_21: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma, torch.bfloat16);  fma = None
        div_1: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_21, 16.0);  convert_element_type_21 = None
        view_19: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_1, _shape_param_19);  div_1 = _shape_param_19 = None
        bmm_2: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg22_1, view_19);  arg22_1 = None
        bmm_3: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_19, arg23_1);  view_19 = arg23_1 = None
        view_20: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_20);  bmm_2 = _shape_param_20 = None
        view_21: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_3, _shape_param_21);  bmm_3 = _shape_param_21 = None
        convert_element_type_22: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_20, torch.float32);  view_20 = None
        permute_4: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_22, [0, 1, 3, 2]);  convert_element_type_22 = None
        convert_element_type_23: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_4, torch.bfloat16);  permute_4 = None
        convert_element_type_24: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_21, torch.bfloat16);  view_21 = None
        permute_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_24, [0, 2, 1, 3]);  convert_element_type_24 = None
        permute_6: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_23, [0, 2, 1, 3]);  convert_element_type_23 = None
        slice_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_5, 3, 0, 64)
        slice_3: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_5, 3, 64, 256);  permute_5 = None
        slice_4: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_6, 3, 0, 64)
        slice_5: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_6, 3, 64, 256);  permute_6 = None
        expand_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg24_1, _shape_param_22);  arg24_1 = _shape_param_22 = None
        clone: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_22: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone, _shape_param_23);  clone = _shape_param_23 = None
        mul_21: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_2, view_22)
        view_23: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_21, _shape_param_24);  mul_21 = _shape_param_24 = None
        select: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_23, -1, 0)
        select_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_23, -1, 1);  view_23 = None
        neg_1: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select);  select = None
        full_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.full.default(_shape_param_25, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_25 = None
        slice_scatter: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_1, 3, 1, 9223372036854775807, 2);  neg_1 = None
        slice_scatter_1: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_1, 3, 0, 9223372036854775807, 2);  select_1 = None
        add_5: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None
        expand_2: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg25_1, _shape_param_26);  arg25_1 = _shape_param_26 = None
        clone_1: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_24: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_1, _shape_param_27);  clone_1 = _shape_param_27 = None
        mul_22: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_2, view_24);  slice_2 = None
        add_6: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_5, mul_22);  add_5 = mul_22 = None
        mul_23: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_4, view_22);  view_22 = None
        view_25: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_23, _shape_param_28);  mul_23 = _shape_param_28 = None
        select_2: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_25, -1, 0)
        select_3: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_25, -1, 1);  view_25 = None
        neg_2: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_2);  select_2 = None
        slice_scatter_2: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_2, 3, 1, 9223372036854775807, 2);  neg_2 = None
        slice_scatter_3: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_3, 3, 0, 9223372036854775807, 2);  select_3 = None
        add_7: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_2, slice_scatter_3);  slice_scatter_2 = slice_scatter_3 = None
        mul_24: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_4, view_24);  slice_4 = view_24 = None
        add_8: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_7, mul_24);  add_7 = mul_24 = None
        full_3: "bf16[1, 128, 16, 256]" = torch.ops.aten.full.default(_shape_param_29, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_29 = None
        slice_scatter_4: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_3, 3, 64, 9223372036854775807);  slice_3 = None
        slice_scatter_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_6, 3, 0, 64);  add_6 = None
        add_9: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_4, slice_scatter_5);  slice_scatter_4 = slice_scatter_5 = None
        slice_scatter_6: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_5, 3, 64, 9223372036854775807);  slice_5 = None
        slice_scatter_7: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_8, 3, 0, 64);  add_8 = None
        add_10: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_6, slice_scatter_7);  slice_scatter_6 = slice_scatter_7 = None
        permute_7: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_2: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        view_26: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_2, _shape_param_30);  clone_2 = _shape_param_30 = None
        view_27: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_10, _shape_param_31);  add_10 = _shape_param_31 = None
        view_28: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_9, _shape_param_32);  add_9 = _shape_param_32 = None
        view_29: "bf16[128, 4096]" = torch.ops.aten.view.default(view_26, _shape_param_33);  view_26 = _shape_param_33 = None
        permute_8: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_29, [1, 0])
        mm_8: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_8, arg16_1);  permute_8 = None
        mm_9: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_29, arg26_1);  view_29 = arg26_1 = None
        view_30: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_9, _shape_param_34);  mm_9 = _shape_param_34 = None
        convert_element_type_25: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_30, torch.float32);  view_30 = None
        add_11: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_16, convert_element_type_25);  convert_element_type_16 = convert_element_type_25 = None
        convert_element_type_26: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_8, torch.float32);  mm_8 = None
        view_31: "bf16[128, 4096]" = torch.ops.aten.view.default(view_27, _shape_param_35);  view_27 = _shape_param_35 = None
        permute_9: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_31, [1, 0])
        mm_10: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_9, arg16_1);  permute_9 = None
        mm_11: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_31, arg27_1);  view_31 = arg27_1 = None
        view_32: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_11, _shape_param_36);  mm_11 = _shape_param_36 = None
        convert_element_type_27: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_32, torch.float32);  view_32 = None
        add_12: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_11, convert_element_type_27);  add_11 = convert_element_type_27 = None
        convert_element_type_28: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_10, torch.float32);  mm_10 = None
        view_33: "bf16[128, 4096]" = torch.ops.aten.view.default(view_28, _shape_param_37);  view_28 = _shape_param_37 = None
        permute_10: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_33, [1, 0])
        mm_12: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_10, arg16_1);  permute_10 = arg16_1 = None
        mm_13: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_33, arg28_1);  view_33 = arg28_1 = None
        view_34: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_13, _shape_param_38);  mm_13 = _shape_param_38 = None
        convert_element_type_29: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_34, torch.float32);  view_34 = None
        add_13: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_12, convert_element_type_29);  add_12 = convert_element_type_29 = None
        convert_element_type_30: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_12, torch.float32);  mm_12 = None
        mul_25: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_13, arg29_1);  arg29_1 = None
        mul_26: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_25, 4096)
        sum_10: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_25, [2], True)
        mul_27: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_25, arg30_1);  mul_25 = None
        sum_11: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_27, [2], True);  mul_27 = None
        mul_28: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg30_1, sum_11);  sum_11 = None
        sub_6: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_26, sum_10);  mul_26 = sum_10 = None
        sub_7: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_6, mul_28);  sub_6 = mul_28 = None
        mul_29: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg31_1, sub_7);  arg31_1 = sub_7 = None
        mul_30: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_13, arg30_1);  arg30_1 = None
        sum_12: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_30, [0, 1]);  mul_30 = None
        sum_13: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_13, [0, 1]);  add_13 = None
        add_14: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_6, mul_29);  mul_6 = mul_29 = None
        convert_element_type_31: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16)
        view_35: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_31, _shape_param_39);  convert_element_type_31 = _shape_param_39 = None
        mm_14: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_35, arg32_1);  arg32_1 = None
        permute_11: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_35, [1, 0])
        mm_15: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_11, arg33_1);  arg33_1 = None
        sum_14: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_35, [0], True, dtype = torch.float32)
        view_36: "f32[4096]" = torch.ops.aten.view.default(sum_14, _shape_param_40);  sum_14 = _shape_param_40 = None
        convert_element_type_32: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_36, torch.bfloat16);  view_36 = None
        view_37: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_14, _shape_param_41);  mm_14 = _shape_param_41 = None
        convert_element_type_33: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_37, torch.float32);  view_37 = None
        convert_element_type_34: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_35: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_32, torch.float32);  convert_element_type_32 = None
        view_38: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg34_1, _shape_param_42);  arg34_1 = _shape_param_42 = None
        mul_31: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
        mul_32: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_33, mul_31);  mul_31 = None
        convert_element_type_36: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_38, torch.float32)
        pow_3: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_36, 3.0)
        mul_33: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_15: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_38, mul_33);  view_38 = mul_33 = None
        mul_34: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
        tanh_1: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_34);  mul_34 = None
        add_16: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_35: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_33, add_16);  convert_element_type_33 = add_16 = None
        convert_element_type_37: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_35, torch.bfloat16);  mul_35 = None
        mul_36: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_8: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_36);  mul_36 = None
        mul_37: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_32, sub_8);  mul_32 = sub_8 = None
        mul_38: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_37, 0.7978845608028654);  mul_37 = None
        convert_element_type_38: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_38, torch.bfloat16)
        mul_39: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_38, 0.044715);  mul_38 = None
        pow_4: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_36, 2.0);  convert_element_type_36 = None
        mul_40: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_4, 3.0);  pow_4 = None
        mul_41: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_39, mul_40);  mul_39 = mul_40 = None
        convert_element_type_39: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_41, torch.bfloat16);  mul_41 = None
        add_17: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_38, convert_element_type_39);  convert_element_type_38 = convert_element_type_39 = None
        mul_42: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 0.5);  convert_element_type_37 = None
        add_18: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_17, mul_42);  add_17 = mul_42 = None
        view_39: "bf16[128, 16384]" = torch.ops.aten.view.default(add_18, _shape_param_43);  add_18 = _shape_param_43 = None
        mm_16: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_39, arg35_1);  arg35_1 = None
        permute_12: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_39, [1, 0])
        mm_17: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_12, arg36_1);  permute_12 = None
        sum_15: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_39, [0], True, dtype = torch.float32);  view_39 = None
        view_40: "f32[16384]" = torch.ops.aten.view.default(sum_15, _shape_param_44);  sum_15 = _shape_param_44 = None
        convert_element_type_40: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_40, torch.bfloat16);  view_40 = None
        view_41: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_16, _shape_param_45);  mm_16 = _shape_param_45 = None
        convert_element_type_41: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        convert_element_type_42: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_43: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_40, torch.float32);  convert_element_type_40 = None
        mm_18: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_11, arg37_1);  permute_11 = arg37_1 = None
        mm_19: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_35, arg38_1);  view_35 = arg38_1 = None
        view_42: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_19, _shape_param_46);  mm_19 = _shape_param_46 = None
        convert_element_type_44: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_18, torch.float32);  mm_18 = None
        view_43: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_42, _shape_param_47);  view_42 = _shape_param_47 = None
        permute_13: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
        view_44: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_13, _shape_param_48);  permute_13 = _shape_param_48 = None
        bmm_4: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg39_1, view_44);  arg39_1 = None
        bmm_5: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_44, arg40_1);  view_44 = arg40_1 = None
        view_45: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_4, _shape_param_49);  bmm_4 = _shape_param_49 = None
        view_46: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_5, _shape_param_50);  bmm_5 = _shape_param_50 = None
        convert_element_type_45: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_46, torch.float32);  view_46 = None
        mul_43: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_45, arg41_1);  convert_element_type_45 = None
        sum_16: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_43, [-1], True)
        neg_3: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg41_1);  arg41_1 = None
        fma_1: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_3, sum_16, mul_43);  neg_3 = sum_16 = mul_43 = None
        convert_element_type_46: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_1, torch.bfloat16);  fma_1 = None
        div_2: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_46, 16.0);  convert_element_type_46 = None
        view_47: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_2, _shape_param_51);  div_2 = _shape_param_51 = None
        bmm_6: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg42_1, view_47);  arg42_1 = None
        bmm_7: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_47, arg43_1);  view_47 = arg43_1 = None
        view_48: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_52);  bmm_6 = _shape_param_52 = None
        view_49: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_7, _shape_param_53);  bmm_7 = _shape_param_53 = None
        convert_element_type_47: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_48, torch.float32);  view_48 = None
        permute_14: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_47, [0, 1, 3, 2]);  convert_element_type_47 = None
        convert_element_type_48: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_14, torch.bfloat16);  permute_14 = None
        convert_element_type_49: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_49, torch.bfloat16);  view_49 = None
        permute_15: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_49, [0, 2, 1, 3]);  convert_element_type_49 = None
        permute_16: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_48, [0, 2, 1, 3]);  convert_element_type_48 = None
        slice_6: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_15, 3, 0, 64)
        slice_7: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_15, 3, 64, 256);  permute_15 = None
        slice_8: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_16, 3, 0, 64)
        slice_9: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_16, 3, 64, 256);  permute_16 = None
        expand_3: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg44_1, _shape_param_54);  arg44_1 = _shape_param_54 = None
        clone_3: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_50: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_3, _shape_param_55);  clone_3 = _shape_param_55 = None
        mul_44: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_6, view_50)
        view_51: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_44, _shape_param_56);  mul_44 = _shape_param_56 = None
        select_4: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_51, -1, 0)
        select_5: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_51, -1, 1);  view_51 = None
        neg_4: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_4);  select_4 = None
        slice_scatter_8: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_4, 3, 1, 9223372036854775807, 2);  neg_4 = None
        slice_scatter_9: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_5, 3, 0, 9223372036854775807, 2);  select_5 = None
        add_19: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_8, slice_scatter_9);  slice_scatter_8 = slice_scatter_9 = None
        expand_4: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg45_1, _shape_param_57);  arg45_1 = _shape_param_57 = None
        clone_4: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_52: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_4, _shape_param_58);  clone_4 = _shape_param_58 = None
        mul_45: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_6, view_52);  slice_6 = None
        add_20: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_19, mul_45);  add_19 = mul_45 = None
        mul_46: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_8, view_50);  view_50 = None
        view_53: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_46, _shape_param_59);  mul_46 = _shape_param_59 = None
        select_6: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_53, -1, 0)
        select_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_53, -1, 1);  view_53 = None
        neg_5: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_6);  select_6 = None
        slice_scatter_10: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_5, 3, 1, 9223372036854775807, 2);  neg_5 = None
        slice_scatter_11: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_7, 3, 0, 9223372036854775807, 2);  select_7 = None
        add_21: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_11);  slice_scatter_10 = slice_scatter_11 = None
        mul_47: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_8, view_52);  slice_8 = view_52 = None
        add_22: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_21, mul_47);  add_21 = mul_47 = None
        slice_scatter_12: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_7, 3, 64, 9223372036854775807);  slice_7 = None
        slice_scatter_13: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_20, 3, 0, 64);  add_20 = None
        add_23: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_12, slice_scatter_13);  slice_scatter_12 = slice_scatter_13 = None
        slice_scatter_14: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_9, 3, 64, 9223372036854775807);  slice_9 = None
        slice_scatter_15: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_22, 3, 0, 64);  add_22 = None
        add_24: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_14, slice_scatter_15);  slice_scatter_14 = slice_scatter_15 = None
        permute_17: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
        clone_5: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
        view_54: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_5, _shape_param_60);  clone_5 = _shape_param_60 = None
        view_55: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_24, _shape_param_61);  add_24 = _shape_param_61 = None
        view_56: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_23, _shape_param_62);  add_23 = _shape_param_62 = None
        view_57: "bf16[128, 4096]" = torch.ops.aten.view.default(view_54, _shape_param_63);  view_54 = _shape_param_63 = None
        permute_18: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_57, [1, 0])
        mm_20: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_18, arg36_1);  permute_18 = None
        mm_21: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_57, arg46_1);  view_57 = arg46_1 = None
        view_58: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_21, _shape_param_64);  mm_21 = _shape_param_64 = None
        convert_element_type_50: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_58, torch.float32);  view_58 = None
        add_25: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_41, convert_element_type_50);  convert_element_type_41 = convert_element_type_50 = None
        convert_element_type_51: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_20, torch.float32);  mm_20 = None
        view_59: "bf16[128, 4096]" = torch.ops.aten.view.default(view_55, _shape_param_65);  view_55 = _shape_param_65 = None
        permute_19: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_59, [1, 0])
        mm_22: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_19, arg36_1);  permute_19 = None
        mm_23: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_59, arg47_1);  view_59 = arg47_1 = None
        view_60: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_23, _shape_param_66);  mm_23 = _shape_param_66 = None
        convert_element_type_52: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_60, torch.float32);  view_60 = None
        add_26: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_25, convert_element_type_52);  add_25 = convert_element_type_52 = None
        convert_element_type_53: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_22, torch.float32);  mm_22 = None
        view_61: "bf16[128, 4096]" = torch.ops.aten.view.default(view_56, _shape_param_67);  view_56 = _shape_param_67 = None
        permute_20: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_61, [1, 0])
        mm_24: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_20, arg36_1);  permute_20 = arg36_1 = None
        mm_25: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_61, arg48_1);  view_61 = arg48_1 = None
        view_62: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_25, _shape_param_68);  mm_25 = _shape_param_68 = None
        convert_element_type_54: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_62, torch.float32);  view_62 = None
        add_27: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_26, convert_element_type_54);  add_26 = convert_element_type_54 = None
        convert_element_type_55: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_24, torch.float32);  mm_24 = None
        mul_48: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_27, arg49_1);  arg49_1 = None
        mul_49: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_48, 4096)
        sum_17: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_48, [2], True)
        mul_50: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_48, arg50_1);  mul_48 = None
        sum_18: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_50, [2], True);  mul_50 = None
        mul_51: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg50_1, sum_18);  sum_18 = None
        sub_9: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_49, sum_17);  mul_49 = sum_17 = None
        sub_10: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_9, mul_51);  sub_9 = mul_51 = None
        mul_52: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg51_1, sub_10);  arg51_1 = sub_10 = None
        mul_53: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_27, arg50_1);  arg50_1 = None
        sum_19: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_53, [0, 1]);  mul_53 = None
        sum_20: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_27, [0, 1]);  add_27 = None
        add_28: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_14, mul_52);  add_14 = mul_52 = None
        convert_element_type_56: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16)
        view_63: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_56, _shape_param_69);  convert_element_type_56 = _shape_param_69 = None
        mm_26: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_63, arg52_1);  arg52_1 = None
        permute_21: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_63, [1, 0])
        mm_27: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_21, arg53_1);  arg53_1 = None
        sum_21: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_63, [0], True, dtype = torch.float32)
        view_64: "f32[4096]" = torch.ops.aten.view.default(sum_21, _shape_param_70);  sum_21 = _shape_param_70 = None
        convert_element_type_57: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_64, torch.bfloat16);  view_64 = None
        view_65: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_26, _shape_param_71);  mm_26 = _shape_param_71 = None
        convert_element_type_58: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_65, torch.float32);  view_65 = None
        convert_element_type_59: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_60: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_57, torch.float32);  convert_element_type_57 = None
        view_66: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg54_1, _shape_param_72);  arg54_1 = _shape_param_72 = None
        mul_54: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_66, 0.5)
        mul_55: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_58, mul_54);  mul_54 = None
        convert_element_type_61: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_66, torch.float32)
        pow_5: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_61, 3.0)
        mul_56: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_29: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_66, mul_56);  view_66 = mul_56 = None
        mul_57: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_29, 0.7978845608028654);  add_29 = None
        tanh_2: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_57);  mul_57 = None
        add_30: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_58: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_58, add_30);  convert_element_type_58 = add_30 = None
        convert_element_type_62: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_58, torch.bfloat16);  mul_58 = None
        mul_59: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_11: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_59);  mul_59 = None
        mul_60: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_55, sub_11);  mul_55 = sub_11 = None
        mul_61: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_60, 0.7978845608028654);  mul_60 = None
        convert_element_type_63: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_61, torch.bfloat16)
        mul_62: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_61, 0.044715);  mul_61 = None
        pow_6: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_61, 2.0);  convert_element_type_61 = None
        mul_63: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_6, 3.0);  pow_6 = None
        mul_64: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
        convert_element_type_64: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_64, torch.bfloat16);  mul_64 = None
        add_31: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_63, convert_element_type_64);  convert_element_type_63 = convert_element_type_64 = None
        mul_65: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 0.5);  convert_element_type_62 = None
        add_32: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_31, mul_65);  add_31 = mul_65 = None
        view_67: "bf16[128, 16384]" = torch.ops.aten.view.default(add_32, _shape_param_73);  add_32 = _shape_param_73 = None
        mm_28: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_67, arg55_1);  arg55_1 = None
        permute_22: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_67, [1, 0])
        mm_29: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_22, arg56_1);  permute_22 = None
        sum_22: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_67, [0], True, dtype = torch.float32);  view_67 = None
        view_68: "f32[16384]" = torch.ops.aten.view.default(sum_22, _shape_param_74);  sum_22 = _shape_param_74 = None
        convert_element_type_65: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_68, torch.bfloat16);  view_68 = None
        view_69: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_28, _shape_param_75);  mm_28 = _shape_param_75 = None
        convert_element_type_66: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_69, torch.float32);  view_69 = None
        convert_element_type_67: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_68: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_65, torch.float32);  convert_element_type_65 = None
        mm_30: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_21, arg57_1);  permute_21 = arg57_1 = None
        mm_31: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_63, arg58_1);  view_63 = arg58_1 = None
        view_70: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_31, _shape_param_76);  mm_31 = _shape_param_76 = None
        convert_element_type_69: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_30, torch.float32);  mm_30 = None
        view_71: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_70, _shape_param_77);  view_70 = _shape_param_77 = None
        permute_23: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
        view_72: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_23, _shape_param_78);  permute_23 = _shape_param_78 = None
        bmm_8: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg59_1, view_72);  arg59_1 = None
        bmm_9: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_72, arg60_1);  view_72 = arg60_1 = None
        view_73: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_8, _shape_param_79);  bmm_8 = _shape_param_79 = None
        view_74: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_9, _shape_param_80);  bmm_9 = _shape_param_80 = None
        convert_element_type_70: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_74, torch.float32);  view_74 = None
        mul_66: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_70, arg61_1);  convert_element_type_70 = None
        sum_23: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_66, [-1], True)
        neg_6: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg61_1);  arg61_1 = None
        fma_2: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_6, sum_23, mul_66);  neg_6 = sum_23 = mul_66 = None
        convert_element_type_71: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_2, torch.bfloat16);  fma_2 = None
        div_3: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_71, 16.0);  convert_element_type_71 = None
        view_75: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_3, _shape_param_81);  div_3 = _shape_param_81 = None
        bmm_10: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg62_1, view_75);  arg62_1 = None
        bmm_11: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_75, arg63_1);  view_75 = arg63_1 = None
        view_76: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_82);  bmm_10 = _shape_param_82 = None
        view_77: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_11, _shape_param_83);  bmm_11 = _shape_param_83 = None
        convert_element_type_72: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_76, torch.float32);  view_76 = None
        permute_24: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_72, [0, 1, 3, 2]);  convert_element_type_72 = None
        convert_element_type_73: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_24, torch.bfloat16);  permute_24 = None
        convert_element_type_74: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_77, torch.bfloat16);  view_77 = None
        permute_25: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_74, [0, 2, 1, 3]);  convert_element_type_74 = None
        permute_26: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_73, [0, 2, 1, 3]);  convert_element_type_73 = None
        slice_10: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_25, 3, 0, 64)
        slice_11: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_25, 3, 64, 256);  permute_25 = None
        slice_12: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_26, 3, 0, 64)
        slice_13: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_26, 3, 64, 256);  permute_26 = None
        expand_5: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg64_1, _shape_param_84);  arg64_1 = _shape_param_84 = None
        clone_6: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_78: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_6, _shape_param_85);  clone_6 = _shape_param_85 = None
        mul_67: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_10, view_78)
        view_79: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_67, _shape_param_86);  mul_67 = _shape_param_86 = None
        select_8: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_79, -1, 0)
        select_9: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_79, -1, 1);  view_79 = None
        neg_7: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_8);  select_8 = None
        slice_scatter_16: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_7, 3, 1, 9223372036854775807, 2);  neg_7 = None
        slice_scatter_17: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_9, 3, 0, 9223372036854775807, 2);  select_9 = None
        add_33: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_16, slice_scatter_17);  slice_scatter_16 = slice_scatter_17 = None
        expand_6: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg65_1, _shape_param_87);  arg65_1 = _shape_param_87 = None
        clone_7: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_80: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_7, _shape_param_88);  clone_7 = _shape_param_88 = None
        mul_68: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_10, view_80);  slice_10 = None
        add_34: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_33, mul_68);  add_33 = mul_68 = None
        mul_69: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_12, view_78);  view_78 = None
        view_81: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_69, _shape_param_89);  mul_69 = _shape_param_89 = None
        select_10: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_81, -1, 0)
        select_11: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_81, -1, 1);  view_81 = None
        neg_8: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_10);  select_10 = None
        slice_scatter_18: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_8, 3, 1, 9223372036854775807, 2);  neg_8 = None
        slice_scatter_19: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_11, 3, 0, 9223372036854775807, 2);  select_11 = None
        add_35: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_18, slice_scatter_19);  slice_scatter_18 = slice_scatter_19 = None
        mul_70: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_12, view_80);  slice_12 = view_80 = None
        add_36: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_35, mul_70);  add_35 = mul_70 = None
        slice_scatter_20: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_11, 3, 64, 9223372036854775807);  slice_11 = None
        slice_scatter_21: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_34, 3, 0, 64);  add_34 = None
        add_37: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_20, slice_scatter_21);  slice_scatter_20 = slice_scatter_21 = None
        slice_scatter_22: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_13, 3, 64, 9223372036854775807);  slice_13 = None
        slice_scatter_23: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_36, 3, 0, 64);  add_36 = None
        add_38: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_22, slice_scatter_23);  slice_scatter_22 = slice_scatter_23 = None
        permute_27: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
        clone_8: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
        view_82: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_8, _shape_param_90);  clone_8 = _shape_param_90 = None
        view_83: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_38, _shape_param_91);  add_38 = _shape_param_91 = None
        view_84: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_37, _shape_param_92);  add_37 = _shape_param_92 = None
        view_85: "bf16[128, 4096]" = torch.ops.aten.view.default(view_82, _shape_param_93);  view_82 = _shape_param_93 = None
        permute_28: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_85, [1, 0])
        mm_32: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_28, arg56_1);  permute_28 = None
        mm_33: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_85, arg66_1);  view_85 = arg66_1 = None
        view_86: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_33, _shape_param_94);  mm_33 = _shape_param_94 = None
        convert_element_type_75: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_86, torch.float32);  view_86 = None
        add_39: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_66, convert_element_type_75);  convert_element_type_66 = convert_element_type_75 = None
        convert_element_type_76: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_32, torch.float32);  mm_32 = None
        view_87: "bf16[128, 4096]" = torch.ops.aten.view.default(view_83, _shape_param_95);  view_83 = _shape_param_95 = None
        permute_29: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_87, [1, 0])
        mm_34: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_29, arg56_1);  permute_29 = None
        mm_35: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_87, arg67_1);  view_87 = arg67_1 = None
        view_88: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_35, _shape_param_96);  mm_35 = _shape_param_96 = None
        convert_element_type_77: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_88, torch.float32);  view_88 = None
        add_40: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_39, convert_element_type_77);  add_39 = convert_element_type_77 = None
        convert_element_type_78: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_34, torch.float32);  mm_34 = None
        view_89: "bf16[128, 4096]" = torch.ops.aten.view.default(view_84, _shape_param_97);  view_84 = _shape_param_97 = None
        permute_30: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_89, [1, 0])
        mm_36: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_30, arg56_1);  permute_30 = arg56_1 = None
        mm_37: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_89, arg68_1);  view_89 = arg68_1 = None
        view_90: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_37, _shape_param_98);  mm_37 = _shape_param_98 = None
        convert_element_type_79: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_90, torch.float32);  view_90 = None
        add_41: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_40, convert_element_type_79);  add_40 = convert_element_type_79 = None
        convert_element_type_80: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_36, torch.float32);  mm_36 = None
        mul_71: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_41, arg69_1);  arg69_1 = None
        mul_72: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_71, 4096)
        sum_24: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_71, [2], True)
        mul_73: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_71, arg70_1);  mul_71 = None
        sum_25: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_73, [2], True);  mul_73 = None
        mul_74: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg70_1, sum_25);  sum_25 = None
        sub_12: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_72, sum_24);  mul_72 = sum_24 = None
        sub_13: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_12, mul_74);  sub_12 = mul_74 = None
        mul_75: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg71_1, sub_13);  arg71_1 = sub_13 = None
        mul_76: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_41, arg70_1);  arg70_1 = None
        sum_26: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_76, [0, 1]);  mul_76 = None
        sum_27: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_41, [0, 1]);  add_41 = None
        add_42: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_28, mul_75);  add_28 = mul_75 = None
        convert_element_type_81: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_42, torch.bfloat16)
        view_91: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_81, _shape_param_99);  convert_element_type_81 = _shape_param_99 = None
        mm_38: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_91, arg72_1);  arg72_1 = None
        permute_31: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_91, [1, 0])
        mm_39: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_31, arg73_1);  arg73_1 = None
        sum_28: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_91, [0], True, dtype = torch.float32)
        view_92: "f32[4096]" = torch.ops.aten.view.default(sum_28, _shape_param_100);  sum_28 = _shape_param_100 = None
        convert_element_type_82: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_92, torch.bfloat16);  view_92 = None
        view_93: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_38, _shape_param_101);  mm_38 = _shape_param_101 = None
        convert_element_type_83: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_93, torch.float32);  view_93 = None
        convert_element_type_84: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_85: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_82, torch.float32);  convert_element_type_82 = None
        view_94: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg74_1, _shape_param_102);  arg74_1 = _shape_param_102 = None
        mul_77: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        mul_78: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_83, mul_77);  mul_77 = None
        convert_element_type_86: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_94, torch.float32)
        pow_7: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_86, 3.0)
        mul_79: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_43: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_94, mul_79);  view_94 = mul_79 = None
        mul_80: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_43, 0.7978845608028654);  add_43 = None
        tanh_3: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_80);  mul_80 = None
        add_44: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_81: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_83, add_44);  convert_element_type_83 = add_44 = None
        convert_element_type_87: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_81, torch.bfloat16);  mul_81 = None
        mul_82: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_14: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_82);  mul_82 = None
        mul_83: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_78, sub_14);  mul_78 = sub_14 = None
        mul_84: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_83, 0.7978845608028654);  mul_83 = None
        convert_element_type_88: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_84, torch.bfloat16)
        mul_85: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_84, 0.044715);  mul_84 = None
        pow_8: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_86, 2.0);  convert_element_type_86 = None
        mul_86: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_8, 3.0);  pow_8 = None
        mul_87: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_85, mul_86);  mul_85 = mul_86 = None
        convert_element_type_89: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_87, torch.bfloat16);  mul_87 = None
        add_45: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_88, convert_element_type_89);  convert_element_type_88 = convert_element_type_89 = None
        mul_88: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 0.5);  convert_element_type_87 = None
        add_46: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_45, mul_88);  add_45 = mul_88 = None
        view_95: "bf16[128, 16384]" = torch.ops.aten.view.default(add_46, _shape_param_103);  add_46 = _shape_param_103 = None
        mm_40: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_95, arg75_1);  arg75_1 = None
        permute_32: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_95, [1, 0])
        mm_41: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_32, arg76_1);  permute_32 = None
        sum_29: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_95, [0], True, dtype = torch.float32);  view_95 = None
        view_96: "f32[16384]" = torch.ops.aten.view.default(sum_29, _shape_param_104);  sum_29 = _shape_param_104 = None
        convert_element_type_90: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_96, torch.bfloat16);  view_96 = None
        view_97: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_40, _shape_param_105);  mm_40 = _shape_param_105 = None
        convert_element_type_91: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_97, torch.float32);  view_97 = None
        convert_element_type_92: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_93: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_90, torch.float32);  convert_element_type_90 = None
        mm_42: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_31, arg77_1);  permute_31 = arg77_1 = None
        mm_43: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_91, arg78_1);  view_91 = arg78_1 = None
        view_98: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_43, _shape_param_106);  mm_43 = _shape_param_106 = None
        convert_element_type_94: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_42, torch.float32);  mm_42 = None
        view_99: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_98, _shape_param_107);  view_98 = _shape_param_107 = None
        permute_33: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
        view_100: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_33, _shape_param_108);  permute_33 = _shape_param_108 = None
        bmm_12: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg79_1, view_100);  arg79_1 = None
        bmm_13: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_100, arg80_1);  view_100 = arg80_1 = None
        view_101: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_12, _shape_param_109);  bmm_12 = _shape_param_109 = None
        view_102: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_13, _shape_param_110);  bmm_13 = _shape_param_110 = None
        convert_element_type_95: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_102, torch.float32);  view_102 = None
        mul_89: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_95, arg81_1);  convert_element_type_95 = None
        sum_30: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [-1], True)
        neg_9: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg81_1);  arg81_1 = None
        fma_3: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_9, sum_30, mul_89);  neg_9 = sum_30 = mul_89 = None
        convert_element_type_96: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_3, torch.bfloat16);  fma_3 = None
        div_4: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_96, 16.0);  convert_element_type_96 = None
        view_103: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_4, _shape_param_111);  div_4 = _shape_param_111 = None
        bmm_14: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg82_1, view_103);  arg82_1 = None
        bmm_15: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_103, arg83_1);  view_103 = arg83_1 = None
        view_104: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_112);  bmm_14 = _shape_param_112 = None
        view_105: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_15, _shape_param_113);  bmm_15 = _shape_param_113 = None
        convert_element_type_97: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_104, torch.float32);  view_104 = None
        permute_34: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_97, [0, 1, 3, 2]);  convert_element_type_97 = None
        convert_element_type_98: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_34, torch.bfloat16);  permute_34 = None
        convert_element_type_99: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_105, torch.bfloat16);  view_105 = None
        permute_35: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_99, [0, 2, 1, 3]);  convert_element_type_99 = None
        permute_36: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_98, [0, 2, 1, 3]);  convert_element_type_98 = None
        slice_14: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_35, 3, 0, 64)
        slice_15: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_35, 3, 64, 256);  permute_35 = None
        slice_16: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_36, 3, 0, 64)
        slice_17: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_36, 3, 64, 256);  permute_36 = None
        expand_7: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg84_1, _shape_param_114);  arg84_1 = _shape_param_114 = None
        clone_9: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_106: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_9, _shape_param_115);  clone_9 = _shape_param_115 = None
        mul_90: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_14, view_106)
        view_107: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_90, _shape_param_116);  mul_90 = _shape_param_116 = None
        select_12: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_107, -1, 0)
        select_13: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_107, -1, 1);  view_107 = None
        neg_10: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_12);  select_12 = None
        slice_scatter_24: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_10, 3, 1, 9223372036854775807, 2);  neg_10 = None
        slice_scatter_25: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_13, 3, 0, 9223372036854775807, 2);  select_13 = None
        add_47: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_24, slice_scatter_25);  slice_scatter_24 = slice_scatter_25 = None
        expand_8: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg85_1, _shape_param_117);  arg85_1 = _shape_param_117 = None
        clone_10: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_108: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_10, _shape_param_118);  clone_10 = _shape_param_118 = None
        mul_91: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_14, view_108);  slice_14 = None
        add_48: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_47, mul_91);  add_47 = mul_91 = None
        mul_92: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_16, view_106);  view_106 = None
        view_109: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_92, _shape_param_119);  mul_92 = _shape_param_119 = None
        select_14: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_109, -1, 0)
        select_15: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_109, -1, 1);  view_109 = None
        neg_11: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_14);  select_14 = None
        slice_scatter_26: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_11, 3, 1, 9223372036854775807, 2);  neg_11 = None
        slice_scatter_27: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_15, 3, 0, 9223372036854775807, 2);  select_15 = None
        add_49: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_26, slice_scatter_27);  slice_scatter_26 = slice_scatter_27 = None
        mul_93: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_16, view_108);  slice_16 = view_108 = None
        add_50: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_49, mul_93);  add_49 = mul_93 = None
        slice_scatter_28: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_15, 3, 64, 9223372036854775807);  slice_15 = None
        slice_scatter_29: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_48, 3, 0, 64);  add_48 = None
        add_51: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_28, slice_scatter_29);  slice_scatter_28 = slice_scatter_29 = None
        slice_scatter_30: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_17, 3, 64, 9223372036854775807);  slice_17 = None
        slice_scatter_31: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_50, 3, 0, 64);  add_50 = None
        add_52: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_30, slice_scatter_31);  slice_scatter_30 = slice_scatter_31 = None
        permute_37: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
        clone_11: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
        view_110: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_11, _shape_param_120);  clone_11 = _shape_param_120 = None
        view_111: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_52, _shape_param_121);  add_52 = _shape_param_121 = None
        view_112: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_51, _shape_param_122);  add_51 = _shape_param_122 = None
        view_113: "bf16[128, 4096]" = torch.ops.aten.view.default(view_110, _shape_param_123);  view_110 = _shape_param_123 = None
        permute_38: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_113, [1, 0])
        mm_44: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_38, arg76_1);  permute_38 = None
        mm_45: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_113, arg86_1);  view_113 = arg86_1 = None
        view_114: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_45, _shape_param_124);  mm_45 = _shape_param_124 = None
        convert_element_type_100: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_114, torch.float32);  view_114 = None
        add_53: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_91, convert_element_type_100);  convert_element_type_91 = convert_element_type_100 = None
        convert_element_type_101: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_44, torch.float32);  mm_44 = None
        view_115: "bf16[128, 4096]" = torch.ops.aten.view.default(view_111, _shape_param_125);  view_111 = _shape_param_125 = None
        permute_39: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_115, [1, 0])
        mm_46: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_39, arg76_1);  permute_39 = None
        mm_47: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_115, arg87_1);  view_115 = arg87_1 = None
        view_116: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_47, _shape_param_126);  mm_47 = _shape_param_126 = None
        convert_element_type_102: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_116, torch.float32);  view_116 = None
        add_54: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_53, convert_element_type_102);  add_53 = convert_element_type_102 = None
        convert_element_type_103: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_46, torch.float32);  mm_46 = None
        view_117: "bf16[128, 4096]" = torch.ops.aten.view.default(view_112, _shape_param_127);  view_112 = _shape_param_127 = None
        permute_40: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_117, [1, 0])
        mm_48: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_40, arg76_1);  permute_40 = arg76_1 = None
        mm_49: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_117, arg88_1);  view_117 = arg88_1 = None
        view_118: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_49, _shape_param_128);  mm_49 = _shape_param_128 = None
        convert_element_type_104: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_118, torch.float32);  view_118 = None
        add_55: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_54, convert_element_type_104);  add_54 = convert_element_type_104 = None
        convert_element_type_105: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_48, torch.float32);  mm_48 = None
        mul_94: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_55, arg89_1);  arg89_1 = None
        mul_95: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_94, 4096)
        sum_31: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_94, [2], True)
        mul_96: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_94, arg90_1);  mul_94 = None
        sum_32: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_96, [2], True);  mul_96 = None
        mul_97: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg90_1, sum_32);  sum_32 = None
        sub_15: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_95, sum_31);  mul_95 = sum_31 = None
        sub_16: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_15, mul_97);  sub_15 = mul_97 = None
        mul_98: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg91_1, sub_16);  arg91_1 = sub_16 = None
        mul_99: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_55, arg90_1);  arg90_1 = None
        sum_33: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_99, [0, 1]);  mul_99 = None
        sum_34: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_55, [0, 1]);  add_55 = None
        add_56: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_42, mul_98);  add_42 = mul_98 = None
        convert_element_type_106: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16)
        view_119: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_106, _shape_param_129);  convert_element_type_106 = _shape_param_129 = None
        mm_50: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_119, arg92_1);  arg92_1 = None
        permute_41: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_119, [1, 0])
        mm_51: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_41, arg93_1);  arg93_1 = None
        sum_35: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_119, [0], True, dtype = torch.float32)
        view_120: "f32[4096]" = torch.ops.aten.view.default(sum_35, _shape_param_130);  sum_35 = _shape_param_130 = None
        convert_element_type_107: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_120, torch.bfloat16);  view_120 = None
        view_121: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_50, _shape_param_131);  mm_50 = _shape_param_131 = None
        convert_element_type_108: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_121, torch.float32);  view_121 = None
        convert_element_type_109: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_110: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_107, torch.float32);  convert_element_type_107 = None
        view_122: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg94_1, _shape_param_132);  arg94_1 = _shape_param_132 = None
        mul_100: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_122, 0.5)
        mul_101: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_108, mul_100);  mul_100 = None
        convert_element_type_111: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_122, torch.float32)
        pow_9: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_111, 3.0)
        mul_102: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_57: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_122, mul_102);  view_122 = mul_102 = None
        mul_103: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_4: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_103);  mul_103 = None
        add_58: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_104: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_108, add_58);  convert_element_type_108 = add_58 = None
        convert_element_type_112: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None
        mul_105: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_17: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_105);  mul_105 = None
        mul_106: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_101, sub_17);  mul_101 = sub_17 = None
        mul_107: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_106, 0.7978845608028654);  mul_106 = None
        convert_element_type_113: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_107, torch.bfloat16)
        mul_108: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_107, 0.044715);  mul_107 = None
        pow_10: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_111, 2.0);  convert_element_type_111 = None
        mul_109: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_10, 3.0);  pow_10 = None
        mul_110: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_108, mul_109);  mul_108 = mul_109 = None
        convert_element_type_114: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_110, torch.bfloat16);  mul_110 = None
        add_59: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_113, convert_element_type_114);  convert_element_type_113 = convert_element_type_114 = None
        mul_111: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_112, 0.5);  convert_element_type_112 = None
        add_60: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_59, mul_111);  add_59 = mul_111 = None
        view_123: "bf16[128, 16384]" = torch.ops.aten.view.default(add_60, _shape_param_133);  add_60 = _shape_param_133 = None
        mm_52: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_123, arg95_1);  arg95_1 = None
        permute_42: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_123, [1, 0])
        mm_53: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_42, arg96_1);  permute_42 = None
        sum_36: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_123, [0], True, dtype = torch.float32);  view_123 = None
        view_124: "f32[16384]" = torch.ops.aten.view.default(sum_36, _shape_param_134);  sum_36 = _shape_param_134 = None
        convert_element_type_115: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_124, torch.bfloat16);  view_124 = None
        view_125: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_52, _shape_param_135);  mm_52 = _shape_param_135 = None
        convert_element_type_116: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_125, torch.float32);  view_125 = None
        convert_element_type_117: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_118: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_115, torch.float32);  convert_element_type_115 = None
        mm_54: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_41, arg97_1);  permute_41 = arg97_1 = None
        mm_55: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_119, arg98_1);  view_119 = arg98_1 = None
        view_126: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_55, _shape_param_136);  mm_55 = _shape_param_136 = None
        convert_element_type_119: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_54, torch.float32);  mm_54 = None
        view_127: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_126, _shape_param_137);  view_126 = _shape_param_137 = None
        permute_43: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
        view_128: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_43, _shape_param_138);  permute_43 = _shape_param_138 = None
        bmm_16: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg99_1, view_128);  arg99_1 = None
        bmm_17: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_128, arg100_1);  view_128 = arg100_1 = None
        view_129: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_16, _shape_param_139);  bmm_16 = _shape_param_139 = None
        view_130: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_17, _shape_param_140);  bmm_17 = _shape_param_140 = None
        convert_element_type_120: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_130, torch.float32);  view_130 = None
        mul_112: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_120, arg101_1);  convert_element_type_120 = None
        sum_37: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_112, [-1], True)
        neg_12: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg101_1);  arg101_1 = None
        fma_4: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_12, sum_37, mul_112);  neg_12 = sum_37 = mul_112 = None
        convert_element_type_121: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_4, torch.bfloat16);  fma_4 = None
        div_5: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_121, 16.0);  convert_element_type_121 = None
        view_131: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_5, _shape_param_141);  div_5 = _shape_param_141 = None
        bmm_18: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg102_1, view_131);  arg102_1 = None
        bmm_19: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_131, arg103_1);  view_131 = arg103_1 = None
        view_132: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_142);  bmm_18 = _shape_param_142 = None
        view_133: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_19, _shape_param_143);  bmm_19 = _shape_param_143 = None
        convert_element_type_122: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_132, torch.float32);  view_132 = None
        permute_44: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_122, [0, 1, 3, 2]);  convert_element_type_122 = None
        convert_element_type_123: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_44, torch.bfloat16);  permute_44 = None
        convert_element_type_124: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_133, torch.bfloat16);  view_133 = None
        permute_45: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_124, [0, 2, 1, 3]);  convert_element_type_124 = None
        permute_46: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_123, [0, 2, 1, 3]);  convert_element_type_123 = None
        slice_18: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_45, 3, 0, 64)
        slice_19: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_45, 3, 64, 256);  permute_45 = None
        slice_20: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_46, 3, 0, 64)
        slice_21: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_46, 3, 64, 256);  permute_46 = None
        expand_9: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg104_1, _shape_param_144);  arg104_1 = _shape_param_144 = None
        clone_12: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_134: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_12, _shape_param_145);  clone_12 = _shape_param_145 = None
        mul_113: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_18, view_134)
        view_135: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_113, _shape_param_146);  mul_113 = _shape_param_146 = None
        select_16: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_135, -1, 0)
        select_17: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_135, -1, 1);  view_135 = None
        neg_13: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_16);  select_16 = None
        slice_scatter_32: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_13, 3, 1, 9223372036854775807, 2);  neg_13 = None
        slice_scatter_33: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_17, 3, 0, 9223372036854775807, 2);  select_17 = None
        add_61: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_33);  slice_scatter_32 = slice_scatter_33 = None
        expand_10: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg105_1, _shape_param_147);  arg105_1 = _shape_param_147 = None
        clone_13: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
        view_136: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_13, _shape_param_148);  clone_13 = _shape_param_148 = None
        mul_114: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_18, view_136);  slice_18 = None
        add_62: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_61, mul_114);  add_61 = mul_114 = None
        mul_115: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_20, view_134);  view_134 = None
        view_137: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_115, _shape_param_149);  mul_115 = _shape_param_149 = None
        select_18: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_137, -1, 0)
        select_19: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_137, -1, 1);  view_137 = None
        neg_14: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_18);  select_18 = None
        slice_scatter_34: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_14, 3, 1, 9223372036854775807, 2);  neg_14 = None
        slice_scatter_35: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_19, 3, 0, 9223372036854775807, 2);  select_19 = None
        add_63: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_34, slice_scatter_35);  slice_scatter_34 = slice_scatter_35 = None
        mul_116: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_20, view_136);  slice_20 = view_136 = None
        add_64: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_63, mul_116);  add_63 = mul_116 = None
        slice_scatter_36: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_19, 3, 64, 9223372036854775807);  slice_19 = None
        slice_scatter_37: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_62, 3, 0, 64);  add_62 = None
        add_65: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_36, slice_scatter_37);  slice_scatter_36 = slice_scatter_37 = None
        slice_scatter_38: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_21, 3, 64, 9223372036854775807);  slice_21 = None
        slice_scatter_39: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_64, 3, 0, 64);  add_64 = None
        add_66: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_38, slice_scatter_39);  slice_scatter_38 = slice_scatter_39 = None
        permute_47: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
        clone_14: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
        view_138: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_14, _shape_param_150);  clone_14 = _shape_param_150 = None
        view_139: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_66, _shape_param_151);  add_66 = _shape_param_151 = None
        view_140: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_65, _shape_param_152);  add_65 = _shape_param_152 = None
        view_141: "bf16[128, 4096]" = torch.ops.aten.view.default(view_138, _shape_param_153);  view_138 = _shape_param_153 = None
        permute_48: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_141, [1, 0])
        mm_56: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_48, arg96_1);  permute_48 = None
        mm_57: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_141, arg106_1);  view_141 = arg106_1 = None
        view_142: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_57, _shape_param_154);  mm_57 = _shape_param_154 = None
        convert_element_type_125: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_142, torch.float32);  view_142 = None
        add_67: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_116, convert_element_type_125);  convert_element_type_116 = convert_element_type_125 = None
        convert_element_type_126: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_56, torch.float32);  mm_56 = None
        view_143: "bf16[128, 4096]" = torch.ops.aten.view.default(view_139, _shape_param_155);  view_139 = _shape_param_155 = None
        permute_49: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_143, [1, 0])
        mm_58: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_49, arg96_1);  permute_49 = None
        mm_59: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_143, arg107_1);  view_143 = arg107_1 = None
        view_144: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_59, _shape_param_156);  mm_59 = _shape_param_156 = None
        convert_element_type_127: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_144, torch.float32);  view_144 = None
        add_68: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_67, convert_element_type_127);  add_67 = convert_element_type_127 = None
        convert_element_type_128: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_58, torch.float32);  mm_58 = None
        view_145: "bf16[128, 4096]" = torch.ops.aten.view.default(view_140, _shape_param_157);  view_140 = _shape_param_157 = None
        permute_50: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_145, [1, 0])
        mm_60: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_50, arg96_1);  permute_50 = arg96_1 = None
        mm_61: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_145, arg108_1);  view_145 = arg108_1 = None
        view_146: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_61, _shape_param_158);  mm_61 = _shape_param_158 = None
        convert_element_type_129: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_146, torch.float32);  view_146 = None
        add_69: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_68, convert_element_type_129);  add_68 = convert_element_type_129 = None
        convert_element_type_130: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_60, torch.float32);  mm_60 = None
        mul_117: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_69, arg109_1);  arg109_1 = None
        mul_118: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_117, 4096)
        sum_38: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [2], True)
        mul_119: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_117, arg110_1);  mul_117 = None
        sum_39: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_119, [2], True);  mul_119 = None
        mul_120: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg110_1, sum_39);  sum_39 = None
        sub_18: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_118, sum_38);  mul_118 = sum_38 = None
        sub_19: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_18, mul_120);  sub_18 = mul_120 = None
        mul_121: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg111_1, sub_19);  arg111_1 = sub_19 = None
        mul_122: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_69, arg110_1);  arg110_1 = None
        sum_40: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_122, [0, 1]);  mul_122 = None
        sum_41: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_69, [0, 1]);  add_69 = None
        add_70: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_56, mul_121);  add_56 = mul_121 = None
        convert_element_type_131: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16)
        view_147: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_131, _shape_param_159);  convert_element_type_131 = _shape_param_159 = None
        mm_62: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_147, arg112_1);  arg112_1 = None
        permute_51: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_147, [1, 0])
        mm_63: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_51, arg113_1);  arg113_1 = None
        sum_42: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_147, [0], True, dtype = torch.float32)
        view_148: "f32[4096]" = torch.ops.aten.view.default(sum_42, _shape_param_160);  sum_42 = _shape_param_160 = None
        convert_element_type_132: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_148, torch.bfloat16);  view_148 = None
        view_149: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_62, _shape_param_161);  mm_62 = _shape_param_161 = None
        convert_element_type_133: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_149, torch.float32);  view_149 = None
        convert_element_type_134: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_135: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_132, torch.float32);  convert_element_type_132 = None
        view_150: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg114_1, _shape_param_162);  arg114_1 = _shape_param_162 = None
        mul_123: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_150, 0.5)
        mul_124: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_133, mul_123);  mul_123 = None
        convert_element_type_136: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_150, torch.float32)
        pow_11: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_136, 3.0)
        mul_125: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_71: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_150, mul_125);  view_150 = mul_125 = None
        mul_126: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_71, 0.7978845608028654);  add_71 = None
        tanh_5: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
        add_72: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_127: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_133, add_72);  convert_element_type_133 = add_72 = None
        convert_element_type_137: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_127, torch.bfloat16);  mul_127 = None
        mul_128: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_20: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_128);  mul_128 = None
        mul_129: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_124, sub_20);  mul_124 = sub_20 = None
        mul_130: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_129, 0.7978845608028654);  mul_129 = None
        convert_element_type_138: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_130, torch.bfloat16)
        mul_131: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_130, 0.044715);  mul_130 = None
        pow_12: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_136, 2.0);  convert_element_type_136 = None
        mul_132: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_12, 3.0);  pow_12 = None
        mul_133: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_131, mul_132);  mul_131 = mul_132 = None
        convert_element_type_139: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_133, torch.bfloat16);  mul_133 = None
        add_73: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_138, convert_element_type_139);  convert_element_type_138 = convert_element_type_139 = None
        mul_134: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_137, 0.5);  convert_element_type_137 = None
        add_74: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_73, mul_134);  add_73 = mul_134 = None
        view_151: "bf16[128, 16384]" = torch.ops.aten.view.default(add_74, _shape_param_163);  add_74 = _shape_param_163 = None
        mm_64: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_151, arg115_1);  arg115_1 = None
        permute_52: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_151, [1, 0])
        mm_65: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_52, arg116_1);  permute_52 = None
        sum_43: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_151, [0], True, dtype = torch.float32);  view_151 = None
        view_152: "f32[16384]" = torch.ops.aten.view.default(sum_43, _shape_param_164);  sum_43 = _shape_param_164 = None
        convert_element_type_140: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_152, torch.bfloat16);  view_152 = None
        view_153: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_64, _shape_param_165);  mm_64 = _shape_param_165 = None
        convert_element_type_141: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_153, torch.float32);  view_153 = None
        convert_element_type_142: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_143: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_140, torch.float32);  convert_element_type_140 = None
        mm_66: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_51, arg117_1);  permute_51 = arg117_1 = None
        mm_67: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_147, arg118_1);  view_147 = arg118_1 = None
        view_154: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_67, _shape_param_166);  mm_67 = _shape_param_166 = None
        convert_element_type_144: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_66, torch.float32);  mm_66 = None
        view_155: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_154, _shape_param_167);  view_154 = _shape_param_167 = None
        permute_53: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
        view_156: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_53, _shape_param_168);  permute_53 = _shape_param_168 = None
        bmm_20: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg119_1, view_156);  arg119_1 = None
        bmm_21: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_156, arg120_1);  view_156 = arg120_1 = None
        view_157: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_20, _shape_param_169);  bmm_20 = _shape_param_169 = None
        view_158: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_21, _shape_param_170);  bmm_21 = _shape_param_170 = None
        convert_element_type_145: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_158, torch.float32);  view_158 = None
        mul_135: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_145, arg121_1);  convert_element_type_145 = None
        sum_44: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [-1], True)
        neg_15: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg121_1);  arg121_1 = None
        fma_5: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_15, sum_44, mul_135);  neg_15 = sum_44 = mul_135 = None
        convert_element_type_146: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_5, torch.bfloat16);  fma_5 = None
        div_6: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_146, 16.0);  convert_element_type_146 = None
        view_159: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_6, _shape_param_171);  div_6 = _shape_param_171 = None
        bmm_22: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg122_1, view_159);  arg122_1 = None
        bmm_23: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_159, arg123_1);  view_159 = arg123_1 = None
        view_160: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_172);  bmm_22 = _shape_param_172 = None
        view_161: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_23, _shape_param_173);  bmm_23 = _shape_param_173 = None
        convert_element_type_147: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_160, torch.float32);  view_160 = None
        permute_54: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_147, [0, 1, 3, 2]);  convert_element_type_147 = None
        convert_element_type_148: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_54, torch.bfloat16);  permute_54 = None
        convert_element_type_149: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_161, torch.bfloat16);  view_161 = None
        permute_55: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_149, [0, 2, 1, 3]);  convert_element_type_149 = None
        permute_56: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_148, [0, 2, 1, 3]);  convert_element_type_148 = None
        slice_22: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_55, 3, 0, 64)
        slice_23: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_55, 3, 64, 256);  permute_55 = None
        slice_24: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_56, 3, 0, 64)
        slice_25: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_56, 3, 64, 256);  permute_56 = None
        expand_11: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg124_1, _shape_param_174);  arg124_1 = _shape_param_174 = None
        clone_15: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_162: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_15, _shape_param_175);  clone_15 = _shape_param_175 = None
        mul_136: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_22, view_162)
        view_163: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_136, _shape_param_176);  mul_136 = _shape_param_176 = None
        select_20: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_163, -1, 0)
        select_21: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_163, -1, 1);  view_163 = None
        neg_16: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_20);  select_20 = None
        slice_scatter_40: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_16, 3, 1, 9223372036854775807, 2);  neg_16 = None
        slice_scatter_41: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_21, 3, 0, 9223372036854775807, 2);  select_21 = None
        add_75: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_40, slice_scatter_41);  slice_scatter_40 = slice_scatter_41 = None
        expand_12: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg125_1, _shape_param_177);  arg125_1 = _shape_param_177 = None
        clone_16: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_164: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_16, _shape_param_178);  clone_16 = _shape_param_178 = None
        mul_137: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_22, view_164);  slice_22 = None
        add_76: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_75, mul_137);  add_75 = mul_137 = None
        mul_138: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_24, view_162);  view_162 = None
        view_165: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_138, _shape_param_179);  mul_138 = _shape_param_179 = None
        select_22: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_165, -1, 0)
        select_23: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_165, -1, 1);  view_165 = None
        neg_17: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_22);  select_22 = None
        slice_scatter_42: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_17, 3, 1, 9223372036854775807, 2);  neg_17 = None
        slice_scatter_43: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_23, 3, 0, 9223372036854775807, 2);  select_23 = None
        add_77: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_42, slice_scatter_43);  slice_scatter_42 = slice_scatter_43 = None
        mul_139: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_24, view_164);  slice_24 = view_164 = None
        add_78: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_77, mul_139);  add_77 = mul_139 = None
        slice_scatter_44: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_23, 3, 64, 9223372036854775807);  slice_23 = None
        slice_scatter_45: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_76, 3, 0, 64);  add_76 = None
        add_79: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_44, slice_scatter_45);  slice_scatter_44 = slice_scatter_45 = None
        slice_scatter_46: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_25, 3, 64, 9223372036854775807);  slice_25 = None
        slice_scatter_47: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_78, 3, 0, 64);  add_78 = None
        add_80: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_46, slice_scatter_47);  slice_scatter_46 = slice_scatter_47 = None
        permute_57: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        clone_17: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
        view_166: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_17, _shape_param_180);  clone_17 = _shape_param_180 = None
        view_167: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_80, _shape_param_181);  add_80 = _shape_param_181 = None
        view_168: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_79, _shape_param_182);  add_79 = _shape_param_182 = None
        view_169: "bf16[128, 4096]" = torch.ops.aten.view.default(view_166, _shape_param_183);  view_166 = _shape_param_183 = None
        permute_58: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_169, [1, 0])
        mm_68: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_58, arg116_1);  permute_58 = None
        mm_69: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_169, arg126_1);  view_169 = arg126_1 = None
        view_170: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_69, _shape_param_184);  mm_69 = _shape_param_184 = None
        convert_element_type_150: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_170, torch.float32);  view_170 = None
        add_81: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_141, convert_element_type_150);  convert_element_type_141 = convert_element_type_150 = None
        convert_element_type_151: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_68, torch.float32);  mm_68 = None
        view_171: "bf16[128, 4096]" = torch.ops.aten.view.default(view_167, _shape_param_185);  view_167 = _shape_param_185 = None
        permute_59: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_171, [1, 0])
        mm_70: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_59, arg116_1);  permute_59 = None
        mm_71: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_171, arg127_1);  view_171 = arg127_1 = None
        view_172: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_71, _shape_param_186);  mm_71 = _shape_param_186 = None
        convert_element_type_152: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_172, torch.float32);  view_172 = None
        add_82: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_81, convert_element_type_152);  add_81 = convert_element_type_152 = None
        convert_element_type_153: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_70, torch.float32);  mm_70 = None
        view_173: "bf16[128, 4096]" = torch.ops.aten.view.default(view_168, _shape_param_187);  view_168 = _shape_param_187 = None
        permute_60: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_173, [1, 0])
        mm_72: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_60, arg116_1);  permute_60 = arg116_1 = None
        mm_73: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_173, arg128_1);  view_173 = arg128_1 = None
        view_174: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_73, _shape_param_188);  mm_73 = _shape_param_188 = None
        convert_element_type_154: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_174, torch.float32);  view_174 = None
        add_83: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_82, convert_element_type_154);  add_82 = convert_element_type_154 = None
        convert_element_type_155: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_72, torch.float32);  mm_72 = None
        mul_140: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_83, arg129_1);  arg129_1 = None
        mul_141: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_140, 4096)
        sum_45: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_140, [2], True)
        mul_142: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_140, arg130_1);  mul_140 = None
        sum_46: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_142, [2], True);  mul_142 = None
        mul_143: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg130_1, sum_46);  sum_46 = None
        sub_21: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_141, sum_45);  mul_141 = sum_45 = None
        sub_22: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_21, mul_143);  sub_21 = mul_143 = None
        mul_144: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg131_1, sub_22);  arg131_1 = sub_22 = None
        mul_145: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_83, arg130_1);  arg130_1 = None
        sum_47: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_145, [0, 1]);  mul_145 = None
        sum_48: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_83, [0, 1]);  add_83 = None
        add_84: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_70, mul_144);  add_70 = mul_144 = None
        convert_element_type_156: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16)
        view_175: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_156, _shape_param_189);  convert_element_type_156 = _shape_param_189 = None
        mm_74: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_175, arg132_1);  arg132_1 = None
        permute_61: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_175, [1, 0])
        mm_75: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_61, arg133_1);  arg133_1 = None
        sum_49: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_175, [0], True, dtype = torch.float32)
        view_176: "f32[4096]" = torch.ops.aten.view.default(sum_49, _shape_param_190);  sum_49 = _shape_param_190 = None
        convert_element_type_157: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_176, torch.bfloat16);  view_176 = None
        view_177: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_74, _shape_param_191);  mm_74 = _shape_param_191 = None
        convert_element_type_158: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_177, torch.float32);  view_177 = None
        convert_element_type_159: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_160: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_157, torch.float32);  convert_element_type_157 = None
        view_178: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg134_1, _shape_param_192);  arg134_1 = _shape_param_192 = None
        mul_146: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
        mul_147: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_158, mul_146);  mul_146 = None
        convert_element_type_161: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_178, torch.float32)
        pow_13: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_161, 3.0)
        mul_148: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_85: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_178, mul_148);  view_178 = mul_148 = None
        mul_149: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_85, 0.7978845608028654);  add_85 = None
        tanh_6: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_149);  mul_149 = None
        add_86: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_150: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_158, add_86);  convert_element_type_158 = add_86 = None
        convert_element_type_162: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_150, torch.bfloat16);  mul_150 = None
        mul_151: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_23: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_151);  mul_151 = None
        mul_152: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_147, sub_23);  mul_147 = sub_23 = None
        mul_153: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_152, 0.7978845608028654);  mul_152 = None
        convert_element_type_163: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_153, torch.bfloat16)
        mul_154: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_153, 0.044715);  mul_153 = None
        pow_14: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_161, 2.0);  convert_element_type_161 = None
        mul_155: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_14, 3.0);  pow_14 = None
        mul_156: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
        convert_element_type_164: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_156, torch.bfloat16);  mul_156 = None
        add_87: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_163, convert_element_type_164);  convert_element_type_163 = convert_element_type_164 = None
        mul_157: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_162, 0.5);  convert_element_type_162 = None
        add_88: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_87, mul_157);  add_87 = mul_157 = None
        view_179: "bf16[128, 16384]" = torch.ops.aten.view.default(add_88, _shape_param_193);  add_88 = _shape_param_193 = None
        mm_76: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_179, arg135_1);  arg135_1 = None
        permute_62: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_179, [1, 0])
        mm_77: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_62, arg136_1);  permute_62 = None
        sum_50: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True, dtype = torch.float32);  view_179 = None
        view_180: "f32[16384]" = torch.ops.aten.view.default(sum_50, _shape_param_194);  sum_50 = _shape_param_194 = None
        convert_element_type_165: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_180, torch.bfloat16);  view_180 = None
        view_181: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_76, _shape_param_195);  mm_76 = _shape_param_195 = None
        convert_element_type_166: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_181, torch.float32);  view_181 = None
        convert_element_type_167: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_168: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_165, torch.float32);  convert_element_type_165 = None
        mm_78: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_61, arg137_1);  permute_61 = arg137_1 = None
        mm_79: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_175, arg138_1);  view_175 = arg138_1 = None
        view_182: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_79, _shape_param_196);  mm_79 = _shape_param_196 = None
        convert_element_type_169: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_78, torch.float32);  mm_78 = None
        view_183: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_182, _shape_param_197);  view_182 = _shape_param_197 = None
        permute_63: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_183, [0, 2, 1, 3]);  view_183 = None
        view_184: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_63, _shape_param_198);  permute_63 = _shape_param_198 = None
        bmm_24: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg139_1, view_184);  arg139_1 = None
        bmm_25: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_184, arg140_1);  view_184 = arg140_1 = None
        view_185: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_24, _shape_param_199);  bmm_24 = _shape_param_199 = None
        view_186: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_25, _shape_param_200);  bmm_25 = _shape_param_200 = None
        convert_element_type_170: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_186, torch.float32);  view_186 = None
        mul_158: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_170, arg141_1);  convert_element_type_170 = None
        sum_51: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_158, [-1], True)
        neg_18: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg141_1);  arg141_1 = None
        fma_6: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_18, sum_51, mul_158);  neg_18 = sum_51 = mul_158 = None
        convert_element_type_171: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_6, torch.bfloat16);  fma_6 = None
        div_7: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_171, 16.0);  convert_element_type_171 = None
        view_187: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_7, _shape_param_201);  div_7 = _shape_param_201 = None
        bmm_26: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg142_1, view_187);  arg142_1 = None
        bmm_27: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_187, arg143_1);  view_187 = arg143_1 = None
        view_188: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_202);  bmm_26 = _shape_param_202 = None
        view_189: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_27, _shape_param_203);  bmm_27 = _shape_param_203 = None
        convert_element_type_172: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_188, torch.float32);  view_188 = None
        permute_64: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_172, [0, 1, 3, 2]);  convert_element_type_172 = None
        convert_element_type_173: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_64, torch.bfloat16);  permute_64 = None
        convert_element_type_174: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_189, torch.bfloat16);  view_189 = None
        permute_65: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_174, [0, 2, 1, 3]);  convert_element_type_174 = None
        permute_66: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_173, [0, 2, 1, 3]);  convert_element_type_173 = None
        slice_26: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_65, 3, 0, 64)
        slice_27: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_65, 3, 64, 256);  permute_65 = None
        slice_28: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_66, 3, 0, 64)
        slice_29: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_66, 3, 64, 256);  permute_66 = None
        expand_13: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg144_1, _shape_param_204);  arg144_1 = _shape_param_204 = None
        clone_18: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_190: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_18, _shape_param_205);  clone_18 = _shape_param_205 = None
        mul_159: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_26, view_190)
        view_191: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_159, _shape_param_206);  mul_159 = _shape_param_206 = None
        select_24: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_191, -1, 0)
        select_25: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_191, -1, 1);  view_191 = None
        neg_19: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_24);  select_24 = None
        slice_scatter_48: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_19, 3, 1, 9223372036854775807, 2);  neg_19 = None
        slice_scatter_49: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_25, 3, 0, 9223372036854775807, 2);  select_25 = None
        add_89: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_48, slice_scatter_49);  slice_scatter_48 = slice_scatter_49 = None
        expand_14: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg145_1, _shape_param_207);  arg145_1 = _shape_param_207 = None
        clone_19: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
        view_192: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_19, _shape_param_208);  clone_19 = _shape_param_208 = None
        mul_160: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_26, view_192);  slice_26 = None
        add_90: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_89, mul_160);  add_89 = mul_160 = None
        mul_161: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_28, view_190);  view_190 = None
        view_193: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_161, _shape_param_209);  mul_161 = _shape_param_209 = None
        select_26: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_193, -1, 0)
        select_27: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_193, -1, 1);  view_193 = None
        neg_20: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_26);  select_26 = None
        slice_scatter_50: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_20, 3, 1, 9223372036854775807, 2);  neg_20 = None
        slice_scatter_51: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_27, 3, 0, 9223372036854775807, 2);  select_27 = None
        add_91: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_50, slice_scatter_51);  slice_scatter_50 = slice_scatter_51 = None
        mul_162: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_28, view_192);  slice_28 = view_192 = None
        add_92: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_91, mul_162);  add_91 = mul_162 = None
        slice_scatter_52: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_27, 3, 64, 9223372036854775807);  slice_27 = None
        slice_scatter_53: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_90, 3, 0, 64);  add_90 = None
        add_93: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_52, slice_scatter_53);  slice_scatter_52 = slice_scatter_53 = None
        slice_scatter_54: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_29, 3, 64, 9223372036854775807);  slice_29 = None
        slice_scatter_55: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_92, 3, 0, 64);  add_92 = None
        add_94: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_54, slice_scatter_55);  slice_scatter_54 = slice_scatter_55 = None
        permute_67: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
        clone_20: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
        view_194: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_20, _shape_param_210);  clone_20 = _shape_param_210 = None
        view_195: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_94, _shape_param_211);  add_94 = _shape_param_211 = None
        view_196: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_93, _shape_param_212);  add_93 = _shape_param_212 = None
        view_197: "bf16[128, 4096]" = torch.ops.aten.view.default(view_194, _shape_param_213);  view_194 = _shape_param_213 = None
        permute_68: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_197, [1, 0])
        mm_80: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_68, arg136_1);  permute_68 = None
        mm_81: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_197, arg146_1);  view_197 = arg146_1 = None
        view_198: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_81, _shape_param_214);  mm_81 = _shape_param_214 = None
        convert_element_type_175: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_198, torch.float32);  view_198 = None
        add_95: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_166, convert_element_type_175);  convert_element_type_166 = convert_element_type_175 = None
        convert_element_type_176: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_80, torch.float32);  mm_80 = None
        view_199: "bf16[128, 4096]" = torch.ops.aten.view.default(view_195, _shape_param_215);  view_195 = _shape_param_215 = None
        permute_69: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_199, [1, 0])
        mm_82: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_69, arg136_1);  permute_69 = None
        mm_83: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_199, arg147_1);  view_199 = arg147_1 = None
        view_200: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_83, _shape_param_216);  mm_83 = _shape_param_216 = None
        convert_element_type_177: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_200, torch.float32);  view_200 = None
        add_96: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_95, convert_element_type_177);  add_95 = convert_element_type_177 = None
        convert_element_type_178: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_82, torch.float32);  mm_82 = None
        view_201: "bf16[128, 4096]" = torch.ops.aten.view.default(view_196, _shape_param_217);  view_196 = _shape_param_217 = None
        permute_70: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_201, [1, 0])
        mm_84: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_70, arg136_1);  permute_70 = arg136_1 = None
        mm_85: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_201, arg148_1);  view_201 = arg148_1 = None
        view_202: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_85, _shape_param_218);  mm_85 = _shape_param_218 = None
        convert_element_type_179: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_202, torch.float32);  view_202 = None
        add_97: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_96, convert_element_type_179);  add_96 = convert_element_type_179 = None
        convert_element_type_180: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_84, torch.float32);  mm_84 = None
        mul_163: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_97, arg149_1);  arg149_1 = None
        mul_164: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_163, 4096)
        sum_52: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_163, [2], True)
        mul_165: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_163, arg150_1);  mul_163 = None
        sum_53: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True);  mul_165 = None
        mul_166: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg150_1, sum_53);  sum_53 = None
        sub_24: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_164, sum_52);  mul_164 = sum_52 = None
        sub_25: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_24, mul_166);  sub_24 = mul_166 = None
        mul_167: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg151_1, sub_25);  arg151_1 = sub_25 = None
        mul_168: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_97, arg150_1);  arg150_1 = None
        sum_54: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_168, [0, 1]);  mul_168 = None
        sum_55: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_97, [0, 1]);  add_97 = None
        add_98: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_84, mul_167);  add_84 = mul_167 = None
        convert_element_type_181: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16)
        view_203: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_181, _shape_param_219);  convert_element_type_181 = _shape_param_219 = None
        mm_86: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_203, arg152_1);  arg152_1 = None
        permute_71: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_203, [1, 0])
        mm_87: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_71, arg153_1);  arg153_1 = None
        sum_56: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_203, [0], True, dtype = torch.float32)
        view_204: "f32[4096]" = torch.ops.aten.view.default(sum_56, _shape_param_220);  sum_56 = _shape_param_220 = None
        convert_element_type_182: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_204, torch.bfloat16);  view_204 = None
        view_205: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_86, _shape_param_221);  mm_86 = _shape_param_221 = None
        convert_element_type_183: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_205, torch.float32);  view_205 = None
        convert_element_type_184: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_185: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_182, torch.float32);  convert_element_type_182 = None
        view_206: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg154_1, _shape_param_222);  arg154_1 = _shape_param_222 = None
        mul_169: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_206, 0.5)
        mul_170: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_183, mul_169);  mul_169 = None
        convert_element_type_186: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_206, torch.float32)
        pow_15: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_186, 3.0)
        mul_171: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_99: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_206, mul_171);  view_206 = mul_171 = None
        mul_172: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_99, 0.7978845608028654);  add_99 = None
        tanh_7: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_172);  mul_172 = None
        add_100: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_173: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_183, add_100);  convert_element_type_183 = add_100 = None
        convert_element_type_187: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_173, torch.bfloat16);  mul_173 = None
        mul_174: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_26: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_174);  mul_174 = None
        mul_175: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_170, sub_26);  mul_170 = sub_26 = None
        mul_176: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_175, 0.7978845608028654);  mul_175 = None
        convert_element_type_188: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_176, torch.bfloat16)
        mul_177: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_176, 0.044715);  mul_176 = None
        pow_16: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_186, 2.0);  convert_element_type_186 = None
        mul_178: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_16, 3.0);  pow_16 = None
        mul_179: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_177, mul_178);  mul_177 = mul_178 = None
        convert_element_type_189: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_179, torch.bfloat16);  mul_179 = None
        add_101: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_188, convert_element_type_189);  convert_element_type_188 = convert_element_type_189 = None
        mul_180: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_187, 0.5);  convert_element_type_187 = None
        add_102: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_101, mul_180);  add_101 = mul_180 = None
        view_207: "bf16[128, 16384]" = torch.ops.aten.view.default(add_102, _shape_param_223);  add_102 = _shape_param_223 = None
        mm_88: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_207, arg155_1);  arg155_1 = None
        permute_72: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_207, [1, 0])
        mm_89: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_72, arg156_1);  permute_72 = None
        sum_57: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_207, [0], True, dtype = torch.float32);  view_207 = None
        view_208: "f32[16384]" = torch.ops.aten.view.default(sum_57, _shape_param_224);  sum_57 = _shape_param_224 = None
        convert_element_type_190: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_208, torch.bfloat16);  view_208 = None
        view_209: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_88, _shape_param_225);  mm_88 = _shape_param_225 = None
        convert_element_type_191: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_209, torch.float32);  view_209 = None
        convert_element_type_192: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_193: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_190, torch.float32);  convert_element_type_190 = None
        mm_90: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_71, arg157_1);  permute_71 = arg157_1 = None
        mm_91: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_203, arg158_1);  view_203 = arg158_1 = None
        view_210: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_91, _shape_param_226);  mm_91 = _shape_param_226 = None
        convert_element_type_194: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_90, torch.float32);  mm_90 = None
        view_211: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_210, _shape_param_227);  view_210 = _shape_param_227 = None
        permute_73: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
        view_212: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_73, _shape_param_228);  permute_73 = _shape_param_228 = None
        bmm_28: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg159_1, view_212);  arg159_1 = None
        bmm_29: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_212, arg160_1);  view_212 = arg160_1 = None
        view_213: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_28, _shape_param_229);  bmm_28 = _shape_param_229 = None
        view_214: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_29, _shape_param_230);  bmm_29 = _shape_param_230 = None
        convert_element_type_195: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_214, torch.float32);  view_214 = None
        mul_181: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_195, arg161_1);  convert_element_type_195 = None
        sum_58: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [-1], True)
        neg_21: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg161_1);  arg161_1 = None
        fma_7: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_21, sum_58, mul_181);  neg_21 = sum_58 = mul_181 = None
        convert_element_type_196: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_7, torch.bfloat16);  fma_7 = None
        div_8: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_196, 16.0);  convert_element_type_196 = None
        view_215: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_8, _shape_param_231);  div_8 = _shape_param_231 = None
        bmm_30: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg162_1, view_215);  arg162_1 = None
        bmm_31: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_215, arg163_1);  view_215 = arg163_1 = None
        view_216: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_232);  bmm_30 = _shape_param_232 = None
        view_217: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_31, _shape_param_233);  bmm_31 = _shape_param_233 = None
        convert_element_type_197: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_216, torch.float32);  view_216 = None
        permute_74: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_197, [0, 1, 3, 2]);  convert_element_type_197 = None
        convert_element_type_198: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_74, torch.bfloat16);  permute_74 = None
        convert_element_type_199: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_217, torch.bfloat16);  view_217 = None
        permute_75: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_199, [0, 2, 1, 3]);  convert_element_type_199 = None
        permute_76: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_198, [0, 2, 1, 3]);  convert_element_type_198 = None
        slice_30: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_75, 3, 0, 64)
        slice_31: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_75, 3, 64, 256);  permute_75 = None
        slice_32: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_76, 3, 0, 64)
        slice_33: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_76, 3, 64, 256);  permute_76 = None
        expand_15: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg164_1, _shape_param_234);  arg164_1 = _shape_param_234 = None
        clone_21: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_218: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_21, _shape_param_235);  clone_21 = _shape_param_235 = None
        mul_182: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_30, view_218)
        view_219: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_182, _shape_param_236);  mul_182 = _shape_param_236 = None
        select_28: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_219, -1, 0)
        select_29: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_219, -1, 1);  view_219 = None
        neg_22: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_28);  select_28 = None
        slice_scatter_56: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_22, 3, 1, 9223372036854775807, 2);  neg_22 = None
        slice_scatter_57: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_29, 3, 0, 9223372036854775807, 2);  select_29 = None
        add_103: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_56, slice_scatter_57);  slice_scatter_56 = slice_scatter_57 = None
        expand_16: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg165_1, _shape_param_237);  arg165_1 = _shape_param_237 = None
        clone_22: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_220: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_22, _shape_param_238);  clone_22 = _shape_param_238 = None
        mul_183: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_30, view_220);  slice_30 = None
        add_104: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_103, mul_183);  add_103 = mul_183 = None
        mul_184: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_32, view_218);  view_218 = None
        view_221: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_184, _shape_param_239);  mul_184 = _shape_param_239 = None
        select_30: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_221, -1, 0)
        select_31: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_221, -1, 1);  view_221 = None
        neg_23: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_30);  select_30 = None
        slice_scatter_58: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_23, 3, 1, 9223372036854775807, 2);  neg_23 = None
        slice_scatter_59: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_31, 3, 0, 9223372036854775807, 2);  select_31 = None
        add_105: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_58, slice_scatter_59);  slice_scatter_58 = slice_scatter_59 = None
        mul_185: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_32, view_220);  slice_32 = view_220 = None
        add_106: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_105, mul_185);  add_105 = mul_185 = None
        slice_scatter_60: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_31, 3, 64, 9223372036854775807);  slice_31 = None
        slice_scatter_61: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_104, 3, 0, 64);  add_104 = None
        add_107: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_60, slice_scatter_61);  slice_scatter_60 = slice_scatter_61 = None
        slice_scatter_62: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_33, 3, 64, 9223372036854775807);  slice_33 = None
        slice_scatter_63: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_106, 3, 0, 64);  add_106 = None
        add_108: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_62, slice_scatter_63);  slice_scatter_62 = slice_scatter_63 = None
        permute_77: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_213, [0, 2, 1, 3]);  view_213 = None
        clone_23: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
        view_222: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_23, _shape_param_240);  clone_23 = _shape_param_240 = None
        view_223: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_108, _shape_param_241);  add_108 = _shape_param_241 = None
        view_224: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_107, _shape_param_242);  add_107 = _shape_param_242 = None
        view_225: "bf16[128, 4096]" = torch.ops.aten.view.default(view_222, _shape_param_243);  view_222 = _shape_param_243 = None
        permute_78: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_225, [1, 0])
        mm_92: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_78, arg156_1);  permute_78 = None
        mm_93: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_225, arg166_1);  view_225 = arg166_1 = None
        view_226: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_93, _shape_param_244);  mm_93 = _shape_param_244 = None
        convert_element_type_200: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_226, torch.float32);  view_226 = None
        add_109: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_191, convert_element_type_200);  convert_element_type_191 = convert_element_type_200 = None
        convert_element_type_201: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_92, torch.float32);  mm_92 = None
        view_227: "bf16[128, 4096]" = torch.ops.aten.view.default(view_223, _shape_param_245);  view_223 = _shape_param_245 = None
        permute_79: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_227, [1, 0])
        mm_94: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_79, arg156_1);  permute_79 = None
        mm_95: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_227, arg167_1);  view_227 = arg167_1 = None
        view_228: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_95, _shape_param_246);  mm_95 = _shape_param_246 = None
        convert_element_type_202: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_228, torch.float32);  view_228 = None
        add_110: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_109, convert_element_type_202);  add_109 = convert_element_type_202 = None
        convert_element_type_203: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_94, torch.float32);  mm_94 = None
        view_229: "bf16[128, 4096]" = torch.ops.aten.view.default(view_224, _shape_param_247);  view_224 = _shape_param_247 = None
        permute_80: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_229, [1, 0])
        mm_96: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_80, arg156_1);  permute_80 = arg156_1 = None
        mm_97: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_229, arg168_1);  view_229 = arg168_1 = None
        view_230: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_97, _shape_param_248);  mm_97 = _shape_param_248 = None
        convert_element_type_204: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_230, torch.float32);  view_230 = None
        add_111: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_110, convert_element_type_204);  add_110 = convert_element_type_204 = None
        convert_element_type_205: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_96, torch.float32);  mm_96 = None
        mul_186: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_111, arg169_1);  arg169_1 = None
        mul_187: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_186, 4096)
        sum_59: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True)
        mul_188: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_186, arg170_1);  mul_186 = None
        sum_60: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True);  mul_188 = None
        mul_189: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg170_1, sum_60);  sum_60 = None
        sub_27: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_187, sum_59);  mul_187 = sum_59 = None
        sub_28: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_27, mul_189);  sub_27 = mul_189 = None
        mul_190: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg171_1, sub_28);  arg171_1 = sub_28 = None
        mul_191: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_111, arg170_1);  arg170_1 = None
        sum_61: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 1]);  mul_191 = None
        sum_62: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_111, [0, 1]);  add_111 = None
        add_112: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_98, mul_190);  add_98 = mul_190 = None
        convert_element_type_206: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16)
        view_231: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_206, _shape_param_249);  convert_element_type_206 = _shape_param_249 = None
        mm_98: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_231, arg172_1);  arg172_1 = None
        permute_81: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_231, [1, 0])
        mm_99: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_81, arg173_1);  arg173_1 = None
        sum_63: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True, dtype = torch.float32)
        view_232: "f32[4096]" = torch.ops.aten.view.default(sum_63, _shape_param_250);  sum_63 = _shape_param_250 = None
        convert_element_type_207: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_232, torch.bfloat16);  view_232 = None
        view_233: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_98, _shape_param_251);  mm_98 = _shape_param_251 = None
        convert_element_type_208: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_233, torch.float32);  view_233 = None
        convert_element_type_209: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_210: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_207, torch.float32);  convert_element_type_207 = None
        view_234: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg174_1, _shape_param_252);  arg174_1 = _shape_param_252 = None
        mul_192: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_234, 0.5)
        mul_193: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_208, mul_192);  mul_192 = None
        convert_element_type_211: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_234, torch.float32)
        pow_17: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_211, 3.0)
        mul_194: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_113: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_234, mul_194);  view_234 = mul_194 = None
        mul_195: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        tanh_8: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_195);  mul_195 = None
        add_114: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_196: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_208, add_114);  convert_element_type_208 = add_114 = None
        convert_element_type_212: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_196, torch.bfloat16);  mul_196 = None
        mul_197: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_29: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_197);  mul_197 = None
        mul_198: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_193, sub_29);  mul_193 = sub_29 = None
        mul_199: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_198, 0.7978845608028654);  mul_198 = None
        convert_element_type_213: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_199, torch.bfloat16)
        mul_200: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_199, 0.044715);  mul_199 = None
        pow_18: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_211, 2.0);  convert_element_type_211 = None
        mul_201: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_18, 3.0);  pow_18 = None
        mul_202: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_200, mul_201);  mul_200 = mul_201 = None
        convert_element_type_214: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_202, torch.bfloat16);  mul_202 = None
        add_115: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_213, convert_element_type_214);  convert_element_type_213 = convert_element_type_214 = None
        mul_203: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_212, 0.5);  convert_element_type_212 = None
        add_116: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_115, mul_203);  add_115 = mul_203 = None
        view_235: "bf16[128, 16384]" = torch.ops.aten.view.default(add_116, _shape_param_253);  add_116 = _shape_param_253 = None
        mm_100: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_235, arg175_1);  arg175_1 = None
        permute_82: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_235, [1, 0])
        mm_101: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_82, arg176_1);  permute_82 = None
        sum_64: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_235, [0], True, dtype = torch.float32);  view_235 = None
        view_236: "f32[16384]" = torch.ops.aten.view.default(sum_64, _shape_param_254);  sum_64 = _shape_param_254 = None
        convert_element_type_215: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_236, torch.bfloat16);  view_236 = None
        view_237: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_100, _shape_param_255);  mm_100 = _shape_param_255 = None
        convert_element_type_216: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_237, torch.float32);  view_237 = None
        convert_element_type_217: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_218: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_215, torch.float32);  convert_element_type_215 = None
        mm_102: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_81, arg177_1);  permute_81 = arg177_1 = None
        mm_103: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_231, arg178_1);  view_231 = arg178_1 = None
        view_238: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_103, _shape_param_256);  mm_103 = _shape_param_256 = None
        convert_element_type_219: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_102, torch.float32);  mm_102 = None
        view_239: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_238, _shape_param_257);  view_238 = _shape_param_257 = None
        permute_83: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
        view_240: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_83, _shape_param_258);  permute_83 = _shape_param_258 = None
        bmm_32: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg179_1, view_240);  arg179_1 = None
        bmm_33: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_240, arg180_1);  view_240 = arg180_1 = None
        view_241: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_32, _shape_param_259);  bmm_32 = _shape_param_259 = None
        view_242: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_33, _shape_param_260);  bmm_33 = _shape_param_260 = None
        convert_element_type_220: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_242, torch.float32);  view_242 = None
        mul_204: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_220, arg181_1);  convert_element_type_220 = None
        sum_65: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_204, [-1], True)
        neg_24: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg181_1);  arg181_1 = None
        fma_8: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_24, sum_65, mul_204);  neg_24 = sum_65 = mul_204 = None
        convert_element_type_221: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_8, torch.bfloat16);  fma_8 = None
        div_9: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_221, 16.0);  convert_element_type_221 = None
        view_243: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_9, _shape_param_261);  div_9 = _shape_param_261 = None
        bmm_34: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg182_1, view_243);  arg182_1 = None
        bmm_35: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_243, arg183_1);  view_243 = arg183_1 = None
        view_244: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_262);  bmm_34 = _shape_param_262 = None
        view_245: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_35, _shape_param_263);  bmm_35 = _shape_param_263 = None
        convert_element_type_222: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_244, torch.float32);  view_244 = None
        permute_84: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_222, [0, 1, 3, 2]);  convert_element_type_222 = None
        convert_element_type_223: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_84, torch.bfloat16);  permute_84 = None
        convert_element_type_224: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_245, torch.bfloat16);  view_245 = None
        permute_85: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_224, [0, 2, 1, 3]);  convert_element_type_224 = None
        permute_86: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_223, [0, 2, 1, 3]);  convert_element_type_223 = None
        slice_34: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_85, 3, 0, 64)
        slice_35: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_85, 3, 64, 256);  permute_85 = None
        slice_36: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_86, 3, 0, 64)
        slice_37: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_86, 3, 64, 256);  permute_86 = None
        expand_17: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg184_1, _shape_param_264);  arg184_1 = _shape_param_264 = None
        clone_24: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_246: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_24, _shape_param_265);  clone_24 = _shape_param_265 = None
        mul_205: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_34, view_246)
        view_247: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_205, _shape_param_266);  mul_205 = _shape_param_266 = None
        select_32: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_247, -1, 0)
        select_33: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_247, -1, 1);  view_247 = None
        neg_25: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_32);  select_32 = None
        slice_scatter_64: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_25, 3, 1, 9223372036854775807, 2);  neg_25 = None
        slice_scatter_65: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_33, 3, 0, 9223372036854775807, 2);  select_33 = None
        add_117: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_64, slice_scatter_65);  slice_scatter_64 = slice_scatter_65 = None
        expand_18: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg185_1, _shape_param_267);  arg185_1 = _shape_param_267 = None
        clone_25: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_248: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_25, _shape_param_268);  clone_25 = _shape_param_268 = None
        mul_206: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_34, view_248);  slice_34 = None
        add_118: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_117, mul_206);  add_117 = mul_206 = None
        mul_207: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_36, view_246);  view_246 = None
        view_249: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_207, _shape_param_269);  mul_207 = _shape_param_269 = None
        select_34: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_249, -1, 0)
        select_35: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_249, -1, 1);  view_249 = None
        neg_26: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_34);  select_34 = None
        slice_scatter_66: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_26, 3, 1, 9223372036854775807, 2);  neg_26 = None
        slice_scatter_67: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_35, 3, 0, 9223372036854775807, 2);  select_35 = None
        add_119: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_66, slice_scatter_67);  slice_scatter_66 = slice_scatter_67 = None
        mul_208: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_36, view_248);  slice_36 = view_248 = None
        add_120: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_119, mul_208);  add_119 = mul_208 = None
        slice_scatter_68: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_35, 3, 64, 9223372036854775807);  slice_35 = None
        slice_scatter_69: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_118, 3, 0, 64);  add_118 = None
        add_121: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_68, slice_scatter_69);  slice_scatter_68 = slice_scatter_69 = None
        slice_scatter_70: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_37, 3, 64, 9223372036854775807);  slice_37 = None
        slice_scatter_71: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_120, 3, 0, 64);  add_120 = None
        add_122: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_70, slice_scatter_71);  slice_scatter_70 = slice_scatter_71 = None
        permute_87: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_241, [0, 2, 1, 3]);  view_241 = None
        clone_26: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_87, memory_format = torch.contiguous_format);  permute_87 = None
        view_250: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_26, _shape_param_270);  clone_26 = _shape_param_270 = None
        view_251: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_122, _shape_param_271);  add_122 = _shape_param_271 = None
        view_252: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_121, _shape_param_272);  add_121 = _shape_param_272 = None
        view_253: "bf16[128, 4096]" = torch.ops.aten.view.default(view_250, _shape_param_273);  view_250 = _shape_param_273 = None
        permute_88: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_253, [1, 0])
        mm_104: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_88, arg176_1);  permute_88 = None
        mm_105: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_253, arg186_1);  view_253 = arg186_1 = None
        view_254: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_105, _shape_param_274);  mm_105 = _shape_param_274 = None
        convert_element_type_225: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_254, torch.float32);  view_254 = None
        add_123: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_216, convert_element_type_225);  convert_element_type_216 = convert_element_type_225 = None
        convert_element_type_226: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_104, torch.float32);  mm_104 = None
        view_255: "bf16[128, 4096]" = torch.ops.aten.view.default(view_251, _shape_param_275);  view_251 = _shape_param_275 = None
        permute_89: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_255, [1, 0])
        mm_106: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_89, arg176_1);  permute_89 = None
        mm_107: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_255, arg187_1);  view_255 = arg187_1 = None
        view_256: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_107, _shape_param_276);  mm_107 = _shape_param_276 = None
        convert_element_type_227: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_256, torch.float32);  view_256 = None
        add_124: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_123, convert_element_type_227);  add_123 = convert_element_type_227 = None
        convert_element_type_228: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_106, torch.float32);  mm_106 = None
        view_257: "bf16[128, 4096]" = torch.ops.aten.view.default(view_252, _shape_param_277);  view_252 = _shape_param_277 = None
        permute_90: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_257, [1, 0])
        mm_108: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_90, arg176_1);  permute_90 = arg176_1 = None
        mm_109: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_257, arg188_1);  view_257 = arg188_1 = None
        view_258: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_109, _shape_param_278);  mm_109 = _shape_param_278 = None
        convert_element_type_229: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_258, torch.float32);  view_258 = None
        add_125: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_124, convert_element_type_229);  add_124 = convert_element_type_229 = None
        convert_element_type_230: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_108, torch.float32);  mm_108 = None
        mul_209: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_125, arg189_1);  arg189_1 = None
        mul_210: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_209, 4096)
        sum_66: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [2], True)
        mul_211: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_209, arg190_1);  mul_209 = None
        sum_67: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
        mul_212: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg190_1, sum_67);  sum_67 = None
        sub_30: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_210, sum_66);  mul_210 = sum_66 = None
        sub_31: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_30, mul_212);  sub_30 = mul_212 = None
        mul_213: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg191_1, sub_31);  arg191_1 = sub_31 = None
        mul_214: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_125, arg190_1);  arg190_1 = None
        sum_68: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1]);  mul_214 = None
        sum_69: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_125, [0, 1]);  add_125 = None
        add_126: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_112, mul_213);  add_112 = mul_213 = None
        convert_element_type_231: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_126, torch.bfloat16)
        view_259: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_231, _shape_param_279);  convert_element_type_231 = _shape_param_279 = None
        mm_110: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_259, arg192_1);  arg192_1 = None
        permute_91: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_259, [1, 0])
        mm_111: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_91, arg193_1);  arg193_1 = None
        sum_70: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_259, [0], True, dtype = torch.float32)
        view_260: "f32[4096]" = torch.ops.aten.view.default(sum_70, _shape_param_280);  sum_70 = _shape_param_280 = None
        convert_element_type_232: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_260, torch.bfloat16);  view_260 = None
        view_261: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_110, _shape_param_281);  mm_110 = _shape_param_281 = None
        convert_element_type_233: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        convert_element_type_234: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_235: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_232, torch.float32);  convert_element_type_232 = None
        view_262: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg194_1, _shape_param_282);  arg194_1 = _shape_param_282 = None
        mul_215: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_262, 0.5)
        mul_216: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_233, mul_215);  mul_215 = None
        convert_element_type_236: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_262, torch.float32)
        pow_19: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_236, 3.0)
        mul_217: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_127: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_262, mul_217);  view_262 = mul_217 = None
        mul_218: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_127, 0.7978845608028654);  add_127 = None
        tanh_9: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_218);  mul_218 = None
        add_128: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_219: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_233, add_128);  convert_element_type_233 = add_128 = None
        convert_element_type_237: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_219, torch.bfloat16);  mul_219 = None
        mul_220: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_32: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_220);  mul_220 = None
        mul_221: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_216, sub_32);  mul_216 = sub_32 = None
        mul_222: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_221, 0.7978845608028654);  mul_221 = None
        convert_element_type_238: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_222, torch.bfloat16)
        mul_223: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_222, 0.044715);  mul_222 = None
        pow_20: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_236, 2.0);  convert_element_type_236 = None
        mul_224: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_20, 3.0);  pow_20 = None
        mul_225: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_223, mul_224);  mul_223 = mul_224 = None
        convert_element_type_239: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_225, torch.bfloat16);  mul_225 = None
        add_129: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_238, convert_element_type_239);  convert_element_type_238 = convert_element_type_239 = None
        mul_226: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_237, 0.5);  convert_element_type_237 = None
        add_130: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_129, mul_226);  add_129 = mul_226 = None
        view_263: "bf16[128, 16384]" = torch.ops.aten.view.default(add_130, _shape_param_283);  add_130 = _shape_param_283 = None
        mm_112: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_263, arg195_1);  arg195_1 = None
        permute_92: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_263, [1, 0])
        mm_113: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_92, arg196_1);  permute_92 = None
        sum_71: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True, dtype = torch.float32);  view_263 = None
        view_264: "f32[16384]" = torch.ops.aten.view.default(sum_71, _shape_param_284);  sum_71 = _shape_param_284 = None
        convert_element_type_240: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_264, torch.bfloat16);  view_264 = None
        view_265: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_112, _shape_param_285);  mm_112 = _shape_param_285 = None
        convert_element_type_241: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_265, torch.float32);  view_265 = None
        convert_element_type_242: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_243: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_240, torch.float32);  convert_element_type_240 = None
        mm_114: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_91, arg197_1);  permute_91 = arg197_1 = None
        mm_115: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_259, arg198_1);  view_259 = arg198_1 = None
        view_266: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_115, _shape_param_286);  mm_115 = _shape_param_286 = None
        convert_element_type_244: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_114, torch.float32);  mm_114 = None
        view_267: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_266, _shape_param_287);  view_266 = _shape_param_287 = None
        permute_93: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
        view_268: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_93, _shape_param_288);  permute_93 = _shape_param_288 = None
        bmm_36: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg199_1, view_268);  arg199_1 = None
        bmm_37: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_268, arg200_1);  view_268 = arg200_1 = None
        view_269: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_36, _shape_param_289);  bmm_36 = _shape_param_289 = None
        view_270: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_37, _shape_param_290);  bmm_37 = _shape_param_290 = None
        convert_element_type_245: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_270, torch.float32);  view_270 = None
        mul_227: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_245, arg201_1);  convert_element_type_245 = None
        sum_72: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [-1], True)
        neg_27: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg201_1);  arg201_1 = None
        fma_9: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_27, sum_72, mul_227);  neg_27 = sum_72 = mul_227 = None
        convert_element_type_246: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_9, torch.bfloat16);  fma_9 = None
        div_10: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_246, 16.0);  convert_element_type_246 = None
        view_271: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_10, _shape_param_291);  div_10 = _shape_param_291 = None
        bmm_38: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg202_1, view_271);  arg202_1 = None
        bmm_39: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_271, arg203_1);  view_271 = arg203_1 = None
        view_272: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_292);  bmm_38 = _shape_param_292 = None
        view_273: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_39, _shape_param_293);  bmm_39 = _shape_param_293 = None
        convert_element_type_247: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_272, torch.float32);  view_272 = None
        permute_94: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_247, [0, 1, 3, 2]);  convert_element_type_247 = None
        convert_element_type_248: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_94, torch.bfloat16);  permute_94 = None
        convert_element_type_249: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_273, torch.bfloat16);  view_273 = None
        permute_95: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_249, [0, 2, 1, 3]);  convert_element_type_249 = None
        permute_96: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_248, [0, 2, 1, 3]);  convert_element_type_248 = None
        slice_38: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_95, 3, 0, 64)
        slice_39: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_95, 3, 64, 256);  permute_95 = None
        slice_40: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_96, 3, 0, 64)
        slice_41: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_96, 3, 64, 256);  permute_96 = None
        expand_19: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg204_1, _shape_param_294);  arg204_1 = _shape_param_294 = None
        clone_27: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_274: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_27, _shape_param_295);  clone_27 = _shape_param_295 = None
        mul_228: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_38, view_274)
        view_275: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_228, _shape_param_296);  mul_228 = _shape_param_296 = None
        select_36: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_275, -1, 0)
        select_37: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_275, -1, 1);  view_275 = None
        neg_28: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_36);  select_36 = None
        slice_scatter_72: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_28, 3, 1, 9223372036854775807, 2);  neg_28 = None
        slice_scatter_73: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_37, 3, 0, 9223372036854775807, 2);  select_37 = None
        add_131: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_72, slice_scatter_73);  slice_scatter_72 = slice_scatter_73 = None
        expand_20: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg205_1, _shape_param_297);  arg205_1 = _shape_param_297 = None
        clone_28: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_276: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_28, _shape_param_298);  clone_28 = _shape_param_298 = None
        mul_229: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_38, view_276);  slice_38 = None
        add_132: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_131, mul_229);  add_131 = mul_229 = None
        mul_230: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_40, view_274);  view_274 = None
        view_277: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_230, _shape_param_299);  mul_230 = _shape_param_299 = None
        select_38: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_277, -1, 0)
        select_39: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_277, -1, 1);  view_277 = None
        neg_29: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_38);  select_38 = None
        slice_scatter_74: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_29, 3, 1, 9223372036854775807, 2);  neg_29 = None
        slice_scatter_75: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_39, 3, 0, 9223372036854775807, 2);  select_39 = None
        add_133: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_74, slice_scatter_75);  slice_scatter_74 = slice_scatter_75 = None
        mul_231: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_40, view_276);  slice_40 = view_276 = None
        add_134: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_133, mul_231);  add_133 = mul_231 = None
        slice_scatter_76: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_39, 3, 64, 9223372036854775807);  slice_39 = None
        slice_scatter_77: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_132, 3, 0, 64);  add_132 = None
        add_135: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_76, slice_scatter_77);  slice_scatter_76 = slice_scatter_77 = None
        slice_scatter_78: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_41, 3, 64, 9223372036854775807);  slice_41 = None
        slice_scatter_79: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_134, 3, 0, 64);  add_134 = None
        add_136: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_78, slice_scatter_79);  slice_scatter_78 = slice_scatter_79 = None
        permute_97: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
        clone_29: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_97, memory_format = torch.contiguous_format);  permute_97 = None
        view_278: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_29, _shape_param_300);  clone_29 = _shape_param_300 = None
        view_279: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_136, _shape_param_301);  add_136 = _shape_param_301 = None
        view_280: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_135, _shape_param_302);  add_135 = _shape_param_302 = None
        view_281: "bf16[128, 4096]" = torch.ops.aten.view.default(view_278, _shape_param_303);  view_278 = _shape_param_303 = None
        permute_98: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_281, [1, 0])
        mm_116: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_98, arg196_1);  permute_98 = None
        mm_117: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_281, arg206_1);  view_281 = arg206_1 = None
        view_282: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_117, _shape_param_304);  mm_117 = _shape_param_304 = None
        convert_element_type_250: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_282, torch.float32);  view_282 = None
        add_137: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_241, convert_element_type_250);  convert_element_type_241 = convert_element_type_250 = None
        convert_element_type_251: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_116, torch.float32);  mm_116 = None
        view_283: "bf16[128, 4096]" = torch.ops.aten.view.default(view_279, _shape_param_305);  view_279 = _shape_param_305 = None
        permute_99: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_283, [1, 0])
        mm_118: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_99, arg196_1);  permute_99 = None
        mm_119: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_283, arg207_1);  view_283 = arg207_1 = None
        view_284: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_119, _shape_param_306);  mm_119 = _shape_param_306 = None
        convert_element_type_252: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_284, torch.float32);  view_284 = None
        add_138: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_137, convert_element_type_252);  add_137 = convert_element_type_252 = None
        convert_element_type_253: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_118, torch.float32);  mm_118 = None
        view_285: "bf16[128, 4096]" = torch.ops.aten.view.default(view_280, _shape_param_307);  view_280 = _shape_param_307 = None
        permute_100: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_285, [1, 0])
        mm_120: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_100, arg196_1);  permute_100 = arg196_1 = None
        mm_121: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_285, arg208_1);  view_285 = arg208_1 = None
        view_286: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_121, _shape_param_308);  mm_121 = _shape_param_308 = None
        convert_element_type_254: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_286, torch.float32);  view_286 = None
        add_139: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_138, convert_element_type_254);  add_138 = convert_element_type_254 = None
        convert_element_type_255: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        mul_232: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_139, arg209_1);  arg209_1 = None
        mul_233: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_232, 4096)
        sum_73: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_232, [2], True)
        mul_234: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_232, arg210_1);  mul_232 = None
        sum_74: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True);  mul_234 = None
        mul_235: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg210_1, sum_74);  sum_74 = None
        sub_33: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_233, sum_73);  mul_233 = sum_73 = None
        sub_34: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_33, mul_235);  sub_33 = mul_235 = None
        mul_236: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg211_1, sub_34);  arg211_1 = sub_34 = None
        mul_237: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_139, arg210_1);  arg210_1 = None
        sum_75: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_237, [0, 1]);  mul_237 = None
        sum_76: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_139, [0, 1]);  add_139 = None
        add_140: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_126, mul_236);  add_126 = mul_236 = None
        convert_element_type_256: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_140, torch.bfloat16)
        view_287: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_256, _shape_param_309);  convert_element_type_256 = _shape_param_309 = None
        mm_122: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_287, arg212_1);  arg212_1 = None
        permute_101: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_287, [1, 0])
        mm_123: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_101, arg213_1);  arg213_1 = None
        sum_77: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True, dtype = torch.float32)
        view_288: "f32[4096]" = torch.ops.aten.view.default(sum_77, _shape_param_310);  sum_77 = _shape_param_310 = None
        convert_element_type_257: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_288, torch.bfloat16);  view_288 = None
        view_289: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_122, _shape_param_311);  mm_122 = _shape_param_311 = None
        convert_element_type_258: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_289, torch.float32);  view_289 = None
        convert_element_type_259: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_260: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_257, torch.float32);  convert_element_type_257 = None
        view_290: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg214_1, _shape_param_312);  arg214_1 = _shape_param_312 = None
        mul_238: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_290, 0.5)
        mul_239: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_258, mul_238);  mul_238 = None
        convert_element_type_261: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_290, torch.float32)
        pow_21: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_261, 3.0)
        mul_240: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_141: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_290, mul_240);  view_290 = mul_240 = None
        mul_241: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_141, 0.7978845608028654);  add_141 = None
        tanh_10: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_241);  mul_241 = None
        add_142: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_242: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_258, add_142);  convert_element_type_258 = add_142 = None
        convert_element_type_262: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_242, torch.bfloat16);  mul_242 = None
        mul_243: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_35: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_243);  mul_243 = None
        mul_244: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_239, sub_35);  mul_239 = sub_35 = None
        mul_245: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_244, 0.7978845608028654);  mul_244 = None
        convert_element_type_263: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_245, torch.bfloat16)
        mul_246: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_245, 0.044715);  mul_245 = None
        pow_22: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_261, 2.0);  convert_element_type_261 = None
        mul_247: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_22, 3.0);  pow_22 = None
        mul_248: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
        convert_element_type_264: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None
        add_143: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_263, convert_element_type_264);  convert_element_type_263 = convert_element_type_264 = None
        mul_249: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_262, 0.5);  convert_element_type_262 = None
        add_144: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_143, mul_249);  add_143 = mul_249 = None
        view_291: "bf16[128, 16384]" = torch.ops.aten.view.default(add_144, _shape_param_313);  add_144 = _shape_param_313 = None
        mm_124: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_291, arg215_1);  arg215_1 = None
        permute_102: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_291, [1, 0])
        mm_125: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_102, arg216_1);  permute_102 = None
        sum_78: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True, dtype = torch.float32);  view_291 = None
        view_292: "f32[16384]" = torch.ops.aten.view.default(sum_78, _shape_param_314);  sum_78 = _shape_param_314 = None
        convert_element_type_265: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_292, torch.bfloat16);  view_292 = None
        view_293: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_124, _shape_param_315);  mm_124 = _shape_param_315 = None
        convert_element_type_266: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_293, torch.float32);  view_293 = None
        convert_element_type_267: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_268: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_265, torch.float32);  convert_element_type_265 = None
        mm_126: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_101, arg217_1);  permute_101 = arg217_1 = None
        mm_127: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_287, arg218_1);  view_287 = arg218_1 = None
        view_294: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_127, _shape_param_316);  mm_127 = _shape_param_316 = None
        convert_element_type_269: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_126, torch.float32);  mm_126 = None
        view_295: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_294, _shape_param_317);  view_294 = _shape_param_317 = None
        permute_103: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
        view_296: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_103, _shape_param_318);  permute_103 = _shape_param_318 = None
        bmm_40: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg219_1, view_296);  arg219_1 = None
        bmm_41: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_296, arg220_1);  view_296 = arg220_1 = None
        view_297: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_40, _shape_param_319);  bmm_40 = _shape_param_319 = None
        view_298: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_41, _shape_param_320);  bmm_41 = _shape_param_320 = None
        convert_element_type_270: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_298, torch.float32);  view_298 = None
        mul_250: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_270, arg221_1);  convert_element_type_270 = None
        sum_79: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [-1], True)
        neg_30: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg221_1);  arg221_1 = None
        fma_10: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_30, sum_79, mul_250);  neg_30 = sum_79 = mul_250 = None
        convert_element_type_271: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_10, torch.bfloat16);  fma_10 = None
        div_11: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_271, 16.0);  convert_element_type_271 = None
        view_299: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_11, _shape_param_321);  div_11 = _shape_param_321 = None
        bmm_42: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg222_1, view_299);  arg222_1 = None
        bmm_43: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_299, arg223_1);  view_299 = arg223_1 = None
        view_300: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_322);  bmm_42 = _shape_param_322 = None
        view_301: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_43, _shape_param_323);  bmm_43 = _shape_param_323 = None
        convert_element_type_272: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_300, torch.float32);  view_300 = None
        permute_104: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_272, [0, 1, 3, 2]);  convert_element_type_272 = None
        convert_element_type_273: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_104, torch.bfloat16);  permute_104 = None
        convert_element_type_274: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_301, torch.bfloat16);  view_301 = None
        permute_105: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_274, [0, 2, 1, 3]);  convert_element_type_274 = None
        permute_106: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_273, [0, 2, 1, 3]);  convert_element_type_273 = None
        slice_42: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_105, 3, 0, 64)
        slice_43: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_105, 3, 64, 256);  permute_105 = None
        slice_44: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_106, 3, 0, 64)
        slice_45: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_106, 3, 64, 256);  permute_106 = None
        expand_21: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg224_1, _shape_param_324);  arg224_1 = _shape_param_324 = None
        clone_30: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_302: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_30, _shape_param_325);  clone_30 = _shape_param_325 = None
        mul_251: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_42, view_302)
        view_303: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_251, _shape_param_326);  mul_251 = _shape_param_326 = None
        select_40: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_303, -1, 0)
        select_41: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_303, -1, 1);  view_303 = None
        neg_31: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_40);  select_40 = None
        slice_scatter_80: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_31, 3, 1, 9223372036854775807, 2);  neg_31 = None
        slice_scatter_81: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_41, 3, 0, 9223372036854775807, 2);  select_41 = None
        add_145: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_80, slice_scatter_81);  slice_scatter_80 = slice_scatter_81 = None
        expand_22: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg225_1, _shape_param_327);  arg225_1 = _shape_param_327 = None
        clone_31: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_304: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_31, _shape_param_328);  clone_31 = _shape_param_328 = None
        mul_252: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_42, view_304);  slice_42 = None
        add_146: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_145, mul_252);  add_145 = mul_252 = None
        mul_253: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_44, view_302);  view_302 = None
        view_305: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_253, _shape_param_329);  mul_253 = _shape_param_329 = None
        select_42: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_305, -1, 0)
        select_43: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_305, -1, 1);  view_305 = None
        neg_32: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_42);  select_42 = None
        slice_scatter_82: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_32, 3, 1, 9223372036854775807, 2);  neg_32 = None
        slice_scatter_83: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_43, 3, 0, 9223372036854775807, 2);  select_43 = None
        add_147: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_82, slice_scatter_83);  slice_scatter_82 = slice_scatter_83 = None
        mul_254: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_44, view_304);  slice_44 = view_304 = None
        add_148: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_147, mul_254);  add_147 = mul_254 = None
        slice_scatter_84: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_43, 3, 64, 9223372036854775807);  slice_43 = None
        slice_scatter_85: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_146, 3, 0, 64);  add_146 = None
        add_149: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_84, slice_scatter_85);  slice_scatter_84 = slice_scatter_85 = None
        slice_scatter_86: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_45, 3, 64, 9223372036854775807);  slice_45 = None
        slice_scatter_87: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_148, 3, 0, 64);  add_148 = None
        add_150: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_86, slice_scatter_87);  slice_scatter_86 = slice_scatter_87 = None
        permute_107: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
        clone_32: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_107, memory_format = torch.contiguous_format);  permute_107 = None
        view_306: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_32, _shape_param_330);  clone_32 = _shape_param_330 = None
        view_307: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_150, _shape_param_331);  add_150 = _shape_param_331 = None
        view_308: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_149, _shape_param_332);  add_149 = _shape_param_332 = None
        view_309: "bf16[128, 4096]" = torch.ops.aten.view.default(view_306, _shape_param_333);  view_306 = _shape_param_333 = None
        permute_108: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_309, [1, 0])
        mm_128: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_108, arg216_1);  permute_108 = None
        mm_129: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_309, arg226_1);  view_309 = arg226_1 = None
        view_310: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_129, _shape_param_334);  mm_129 = _shape_param_334 = None
        convert_element_type_275: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_310, torch.float32);  view_310 = None
        add_151: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_266, convert_element_type_275);  convert_element_type_266 = convert_element_type_275 = None
        convert_element_type_276: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        view_311: "bf16[128, 4096]" = torch.ops.aten.view.default(view_307, _shape_param_335);  view_307 = _shape_param_335 = None
        permute_109: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_311, [1, 0])
        mm_130: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_109, arg216_1);  permute_109 = None
        mm_131: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_311, arg227_1);  view_311 = arg227_1 = None
        view_312: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_131, _shape_param_336);  mm_131 = _shape_param_336 = None
        convert_element_type_277: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_312, torch.float32);  view_312 = None
        add_152: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_151, convert_element_type_277);  add_151 = convert_element_type_277 = None
        convert_element_type_278: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_130, torch.float32);  mm_130 = None
        view_313: "bf16[128, 4096]" = torch.ops.aten.view.default(view_308, _shape_param_337);  view_308 = _shape_param_337 = None
        permute_110: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_313, [1, 0])
        mm_132: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_110, arg216_1);  permute_110 = arg216_1 = None
        mm_133: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_313, arg228_1);  view_313 = arg228_1 = None
        view_314: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_133, _shape_param_338);  mm_133 = _shape_param_338 = None
        convert_element_type_279: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_314, torch.float32);  view_314 = None
        add_153: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_152, convert_element_type_279);  add_152 = convert_element_type_279 = None
        convert_element_type_280: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_132, torch.float32);  mm_132 = None
        mul_255: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_153, arg229_1);  arg229_1 = None
        mul_256: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_255, 4096)
        sum_80: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
        mul_257: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_255, arg230_1);  mul_255 = None
        sum_81: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
        mul_258: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg230_1, sum_81);  sum_81 = None
        sub_36: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_256, sum_80);  mul_256 = sum_80 = None
        sub_37: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_36, mul_258);  sub_36 = mul_258 = None
        mul_259: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg231_1, sub_37);  arg231_1 = sub_37 = None
        mul_260: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_153, arg230_1);  arg230_1 = None
        sum_82: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
        sum_83: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
        add_154: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_140, mul_259);  add_140 = mul_259 = None
        convert_element_type_281: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16)
        view_315: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_281, _shape_param_339);  convert_element_type_281 = _shape_param_339 = None
        mm_134: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_315, arg232_1);  arg232_1 = None
        permute_111: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_315, [1, 0])
        mm_135: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_111, arg233_1);  arg233_1 = None
        sum_84: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True, dtype = torch.float32)
        view_316: "f32[4096]" = torch.ops.aten.view.default(sum_84, _shape_param_340);  sum_84 = _shape_param_340 = None
        convert_element_type_282: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_316, torch.bfloat16);  view_316 = None
        view_317: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_134, _shape_param_341);  mm_134 = _shape_param_341 = None
        convert_element_type_283: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_317, torch.float32);  view_317 = None
        convert_element_type_284: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_285: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_282, torch.float32);  convert_element_type_282 = None
        view_318: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg234_1, _shape_param_342);  arg234_1 = _shape_param_342 = None
        mul_261: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_318, 0.5)
        mul_262: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_283, mul_261);  mul_261 = None
        convert_element_type_286: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_318, torch.float32)
        pow_23: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_286, 3.0)
        mul_263: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_155: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_318, mul_263);  view_318 = mul_263 = None
        mul_264: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_155, 0.7978845608028654);  add_155 = None
        tanh_11: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_264);  mul_264 = None
        add_156: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_265: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_283, add_156);  convert_element_type_283 = add_156 = None
        convert_element_type_287: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_265, torch.bfloat16);  mul_265 = None
        mul_266: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_38: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_266);  mul_266 = None
        mul_267: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_262, sub_38);  mul_262 = sub_38 = None
        mul_268: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_267, 0.7978845608028654);  mul_267 = None
        convert_element_type_288: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_268, torch.bfloat16)
        mul_269: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_268, 0.044715);  mul_268 = None
        pow_24: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_286, 2.0);  convert_element_type_286 = None
        mul_270: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_24, 3.0);  pow_24 = None
        mul_271: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_269, mul_270);  mul_269 = mul_270 = None
        convert_element_type_289: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_271, torch.bfloat16);  mul_271 = None
        add_157: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_288, convert_element_type_289);  convert_element_type_288 = convert_element_type_289 = None
        mul_272: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_287, 0.5);  convert_element_type_287 = None
        add_158: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_157, mul_272);  add_157 = mul_272 = None
        view_319: "bf16[128, 16384]" = torch.ops.aten.view.default(add_158, _shape_param_343);  add_158 = _shape_param_343 = None
        mm_136: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_319, arg235_1);  arg235_1 = None
        permute_112: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_319, [1, 0])
        mm_137: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_112, arg236_1);  permute_112 = None
        sum_85: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_319, [0], True, dtype = torch.float32);  view_319 = None
        view_320: "f32[16384]" = torch.ops.aten.view.default(sum_85, _shape_param_344);  sum_85 = _shape_param_344 = None
        convert_element_type_290: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_320, torch.bfloat16);  view_320 = None
        view_321: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_136, _shape_param_345);  mm_136 = _shape_param_345 = None
        convert_element_type_291: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_321, torch.float32);  view_321 = None
        convert_element_type_292: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_293: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_290, torch.float32);  convert_element_type_290 = None
        mm_138: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_111, arg237_1);  permute_111 = arg237_1 = None
        mm_139: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_315, arg238_1);  view_315 = arg238_1 = None
        view_322: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_139, _shape_param_346);  mm_139 = _shape_param_346 = None
        convert_element_type_294: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_138, torch.float32);  mm_138 = None
        view_323: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_322, _shape_param_347);  view_322 = _shape_param_347 = None
        permute_113: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
        view_324: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_113, _shape_param_348);  permute_113 = _shape_param_348 = None
        bmm_44: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg239_1, view_324);  arg239_1 = None
        bmm_45: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_324, arg240_1);  view_324 = arg240_1 = None
        view_325: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_44, _shape_param_349);  bmm_44 = _shape_param_349 = None
        view_326: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_45, _shape_param_350);  bmm_45 = _shape_param_350 = None
        convert_element_type_295: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_326, torch.float32);  view_326 = None
        mul_273: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_295, arg241_1);  convert_element_type_295 = None
        sum_86: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [-1], True)
        neg_33: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg241_1);  arg241_1 = None
        fma_11: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_33, sum_86, mul_273);  neg_33 = sum_86 = mul_273 = None
        convert_element_type_296: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_11, torch.bfloat16);  fma_11 = None
        div_12: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_296, 16.0);  convert_element_type_296 = None
        view_327: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_12, _shape_param_351);  div_12 = _shape_param_351 = None
        bmm_46: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg242_1, view_327);  arg242_1 = None
        bmm_47: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_327, arg243_1);  view_327 = arg243_1 = None
        view_328: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_352);  bmm_46 = _shape_param_352 = None
        view_329: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_47, _shape_param_353);  bmm_47 = _shape_param_353 = None
        convert_element_type_297: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_328, torch.float32);  view_328 = None
        permute_114: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_297, [0, 1, 3, 2]);  convert_element_type_297 = None
        convert_element_type_298: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_114, torch.bfloat16);  permute_114 = None
        convert_element_type_299: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_329, torch.bfloat16);  view_329 = None
        permute_115: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_299, [0, 2, 1, 3]);  convert_element_type_299 = None
        permute_116: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_298, [0, 2, 1, 3]);  convert_element_type_298 = None
        slice_46: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_115, 3, 0, 64)
        slice_47: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_115, 3, 64, 256);  permute_115 = None
        slice_48: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_116, 3, 0, 64)
        slice_49: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_116, 3, 64, 256);  permute_116 = None
        expand_23: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg244_1, _shape_param_354);  arg244_1 = _shape_param_354 = None
        clone_33: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_330: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_33, _shape_param_355);  clone_33 = _shape_param_355 = None
        mul_274: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_46, view_330)
        view_331: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_274, _shape_param_356);  mul_274 = _shape_param_356 = None
        select_44: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_331, -1, 0)
        select_45: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_331, -1, 1);  view_331 = None
        neg_34: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_44);  select_44 = None
        slice_scatter_88: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_34, 3, 1, 9223372036854775807, 2);  neg_34 = None
        slice_scatter_89: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_45, 3, 0, 9223372036854775807, 2);  select_45 = None
        add_159: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_88, slice_scatter_89);  slice_scatter_88 = slice_scatter_89 = None
        expand_24: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg245_1, _shape_param_357);  arg245_1 = _shape_param_357 = None
        clone_34: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_332: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_34, _shape_param_358);  clone_34 = _shape_param_358 = None
        mul_275: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_46, view_332);  slice_46 = None
        add_160: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_159, mul_275);  add_159 = mul_275 = None
        mul_276: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_48, view_330);  view_330 = None
        view_333: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_276, _shape_param_359);  mul_276 = _shape_param_359 = None
        select_46: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_333, -1, 0)
        select_47: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_333, -1, 1);  view_333 = None
        neg_35: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_46);  select_46 = None
        slice_scatter_90: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_35, 3, 1, 9223372036854775807, 2);  neg_35 = None
        slice_scatter_91: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_47, 3, 0, 9223372036854775807, 2);  select_47 = None
        add_161: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_90, slice_scatter_91);  slice_scatter_90 = slice_scatter_91 = None
        mul_277: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_48, view_332);  slice_48 = view_332 = None
        add_162: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_161, mul_277);  add_161 = mul_277 = None
        slice_scatter_92: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_47, 3, 64, 9223372036854775807);  slice_47 = None
        slice_scatter_93: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_160, 3, 0, 64);  add_160 = None
        add_163: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_92, slice_scatter_93);  slice_scatter_92 = slice_scatter_93 = None
        slice_scatter_94: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_49, 3, 64, 9223372036854775807);  slice_49 = None
        slice_scatter_95: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_162, 3, 0, 64);  add_162 = None
        add_164: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_94, slice_scatter_95);  slice_scatter_94 = slice_scatter_95 = None
        permute_117: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
        clone_35: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
        view_334: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_35, _shape_param_360);  clone_35 = _shape_param_360 = None
        view_335: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_164, _shape_param_361);  add_164 = _shape_param_361 = None
        view_336: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_163, _shape_param_362);  add_163 = _shape_param_362 = None
        view_337: "bf16[128, 4096]" = torch.ops.aten.view.default(view_334, _shape_param_363);  view_334 = _shape_param_363 = None
        permute_118: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_337, [1, 0])
        mm_140: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_118, arg236_1);  permute_118 = None
        mm_141: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_337, arg246_1);  view_337 = arg246_1 = None
        view_338: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_141, _shape_param_364);  mm_141 = _shape_param_364 = None
        convert_element_type_300: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_338, torch.float32);  view_338 = None
        add_165: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_291, convert_element_type_300);  convert_element_type_291 = convert_element_type_300 = None
        convert_element_type_301: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_140, torch.float32);  mm_140 = None
        view_339: "bf16[128, 4096]" = torch.ops.aten.view.default(view_335, _shape_param_365);  view_335 = _shape_param_365 = None
        permute_119: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_339, [1, 0])
        mm_142: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_119, arg236_1);  permute_119 = None
        mm_143: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_339, arg247_1);  view_339 = arg247_1 = None
        view_340: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_143, _shape_param_366);  mm_143 = _shape_param_366 = None
        convert_element_type_302: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_340, torch.float32);  view_340 = None
        add_166: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_165, convert_element_type_302);  add_165 = convert_element_type_302 = None
        convert_element_type_303: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_142, torch.float32);  mm_142 = None
        view_341: "bf16[128, 4096]" = torch.ops.aten.view.default(view_336, _shape_param_367);  view_336 = _shape_param_367 = None
        permute_120: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_341, [1, 0])
        mm_144: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_120, arg236_1);  permute_120 = arg236_1 = None
        mm_145: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_341, arg248_1);  view_341 = arg248_1 = None
        view_342: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_145, _shape_param_368);  mm_145 = _shape_param_368 = None
        convert_element_type_304: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_342, torch.float32);  view_342 = None
        add_167: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_166, convert_element_type_304);  add_166 = convert_element_type_304 = None
        convert_element_type_305: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        mul_278: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_167, arg249_1);  arg249_1 = None
        mul_279: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_278, 4096)
        sum_87: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True)
        mul_280: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_278, arg250_1);  mul_278 = None
        sum_88: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True);  mul_280 = None
        mul_281: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg250_1, sum_88);  sum_88 = None
        sub_39: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_279, sum_87);  mul_279 = sum_87 = None
        sub_40: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_39, mul_281);  sub_39 = mul_281 = None
        mul_282: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg251_1, sub_40);  arg251_1 = sub_40 = None
        mul_283: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_167, arg250_1);  arg250_1 = None
        sum_89: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_283, [0, 1]);  mul_283 = None
        sum_90: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
        add_168: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_154, mul_282);  add_154 = mul_282 = None
        convert_element_type_306: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_168, torch.bfloat16)
        view_343: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_306, _shape_param_369);  convert_element_type_306 = _shape_param_369 = None
        mm_146: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_343, arg252_1);  arg252_1 = None
        permute_121: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_343, [1, 0])
        mm_147: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_121, arg253_1);  arg253_1 = None
        sum_91: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_343, [0], True, dtype = torch.float32)
        view_344: "f32[4096]" = torch.ops.aten.view.default(sum_91, _shape_param_370);  sum_91 = _shape_param_370 = None
        convert_element_type_307: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_344, torch.bfloat16);  view_344 = None
        view_345: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_146, _shape_param_371);  mm_146 = _shape_param_371 = None
        convert_element_type_308: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_345, torch.float32);  view_345 = None
        convert_element_type_309: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_310: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_307, torch.float32);  convert_element_type_307 = None
        view_346: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg254_1, _shape_param_372);  arg254_1 = _shape_param_372 = None
        mul_284: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_346, 0.5)
        mul_285: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_308, mul_284);  mul_284 = None
        convert_element_type_311: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_346, torch.float32)
        pow_25: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_311, 3.0)
        mul_286: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_169: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_346, mul_286);  view_346 = mul_286 = None
        mul_287: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_169, 0.7978845608028654);  add_169 = None
        tanh_12: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_287);  mul_287 = None
        add_170: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_288: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_308, add_170);  convert_element_type_308 = add_170 = None
        convert_element_type_312: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_288, torch.bfloat16);  mul_288 = None
        mul_289: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_41: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_289);  mul_289 = None
        mul_290: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_285, sub_41);  mul_285 = sub_41 = None
        mul_291: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_290, 0.7978845608028654);  mul_290 = None
        convert_element_type_313: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_291, torch.bfloat16)
        mul_292: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_291, 0.044715);  mul_291 = None
        pow_26: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_311, 2.0);  convert_element_type_311 = None
        mul_293: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_26, 3.0);  pow_26 = None
        mul_294: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_292, mul_293);  mul_292 = mul_293 = None
        convert_element_type_314: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_294, torch.bfloat16);  mul_294 = None
        add_171: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_313, convert_element_type_314);  convert_element_type_313 = convert_element_type_314 = None
        mul_295: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_312, 0.5);  convert_element_type_312 = None
        add_172: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_171, mul_295);  add_171 = mul_295 = None
        view_347: "bf16[128, 16384]" = torch.ops.aten.view.default(add_172, _shape_param_373);  add_172 = _shape_param_373 = None
        mm_148: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_347, arg255_1);  arg255_1 = None
        permute_122: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_347, [1, 0])
        mm_149: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_122, arg256_1);  permute_122 = None
        sum_92: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True, dtype = torch.float32);  view_347 = None
        view_348: "f32[16384]" = torch.ops.aten.view.default(sum_92, _shape_param_374);  sum_92 = _shape_param_374 = None
        convert_element_type_315: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_348, torch.bfloat16);  view_348 = None
        view_349: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_148, _shape_param_375);  mm_148 = _shape_param_375 = None
        convert_element_type_316: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        convert_element_type_317: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_318: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_315, torch.float32);  convert_element_type_315 = None
        mm_150: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_121, arg257_1);  permute_121 = arg257_1 = None
        mm_151: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_343, arg258_1);  view_343 = arg258_1 = None
        view_350: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_151, _shape_param_376);  mm_151 = _shape_param_376 = None
        convert_element_type_319: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_150, torch.float32);  mm_150 = None
        view_351: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_350, _shape_param_377);  view_350 = _shape_param_377 = None
        permute_123: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_351, [0, 2, 1, 3]);  view_351 = None
        view_352: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_123, _shape_param_378);  permute_123 = _shape_param_378 = None
        bmm_48: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg259_1, view_352);  arg259_1 = None
        bmm_49: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_352, arg260_1);  view_352 = arg260_1 = None
        view_353: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_48, _shape_param_379);  bmm_48 = _shape_param_379 = None
        view_354: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_49, _shape_param_380);  bmm_49 = _shape_param_380 = None
        convert_element_type_320: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_354, torch.float32);  view_354 = None
        mul_296: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_320, arg261_1);  convert_element_type_320 = None
        sum_93: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [-1], True)
        neg_36: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg261_1);  arg261_1 = None
        fma_12: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_36, sum_93, mul_296);  neg_36 = sum_93 = mul_296 = None
        convert_element_type_321: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_12, torch.bfloat16);  fma_12 = None
        div_13: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_321, 16.0);  convert_element_type_321 = None
        view_355: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_13, _shape_param_381);  div_13 = _shape_param_381 = None
        bmm_50: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg262_1, view_355);  arg262_1 = None
        bmm_51: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_355, arg263_1);  view_355 = arg263_1 = None
        view_356: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_50, _shape_param_382);  bmm_50 = _shape_param_382 = None
        view_357: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_51, _shape_param_383);  bmm_51 = _shape_param_383 = None
        convert_element_type_322: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_356, torch.float32);  view_356 = None
        permute_124: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_322, [0, 1, 3, 2]);  convert_element_type_322 = None
        convert_element_type_323: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_124, torch.bfloat16);  permute_124 = None
        convert_element_type_324: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_357, torch.bfloat16);  view_357 = None
        permute_125: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_324, [0, 2, 1, 3]);  convert_element_type_324 = None
        permute_126: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_323, [0, 2, 1, 3]);  convert_element_type_323 = None
        slice_50: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_125, 3, 0, 64)
        slice_51: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_125, 3, 64, 256);  permute_125 = None
        slice_52: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_126, 3, 0, 64)
        slice_53: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_126, 3, 64, 256);  permute_126 = None
        expand_25: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg264_1, _shape_param_384);  arg264_1 = _shape_param_384 = None
        clone_36: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_358: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_36, _shape_param_385);  clone_36 = _shape_param_385 = None
        mul_297: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_50, view_358)
        view_359: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_297, _shape_param_386);  mul_297 = _shape_param_386 = None
        select_48: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_359, -1, 0)
        select_49: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_359, -1, 1);  view_359 = None
        neg_37: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_48);  select_48 = None
        slice_scatter_96: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_37, 3, 1, 9223372036854775807, 2);  neg_37 = None
        slice_scatter_97: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_49, 3, 0, 9223372036854775807, 2);  select_49 = None
        add_173: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_96, slice_scatter_97);  slice_scatter_96 = slice_scatter_97 = None
        expand_26: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg265_1, _shape_param_387);  arg265_1 = _shape_param_387 = None
        clone_37: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
        view_360: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_37, _shape_param_388);  clone_37 = _shape_param_388 = None
        mul_298: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_50, view_360);  slice_50 = None
        add_174: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_173, mul_298);  add_173 = mul_298 = None
        mul_299: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_52, view_358);  view_358 = None
        view_361: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_299, _shape_param_389);  mul_299 = _shape_param_389 = None
        select_50: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_361, -1, 0)
        select_51: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_361, -1, 1);  view_361 = None
        neg_38: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_50);  select_50 = None
        slice_scatter_98: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_38, 3, 1, 9223372036854775807, 2);  neg_38 = None
        slice_scatter_99: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_51, 3, 0, 9223372036854775807, 2);  select_51 = None
        add_175: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_98, slice_scatter_99);  slice_scatter_98 = slice_scatter_99 = None
        mul_300: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_52, view_360);  slice_52 = view_360 = None
        add_176: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_175, mul_300);  add_175 = mul_300 = None
        slice_scatter_100: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_51, 3, 64, 9223372036854775807);  slice_51 = None
        slice_scatter_101: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_174, 3, 0, 64);  add_174 = None
        add_177: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_100, slice_scatter_101);  slice_scatter_100 = slice_scatter_101 = None
        slice_scatter_102: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_53, 3, 64, 9223372036854775807);  slice_53 = None
        slice_scatter_103: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_176, 3, 0, 64);  add_176 = None
        add_178: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_102, slice_scatter_103);  slice_scatter_102 = slice_scatter_103 = None
        permute_127: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
        clone_38: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
        view_362: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_38, _shape_param_390);  clone_38 = _shape_param_390 = None
        view_363: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_178, _shape_param_391);  add_178 = _shape_param_391 = None
        view_364: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_177, _shape_param_392);  add_177 = _shape_param_392 = None
        view_365: "bf16[128, 4096]" = torch.ops.aten.view.default(view_362, _shape_param_393);  view_362 = _shape_param_393 = None
        permute_128: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_365, [1, 0])
        mm_152: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_128, arg256_1);  permute_128 = None
        mm_153: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_365, arg266_1);  view_365 = arg266_1 = None
        view_366: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_153, _shape_param_394);  mm_153 = _shape_param_394 = None
        convert_element_type_325: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_366, torch.float32);  view_366 = None
        add_179: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_316, convert_element_type_325);  convert_element_type_316 = convert_element_type_325 = None
        convert_element_type_326: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        view_367: "bf16[128, 4096]" = torch.ops.aten.view.default(view_363, _shape_param_395);  view_363 = _shape_param_395 = None
        permute_129: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_367, [1, 0])
        mm_154: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_129, arg256_1);  permute_129 = None
        mm_155: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_367, arg267_1);  view_367 = arg267_1 = None
        view_368: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_155, _shape_param_396);  mm_155 = _shape_param_396 = None
        convert_element_type_327: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_368, torch.float32);  view_368 = None
        add_180: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_179, convert_element_type_327);  add_179 = convert_element_type_327 = None
        convert_element_type_328: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_154, torch.float32);  mm_154 = None
        view_369: "bf16[128, 4096]" = torch.ops.aten.view.default(view_364, _shape_param_397);  view_364 = _shape_param_397 = None
        permute_130: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_369, [1, 0])
        mm_156: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_130, arg256_1);  permute_130 = arg256_1 = None
        mm_157: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_369, arg268_1);  view_369 = arg268_1 = None
        view_370: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_157, _shape_param_398);  mm_157 = _shape_param_398 = None
        convert_element_type_329: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_370, torch.float32);  view_370 = None
        add_181: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_180, convert_element_type_329);  add_180 = convert_element_type_329 = None
        convert_element_type_330: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_156, torch.float32);  mm_156 = None
        mul_301: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_181, arg269_1);  arg269_1 = None
        mul_302: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_301, 4096)
        sum_94: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_301, [2], True)
        mul_303: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_301, arg270_1);  mul_301 = None
        sum_95: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_303, [2], True);  mul_303 = None
        mul_304: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg270_1, sum_95);  sum_95 = None
        sub_42: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_302, sum_94);  mul_302 = sum_94 = None
        sub_43: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_42, mul_304);  sub_42 = mul_304 = None
        mul_305: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg271_1, sub_43);  arg271_1 = sub_43 = None
        mul_306: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_181, arg270_1);  arg270_1 = None
        sum_96: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_306, [0, 1]);  mul_306 = None
        sum_97: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_181, [0, 1]);  add_181 = None
        add_182: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_168, mul_305);  add_168 = mul_305 = None
        convert_element_type_331: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_182, torch.bfloat16)
        view_371: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_331, _shape_param_399);  convert_element_type_331 = _shape_param_399 = None
        mm_158: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_371, arg272_1);  arg272_1 = None
        permute_131: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_371, [1, 0])
        mm_159: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_131, arg273_1);  arg273_1 = None
        sum_98: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_371, [0], True, dtype = torch.float32)
        view_372: "f32[4096]" = torch.ops.aten.view.default(sum_98, _shape_param_400);  sum_98 = _shape_param_400 = None
        convert_element_type_332: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_372, torch.bfloat16);  view_372 = None
        view_373: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_158, _shape_param_401);  mm_158 = _shape_param_401 = None
        convert_element_type_333: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_373, torch.float32);  view_373 = None
        convert_element_type_334: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_335: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_332, torch.float32);  convert_element_type_332 = None
        view_374: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg274_1, _shape_param_402);  arg274_1 = _shape_param_402 = None
        mul_307: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_374, 0.5)
        mul_308: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_333, mul_307);  mul_307 = None
        convert_element_type_336: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_374, torch.float32)
        pow_27: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_336, 3.0)
        mul_309: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_183: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_374, mul_309);  view_374 = mul_309 = None
        mul_310: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_183, 0.7978845608028654);  add_183 = None
        tanh_13: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_310);  mul_310 = None
        add_184: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_311: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_333, add_184);  convert_element_type_333 = add_184 = None
        convert_element_type_337: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_311, torch.bfloat16);  mul_311 = None
        mul_312: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_44: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_312);  mul_312 = None
        mul_313: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_308, sub_44);  mul_308 = sub_44 = None
        mul_314: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_313, 0.7978845608028654);  mul_313 = None
        convert_element_type_338: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_314, torch.bfloat16)
        mul_315: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_314, 0.044715);  mul_314 = None
        pow_28: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_336, 2.0);  convert_element_type_336 = None
        mul_316: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_28, 3.0);  pow_28 = None
        mul_317: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_315, mul_316);  mul_315 = mul_316 = None
        convert_element_type_339: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_317, torch.bfloat16);  mul_317 = None
        add_185: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_338, convert_element_type_339);  convert_element_type_338 = convert_element_type_339 = None
        mul_318: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_337, 0.5);  convert_element_type_337 = None
        add_186: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_185, mul_318);  add_185 = mul_318 = None
        view_375: "bf16[128, 16384]" = torch.ops.aten.view.default(add_186, _shape_param_403);  add_186 = _shape_param_403 = None
        mm_160: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_375, arg275_1);  arg275_1 = None
        permute_132: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_375, [1, 0])
        mm_161: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_132, arg276_1);  permute_132 = None
        sum_99: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_375, [0], True, dtype = torch.float32);  view_375 = None
        view_376: "f32[16384]" = torch.ops.aten.view.default(sum_99, _shape_param_404);  sum_99 = _shape_param_404 = None
        convert_element_type_340: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_376, torch.bfloat16);  view_376 = None
        view_377: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_160, _shape_param_405);  mm_160 = _shape_param_405 = None
        convert_element_type_341: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_377, torch.float32);  view_377 = None
        convert_element_type_342: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None
        convert_element_type_343: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_340, torch.float32);  convert_element_type_340 = None
        mm_162: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_131, arg277_1);  permute_131 = arg277_1 = None
        mm_163: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_371, arg278_1);  view_371 = arg278_1 = None
        view_378: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_163, _shape_param_406);  mm_163 = _shape_param_406 = None
        convert_element_type_344: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_162, torch.float32);  mm_162 = None
        view_379: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_378, _shape_param_407);  view_378 = _shape_param_407 = None
        permute_133: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        view_380: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_133, _shape_param_408);  permute_133 = _shape_param_408 = None
        bmm_52: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg279_1, view_380);  arg279_1 = None
        bmm_53: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_380, arg280_1);  view_380 = arg280_1 = None
        view_381: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_52, _shape_param_409);  bmm_52 = _shape_param_409 = None
        view_382: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_53, _shape_param_410);  bmm_53 = _shape_param_410 = None
        convert_element_type_345: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_382, torch.float32);  view_382 = None
        mul_319: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_345, arg281_1);  convert_element_type_345 = None
        sum_100: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [-1], True)
        neg_39: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg281_1);  arg281_1 = None
        fma_13: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_39, sum_100, mul_319);  neg_39 = sum_100 = mul_319 = None
        convert_element_type_346: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_13, torch.bfloat16);  fma_13 = None
        div_14: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_346, 16.0);  convert_element_type_346 = None
        view_383: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_14, _shape_param_411);  div_14 = _shape_param_411 = None
        bmm_54: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg282_1, view_383);  arg282_1 = None
        bmm_55: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_383, arg283_1);  view_383 = arg283_1 = None
        view_384: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_54, _shape_param_412);  bmm_54 = _shape_param_412 = None
        view_385: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_55, _shape_param_413);  bmm_55 = _shape_param_413 = None
        convert_element_type_347: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_384, torch.float32);  view_384 = None
        permute_134: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_347, [0, 1, 3, 2]);  convert_element_type_347 = None
        convert_element_type_348: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_134, torch.bfloat16);  permute_134 = None
        convert_element_type_349: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_385, torch.bfloat16);  view_385 = None
        permute_135: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_349, [0, 2, 1, 3]);  convert_element_type_349 = None
        permute_136: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_348, [0, 2, 1, 3]);  convert_element_type_348 = None
        slice_54: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_135, 3, 0, 64)
        slice_55: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_135, 3, 64, 256);  permute_135 = None
        slice_56: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_136, 3, 0, 64)
        slice_57: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_136, 3, 64, 256);  permute_136 = None
        expand_27: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg284_1, _shape_param_414);  arg284_1 = _shape_param_414 = None
        clone_39: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_386: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_39, _shape_param_415);  clone_39 = _shape_param_415 = None
        mul_320: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_54, view_386)
        view_387: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_320, _shape_param_416);  mul_320 = _shape_param_416 = None
        select_52: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_387, -1, 0)
        select_53: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_387, -1, 1);  view_387 = None
        neg_40: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_52);  select_52 = None
        slice_scatter_104: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_40, 3, 1, 9223372036854775807, 2);  neg_40 = None
        slice_scatter_105: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_53, 3, 0, 9223372036854775807, 2);  select_53 = None
        add_187: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_104, slice_scatter_105);  slice_scatter_104 = slice_scatter_105 = None
        expand_28: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg285_1, _shape_param_417);  arg285_1 = _shape_param_417 = None
        clone_40: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_388: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_40, _shape_param_418);  clone_40 = _shape_param_418 = None
        mul_321: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_54, view_388);  slice_54 = None
        add_188: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_187, mul_321);  add_187 = mul_321 = None
        mul_322: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_56, view_386);  view_386 = None
        view_389: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_322, _shape_param_419);  mul_322 = _shape_param_419 = None
        select_54: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_389, -1, 0)
        select_55: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_389, -1, 1);  view_389 = None
        neg_41: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_54);  select_54 = None
        slice_scatter_106: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_41, 3, 1, 9223372036854775807, 2);  neg_41 = None
        slice_scatter_107: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_55, 3, 0, 9223372036854775807, 2);  select_55 = None
        add_189: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_106, slice_scatter_107);  slice_scatter_106 = slice_scatter_107 = None
        mul_323: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_56, view_388);  slice_56 = view_388 = None
        add_190: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_189, mul_323);  add_189 = mul_323 = None
        slice_scatter_108: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_55, 3, 64, 9223372036854775807);  slice_55 = None
        slice_scatter_109: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_188, 3, 0, 64);  add_188 = None
        add_191: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_108, slice_scatter_109);  slice_scatter_108 = slice_scatter_109 = None
        slice_scatter_110: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_57, 3, 64, 9223372036854775807);  slice_57 = None
        slice_scatter_111: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_190, 3, 0, 64);  add_190 = None
        add_192: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_110, slice_scatter_111);  slice_scatter_110 = slice_scatter_111 = None
        permute_137: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None
        clone_41: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
        view_390: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_41, _shape_param_420);  clone_41 = _shape_param_420 = None
        view_391: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_192, _shape_param_421);  add_192 = _shape_param_421 = None
        view_392: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_191, _shape_param_422);  add_191 = _shape_param_422 = None
        view_393: "bf16[128, 4096]" = torch.ops.aten.view.default(view_390, _shape_param_423);  view_390 = _shape_param_423 = None
        permute_138: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_393, [1, 0])
        mm_164: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_138, arg276_1);  permute_138 = None
        mm_165: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_393, arg286_1);  view_393 = arg286_1 = None
        view_394: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_165, _shape_param_424);  mm_165 = _shape_param_424 = None
        convert_element_type_350: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_394, torch.float32);  view_394 = None
        add_193: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_341, convert_element_type_350);  convert_element_type_341 = convert_element_type_350 = None
        convert_element_type_351: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_164, torch.float32);  mm_164 = None
        view_395: "bf16[128, 4096]" = torch.ops.aten.view.default(view_391, _shape_param_425);  view_391 = _shape_param_425 = None
        permute_139: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_395, [1, 0])
        mm_166: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_139, arg276_1);  permute_139 = None
        mm_167: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_395, arg287_1);  view_395 = arg287_1 = None
        view_396: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_167, _shape_param_426);  mm_167 = _shape_param_426 = None
        convert_element_type_352: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_396, torch.float32);  view_396 = None
        add_194: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_193, convert_element_type_352);  add_193 = convert_element_type_352 = None
        convert_element_type_353: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_166, torch.float32);  mm_166 = None
        view_397: "bf16[128, 4096]" = torch.ops.aten.view.default(view_392, _shape_param_427);  view_392 = _shape_param_427 = None
        permute_140: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_397, [1, 0])
        mm_168: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_140, arg276_1);  permute_140 = arg276_1 = None
        mm_169: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_397, arg288_1);  view_397 = arg288_1 = None
        view_398: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_169, _shape_param_428);  mm_169 = _shape_param_428 = None
        convert_element_type_354: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        add_195: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_194, convert_element_type_354);  add_194 = convert_element_type_354 = None
        convert_element_type_355: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        mul_324: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_195, arg289_1);  arg289_1 = None
        mul_325: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_324, 4096)
        sum_101: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_324, [2], True)
        mul_326: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_324, arg290_1);  mul_324 = None
        sum_102: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True);  mul_326 = None
        mul_327: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg290_1, sum_102);  sum_102 = None
        sub_45: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_325, sum_101);  mul_325 = sum_101 = None
        sub_46: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_45, mul_327);  sub_45 = mul_327 = None
        mul_328: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg291_1, sub_46);  arg291_1 = sub_46 = None
        mul_329: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_195, arg290_1);  arg290_1 = None
        sum_103: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_329, [0, 1]);  mul_329 = None
        sum_104: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_195, [0, 1]);  add_195 = None
        add_196: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_182, mul_328);  add_182 = mul_328 = None
        convert_element_type_356: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16)
        view_399: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_356, _shape_param_429);  convert_element_type_356 = _shape_param_429 = None
        mm_170: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_399, arg292_1);  arg292_1 = None
        permute_141: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_399, [1, 0])
        mm_171: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_141, arg293_1);  arg293_1 = None
        sum_105: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_399, [0], True, dtype = torch.float32)
        view_400: "f32[4096]" = torch.ops.aten.view.default(sum_105, _shape_param_430);  sum_105 = _shape_param_430 = None
        convert_element_type_357: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_400, torch.bfloat16);  view_400 = None
        view_401: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_170, _shape_param_431);  mm_170 = _shape_param_431 = None
        convert_element_type_358: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_401, torch.float32);  view_401 = None
        convert_element_type_359: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        convert_element_type_360: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_357, torch.float32);  convert_element_type_357 = None
        view_402: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg294_1, _shape_param_432);  arg294_1 = _shape_param_432 = None
        mul_330: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_402, 0.5)
        mul_331: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_358, mul_330);  mul_330 = None
        convert_element_type_361: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_402, torch.float32)
        pow_29: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_361, 3.0)
        mul_332: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_197: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_402, mul_332);  view_402 = mul_332 = None
        mul_333: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_197, 0.7978845608028654);  add_197 = None
        tanh_14: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_333);  mul_333 = None
        add_198: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_334: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_358, add_198);  convert_element_type_358 = add_198 = None
        convert_element_type_362: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_334, torch.bfloat16);  mul_334 = None
        mul_335: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_47: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_335);  mul_335 = None
        mul_336: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_331, sub_47);  mul_331 = sub_47 = None
        mul_337: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_336, 0.7978845608028654);  mul_336 = None
        convert_element_type_363: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_337, torch.bfloat16)
        mul_338: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_337, 0.044715);  mul_337 = None
        pow_30: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_361, 2.0);  convert_element_type_361 = None
        mul_339: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_30, 3.0);  pow_30 = None
        mul_340: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_338, mul_339);  mul_338 = mul_339 = None
        convert_element_type_364: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_340, torch.bfloat16);  mul_340 = None
        add_199: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_363, convert_element_type_364);  convert_element_type_363 = convert_element_type_364 = None
        mul_341: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_362, 0.5);  convert_element_type_362 = None
        add_200: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_199, mul_341);  add_199 = mul_341 = None
        view_403: "bf16[128, 16384]" = torch.ops.aten.view.default(add_200, _shape_param_433);  add_200 = _shape_param_433 = None
        mm_172: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_403, arg295_1);  arg295_1 = None
        permute_142: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_403, [1, 0])
        mm_173: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_142, arg296_1);  permute_142 = None
        sum_106: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_403, [0], True, dtype = torch.float32);  view_403 = None
        view_404: "f32[16384]" = torch.ops.aten.view.default(sum_106, _shape_param_434);  sum_106 = _shape_param_434 = None
        convert_element_type_365: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_404, torch.bfloat16);  view_404 = None
        view_405: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_172, _shape_param_435);  mm_172 = _shape_param_435 = None
        convert_element_type_366: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_405, torch.float32);  view_405 = None
        convert_element_type_367: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        convert_element_type_368: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_365, torch.float32);  convert_element_type_365 = None
        mm_174: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_141, arg297_1);  permute_141 = arg297_1 = None
        mm_175: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_399, arg298_1);  view_399 = arg298_1 = None
        view_406: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_175, _shape_param_436);  mm_175 = _shape_param_436 = None
        convert_element_type_369: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_174, torch.float32);  mm_174 = None
        view_407: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_406, _shape_param_437);  view_406 = _shape_param_437 = None
        permute_143: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_407, [0, 2, 1, 3]);  view_407 = None
        view_408: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_143, _shape_param_438);  permute_143 = _shape_param_438 = None
        bmm_56: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg299_1, view_408);  arg299_1 = None
        bmm_57: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_408, arg300_1);  view_408 = arg300_1 = None
        view_409: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_56, _shape_param_439);  bmm_56 = _shape_param_439 = None
        view_410: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_57, _shape_param_440);  bmm_57 = _shape_param_440 = None
        convert_element_type_370: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_410, torch.float32);  view_410 = None
        mul_342: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_370, arg301_1);  convert_element_type_370 = None
        sum_107: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_342, [-1], True)
        neg_42: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg301_1);  arg301_1 = None
        fma_14: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_42, sum_107, mul_342);  neg_42 = sum_107 = mul_342 = None
        convert_element_type_371: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_14, torch.bfloat16);  fma_14 = None
        div_15: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_371, 16.0);  convert_element_type_371 = None
        view_411: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_15, _shape_param_441);  div_15 = _shape_param_441 = None
        bmm_58: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg302_1, view_411);  arg302_1 = None
        bmm_59: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_411, arg303_1);  view_411 = arg303_1 = None
        view_412: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_58, _shape_param_442);  bmm_58 = _shape_param_442 = None
        view_413: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_59, _shape_param_443);  bmm_59 = _shape_param_443 = None
        convert_element_type_372: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_412, torch.float32);  view_412 = None
        permute_144: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_372, [0, 1, 3, 2]);  convert_element_type_372 = None
        convert_element_type_373: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_144, torch.bfloat16);  permute_144 = None
        convert_element_type_374: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_413, torch.bfloat16);  view_413 = None
        permute_145: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_374, [0, 2, 1, 3]);  convert_element_type_374 = None
        permute_146: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_373, [0, 2, 1, 3]);  convert_element_type_373 = None
        slice_58: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_145, 3, 0, 64)
        slice_59: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_145, 3, 64, 256);  permute_145 = None
        slice_60: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_146, 3, 0, 64)
        slice_61: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_146, 3, 64, 256);  permute_146 = None
        expand_29: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg304_1, _shape_param_444);  arg304_1 = _shape_param_444 = None
        clone_42: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_414: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_42, _shape_param_445);  clone_42 = _shape_param_445 = None
        mul_343: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_58, view_414)
        view_415: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_343, _shape_param_446);  mul_343 = _shape_param_446 = None
        select_56: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_415, -1, 0)
        select_57: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_415, -1, 1);  view_415 = None
        neg_43: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_56);  select_56 = None
        slice_scatter_112: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_43, 3, 1, 9223372036854775807, 2);  neg_43 = None
        slice_scatter_113: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_57, 3, 0, 9223372036854775807, 2);  select_57 = None
        add_201: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_112, slice_scatter_113);  slice_scatter_112 = slice_scatter_113 = None
        expand_30: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg305_1, _shape_param_447);  arg305_1 = _shape_param_447 = None
        clone_43: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_416: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_43, _shape_param_448);  clone_43 = _shape_param_448 = None
        mul_344: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_58, view_416);  slice_58 = None
        add_202: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_201, mul_344);  add_201 = mul_344 = None
        mul_345: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_60, view_414);  view_414 = None
        view_417: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_345, _shape_param_449);  mul_345 = _shape_param_449 = None
        select_58: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_417, -1, 0)
        select_59: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_417, -1, 1);  view_417 = None
        neg_44: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_58);  select_58 = None
        slice_scatter_114: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_44, 3, 1, 9223372036854775807, 2);  neg_44 = None
        slice_scatter_115: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_59, 3, 0, 9223372036854775807, 2);  select_59 = None
        add_203: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_114, slice_scatter_115);  slice_scatter_114 = slice_scatter_115 = None
        mul_346: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_60, view_416);  slice_60 = view_416 = None
        add_204: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_203, mul_346);  add_203 = mul_346 = None
        slice_scatter_116: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_59, 3, 64, 9223372036854775807);  slice_59 = None
        slice_scatter_117: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_202, 3, 0, 64);  add_202 = None
        add_205: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_116, slice_scatter_117);  slice_scatter_116 = slice_scatter_117 = None
        slice_scatter_118: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_61, 3, 64, 9223372036854775807);  slice_61 = None
        slice_scatter_119: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_204, 3, 0, 64);  add_204 = None
        add_206: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_118, slice_scatter_119);  slice_scatter_118 = slice_scatter_119 = None
        permute_147: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
        clone_44: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
        view_418: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_44, _shape_param_450);  clone_44 = _shape_param_450 = None
        view_419: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_206, _shape_param_451);  add_206 = _shape_param_451 = None
        view_420: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_205, _shape_param_452);  add_205 = _shape_param_452 = None
        view_421: "bf16[128, 4096]" = torch.ops.aten.view.default(view_418, _shape_param_453);  view_418 = _shape_param_453 = None
        permute_148: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_421, [1, 0])
        mm_176: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_148, arg296_1);  permute_148 = None
        mm_177: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_421, arg306_1);  view_421 = arg306_1 = None
        view_422: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_177, _shape_param_454);  mm_177 = _shape_param_454 = None
        convert_element_type_375: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_422, torch.float32);  view_422 = None
        add_207: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_366, convert_element_type_375);  convert_element_type_366 = convert_element_type_375 = None
        convert_element_type_376: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        view_423: "bf16[128, 4096]" = torch.ops.aten.view.default(view_419, _shape_param_455);  view_419 = _shape_param_455 = None
        permute_149: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_423, [1, 0])
        mm_178: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_149, arg296_1);  permute_149 = None
        mm_179: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_423, arg307_1);  view_423 = arg307_1 = None
        view_424: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_179, _shape_param_456);  mm_179 = _shape_param_456 = None
        convert_element_type_377: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_424, torch.float32);  view_424 = None
        add_208: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_207, convert_element_type_377);  add_207 = convert_element_type_377 = None
        convert_element_type_378: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_178, torch.float32);  mm_178 = None
        view_425: "bf16[128, 4096]" = torch.ops.aten.view.default(view_420, _shape_param_457);  view_420 = _shape_param_457 = None
        permute_150: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_425, [1, 0])
        mm_180: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_150, arg296_1);  permute_150 = arg296_1 = None
        mm_181: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_425, arg308_1);  view_425 = arg308_1 = None
        view_426: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_181, _shape_param_458);  mm_181 = _shape_param_458 = None
        convert_element_type_379: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_426, torch.float32);  view_426 = None
        add_209: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_208, convert_element_type_379);  add_208 = convert_element_type_379 = None
        convert_element_type_380: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_180, torch.float32);  mm_180 = None
        mul_347: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_209, arg309_1);  arg309_1 = None
        mul_348: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_347, 4096)
        sum_108: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True)
        mul_349: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_347, arg310_1);  mul_347 = None
        sum_109: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_349, [2], True);  mul_349 = None
        mul_350: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg310_1, sum_109);  sum_109 = None
        sub_48: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_348, sum_108);  mul_348 = sum_108 = None
        sub_49: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_48, mul_350);  sub_48 = mul_350 = None
        mul_351: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg311_1, sub_49);  arg311_1 = sub_49 = None
        mul_352: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_209, arg310_1);  arg310_1 = None
        sum_110: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 1]);  mul_352 = None
        sum_111: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_209, [0, 1]);  add_209 = None
        add_210: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_196, mul_351);  add_196 = mul_351 = None
        convert_element_type_381: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16)
        view_427: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_381, _shape_param_459);  convert_element_type_381 = _shape_param_459 = None
        mm_182: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_427, arg312_1);  arg312_1 = None
        permute_151: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_427, [1, 0])
        mm_183: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_151, arg313_1);  arg313_1 = None
        sum_112: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_427, [0], True, dtype = torch.float32)
        view_428: "f32[4096]" = torch.ops.aten.view.default(sum_112, _shape_param_460);  sum_112 = _shape_param_460 = None
        convert_element_type_382: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_428, torch.bfloat16);  view_428 = None
        view_429: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_182, _shape_param_461);  mm_182 = _shape_param_461 = None
        convert_element_type_383: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_429, torch.float32);  view_429 = None
        convert_element_type_384: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_385: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_382, torch.float32);  convert_element_type_382 = None
        view_430: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg314_1, _shape_param_462);  arg314_1 = _shape_param_462 = None
        mul_353: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_430, 0.5)
        mul_354: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_383, mul_353);  mul_353 = None
        convert_element_type_386: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_430, torch.float32)
        pow_31: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_386, 3.0)
        mul_355: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_31, 0.044715);  pow_31 = None
        add_211: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_430, mul_355);  view_430 = mul_355 = None
        mul_356: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_211, 0.7978845608028654);  add_211 = None
        tanh_15: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_356);  mul_356 = None
        add_212: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_357: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_383, add_212);  convert_element_type_383 = add_212 = None
        convert_element_type_387: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_357, torch.bfloat16);  mul_357 = None
        mul_358: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_50: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_358);  mul_358 = None
        mul_359: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_354, sub_50);  mul_354 = sub_50 = None
        mul_360: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_359, 0.7978845608028654);  mul_359 = None
        convert_element_type_388: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16)
        mul_361: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_360, 0.044715);  mul_360 = None
        pow_32: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_386, 2.0);  convert_element_type_386 = None
        mul_362: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_32, 3.0);  pow_32 = None
        mul_363: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
        convert_element_type_389: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_363, torch.bfloat16);  mul_363 = None
        add_213: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_388, convert_element_type_389);  convert_element_type_388 = convert_element_type_389 = None
        mul_364: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_387, 0.5);  convert_element_type_387 = None
        add_214: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_213, mul_364);  add_213 = mul_364 = None
        view_431: "bf16[128, 16384]" = torch.ops.aten.view.default(add_214, _shape_param_463);  add_214 = _shape_param_463 = None
        mm_184: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_431, arg315_1);  arg315_1 = None
        permute_152: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_431, [1, 0])
        mm_185: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_152, arg316_1);  permute_152 = None
        sum_113: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True, dtype = torch.float32);  view_431 = None
        view_432: "f32[16384]" = torch.ops.aten.view.default(sum_113, _shape_param_464);  sum_113 = _shape_param_464 = None
        convert_element_type_390: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_432, torch.bfloat16);  view_432 = None
        view_433: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_184, _shape_param_465);  mm_184 = _shape_param_465 = None
        convert_element_type_391: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_433, torch.float32);  view_433 = None
        convert_element_type_392: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None
        convert_element_type_393: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_390, torch.float32);  convert_element_type_390 = None
        mm_186: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_151, arg317_1);  permute_151 = arg317_1 = None
        mm_187: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_427, arg318_1);  view_427 = arg318_1 = None
        view_434: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_187, _shape_param_466);  mm_187 = _shape_param_466 = None
        convert_element_type_394: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_186, torch.float32);  mm_186 = None
        view_435: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_434, _shape_param_467);  view_434 = _shape_param_467 = None
        permute_153: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_435, [0, 2, 1, 3]);  view_435 = None
        view_436: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_153, _shape_param_468);  permute_153 = _shape_param_468 = None
        bmm_60: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg319_1, view_436);  arg319_1 = None
        bmm_61: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_436, arg320_1);  view_436 = arg320_1 = None
        view_437: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_60, _shape_param_469);  bmm_60 = _shape_param_469 = None
        view_438: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_61, _shape_param_470);  bmm_61 = _shape_param_470 = None
        convert_element_type_395: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_438, torch.float32);  view_438 = None
        mul_365: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_395, arg321_1);  convert_element_type_395 = None
        sum_114: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_365, [-1], True)
        neg_45: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg321_1);  arg321_1 = None
        fma_15: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_45, sum_114, mul_365);  neg_45 = sum_114 = mul_365 = None
        convert_element_type_396: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_15, torch.bfloat16);  fma_15 = None
        div_16: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_396, 16.0);  convert_element_type_396 = None
        view_439: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_16, _shape_param_471);  div_16 = _shape_param_471 = None
        bmm_62: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg322_1, view_439);  arg322_1 = None
        bmm_63: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_439, arg323_1);  view_439 = arg323_1 = None
        view_440: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_62, _shape_param_472);  bmm_62 = _shape_param_472 = None
        view_441: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_63, _shape_param_473);  bmm_63 = _shape_param_473 = None
        convert_element_type_397: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_440, torch.float32);  view_440 = None
        permute_154: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_397, [0, 1, 3, 2]);  convert_element_type_397 = None
        convert_element_type_398: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_154, torch.bfloat16);  permute_154 = None
        convert_element_type_399: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_441, torch.bfloat16);  view_441 = None
        permute_155: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_399, [0, 2, 1, 3]);  convert_element_type_399 = None
        permute_156: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_398, [0, 2, 1, 3]);  convert_element_type_398 = None
        slice_62: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_155, 3, 0, 64)
        slice_63: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_155, 3, 64, 256);  permute_155 = None
        slice_64: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_156, 3, 0, 64)
        slice_65: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_156, 3, 64, 256);  permute_156 = None
        expand_31: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg324_1, _shape_param_474);  arg324_1 = _shape_param_474 = None
        clone_45: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_442: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_45, _shape_param_475);  clone_45 = _shape_param_475 = None
        mul_366: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_62, view_442)
        view_443: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_366, _shape_param_476);  mul_366 = _shape_param_476 = None
        select_60: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_443, -1, 0)
        select_61: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_443, -1, 1);  view_443 = None
        neg_46: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_60);  select_60 = None
        slice_scatter_120: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_46, 3, 1, 9223372036854775807, 2);  neg_46 = None
        slice_scatter_121: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_61, 3, 0, 9223372036854775807, 2);  select_61 = None
        add_215: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_120, slice_scatter_121);  slice_scatter_120 = slice_scatter_121 = None
        expand_32: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg325_1, _shape_param_477);  arg325_1 = _shape_param_477 = None
        clone_46: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_444: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_46, _shape_param_478);  clone_46 = _shape_param_478 = None
        mul_367: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_62, view_444);  slice_62 = None
        add_216: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_215, mul_367);  add_215 = mul_367 = None
        mul_368: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_64, view_442);  view_442 = None
        view_445: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_368, _shape_param_479);  mul_368 = _shape_param_479 = None
        select_62: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_445, -1, 0)
        select_63: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_445, -1, 1);  view_445 = None
        neg_47: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_62);  select_62 = None
        slice_scatter_122: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_47, 3, 1, 9223372036854775807, 2);  neg_47 = None
        slice_scatter_123: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_63, 3, 0, 9223372036854775807, 2);  select_63 = None
        add_217: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_122, slice_scatter_123);  slice_scatter_122 = slice_scatter_123 = None
        mul_369: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_64, view_444);  slice_64 = view_444 = None
        add_218: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_217, mul_369);  add_217 = mul_369 = None
        slice_scatter_124: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_63, 3, 64, 9223372036854775807);  slice_63 = None
        slice_scatter_125: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_216, 3, 0, 64);  add_216 = None
        add_219: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_124, slice_scatter_125);  slice_scatter_124 = slice_scatter_125 = None
        slice_scatter_126: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_65, 3, 64, 9223372036854775807);  slice_65 = None
        slice_scatter_127: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_218, 3, 0, 64);  add_218 = None
        add_220: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_126, slice_scatter_127);  slice_scatter_126 = slice_scatter_127 = None
        permute_157: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
        clone_47: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_157, memory_format = torch.contiguous_format);  permute_157 = None
        view_446: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_47, _shape_param_480);  clone_47 = _shape_param_480 = None
        view_447: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_220, _shape_param_481);  add_220 = _shape_param_481 = None
        view_448: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_219, _shape_param_482);  add_219 = _shape_param_482 = None
        view_449: "bf16[128, 4096]" = torch.ops.aten.view.default(view_446, _shape_param_483);  view_446 = _shape_param_483 = None
        permute_158: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_449, [1, 0])
        mm_188: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_158, arg316_1);  permute_158 = None
        mm_189: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_449, arg326_1);  view_449 = arg326_1 = None
        view_450: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_189, _shape_param_484);  mm_189 = _shape_param_484 = None
        convert_element_type_400: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_450, torch.float32);  view_450 = None
        add_221: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_391, convert_element_type_400);  convert_element_type_391 = convert_element_type_400 = None
        convert_element_type_401: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_188, torch.float32);  mm_188 = None
        view_451: "bf16[128, 4096]" = torch.ops.aten.view.default(view_447, _shape_param_485);  view_447 = _shape_param_485 = None
        permute_159: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_451, [1, 0])
        mm_190: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_159, arg316_1);  permute_159 = None
        mm_191: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_451, arg327_1);  view_451 = arg327_1 = None
        view_452: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_191, _shape_param_486);  mm_191 = _shape_param_486 = None
        convert_element_type_402: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_452, torch.float32);  view_452 = None
        add_222: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_221, convert_element_type_402);  add_221 = convert_element_type_402 = None
        convert_element_type_403: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_190, torch.float32);  mm_190 = None
        view_453: "bf16[128, 4096]" = torch.ops.aten.view.default(view_448, _shape_param_487);  view_448 = _shape_param_487 = None
        permute_160: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_453, [1, 0])
        mm_192: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_160, arg316_1);  permute_160 = arg316_1 = None
        mm_193: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_453, arg328_1);  view_453 = arg328_1 = None
        view_454: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_193, _shape_param_488);  mm_193 = _shape_param_488 = None
        convert_element_type_404: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_454, torch.float32);  view_454 = None
        add_223: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_222, convert_element_type_404);  add_222 = convert_element_type_404 = None
        convert_element_type_405: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        mul_370: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_223, arg329_1);  arg329_1 = None
        mul_371: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_370, 4096)
        sum_115: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_370, [2], True)
        mul_372: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_370, arg330_1);  mul_370 = None
        sum_116: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_372, [2], True);  mul_372 = None
        mul_373: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg330_1, sum_116);  sum_116 = None
        sub_51: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_371, sum_115);  mul_371 = sum_115 = None
        sub_52: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_51, mul_373);  sub_51 = mul_373 = None
        mul_374: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg331_1, sub_52);  arg331_1 = sub_52 = None
        mul_375: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_223, arg330_1);  arg330_1 = None
        sum_117: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_375, [0, 1]);  mul_375 = None
        sum_118: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_223, [0, 1]);  add_223 = None
        add_224: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_210, mul_374);  add_210 = mul_374 = None
        convert_element_type_406: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_224, torch.bfloat16)
        view_455: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_406, _shape_param_489);  convert_element_type_406 = _shape_param_489 = None
        mm_194: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_455, arg332_1);  arg332_1 = None
        permute_161: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_455, [1, 0])
        mm_195: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_161, arg333_1);  arg333_1 = None
        sum_119: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_455, [0], True, dtype = torch.float32)
        view_456: "f32[4096]" = torch.ops.aten.view.default(sum_119, _shape_param_490);  sum_119 = _shape_param_490 = None
        convert_element_type_407: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_456, torch.bfloat16);  view_456 = None
        view_457: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_194, _shape_param_491);  mm_194 = _shape_param_491 = None
        convert_element_type_408: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_457, torch.float32);  view_457 = None
        convert_element_type_409: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        convert_element_type_410: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_407, torch.float32);  convert_element_type_407 = None
        view_458: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg334_1, _shape_param_492);  arg334_1 = _shape_param_492 = None
        mul_376: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_458, 0.5)
        mul_377: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_408, mul_376);  mul_376 = None
        convert_element_type_411: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_458, torch.float32)
        pow_33: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_411, 3.0)
        mul_378: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_225: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_458, mul_378);  view_458 = mul_378 = None
        mul_379: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_225, 0.7978845608028654);  add_225 = None
        tanh_16: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_379);  mul_379 = None
        add_226: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_380: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_408, add_226);  convert_element_type_408 = add_226 = None
        convert_element_type_412: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_380, torch.bfloat16);  mul_380 = None
        mul_381: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_53: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_381);  mul_381 = None
        mul_382: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_377, sub_53);  mul_377 = sub_53 = None
        mul_383: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_382, 0.7978845608028654);  mul_382 = None
        convert_element_type_413: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_383, torch.bfloat16)
        mul_384: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_383, 0.044715);  mul_383 = None
        pow_34: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_411, 2.0);  convert_element_type_411 = None
        mul_385: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_34, 3.0);  pow_34 = None
        mul_386: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
        convert_element_type_414: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_386, torch.bfloat16);  mul_386 = None
        add_227: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_413, convert_element_type_414);  convert_element_type_413 = convert_element_type_414 = None
        mul_387: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_412, 0.5);  convert_element_type_412 = None
        add_228: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_227, mul_387);  add_227 = mul_387 = None
        view_459: "bf16[128, 16384]" = torch.ops.aten.view.default(add_228, _shape_param_493);  add_228 = _shape_param_493 = None
        mm_196: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_459, arg335_1);  arg335_1 = None
        permute_162: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_459, [1, 0])
        mm_197: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_162, arg336_1);  permute_162 = None
        sum_120: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True, dtype = torch.float32);  view_459 = None
        view_460: "f32[16384]" = torch.ops.aten.view.default(sum_120, _shape_param_494);  sum_120 = _shape_param_494 = None
        convert_element_type_415: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_460, torch.bfloat16);  view_460 = None
        view_461: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_196, _shape_param_495);  mm_196 = _shape_param_495 = None
        convert_element_type_416: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_461, torch.float32);  view_461 = None
        convert_element_type_417: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None
        convert_element_type_418: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_415, torch.float32);  convert_element_type_415 = None
        mm_198: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_161, arg337_1);  permute_161 = arg337_1 = None
        mm_199: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_455, arg338_1);  view_455 = arg338_1 = None
        view_462: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_199, _shape_param_496);  mm_199 = _shape_param_496 = None
        convert_element_type_419: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_198, torch.float32);  mm_198 = None
        view_463: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_462, _shape_param_497);  view_462 = _shape_param_497 = None
        permute_163: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_463, [0, 2, 1, 3]);  view_463 = None
        view_464: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_163, _shape_param_498);  permute_163 = _shape_param_498 = None
        bmm_64: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg339_1, view_464);  arg339_1 = None
        bmm_65: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_464, arg340_1);  view_464 = arg340_1 = None
        view_465: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_64, _shape_param_499);  bmm_64 = _shape_param_499 = None
        view_466: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_65, _shape_param_500);  bmm_65 = _shape_param_500 = None
        convert_element_type_420: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_466, torch.float32);  view_466 = None
        mul_388: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_420, arg341_1);  convert_element_type_420 = None
        sum_121: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [-1], True)
        neg_48: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg341_1);  arg341_1 = None
        fma_16: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_48, sum_121, mul_388);  neg_48 = sum_121 = mul_388 = None
        convert_element_type_421: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_16, torch.bfloat16);  fma_16 = None
        div_17: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_421, 16.0);  convert_element_type_421 = None
        view_467: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_17, _shape_param_501);  div_17 = _shape_param_501 = None
        bmm_66: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg342_1, view_467);  arg342_1 = None
        bmm_67: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_467, arg343_1);  view_467 = arg343_1 = None
        view_468: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_66, _shape_param_502);  bmm_66 = _shape_param_502 = None
        view_469: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_67, _shape_param_503);  bmm_67 = _shape_param_503 = None
        convert_element_type_422: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_468, torch.float32);  view_468 = None
        permute_164: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_422, [0, 1, 3, 2]);  convert_element_type_422 = None
        convert_element_type_423: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_164, torch.bfloat16);  permute_164 = None
        convert_element_type_424: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_469, torch.bfloat16);  view_469 = None
        permute_165: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_424, [0, 2, 1, 3]);  convert_element_type_424 = None
        permute_166: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_423, [0, 2, 1, 3]);  convert_element_type_423 = None
        slice_66: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_165, 3, 0, 64)
        slice_67: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_165, 3, 64, 256);  permute_165 = None
        slice_68: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_166, 3, 0, 64)
        slice_69: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_166, 3, 64, 256);  permute_166 = None
        expand_33: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg344_1, _shape_param_504);  arg344_1 = _shape_param_504 = None
        clone_48: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_470: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_48, _shape_param_505);  clone_48 = _shape_param_505 = None
        mul_389: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_66, view_470)
        view_471: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_389, _shape_param_506);  mul_389 = _shape_param_506 = None
        select_64: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_471, -1, 0)
        select_65: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_471, -1, 1);  view_471 = None
        neg_49: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_64);  select_64 = None
        slice_scatter_128: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_49, 3, 1, 9223372036854775807, 2);  neg_49 = None
        slice_scatter_129: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_65, 3, 0, 9223372036854775807, 2);  select_65 = None
        add_229: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_128, slice_scatter_129);  slice_scatter_128 = slice_scatter_129 = None
        expand_34: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg345_1, _shape_param_507);  arg345_1 = _shape_param_507 = None
        clone_49: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
        view_472: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_49, _shape_param_508);  clone_49 = _shape_param_508 = None
        mul_390: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_66, view_472);  slice_66 = None
        add_230: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_229, mul_390);  add_229 = mul_390 = None
        mul_391: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_68, view_470);  view_470 = None
        view_473: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_391, _shape_param_509);  mul_391 = _shape_param_509 = None
        select_66: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_473, -1, 0)
        select_67: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_473, -1, 1);  view_473 = None
        neg_50: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_66);  select_66 = None
        slice_scatter_130: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_50, 3, 1, 9223372036854775807, 2);  neg_50 = None
        slice_scatter_131: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_67, 3, 0, 9223372036854775807, 2);  select_67 = None
        add_231: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_130, slice_scatter_131);  slice_scatter_130 = slice_scatter_131 = None
        mul_392: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_68, view_472);  slice_68 = view_472 = None
        add_232: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_231, mul_392);  add_231 = mul_392 = None
        slice_scatter_132: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_67, 3, 64, 9223372036854775807);  slice_67 = None
        slice_scatter_133: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_230, 3, 0, 64);  add_230 = None
        add_233: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_132, slice_scatter_133);  slice_scatter_132 = slice_scatter_133 = None
        slice_scatter_134: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_69, 3, 64, 9223372036854775807);  slice_69 = None
        slice_scatter_135: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_232, 3, 0, 64);  add_232 = None
        add_234: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_134, slice_scatter_135);  slice_scatter_134 = slice_scatter_135 = None
        permute_167: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
        clone_50: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        view_474: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_50, _shape_param_510);  clone_50 = _shape_param_510 = None
        view_475: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_234, _shape_param_511);  add_234 = _shape_param_511 = None
        view_476: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_233, _shape_param_512);  add_233 = _shape_param_512 = None
        view_477: "bf16[128, 4096]" = torch.ops.aten.view.default(view_474, _shape_param_513);  view_474 = _shape_param_513 = None
        permute_168: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_477, [1, 0])
        mm_200: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_168, arg336_1);  permute_168 = None
        mm_201: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_477, arg346_1);  view_477 = arg346_1 = None
        view_478: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_201, _shape_param_514);  mm_201 = _shape_param_514 = None
        convert_element_type_425: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_478, torch.float32);  view_478 = None
        add_235: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_416, convert_element_type_425);  convert_element_type_416 = convert_element_type_425 = None
        convert_element_type_426: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        view_479: "bf16[128, 4096]" = torch.ops.aten.view.default(view_475, _shape_param_515);  view_475 = _shape_param_515 = None
        permute_169: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_479, [1, 0])
        mm_202: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_169, arg336_1);  permute_169 = None
        mm_203: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_479, arg347_1);  view_479 = arg347_1 = None
        view_480: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_203, _shape_param_516);  mm_203 = _shape_param_516 = None
        convert_element_type_427: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_480, torch.float32);  view_480 = None
        add_236: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_235, convert_element_type_427);  add_235 = convert_element_type_427 = None
        convert_element_type_428: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_202, torch.float32);  mm_202 = None
        view_481: "bf16[128, 4096]" = torch.ops.aten.view.default(view_476, _shape_param_517);  view_476 = _shape_param_517 = None
        permute_170: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_481, [1, 0])
        mm_204: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_170, arg336_1);  permute_170 = arg336_1 = None
        mm_205: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_481, arg348_1);  view_481 = arg348_1 = None
        view_482: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_205, _shape_param_518);  mm_205 = _shape_param_518 = None
        convert_element_type_429: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_482, torch.float32);  view_482 = None
        add_237: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_236, convert_element_type_429);  add_236 = convert_element_type_429 = None
        convert_element_type_430: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_204, torch.float32);  mm_204 = None
        mul_393: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_237, arg349_1);  arg349_1 = None
        mul_394: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_393, 4096)
        sum_122: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_393, [2], True)
        mul_395: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_393, arg350_1);  mul_393 = None
        sum_123: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True);  mul_395 = None
        mul_396: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg350_1, sum_123);  sum_123 = None
        sub_54: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_394, sum_122);  mul_394 = sum_122 = None
        sub_55: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_54, mul_396);  sub_54 = mul_396 = None
        mul_397: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg351_1, sub_55);  arg351_1 = sub_55 = None
        mul_398: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_237, arg350_1);  arg350_1 = None
        sum_124: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_398, [0, 1]);  mul_398 = None
        sum_125: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_237, [0, 1]);  add_237 = None
        add_238: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_224, mul_397);  add_224 = mul_397 = None
        convert_element_type_431: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_238, torch.bfloat16)
        view_483: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_431, _shape_param_519);  convert_element_type_431 = _shape_param_519 = None
        mm_206: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_483, arg352_1);  arg352_1 = None
        permute_171: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_483, [1, 0])
        mm_207: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_171, arg353_1);  arg353_1 = None
        sum_126: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_483, [0], True, dtype = torch.float32)
        view_484: "f32[4096]" = torch.ops.aten.view.default(sum_126, _shape_param_520);  sum_126 = _shape_param_520 = None
        convert_element_type_432: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_484, torch.bfloat16);  view_484 = None
        view_485: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_206, _shape_param_521);  mm_206 = _shape_param_521 = None
        convert_element_type_433: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_485, torch.float32);  view_485 = None
        convert_element_type_434: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        convert_element_type_435: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_432, torch.float32);  convert_element_type_432 = None
        view_486: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg354_1, _shape_param_522);  arg354_1 = _shape_param_522 = None
        mul_399: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_486, 0.5)
        mul_400: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_433, mul_399);  mul_399 = None
        convert_element_type_436: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_486, torch.float32)
        pow_35: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_436, 3.0)
        mul_401: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_35, 0.044715);  pow_35 = None
        add_239: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_486, mul_401);  view_486 = mul_401 = None
        mul_402: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_239, 0.7978845608028654);  add_239 = None
        tanh_17: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_402);  mul_402 = None
        add_240: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_403: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_433, add_240);  convert_element_type_433 = add_240 = None
        convert_element_type_437: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_403, torch.bfloat16);  mul_403 = None
        mul_404: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_56: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_404);  mul_404 = None
        mul_405: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_400, sub_56);  mul_400 = sub_56 = None
        mul_406: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_405, 0.7978845608028654);  mul_405 = None
        convert_element_type_438: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_406, torch.bfloat16)
        mul_407: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_406, 0.044715);  mul_406 = None
        pow_36: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_436, 2.0);  convert_element_type_436 = None
        mul_408: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_36, 3.0);  pow_36 = None
        mul_409: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
        convert_element_type_439: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_409, torch.bfloat16);  mul_409 = None
        add_241: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_438, convert_element_type_439);  convert_element_type_438 = convert_element_type_439 = None
        mul_410: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_437, 0.5);  convert_element_type_437 = None
        add_242: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_241, mul_410);  add_241 = mul_410 = None
        view_487: "bf16[128, 16384]" = torch.ops.aten.view.default(add_242, _shape_param_523);  add_242 = _shape_param_523 = None
        mm_208: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_487, arg355_1);  arg355_1 = None
        permute_172: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_487, [1, 0])
        mm_209: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_172, arg356_1);  permute_172 = None
        sum_127: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True, dtype = torch.float32);  view_487 = None
        view_488: "f32[16384]" = torch.ops.aten.view.default(sum_127, _shape_param_524);  sum_127 = _shape_param_524 = None
        convert_element_type_440: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_488, torch.bfloat16);  view_488 = None
        view_489: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_208, _shape_param_525);  mm_208 = _shape_param_525 = None
        convert_element_type_441: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_489, torch.float32);  view_489 = None
        convert_element_type_442: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_209, torch.float32);  mm_209 = None
        convert_element_type_443: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_440, torch.float32);  convert_element_type_440 = None
        mm_210: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_171, arg357_1);  permute_171 = arg357_1 = None
        mm_211: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_483, arg358_1);  view_483 = arg358_1 = None
        view_490: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_211, _shape_param_526);  mm_211 = _shape_param_526 = None
        convert_element_type_444: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_210, torch.float32);  mm_210 = None
        view_491: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_490, _shape_param_527);  view_490 = _shape_param_527 = None
        permute_173: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
        view_492: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_173, _shape_param_528);  permute_173 = _shape_param_528 = None
        bmm_68: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg359_1, view_492);  arg359_1 = None
        bmm_69: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_492, arg360_1);  view_492 = arg360_1 = None
        view_493: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_68, _shape_param_529);  bmm_68 = _shape_param_529 = None
        view_494: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_69, _shape_param_530);  bmm_69 = _shape_param_530 = None
        convert_element_type_445: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_494, torch.float32);  view_494 = None
        mul_411: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_445, arg361_1);  convert_element_type_445 = None
        sum_128: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [-1], True)
        neg_51: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg361_1);  arg361_1 = None
        fma_17: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_51, sum_128, mul_411);  neg_51 = sum_128 = mul_411 = None
        convert_element_type_446: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_17, torch.bfloat16);  fma_17 = None
        div_18: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_446, 16.0);  convert_element_type_446 = None
        view_495: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_18, _shape_param_531);  div_18 = _shape_param_531 = None
        bmm_70: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg362_1, view_495);  arg362_1 = None
        bmm_71: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_495, arg363_1);  view_495 = arg363_1 = None
        view_496: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_70, _shape_param_532);  bmm_70 = _shape_param_532 = None
        view_497: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_71, _shape_param_533);  bmm_71 = _shape_param_533 = None
        convert_element_type_447: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_496, torch.float32);  view_496 = None
        permute_174: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_447, [0, 1, 3, 2]);  convert_element_type_447 = None
        convert_element_type_448: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_174, torch.bfloat16);  permute_174 = None
        convert_element_type_449: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_497, torch.bfloat16);  view_497 = None
        permute_175: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_449, [0, 2, 1, 3]);  convert_element_type_449 = None
        permute_176: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_448, [0, 2, 1, 3]);  convert_element_type_448 = None
        slice_70: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_175, 3, 0, 64)
        slice_71: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_175, 3, 64, 256);  permute_175 = None
        slice_72: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_176, 3, 0, 64)
        slice_73: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_176, 3, 64, 256);  permute_176 = None
        expand_35: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg364_1, _shape_param_534);  arg364_1 = _shape_param_534 = None
        clone_51: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_498: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_51, _shape_param_535);  clone_51 = _shape_param_535 = None
        mul_412: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_70, view_498)
        view_499: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_412, _shape_param_536);  mul_412 = _shape_param_536 = None
        select_68: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_499, -1, 0)
        select_69: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_499, -1, 1);  view_499 = None
        neg_52: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_68);  select_68 = None
        slice_scatter_136: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_52, 3, 1, 9223372036854775807, 2);  neg_52 = None
        slice_scatter_137: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_69, 3, 0, 9223372036854775807, 2);  select_69 = None
        add_243: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_136, slice_scatter_137);  slice_scatter_136 = slice_scatter_137 = None
        expand_36: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg365_1, _shape_param_537);  arg365_1 = _shape_param_537 = None
        clone_52: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_500: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_52, _shape_param_538);  clone_52 = _shape_param_538 = None
        mul_413: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_70, view_500);  slice_70 = None
        add_244: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_243, mul_413);  add_243 = mul_413 = None
        mul_414: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_72, view_498);  view_498 = None
        view_501: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_414, _shape_param_539);  mul_414 = _shape_param_539 = None
        select_70: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_501, -1, 0)
        select_71: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_501, -1, 1);  view_501 = None
        neg_53: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_70);  select_70 = None
        slice_scatter_138: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_53, 3, 1, 9223372036854775807, 2);  neg_53 = None
        slice_scatter_139: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_71, 3, 0, 9223372036854775807, 2);  select_71 = None
        add_245: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_138, slice_scatter_139);  slice_scatter_138 = slice_scatter_139 = None
        mul_415: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_72, view_500);  slice_72 = view_500 = None
        add_246: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_245, mul_415);  add_245 = mul_415 = None
        slice_scatter_140: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_71, 3, 64, 9223372036854775807);  slice_71 = None
        slice_scatter_141: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_244, 3, 0, 64);  add_244 = None
        add_247: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_140, slice_scatter_141);  slice_scatter_140 = slice_scatter_141 = None
        slice_scatter_142: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_73, 3, 64, 9223372036854775807);  slice_73 = None
        slice_scatter_143: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_246, 3, 0, 64);  add_246 = None
        add_248: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_142, slice_scatter_143);  slice_scatter_142 = slice_scatter_143 = None
        permute_177: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
        clone_53: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
        view_502: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_53, _shape_param_540);  clone_53 = _shape_param_540 = None
        view_503: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_248, _shape_param_541);  add_248 = _shape_param_541 = None
        view_504: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_247, _shape_param_542);  add_247 = _shape_param_542 = None
        view_505: "bf16[128, 4096]" = torch.ops.aten.view.default(view_502, _shape_param_543);  view_502 = _shape_param_543 = None
        permute_178: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_505, [1, 0])
        mm_212: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_178, arg356_1);  permute_178 = None
        mm_213: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_505, arg366_1);  view_505 = arg366_1 = None
        view_506: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_213, _shape_param_544);  mm_213 = _shape_param_544 = None
        convert_element_type_450: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_506, torch.float32);  view_506 = None
        add_249: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_441, convert_element_type_450);  convert_element_type_441 = convert_element_type_450 = None
        convert_element_type_451: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_212, torch.float32);  mm_212 = None
        view_507: "bf16[128, 4096]" = torch.ops.aten.view.default(view_503, _shape_param_545);  view_503 = _shape_param_545 = None
        permute_179: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_507, [1, 0])
        mm_214: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_179, arg356_1);  permute_179 = None
        mm_215: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_507, arg367_1);  view_507 = arg367_1 = None
        view_508: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_215, _shape_param_546);  mm_215 = _shape_param_546 = None
        convert_element_type_452: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_508, torch.float32);  view_508 = None
        add_250: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_249, convert_element_type_452);  add_249 = convert_element_type_452 = None
        convert_element_type_453: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_214, torch.float32);  mm_214 = None
        view_509: "bf16[128, 4096]" = torch.ops.aten.view.default(view_504, _shape_param_547);  view_504 = _shape_param_547 = None
        permute_180: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_509, [1, 0])
        mm_216: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_180, arg356_1);  permute_180 = arg356_1 = None
        mm_217: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_509, arg368_1);  view_509 = arg368_1 = None
        view_510: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_217, _shape_param_548);  mm_217 = _shape_param_548 = None
        convert_element_type_454: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_510, torch.float32);  view_510 = None
        add_251: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_250, convert_element_type_454);  add_250 = convert_element_type_454 = None
        convert_element_type_455: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_216, torch.float32);  mm_216 = None
        mul_416: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_251, arg369_1);  arg369_1 = None
        mul_417: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_416, 4096)
        sum_129: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
        mul_418: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_416, arg370_1);  mul_416 = None
        sum_130: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
        mul_419: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg370_1, sum_130);  sum_130 = None
        sub_57: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_417, sum_129);  mul_417 = sum_129 = None
        sub_58: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_57, mul_419);  sub_57 = mul_419 = None
        mul_420: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg371_1, sub_58);  arg371_1 = sub_58 = None
        mul_421: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_251, arg370_1);  arg370_1 = None
        sum_131: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
        sum_132: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None
        add_252: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_238, mul_420);  add_238 = mul_420 = None
        convert_element_type_456: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16)
        view_511: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_456, _shape_param_549);  convert_element_type_456 = _shape_param_549 = None
        mm_218: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_511, arg372_1);  arg372_1 = None
        permute_181: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_511, [1, 0])
        mm_219: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_181, arg373_1);  arg373_1 = None
        sum_133: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True, dtype = torch.float32)
        view_512: "f32[4096]" = torch.ops.aten.view.default(sum_133, _shape_param_550);  sum_133 = _shape_param_550 = None
        convert_element_type_457: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_512, torch.bfloat16);  view_512 = None
        view_513: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_218, _shape_param_551);  mm_218 = _shape_param_551 = None
        convert_element_type_458: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_513, torch.float32);  view_513 = None
        convert_element_type_459: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_219, torch.float32);  mm_219 = None
        convert_element_type_460: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_457, torch.float32);  convert_element_type_457 = None
        view_514: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg374_1, _shape_param_552);  arg374_1 = _shape_param_552 = None
        mul_422: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_514, 0.5)
        mul_423: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_458, mul_422);  mul_422 = None
        convert_element_type_461: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_514, torch.float32)
        pow_37: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_461, 3.0)
        mul_424: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_253: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_514, mul_424);  view_514 = mul_424 = None
        mul_425: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_253, 0.7978845608028654);  add_253 = None
        tanh_18: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_425);  mul_425 = None
        add_254: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_426: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_458, add_254);  convert_element_type_458 = add_254 = None
        convert_element_type_462: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_426, torch.bfloat16);  mul_426 = None
        mul_427: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_59: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_427);  mul_427 = None
        mul_428: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_423, sub_59);  mul_423 = sub_59 = None
        mul_429: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_428, 0.7978845608028654);  mul_428 = None
        convert_element_type_463: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_429, torch.bfloat16)
        mul_430: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_429, 0.044715);  mul_429 = None
        pow_38: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_461, 2.0);  convert_element_type_461 = None
        mul_431: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_432: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_430, mul_431);  mul_430 = mul_431 = None
        convert_element_type_464: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_432, torch.bfloat16);  mul_432 = None
        add_255: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_463, convert_element_type_464);  convert_element_type_463 = convert_element_type_464 = None
        mul_433: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_462, 0.5);  convert_element_type_462 = None
        add_256: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_255, mul_433);  add_255 = mul_433 = None
        view_515: "bf16[128, 16384]" = torch.ops.aten.view.default(add_256, _shape_param_553);  add_256 = _shape_param_553 = None
        mm_220: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_515, arg375_1);  arg375_1 = None
        permute_182: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_515, [1, 0])
        mm_221: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_182, arg376_1);  permute_182 = None
        sum_134: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True, dtype = torch.float32);  view_515 = None
        view_516: "f32[16384]" = torch.ops.aten.view.default(sum_134, _shape_param_554);  sum_134 = _shape_param_554 = None
        convert_element_type_465: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_516, torch.bfloat16);  view_516 = None
        view_517: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_220, _shape_param_555);  mm_220 = _shape_param_555 = None
        convert_element_type_466: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_517, torch.float32);  view_517 = None
        convert_element_type_467: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_221, torch.float32);  mm_221 = None
        convert_element_type_468: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_465, torch.float32);  convert_element_type_465 = None
        mm_222: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_181, arg377_1);  permute_181 = arg377_1 = None
        mm_223: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_511, arg378_1);  view_511 = arg378_1 = None
        view_518: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_223, _shape_param_556);  mm_223 = _shape_param_556 = None
        convert_element_type_469: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_222, torch.float32);  mm_222 = None
        view_519: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_518, _shape_param_557);  view_518 = _shape_param_557 = None
        permute_183: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_519, [0, 2, 1, 3]);  view_519 = None
        view_520: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_183, _shape_param_558);  permute_183 = _shape_param_558 = None
        bmm_72: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg379_1, view_520);  arg379_1 = None
        bmm_73: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_520, arg380_1);  view_520 = arg380_1 = None
        view_521: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_72, _shape_param_559);  bmm_72 = _shape_param_559 = None
        view_522: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_73, _shape_param_560);  bmm_73 = _shape_param_560 = None
        convert_element_type_470: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_522, torch.float32);  view_522 = None
        mul_434: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_470, arg381_1);  convert_element_type_470 = None
        sum_135: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [-1], True)
        neg_54: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg381_1);  arg381_1 = None
        fma_18: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_54, sum_135, mul_434);  neg_54 = sum_135 = mul_434 = None
        convert_element_type_471: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_18, torch.bfloat16);  fma_18 = None
        div_19: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_471, 16.0);  convert_element_type_471 = None
        view_523: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_19, _shape_param_561);  div_19 = _shape_param_561 = None
        bmm_74: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg382_1, view_523);  arg382_1 = None
        bmm_75: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_523, arg383_1);  view_523 = arg383_1 = None
        view_524: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_74, _shape_param_562);  bmm_74 = _shape_param_562 = None
        view_525: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_75, _shape_param_563);  bmm_75 = _shape_param_563 = None
        convert_element_type_472: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_524, torch.float32);  view_524 = None
        permute_184: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_472, [0, 1, 3, 2]);  convert_element_type_472 = None
        convert_element_type_473: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_184, torch.bfloat16);  permute_184 = None
        convert_element_type_474: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_525, torch.bfloat16);  view_525 = None
        permute_185: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_474, [0, 2, 1, 3]);  convert_element_type_474 = None
        permute_186: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_473, [0, 2, 1, 3]);  convert_element_type_473 = None
        slice_74: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_185, 3, 0, 64)
        slice_75: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_185, 3, 64, 256);  permute_185 = None
        slice_76: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_186, 3, 0, 64)
        slice_77: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_186, 3, 64, 256);  permute_186 = None
        expand_37: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg384_1, _shape_param_564);  arg384_1 = _shape_param_564 = None
        clone_54: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_526: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_54, _shape_param_565);  clone_54 = _shape_param_565 = None
        mul_435: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_74, view_526)
        view_527: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_435, _shape_param_566);  mul_435 = _shape_param_566 = None
        select_72: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_527, -1, 0)
        select_73: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_527, -1, 1);  view_527 = None
        neg_55: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_72);  select_72 = None
        slice_scatter_144: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_55, 3, 1, 9223372036854775807, 2);  neg_55 = None
        slice_scatter_145: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_73, 3, 0, 9223372036854775807, 2);  select_73 = None
        add_257: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_144, slice_scatter_145);  slice_scatter_144 = slice_scatter_145 = None
        expand_38: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg385_1, _shape_param_567);  arg385_1 = _shape_param_567 = None
        clone_55: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_528: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_55, _shape_param_568);  clone_55 = _shape_param_568 = None
        mul_436: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_74, view_528);  slice_74 = None
        add_258: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_257, mul_436);  add_257 = mul_436 = None
        mul_437: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_76, view_526);  view_526 = None
        view_529: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_437, _shape_param_569);  mul_437 = _shape_param_569 = None
        select_74: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_529, -1, 0)
        select_75: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_529, -1, 1);  view_529 = None
        neg_56: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_74);  select_74 = None
        slice_scatter_146: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_56, 3, 1, 9223372036854775807, 2);  neg_56 = None
        slice_scatter_147: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_75, 3, 0, 9223372036854775807, 2);  select_75 = None
        add_259: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_146, slice_scatter_147);  slice_scatter_146 = slice_scatter_147 = None
        mul_438: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_76, view_528);  slice_76 = view_528 = None
        add_260: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_259, mul_438);  add_259 = mul_438 = None
        slice_scatter_148: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_75, 3, 64, 9223372036854775807);  slice_75 = None
        slice_scatter_149: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_258, 3, 0, 64);  add_258 = None
        add_261: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_148, slice_scatter_149);  slice_scatter_148 = slice_scatter_149 = None
        slice_scatter_150: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_77, 3, 64, 9223372036854775807);  slice_77 = None
        slice_scatter_151: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_260, 3, 0, 64);  add_260 = None
        add_262: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_150, slice_scatter_151);  slice_scatter_150 = slice_scatter_151 = None
        permute_187: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_521, [0, 2, 1, 3]);  view_521 = None
        clone_56: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_187, memory_format = torch.contiguous_format);  permute_187 = None
        view_530: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_56, _shape_param_570);  clone_56 = _shape_param_570 = None
        view_531: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_262, _shape_param_571);  add_262 = _shape_param_571 = None
        view_532: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_261, _shape_param_572);  add_261 = _shape_param_572 = None
        view_533: "bf16[128, 4096]" = torch.ops.aten.view.default(view_530, _shape_param_573);  view_530 = _shape_param_573 = None
        permute_188: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_533, [1, 0])
        mm_224: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_188, arg376_1);  permute_188 = None
        mm_225: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_533, arg386_1);  view_533 = arg386_1 = None
        view_534: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_225, _shape_param_574);  mm_225 = _shape_param_574 = None
        convert_element_type_475: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_534, torch.float32);  view_534 = None
        add_263: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_466, convert_element_type_475);  convert_element_type_466 = convert_element_type_475 = None
        convert_element_type_476: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_224, torch.float32);  mm_224 = None
        view_535: "bf16[128, 4096]" = torch.ops.aten.view.default(view_531, _shape_param_575);  view_531 = _shape_param_575 = None
        permute_189: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_226: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_189, arg376_1);  permute_189 = None
        mm_227: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_535, arg387_1);  view_535 = arg387_1 = None
        view_536: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_227, _shape_param_576);  mm_227 = _shape_param_576 = None
        convert_element_type_477: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_536, torch.float32);  view_536 = None
        add_264: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_263, convert_element_type_477);  add_263 = convert_element_type_477 = None
        convert_element_type_478: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_226, torch.float32);  mm_226 = None
        view_537: "bf16[128, 4096]" = torch.ops.aten.view.default(view_532, _shape_param_577);  view_532 = _shape_param_577 = None
        permute_190: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_537, [1, 0])
        mm_228: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_190, arg376_1);  permute_190 = arg376_1 = None
        mm_229: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_537, arg388_1);  view_537 = arg388_1 = None
        view_538: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_229, _shape_param_578);  mm_229 = _shape_param_578 = None
        convert_element_type_479: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_538, torch.float32);  view_538 = None
        add_265: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_264, convert_element_type_479);  add_264 = convert_element_type_479 = None
        convert_element_type_480: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_228, torch.float32);  mm_228 = None
        mul_439: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_265, arg389_1);  arg389_1 = None
        mul_440: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_439, 4096)
        sum_136: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True)
        mul_441: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_439, arg390_1);  mul_439 = None
        sum_137: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True);  mul_441 = None
        mul_442: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg390_1, sum_137);  sum_137 = None
        sub_60: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_440, sum_136);  mul_440 = sum_136 = None
        sub_61: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_60, mul_442);  sub_60 = mul_442 = None
        mul_443: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg391_1, sub_61);  arg391_1 = sub_61 = None
        mul_444: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_265, arg390_1);  arg390_1 = None
        sum_138: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 1]);  mul_444 = None
        sum_139: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_265, [0, 1]);  add_265 = None
        add_266: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_252, mul_443);  add_252 = mul_443 = None
        convert_element_type_481: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_266, torch.bfloat16)
        view_539: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_481, _shape_param_579);  convert_element_type_481 = _shape_param_579 = None
        mm_230: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_539, arg392_1);  arg392_1 = None
        permute_191: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_539, [1, 0])
        mm_231: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_191, arg393_1);  arg393_1 = None
        sum_140: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_539, [0], True, dtype = torch.float32)
        view_540: "f32[4096]" = torch.ops.aten.view.default(sum_140, _shape_param_580);  sum_140 = _shape_param_580 = None
        convert_element_type_482: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_540, torch.bfloat16);  view_540 = None
        view_541: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_230, _shape_param_581);  mm_230 = _shape_param_581 = None
        convert_element_type_483: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_541, torch.float32);  view_541 = None
        convert_element_type_484: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_231, torch.float32);  mm_231 = None
        convert_element_type_485: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_482, torch.float32);  convert_element_type_482 = None
        view_542: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg394_1, _shape_param_582);  arg394_1 = _shape_param_582 = None
        mul_445: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_542, 0.5)
        mul_446: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_483, mul_445);  mul_445 = None
        convert_element_type_486: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_542, torch.float32)
        pow_39: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_486, 3.0)
        mul_447: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_39, 0.044715);  pow_39 = None
        add_267: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_542, mul_447);  view_542 = mul_447 = None
        mul_448: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_267, 0.7978845608028654);  add_267 = None
        tanh_19: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_448);  mul_448 = None
        add_268: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_449: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_483, add_268);  convert_element_type_483 = add_268 = None
        convert_element_type_487: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_449, torch.bfloat16);  mul_449 = None
        mul_450: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_62: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_450);  mul_450 = None
        mul_451: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_446, sub_62);  mul_446 = sub_62 = None
        mul_452: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_451, 0.7978845608028654);  mul_451 = None
        convert_element_type_488: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_452, torch.bfloat16)
        mul_453: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_452, 0.044715);  mul_452 = None
        pow_40: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_486, 2.0);  convert_element_type_486 = None
        mul_454: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_455: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_453, mul_454);  mul_453 = mul_454 = None
        convert_element_type_489: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_455, torch.bfloat16);  mul_455 = None
        add_269: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_488, convert_element_type_489);  convert_element_type_488 = convert_element_type_489 = None
        mul_456: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_487, 0.5);  convert_element_type_487 = None
        add_270: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_269, mul_456);  add_269 = mul_456 = None
        view_543: "bf16[128, 16384]" = torch.ops.aten.view.default(add_270, _shape_param_583);  add_270 = _shape_param_583 = None
        mm_232: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_543, arg395_1);  arg395_1 = None
        permute_192: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_543, [1, 0])
        mm_233: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_192, arg396_1);  permute_192 = None
        sum_141: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_543, [0], True, dtype = torch.float32);  view_543 = None
        view_544: "f32[16384]" = torch.ops.aten.view.default(sum_141, _shape_param_584);  sum_141 = _shape_param_584 = None
        convert_element_type_490: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_544, torch.bfloat16);  view_544 = None
        view_545: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_232, _shape_param_585);  mm_232 = _shape_param_585 = None
        convert_element_type_491: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_545, torch.float32);  view_545 = None
        convert_element_type_492: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_233, torch.float32);  mm_233 = None
        convert_element_type_493: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_490, torch.float32);  convert_element_type_490 = None
        mm_234: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_191, arg397_1);  permute_191 = arg397_1 = None
        mm_235: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_539, arg398_1);  view_539 = arg398_1 = None
        view_546: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_235, _shape_param_586);  mm_235 = _shape_param_586 = None
        convert_element_type_494: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_234, torch.float32);  mm_234 = None
        view_547: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_546, _shape_param_587);  view_546 = _shape_param_587 = None
        permute_193: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_547, [0, 2, 1, 3]);  view_547 = None
        view_548: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_193, _shape_param_588);  permute_193 = _shape_param_588 = None
        bmm_76: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg399_1, view_548);  arg399_1 = None
        bmm_77: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_548, arg400_1);  view_548 = arg400_1 = None
        view_549: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_76, _shape_param_589);  bmm_76 = _shape_param_589 = None
        view_550: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_77, _shape_param_590);  bmm_77 = _shape_param_590 = None
        convert_element_type_495: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_550, torch.float32);  view_550 = None
        mul_457: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_495, arg401_1);  convert_element_type_495 = None
        sum_142: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_457, [-1], True)
        neg_57: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg401_1);  arg401_1 = None
        fma_19: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_57, sum_142, mul_457);  neg_57 = sum_142 = mul_457 = None
        convert_element_type_496: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_19, torch.bfloat16);  fma_19 = None
        div_20: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_496, 16.0);  convert_element_type_496 = None
        view_551: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_20, _shape_param_591);  div_20 = _shape_param_591 = None
        bmm_78: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg402_1, view_551);  arg402_1 = None
        bmm_79: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_551, arg403_1);  view_551 = arg403_1 = None
        view_552: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_78, _shape_param_592);  bmm_78 = _shape_param_592 = None
        view_553: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_79, _shape_param_593);  bmm_79 = _shape_param_593 = None
        convert_element_type_497: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_552, torch.float32);  view_552 = None
        permute_194: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_497, [0, 1, 3, 2]);  convert_element_type_497 = None
        convert_element_type_498: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_194, torch.bfloat16);  permute_194 = None
        convert_element_type_499: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_553, torch.bfloat16);  view_553 = None
        permute_195: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_499, [0, 2, 1, 3]);  convert_element_type_499 = None
        permute_196: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_498, [0, 2, 1, 3]);  convert_element_type_498 = None
        slice_78: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_195, 3, 0, 64)
        slice_79: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_195, 3, 64, 256);  permute_195 = None
        slice_80: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_196, 3, 0, 64)
        slice_81: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_196, 3, 64, 256);  permute_196 = None
        expand_39: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg404_1, _shape_param_594);  arg404_1 = _shape_param_594 = None
        clone_57: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_554: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_57, _shape_param_595);  clone_57 = _shape_param_595 = None
        mul_458: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_78, view_554)
        view_555: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_458, _shape_param_596);  mul_458 = _shape_param_596 = None
        select_76: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_555, -1, 0)
        select_77: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_555, -1, 1);  view_555 = None
        neg_58: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_76);  select_76 = None
        slice_scatter_152: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_58, 3, 1, 9223372036854775807, 2);  neg_58 = None
        slice_scatter_153: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_77, 3, 0, 9223372036854775807, 2);  select_77 = None
        add_271: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_152, slice_scatter_153);  slice_scatter_152 = slice_scatter_153 = None
        expand_40: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg405_1, _shape_param_597);  arg405_1 = _shape_param_597 = None
        clone_58: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_556: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_58, _shape_param_598);  clone_58 = _shape_param_598 = None
        mul_459: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_78, view_556);  slice_78 = None
        add_272: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_271, mul_459);  add_271 = mul_459 = None
        mul_460: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_80, view_554);  view_554 = None
        view_557: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_460, _shape_param_599);  mul_460 = _shape_param_599 = None
        select_78: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_557, -1, 0)
        select_79: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_557, -1, 1);  view_557 = None
        neg_59: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_78);  select_78 = None
        slice_scatter_154: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_59, 3, 1, 9223372036854775807, 2);  neg_59 = None
        slice_scatter_155: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_79, 3, 0, 9223372036854775807, 2);  select_79 = None
        add_273: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_154, slice_scatter_155);  slice_scatter_154 = slice_scatter_155 = None
        mul_461: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_80, view_556);  slice_80 = view_556 = None
        add_274: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_273, mul_461);  add_273 = mul_461 = None
        slice_scatter_156: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_79, 3, 64, 9223372036854775807);  slice_79 = None
        slice_scatter_157: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_272, 3, 0, 64);  add_272 = None
        add_275: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_156, slice_scatter_157);  slice_scatter_156 = slice_scatter_157 = None
        slice_scatter_158: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_81, 3, 64, 9223372036854775807);  slice_81 = None
        slice_scatter_159: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_274, 3, 0, 64);  add_274 = None
        add_276: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_158, slice_scatter_159);  slice_scatter_158 = slice_scatter_159 = None
        permute_197: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_549, [0, 2, 1, 3]);  view_549 = None
        clone_59: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_197, memory_format = torch.contiguous_format);  permute_197 = None
        view_558: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_59, _shape_param_600);  clone_59 = _shape_param_600 = None
        view_559: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_276, _shape_param_601);  add_276 = _shape_param_601 = None
        view_560: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_275, _shape_param_602);  add_275 = _shape_param_602 = None
        view_561: "bf16[128, 4096]" = torch.ops.aten.view.default(view_558, _shape_param_603);  view_558 = _shape_param_603 = None
        permute_198: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_561, [1, 0])
        mm_236: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_198, arg396_1);  permute_198 = None
        mm_237: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_561, arg406_1);  view_561 = arg406_1 = None
        view_562: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_237, _shape_param_604);  mm_237 = _shape_param_604 = None
        convert_element_type_500: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_562, torch.float32);  view_562 = None
        add_277: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_491, convert_element_type_500);  convert_element_type_491 = convert_element_type_500 = None
        convert_element_type_501: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_236, torch.float32);  mm_236 = None
        view_563: "bf16[128, 4096]" = torch.ops.aten.view.default(view_559, _shape_param_605);  view_559 = _shape_param_605 = None
        permute_199: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_563, [1, 0])
        mm_238: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_199, arg396_1);  permute_199 = None
        mm_239: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_563, arg407_1);  view_563 = arg407_1 = None
        view_564: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_239, _shape_param_606);  mm_239 = _shape_param_606 = None
        convert_element_type_502: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_564, torch.float32);  view_564 = None
        add_278: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_277, convert_element_type_502);  add_277 = convert_element_type_502 = None
        convert_element_type_503: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_238, torch.float32);  mm_238 = None
        view_565: "bf16[128, 4096]" = torch.ops.aten.view.default(view_560, _shape_param_607);  view_560 = _shape_param_607 = None
        permute_200: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_565, [1, 0])
        mm_240: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_200, arg396_1);  permute_200 = arg396_1 = None
        mm_241: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_565, arg408_1);  view_565 = arg408_1 = None
        view_566: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_241, _shape_param_608);  mm_241 = _shape_param_608 = None
        convert_element_type_504: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_566, torch.float32);  view_566 = None
        add_279: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_278, convert_element_type_504);  add_278 = convert_element_type_504 = None
        convert_element_type_505: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_240, torch.float32);  mm_240 = None
        mul_462: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_279, arg409_1);  arg409_1 = None
        mul_463: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_462, 4096)
        sum_143: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True)
        mul_464: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_462, arg410_1);  mul_462 = None
        sum_144: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True);  mul_464 = None
        mul_465: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg410_1, sum_144);  sum_144 = None
        sub_63: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_463, sum_143);  mul_463 = sum_143 = None
        sub_64: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_63, mul_465);  sub_63 = mul_465 = None
        mul_466: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg411_1, sub_64);  arg411_1 = sub_64 = None
        mul_467: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_279, arg410_1);  arg410_1 = None
        sum_145: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 1]);  mul_467 = None
        sum_146: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_279, [0, 1]);  add_279 = None
        add_280: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_266, mul_466);  add_266 = mul_466 = None
        convert_element_type_506: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_280, torch.bfloat16)
        view_567: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_506, _shape_param_609);  convert_element_type_506 = _shape_param_609 = None
        mm_242: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_567, arg412_1);  arg412_1 = None
        permute_201: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_567, [1, 0])
        mm_243: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_201, arg413_1);  arg413_1 = None
        sum_147: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True, dtype = torch.float32)
        view_568: "f32[4096]" = torch.ops.aten.view.default(sum_147, _shape_param_610);  sum_147 = _shape_param_610 = None
        convert_element_type_507: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_568, torch.bfloat16);  view_568 = None
        view_569: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_242, _shape_param_611);  mm_242 = _shape_param_611 = None
        convert_element_type_508: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_569, torch.float32);  view_569 = None
        convert_element_type_509: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_243, torch.float32);  mm_243 = None
        convert_element_type_510: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_507, torch.float32);  convert_element_type_507 = None
        view_570: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg414_1, _shape_param_612);  arg414_1 = _shape_param_612 = None
        mul_468: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_570, 0.5)
        mul_469: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_508, mul_468);  mul_468 = None
        convert_element_type_511: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_570, torch.float32)
        pow_41: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_511, 3.0)
        mul_470: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_281: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_570, mul_470);  view_570 = mul_470 = None
        mul_471: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_281, 0.7978845608028654);  add_281 = None
        tanh_20: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_471);  mul_471 = None
        add_282: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_472: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_508, add_282);  convert_element_type_508 = add_282 = None
        convert_element_type_512: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_472, torch.bfloat16);  mul_472 = None
        mul_473: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_65: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_473);  mul_473 = None
        mul_474: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_469, sub_65);  mul_469 = sub_65 = None
        mul_475: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_474, 0.7978845608028654);  mul_474 = None
        convert_element_type_513: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_475, torch.bfloat16)
        mul_476: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_475, 0.044715);  mul_475 = None
        pow_42: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_511, 2.0);  convert_element_type_511 = None
        mul_477: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_478: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_476, mul_477);  mul_476 = mul_477 = None
        convert_element_type_514: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_478, torch.bfloat16);  mul_478 = None
        add_283: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_513, convert_element_type_514);  convert_element_type_513 = convert_element_type_514 = None
        mul_479: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_512, 0.5);  convert_element_type_512 = None
        add_284: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_283, mul_479);  add_283 = mul_479 = None
        view_571: "bf16[128, 16384]" = torch.ops.aten.view.default(add_284, _shape_param_613);  add_284 = _shape_param_613 = None
        mm_244: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_571, arg415_1);  arg415_1 = None
        permute_202: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_571, [1, 0])
        mm_245: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_202, arg416_1);  permute_202 = None
        sum_148: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_571, [0], True, dtype = torch.float32);  view_571 = None
        view_572: "f32[16384]" = torch.ops.aten.view.default(sum_148, _shape_param_614);  sum_148 = _shape_param_614 = None
        convert_element_type_515: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_572, torch.bfloat16);  view_572 = None
        view_573: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_244, _shape_param_615);  mm_244 = _shape_param_615 = None
        convert_element_type_516: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_573, torch.float32);  view_573 = None
        convert_element_type_517: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_245, torch.float32);  mm_245 = None
        convert_element_type_518: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_515, torch.float32);  convert_element_type_515 = None
        mm_246: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_201, arg417_1);  permute_201 = arg417_1 = None
        mm_247: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_567, arg418_1);  view_567 = arg418_1 = None
        view_574: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_247, _shape_param_616);  mm_247 = _shape_param_616 = None
        convert_element_type_519: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_246, torch.float32);  mm_246 = None
        view_575: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_574, _shape_param_617);  view_574 = _shape_param_617 = None
        permute_203: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_575, [0, 2, 1, 3]);  view_575 = None
        view_576: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_203, _shape_param_618);  permute_203 = _shape_param_618 = None
        bmm_80: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg419_1, view_576);  arg419_1 = None
        bmm_81: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_576, arg420_1);  view_576 = arg420_1 = None
        view_577: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_80, _shape_param_619);  bmm_80 = _shape_param_619 = None
        view_578: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_81, _shape_param_620);  bmm_81 = _shape_param_620 = None
        convert_element_type_520: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_578, torch.float32);  view_578 = None
        mul_480: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_520, arg421_1);  convert_element_type_520 = None
        sum_149: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [-1], True)
        neg_60: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg421_1);  arg421_1 = None
        fma_20: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_60, sum_149, mul_480);  neg_60 = sum_149 = mul_480 = None
        convert_element_type_521: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_20, torch.bfloat16);  fma_20 = None
        div_21: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_521, 16.0);  convert_element_type_521 = None
        view_579: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_21, _shape_param_621);  div_21 = _shape_param_621 = None
        bmm_82: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg422_1, view_579);  arg422_1 = None
        bmm_83: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_579, arg423_1);  view_579 = arg423_1 = None
        view_580: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_82, _shape_param_622);  bmm_82 = _shape_param_622 = None
        view_581: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_83, _shape_param_623);  bmm_83 = _shape_param_623 = None
        convert_element_type_522: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_580, torch.float32);  view_580 = None
        permute_204: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_522, [0, 1, 3, 2]);  convert_element_type_522 = None
        convert_element_type_523: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_204, torch.bfloat16);  permute_204 = None
        convert_element_type_524: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_581, torch.bfloat16);  view_581 = None
        permute_205: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_524, [0, 2, 1, 3]);  convert_element_type_524 = None
        permute_206: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_523, [0, 2, 1, 3]);  convert_element_type_523 = None
        slice_82: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_205, 3, 0, 64)
        slice_83: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_205, 3, 64, 256);  permute_205 = None
        slice_84: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_206, 3, 0, 64)
        slice_85: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_206, 3, 64, 256);  permute_206 = None
        expand_41: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg424_1, _shape_param_624);  arg424_1 = _shape_param_624 = None
        clone_60: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_582: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_60, _shape_param_625);  clone_60 = _shape_param_625 = None
        mul_481: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_82, view_582)
        view_583: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_481, _shape_param_626);  mul_481 = _shape_param_626 = None
        select_80: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_583, -1, 0)
        select_81: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_583, -1, 1);  view_583 = None
        neg_61: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_80);  select_80 = None
        slice_scatter_160: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_61, 3, 1, 9223372036854775807, 2);  neg_61 = None
        slice_scatter_161: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_81, 3, 0, 9223372036854775807, 2);  select_81 = None
        add_285: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_160, slice_scatter_161);  slice_scatter_160 = slice_scatter_161 = None
        expand_42: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg425_1, _shape_param_627);  arg425_1 = _shape_param_627 = None
        clone_61: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_584: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_61, _shape_param_628);  clone_61 = _shape_param_628 = None
        mul_482: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_82, view_584);  slice_82 = None
        add_286: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_285, mul_482);  add_285 = mul_482 = None
        mul_483: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_84, view_582);  view_582 = None
        view_585: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_483, _shape_param_629);  mul_483 = _shape_param_629 = None
        select_82: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_585, -1, 0)
        select_83: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_585, -1, 1);  view_585 = None
        neg_62: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_82);  select_82 = None
        slice_scatter_162: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_62, 3, 1, 9223372036854775807, 2);  neg_62 = None
        slice_scatter_163: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_83, 3, 0, 9223372036854775807, 2);  select_83 = None
        add_287: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_162, slice_scatter_163);  slice_scatter_162 = slice_scatter_163 = None
        mul_484: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_84, view_584);  slice_84 = view_584 = None
        add_288: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_287, mul_484);  add_287 = mul_484 = None
        slice_scatter_164: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_83, 3, 64, 9223372036854775807);  slice_83 = None
        slice_scatter_165: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_286, 3, 0, 64);  add_286 = None
        add_289: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_164, slice_scatter_165);  slice_scatter_164 = slice_scatter_165 = None
        slice_scatter_166: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_85, 3, 64, 9223372036854775807);  slice_85 = None
        slice_scatter_167: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_288, 3, 0, 64);  add_288 = None
        add_290: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_166, slice_scatter_167);  slice_scatter_166 = slice_scatter_167 = None
        permute_207: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_577, [0, 2, 1, 3]);  view_577 = None
        clone_62: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_207, memory_format = torch.contiguous_format);  permute_207 = None
        view_586: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_62, _shape_param_630);  clone_62 = _shape_param_630 = None
        view_587: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_290, _shape_param_631);  add_290 = _shape_param_631 = None
        view_588: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_289, _shape_param_632);  add_289 = _shape_param_632 = None
        view_589: "bf16[128, 4096]" = torch.ops.aten.view.default(view_586, _shape_param_633);  view_586 = _shape_param_633 = None
        permute_208: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_589, [1, 0])
        mm_248: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_208, arg416_1);  permute_208 = None
        mm_249: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_589, arg426_1);  view_589 = arg426_1 = None
        view_590: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_249, _shape_param_634);  mm_249 = _shape_param_634 = None
        convert_element_type_525: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_590, torch.float32);  view_590 = None
        add_291: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_516, convert_element_type_525);  convert_element_type_516 = convert_element_type_525 = None
        convert_element_type_526: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_248, torch.float32);  mm_248 = None
        view_591: "bf16[128, 4096]" = torch.ops.aten.view.default(view_587, _shape_param_635);  view_587 = _shape_param_635 = None
        permute_209: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_591, [1, 0])
        mm_250: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_209, arg416_1);  permute_209 = None
        mm_251: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_591, arg427_1);  view_591 = arg427_1 = None
        view_592: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_251, _shape_param_636);  mm_251 = _shape_param_636 = None
        convert_element_type_527: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_592, torch.float32);  view_592 = None
        add_292: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_291, convert_element_type_527);  add_291 = convert_element_type_527 = None
        convert_element_type_528: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_250, torch.float32);  mm_250 = None
        view_593: "bf16[128, 4096]" = torch.ops.aten.view.default(view_588, _shape_param_637);  view_588 = _shape_param_637 = None
        permute_210: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_252: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_210, arg416_1);  permute_210 = arg416_1 = None
        mm_253: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_593, arg428_1);  view_593 = arg428_1 = None
        view_594: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_253, _shape_param_638);  mm_253 = _shape_param_638 = None
        convert_element_type_529: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_594, torch.float32);  view_594 = None
        add_293: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_292, convert_element_type_529);  add_292 = convert_element_type_529 = None
        convert_element_type_530: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_252, torch.float32);  mm_252 = None
        mul_485: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_293, arg429_1);  arg429_1 = None
        mul_486: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_485, 4096)
        sum_150: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True)
        mul_487: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_485, arg430_1);  mul_485 = None
        sum_151: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True);  mul_487 = None
        mul_488: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg430_1, sum_151);  sum_151 = None
        sub_66: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_486, sum_150);  mul_486 = sum_150 = None
        sub_67: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_66, mul_488);  sub_66 = mul_488 = None
        mul_489: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg431_1, sub_67);  arg431_1 = sub_67 = None
        mul_490: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_293, arg430_1);  arg430_1 = None
        sum_152: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1]);  mul_490 = None
        sum_153: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None
        add_294: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_280, mul_489);  add_280 = mul_489 = None
        convert_element_type_531: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_294, torch.bfloat16)
        view_595: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_531, _shape_param_639);  convert_element_type_531 = _shape_param_639 = None
        mm_254: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_595, arg432_1);  arg432_1 = None
        permute_211: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_595, [1, 0])
        mm_255: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_211, arg433_1);  arg433_1 = None
        sum_154: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_595, [0], True, dtype = torch.float32)
        view_596: "f32[4096]" = torch.ops.aten.view.default(sum_154, _shape_param_640);  sum_154 = _shape_param_640 = None
        convert_element_type_532: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_596, torch.bfloat16);  view_596 = None
        view_597: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_254, _shape_param_641);  mm_254 = _shape_param_641 = None
        convert_element_type_533: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_597, torch.float32);  view_597 = None
        convert_element_type_534: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_255, torch.float32);  mm_255 = None
        convert_element_type_535: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_532, torch.float32);  convert_element_type_532 = None
        view_598: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg434_1, _shape_param_642);  arg434_1 = _shape_param_642 = None
        mul_491: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_598, 0.5)
        mul_492: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_533, mul_491);  mul_491 = None
        convert_element_type_536: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_598, torch.float32)
        pow_43: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_536, 3.0)
        mul_493: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_43, 0.044715);  pow_43 = None
        add_295: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_598, mul_493);  view_598 = mul_493 = None
        mul_494: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_295, 0.7978845608028654);  add_295 = None
        tanh_21: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_494);  mul_494 = None
        add_296: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_495: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_533, add_296);  convert_element_type_533 = add_296 = None
        convert_element_type_537: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_495, torch.bfloat16);  mul_495 = None
        mul_496: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_68: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_496);  mul_496 = None
        mul_497: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_492, sub_68);  mul_492 = sub_68 = None
        mul_498: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_497, 0.7978845608028654);  mul_497 = None
        convert_element_type_538: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_498, torch.bfloat16)
        mul_499: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_498, 0.044715);  mul_498 = None
        pow_44: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_536, 2.0);  convert_element_type_536 = None
        mul_500: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_501: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_499, mul_500);  mul_499 = mul_500 = None
        convert_element_type_539: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_501, torch.bfloat16);  mul_501 = None
        add_297: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_538, convert_element_type_539);  convert_element_type_538 = convert_element_type_539 = None
        mul_502: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_537, 0.5);  convert_element_type_537 = None
        add_298: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_297, mul_502);  add_297 = mul_502 = None
        view_599: "bf16[128, 16384]" = torch.ops.aten.view.default(add_298, _shape_param_643);  add_298 = _shape_param_643 = None
        mm_256: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_599, arg435_1);  arg435_1 = None
        permute_212: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_599, [1, 0])
        mm_257: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_212, arg436_1);  permute_212 = None
        sum_155: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_599, [0], True, dtype = torch.float32);  view_599 = None
        view_600: "f32[16384]" = torch.ops.aten.view.default(sum_155, _shape_param_644);  sum_155 = _shape_param_644 = None
        convert_element_type_540: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_600, torch.bfloat16);  view_600 = None
        view_601: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_256, _shape_param_645);  mm_256 = _shape_param_645 = None
        convert_element_type_541: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_601, torch.float32);  view_601 = None
        convert_element_type_542: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_257, torch.float32);  mm_257 = None
        convert_element_type_543: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_540, torch.float32);  convert_element_type_540 = None
        mm_258: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_211, arg437_1);  permute_211 = arg437_1 = None
        mm_259: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_595, arg438_1);  view_595 = arg438_1 = None
        view_602: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_259, _shape_param_646);  mm_259 = _shape_param_646 = None
        convert_element_type_544: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_258, torch.float32);  mm_258 = None
        view_603: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_602, _shape_param_647);  view_602 = _shape_param_647 = None
        permute_213: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None
        view_604: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_213, _shape_param_648);  permute_213 = _shape_param_648 = None
        bmm_84: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg439_1, view_604);  arg439_1 = None
        bmm_85: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_604, arg440_1);  view_604 = arg440_1 = None
        view_605: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_84, _shape_param_649);  bmm_84 = _shape_param_649 = None
        view_606: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_85, _shape_param_650);  bmm_85 = _shape_param_650 = None
        convert_element_type_545: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_606, torch.float32);  view_606 = None
        mul_503: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_545, arg441_1);  convert_element_type_545 = None
        sum_156: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_503, [-1], True)
        neg_63: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg441_1);  arg441_1 = None
        fma_21: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_63, sum_156, mul_503);  neg_63 = sum_156 = mul_503 = None
        convert_element_type_546: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_21, torch.bfloat16);  fma_21 = None
        div_22: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_546, 16.0);  convert_element_type_546 = None
        view_607: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_22, _shape_param_651);  div_22 = _shape_param_651 = None
        bmm_86: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg442_1, view_607);  arg442_1 = None
        bmm_87: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_607, arg443_1);  view_607 = arg443_1 = None
        view_608: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_86, _shape_param_652);  bmm_86 = _shape_param_652 = None
        view_609: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_87, _shape_param_653);  bmm_87 = _shape_param_653 = None
        convert_element_type_547: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_608, torch.float32);  view_608 = None
        permute_214: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_547, [0, 1, 3, 2]);  convert_element_type_547 = None
        convert_element_type_548: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_214, torch.bfloat16);  permute_214 = None
        convert_element_type_549: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_609, torch.bfloat16);  view_609 = None
        permute_215: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_549, [0, 2, 1, 3]);  convert_element_type_549 = None
        permute_216: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_548, [0, 2, 1, 3]);  convert_element_type_548 = None
        slice_86: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_215, 3, 0, 64)
        slice_87: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_215, 3, 64, 256);  permute_215 = None
        slice_88: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_216, 3, 0, 64)
        slice_89: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_216, 3, 64, 256);  permute_216 = None
        expand_43: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg444_1, _shape_param_654);  arg444_1 = _shape_param_654 = None
        clone_63: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_610: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_63, _shape_param_655);  clone_63 = _shape_param_655 = None
        mul_504: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_86, view_610)
        view_611: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_504, _shape_param_656);  mul_504 = _shape_param_656 = None
        select_84: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_611, -1, 0)
        select_85: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_611, -1, 1);  view_611 = None
        neg_64: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_84);  select_84 = None
        slice_scatter_168: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_64, 3, 1, 9223372036854775807, 2);  neg_64 = None
        slice_scatter_169: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_85, 3, 0, 9223372036854775807, 2);  select_85 = None
        add_299: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_168, slice_scatter_169);  slice_scatter_168 = slice_scatter_169 = None
        expand_44: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg445_1, _shape_param_657);  arg445_1 = _shape_param_657 = None
        clone_64: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_612: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_64, _shape_param_658);  clone_64 = _shape_param_658 = None
        mul_505: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_86, view_612);  slice_86 = None
        add_300: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_299, mul_505);  add_299 = mul_505 = None
        mul_506: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_88, view_610);  view_610 = None
        view_613: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_506, _shape_param_659);  mul_506 = _shape_param_659 = None
        select_86: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_613, -1, 0)
        select_87: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_613, -1, 1);  view_613 = None
        neg_65: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_86);  select_86 = None
        slice_scatter_170: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_65, 3, 1, 9223372036854775807, 2);  neg_65 = None
        slice_scatter_171: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_87, 3, 0, 9223372036854775807, 2);  select_87 = None
        add_301: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_170, slice_scatter_171);  slice_scatter_170 = slice_scatter_171 = None
        mul_507: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_88, view_612);  slice_88 = view_612 = None
        add_302: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_301, mul_507);  add_301 = mul_507 = None
        slice_scatter_172: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_87, 3, 64, 9223372036854775807);  slice_87 = None
        slice_scatter_173: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_300, 3, 0, 64);  add_300 = None
        add_303: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_172, slice_scatter_173);  slice_scatter_172 = slice_scatter_173 = None
        slice_scatter_174: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_89, 3, 64, 9223372036854775807);  slice_89 = None
        slice_scatter_175: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_302, 3, 0, 64);  add_302 = None
        add_304: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_174, slice_scatter_175);  slice_scatter_174 = slice_scatter_175 = None
        permute_217: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_605, [0, 2, 1, 3]);  view_605 = None
        clone_65: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
        view_614: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_65, _shape_param_660);  clone_65 = _shape_param_660 = None
        view_615: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_304, _shape_param_661);  add_304 = _shape_param_661 = None
        view_616: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_303, _shape_param_662);  add_303 = _shape_param_662 = None
        view_617: "bf16[128, 4096]" = torch.ops.aten.view.default(view_614, _shape_param_663);  view_614 = _shape_param_663 = None
        permute_218: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_617, [1, 0])
        mm_260: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_218, arg436_1);  permute_218 = None
        mm_261: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_617, arg446_1);  view_617 = arg446_1 = None
        view_618: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_261, _shape_param_664);  mm_261 = _shape_param_664 = None
        convert_element_type_550: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_618, torch.float32);  view_618 = None
        add_305: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_541, convert_element_type_550);  convert_element_type_541 = convert_element_type_550 = None
        convert_element_type_551: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_260, torch.float32);  mm_260 = None
        view_619: "bf16[128, 4096]" = torch.ops.aten.view.default(view_615, _shape_param_665);  view_615 = _shape_param_665 = None
        permute_219: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_619, [1, 0])
        mm_262: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_219, arg436_1);  permute_219 = None
        mm_263: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_619, arg447_1);  view_619 = arg447_1 = None
        view_620: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_263, _shape_param_666);  mm_263 = _shape_param_666 = None
        convert_element_type_552: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_620, torch.float32);  view_620 = None
        add_306: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_305, convert_element_type_552);  add_305 = convert_element_type_552 = None
        convert_element_type_553: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_262, torch.float32);  mm_262 = None
        view_621: "bf16[128, 4096]" = torch.ops.aten.view.default(view_616, _shape_param_667);  view_616 = _shape_param_667 = None
        permute_220: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_621, [1, 0])
        mm_264: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_220, arg436_1);  permute_220 = arg436_1 = None
        mm_265: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_621, arg448_1);  view_621 = arg448_1 = None
        view_622: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_265, _shape_param_668);  mm_265 = _shape_param_668 = None
        convert_element_type_554: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_622, torch.float32);  view_622 = None
        add_307: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_306, convert_element_type_554);  add_306 = convert_element_type_554 = None
        convert_element_type_555: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_264, torch.float32);  mm_264 = None
        mul_508: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_307, arg449_1);  arg449_1 = None
        mul_509: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_508, 4096)
        sum_157: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True)
        mul_510: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_508, arg450_1);  mul_508 = None
        sum_158: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_510, [2], True);  mul_510 = None
        mul_511: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg450_1, sum_158);  sum_158 = None
        sub_69: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_509, sum_157);  mul_509 = sum_157 = None
        sub_70: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_69, mul_511);  sub_69 = mul_511 = None
        mul_512: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg451_1, sub_70);  arg451_1 = sub_70 = None
        mul_513: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_307, arg450_1);  arg450_1 = None
        sum_159: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_513, [0, 1]);  mul_513 = None
        sum_160: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_307, [0, 1]);  add_307 = None
        add_308: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_294, mul_512);  add_294 = mul_512 = None
        convert_element_type_556: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_308, torch.bfloat16)
        view_623: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_556, _shape_param_669);  convert_element_type_556 = _shape_param_669 = None
        mm_266: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_623, arg452_1);  arg452_1 = None
        permute_221: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_623, [1, 0])
        mm_267: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_221, arg453_1);  arg453_1 = None
        sum_161: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_623, [0], True, dtype = torch.float32)
        view_624: "f32[4096]" = torch.ops.aten.view.default(sum_161, _shape_param_670);  sum_161 = _shape_param_670 = None
        convert_element_type_557: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_624, torch.bfloat16);  view_624 = None
        view_625: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_266, _shape_param_671);  mm_266 = _shape_param_671 = None
        convert_element_type_558: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_625, torch.float32);  view_625 = None
        convert_element_type_559: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_267, torch.float32);  mm_267 = None
        convert_element_type_560: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_557, torch.float32);  convert_element_type_557 = None
        view_626: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg454_1, _shape_param_672);  arg454_1 = _shape_param_672 = None
        mul_514: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_626, 0.5)
        mul_515: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_558, mul_514);  mul_514 = None
        convert_element_type_561: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_626, torch.float32)
        pow_45: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_561, 3.0)
        mul_516: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_309: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_626, mul_516);  view_626 = mul_516 = None
        mul_517: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_309, 0.7978845608028654);  add_309 = None
        tanh_22: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_517);  mul_517 = None
        add_310: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_518: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_558, add_310);  convert_element_type_558 = add_310 = None
        convert_element_type_562: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_518, torch.bfloat16);  mul_518 = None
        mul_519: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_71: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_519);  mul_519 = None
        mul_520: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_515, sub_71);  mul_515 = sub_71 = None
        mul_521: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_520, 0.7978845608028654);  mul_520 = None
        convert_element_type_563: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_521, torch.bfloat16)
        mul_522: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_521, 0.044715);  mul_521 = None
        pow_46: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_561, 2.0);  convert_element_type_561 = None
        mul_523: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_524: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
        convert_element_type_564: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_524, torch.bfloat16);  mul_524 = None
        add_311: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_563, convert_element_type_564);  convert_element_type_563 = convert_element_type_564 = None
        mul_525: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_562, 0.5);  convert_element_type_562 = None
        add_312: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_311, mul_525);  add_311 = mul_525 = None
        view_627: "bf16[128, 16384]" = torch.ops.aten.view.default(add_312, _shape_param_673);  add_312 = _shape_param_673 = None
        mm_268: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_627, arg455_1);  arg455_1 = None
        permute_222: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_627, [1, 0])
        mm_269: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_222, arg456_1);  permute_222 = None
        sum_162: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_627, [0], True, dtype = torch.float32);  view_627 = None
        view_628: "f32[16384]" = torch.ops.aten.view.default(sum_162, _shape_param_674);  sum_162 = _shape_param_674 = None
        convert_element_type_565: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_628, torch.bfloat16);  view_628 = None
        view_629: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_268, _shape_param_675);  mm_268 = _shape_param_675 = None
        convert_element_type_566: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_629, torch.float32);  view_629 = None
        convert_element_type_567: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_269, torch.float32);  mm_269 = None
        convert_element_type_568: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_565, torch.float32);  convert_element_type_565 = None
        mm_270: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_221, arg457_1);  permute_221 = arg457_1 = None
        mm_271: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_623, arg458_1);  view_623 = arg458_1 = None
        view_630: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_271, _shape_param_676);  mm_271 = _shape_param_676 = None
        convert_element_type_569: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_270, torch.float32);  mm_270 = None
        view_631: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_630, _shape_param_677);  view_630 = _shape_param_677 = None
        permute_223: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_631, [0, 2, 1, 3]);  view_631 = None
        view_632: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_223, _shape_param_678);  permute_223 = _shape_param_678 = None
        bmm_88: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg459_1, view_632);  arg459_1 = None
        bmm_89: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_632, arg460_1);  view_632 = arg460_1 = None
        view_633: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_88, _shape_param_679);  bmm_88 = _shape_param_679 = None
        view_634: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_89, _shape_param_680);  bmm_89 = _shape_param_680 = None
        convert_element_type_570: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_634, torch.float32);  view_634 = None
        mul_526: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_570, arg461_1);  convert_element_type_570 = None
        sum_163: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_526, [-1], True)
        neg_66: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg461_1);  arg461_1 = None
        fma_22: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_66, sum_163, mul_526);  neg_66 = sum_163 = mul_526 = None
        convert_element_type_571: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_22, torch.bfloat16);  fma_22 = None
        div_23: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_571, 16.0);  convert_element_type_571 = None
        view_635: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_23, _shape_param_681);  div_23 = _shape_param_681 = None
        bmm_90: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg462_1, view_635);  arg462_1 = None
        bmm_91: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_635, arg463_1);  view_635 = arg463_1 = None
        view_636: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_90, _shape_param_682);  bmm_90 = _shape_param_682 = None
        view_637: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_91, _shape_param_683);  bmm_91 = _shape_param_683 = None
        convert_element_type_572: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_636, torch.float32);  view_636 = None
        permute_224: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_572, [0, 1, 3, 2]);  convert_element_type_572 = None
        convert_element_type_573: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_224, torch.bfloat16);  permute_224 = None
        convert_element_type_574: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_637, torch.bfloat16);  view_637 = None
        permute_225: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_574, [0, 2, 1, 3]);  convert_element_type_574 = None
        permute_226: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_573, [0, 2, 1, 3]);  convert_element_type_573 = None
        slice_90: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_225, 3, 0, 64)
        slice_91: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_225, 3, 64, 256);  permute_225 = None
        slice_92: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_226, 3, 0, 64)
        slice_93: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_226, 3, 64, 256);  permute_226 = None
        expand_45: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg464_1, _shape_param_684);  arg464_1 = _shape_param_684 = None
        clone_66: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_638: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_66, _shape_param_685);  clone_66 = _shape_param_685 = None
        mul_527: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_90, view_638)
        view_639: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_527, _shape_param_686);  mul_527 = _shape_param_686 = None
        select_88: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_639, -1, 0)
        select_89: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_639, -1, 1);  view_639 = None
        neg_67: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_88);  select_88 = None
        slice_scatter_176: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_67, 3, 1, 9223372036854775807, 2);  neg_67 = None
        slice_scatter_177: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_89, 3, 0, 9223372036854775807, 2);  select_89 = None
        add_313: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_176, slice_scatter_177);  slice_scatter_176 = slice_scatter_177 = None
        expand_46: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg465_1, _shape_param_687);  arg465_1 = _shape_param_687 = None
        clone_67: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
        view_640: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_67, _shape_param_688);  clone_67 = _shape_param_688 = None
        mul_528: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_90, view_640);  slice_90 = None
        add_314: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_313, mul_528);  add_313 = mul_528 = None
        mul_529: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_92, view_638);  view_638 = None
        view_641: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_529, _shape_param_689);  mul_529 = _shape_param_689 = None
        select_90: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_641, -1, 0)
        select_91: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_641, -1, 1);  view_641 = None
        neg_68: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_90);  select_90 = None
        slice_scatter_178: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_68, 3, 1, 9223372036854775807, 2);  neg_68 = None
        slice_scatter_179: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_91, 3, 0, 9223372036854775807, 2);  select_91 = None
        add_315: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_178, slice_scatter_179);  slice_scatter_178 = slice_scatter_179 = None
        mul_530: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_92, view_640);  slice_92 = view_640 = None
        add_316: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_315, mul_530);  add_315 = mul_530 = None
        slice_scatter_180: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_91, 3, 64, 9223372036854775807);  slice_91 = None
        slice_scatter_181: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_314, 3, 0, 64);  add_314 = None
        add_317: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_180, slice_scatter_181);  slice_scatter_180 = slice_scatter_181 = None
        slice_scatter_182: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_93, 3, 64, 9223372036854775807);  slice_93 = None
        slice_scatter_183: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_316, 3, 0, 64);  add_316 = None
        add_318: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_182, slice_scatter_183);  slice_scatter_182 = slice_scatter_183 = None
        permute_227: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_633, [0, 2, 1, 3]);  view_633 = None
        clone_68: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
        view_642: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_68, _shape_param_690);  clone_68 = _shape_param_690 = None
        view_643: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_318, _shape_param_691);  add_318 = _shape_param_691 = None
        view_644: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_317, _shape_param_692);  add_317 = _shape_param_692 = None
        view_645: "bf16[128, 4096]" = torch.ops.aten.view.default(view_642, _shape_param_693);  view_642 = _shape_param_693 = None
        permute_228: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_645, [1, 0])
        mm_272: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_228, arg456_1);  permute_228 = None
        mm_273: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_645, arg466_1);  view_645 = arg466_1 = None
        view_646: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_273, _shape_param_694);  mm_273 = _shape_param_694 = None
        convert_element_type_575: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_646, torch.float32);  view_646 = None
        add_319: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_566, convert_element_type_575);  convert_element_type_566 = convert_element_type_575 = None
        convert_element_type_576: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_272, torch.float32);  mm_272 = None
        view_647: "bf16[128, 4096]" = torch.ops.aten.view.default(view_643, _shape_param_695);  view_643 = _shape_param_695 = None
        permute_229: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_647, [1, 0])
        mm_274: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_229, arg456_1);  permute_229 = None
        mm_275: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_647, arg467_1);  view_647 = arg467_1 = None
        view_648: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_275, _shape_param_696);  mm_275 = _shape_param_696 = None
        convert_element_type_577: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_648, torch.float32);  view_648 = None
        add_320: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_319, convert_element_type_577);  add_319 = convert_element_type_577 = None
        convert_element_type_578: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_274, torch.float32);  mm_274 = None
        view_649: "bf16[128, 4096]" = torch.ops.aten.view.default(view_644, _shape_param_697);  view_644 = _shape_param_697 = None
        permute_230: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_649, [1, 0])
        mm_276: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_230, arg456_1);  permute_230 = arg456_1 = None
        mm_277: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_649, arg468_1);  view_649 = arg468_1 = None
        view_650: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_277, _shape_param_698);  mm_277 = _shape_param_698 = None
        convert_element_type_579: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_650, torch.float32);  view_650 = None
        add_321: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_320, convert_element_type_579);  add_320 = convert_element_type_579 = None
        convert_element_type_580: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_276, torch.float32);  mm_276 = None
        mul_531: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_321, arg469_1);  arg469_1 = None
        mul_532: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_531, 4096)
        sum_164: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_531, [2], True)
        mul_533: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_531, arg470_1);  mul_531 = None
        sum_165: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_533, [2], True);  mul_533 = None
        mul_534: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg470_1, sum_165);  sum_165 = None
        sub_72: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_532, sum_164);  mul_532 = sum_164 = None
        sub_73: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_72, mul_534);  sub_72 = mul_534 = None
        mul_535: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg471_1, sub_73);  arg471_1 = sub_73 = None
        mul_536: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_321, arg470_1);  arg470_1 = None
        sum_166: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 1]);  mul_536 = None
        sum_167: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_321, [0, 1]);  add_321 = None
        add_322: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_308, mul_535);  add_308 = mul_535 = None
        convert_element_type_581: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_322, torch.bfloat16)
        view_651: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_581, _shape_param_699);  convert_element_type_581 = _shape_param_699 = None
        mm_278: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_651, arg472_1);  arg472_1 = None
        permute_231: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_651, [1, 0])
        mm_279: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_231, arg473_1);  arg473_1 = None
        sum_168: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_651, [0], True, dtype = torch.float32)
        view_652: "f32[4096]" = torch.ops.aten.view.default(sum_168, _shape_param_700);  sum_168 = _shape_param_700 = None
        convert_element_type_582: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_652, torch.bfloat16);  view_652 = None
        view_653: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_278, _shape_param_701);  mm_278 = _shape_param_701 = None
        convert_element_type_583: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_653, torch.float32);  view_653 = None
        convert_element_type_584: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None
        convert_element_type_585: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_582, torch.float32);  convert_element_type_582 = None
        view_654: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg474_1, _shape_param_702);  arg474_1 = _shape_param_702 = None
        mul_537: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_654, 0.5)
        mul_538: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_583, mul_537);  mul_537 = None
        convert_element_type_586: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_654, torch.float32)
        pow_47: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_586, 3.0)
        mul_539: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_47, 0.044715);  pow_47 = None
        add_323: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_654, mul_539);  view_654 = mul_539 = None
        mul_540: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_323, 0.7978845608028654);  add_323 = None
        tanh_23: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_540);  mul_540 = None
        add_324: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_541: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_583, add_324);  convert_element_type_583 = add_324 = None
        convert_element_type_587: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_541, torch.bfloat16);  mul_541 = None
        mul_542: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_74: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_542);  mul_542 = None
        mul_543: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_538, sub_74);  mul_538 = sub_74 = None
        mul_544: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_543, 0.7978845608028654);  mul_543 = None
        convert_element_type_588: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_544, torch.bfloat16)
        mul_545: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_544, 0.044715);  mul_544 = None
        pow_48: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_586, 2.0);  convert_element_type_586 = None
        mul_546: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_547: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_545, mul_546);  mul_545 = mul_546 = None
        convert_element_type_589: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_547, torch.bfloat16);  mul_547 = None
        add_325: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_588, convert_element_type_589);  convert_element_type_588 = convert_element_type_589 = None
        mul_548: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_587, 0.5);  convert_element_type_587 = None
        add_326: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_325, mul_548);  add_325 = mul_548 = None
        view_655: "bf16[128, 16384]" = torch.ops.aten.view.default(add_326, _shape_param_703);  add_326 = _shape_param_703 = None
        mm_280: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_655, arg475_1);  arg475_1 = None
        permute_232: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_655, [1, 0])
        mm_281: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_232, arg476_1);  permute_232 = None
        sum_169: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_655, [0], True, dtype = torch.float32);  view_655 = None
        view_656: "f32[16384]" = torch.ops.aten.view.default(sum_169, _shape_param_704);  sum_169 = _shape_param_704 = None
        convert_element_type_590: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_656, torch.bfloat16);  view_656 = None
        view_657: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_280, _shape_param_705);  mm_280 = _shape_param_705 = None
        convert_element_type_591: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_657, torch.float32);  view_657 = None
        convert_element_type_592: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_281, torch.float32);  mm_281 = None
        convert_element_type_593: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_590, torch.float32);  convert_element_type_590 = None
        mm_282: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_231, arg477_1);  permute_231 = arg477_1 = None
        mm_283: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_651, arg478_1);  view_651 = arg478_1 = None
        view_658: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_283, _shape_param_706);  mm_283 = _shape_param_706 = None
        convert_element_type_594: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_282, torch.float32);  mm_282 = None
        view_659: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_658, _shape_param_707);  view_658 = _shape_param_707 = None
        permute_233: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_659, [0, 2, 1, 3]);  view_659 = None
        view_660: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_233, _shape_param_708);  permute_233 = _shape_param_708 = None
        bmm_92: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg479_1, view_660);  arg479_1 = None
        bmm_93: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_660, arg480_1);  view_660 = arg480_1 = None
        view_661: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_92, _shape_param_709);  bmm_92 = _shape_param_709 = None
        view_662: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_93, _shape_param_710);  bmm_93 = _shape_param_710 = None
        convert_element_type_595: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_662, torch.float32);  view_662 = None
        mul_549: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_595, arg481_1);  convert_element_type_595 = None
        sum_170: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_549, [-1], True)
        neg_69: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg481_1);  arg481_1 = None
        fma_23: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_69, sum_170, mul_549);  neg_69 = sum_170 = mul_549 = None
        convert_element_type_596: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_23, torch.bfloat16);  fma_23 = None
        div_24: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_596, 16.0);  convert_element_type_596 = None
        view_663: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_24, _shape_param_711);  div_24 = _shape_param_711 = None
        bmm_94: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg482_1, view_663);  arg482_1 = None
        bmm_95: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_663, arg483_1);  view_663 = arg483_1 = None
        view_664: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_94, _shape_param_712);  bmm_94 = _shape_param_712 = None
        view_665: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_95, _shape_param_713);  bmm_95 = _shape_param_713 = None
        convert_element_type_597: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_664, torch.float32);  view_664 = None
        permute_234: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_597, [0, 1, 3, 2]);  convert_element_type_597 = None
        convert_element_type_598: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_234, torch.bfloat16);  permute_234 = None
        convert_element_type_599: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_665, torch.bfloat16);  view_665 = None
        permute_235: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_599, [0, 2, 1, 3]);  convert_element_type_599 = None
        permute_236: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_598, [0, 2, 1, 3]);  convert_element_type_598 = None
        slice_94: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_235, 3, 0, 64)
        slice_95: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_235, 3, 64, 256);  permute_235 = None
        slice_96: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_236, 3, 0, 64)
        slice_97: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_236, 3, 64, 256);  permute_236 = None
        expand_47: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg484_1, _shape_param_714);  arg484_1 = _shape_param_714 = None
        clone_69: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_666: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_69, _shape_param_715);  clone_69 = _shape_param_715 = None
        mul_550: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_94, view_666)
        view_667: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_550, _shape_param_716);  mul_550 = _shape_param_716 = None
        select_92: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_667, -1, 0)
        select_93: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_667, -1, 1);  view_667 = None
        neg_70: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_92);  select_92 = None
        slice_scatter_184: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_70, 3, 1, 9223372036854775807, 2);  neg_70 = None
        slice_scatter_185: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_93, 3, 0, 9223372036854775807, 2);  select_93 = None
        add_327: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_184, slice_scatter_185);  slice_scatter_184 = slice_scatter_185 = None
        expand_48: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg485_1, _shape_param_717);  arg485_1 = _shape_param_717 = None
        clone_70: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_668: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_70, _shape_param_718);  clone_70 = _shape_param_718 = None
        mul_551: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_94, view_668);  slice_94 = None
        add_328: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_327, mul_551);  add_327 = mul_551 = None
        mul_552: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_96, view_666);  view_666 = None
        view_669: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_552, _shape_param_719);  mul_552 = _shape_param_719 = None
        select_94: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_669, -1, 0)
        select_95: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_669, -1, 1);  view_669 = None
        neg_71: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_94);  select_94 = None
        slice_scatter_186: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_71, 3, 1, 9223372036854775807, 2);  neg_71 = None
        slice_scatter_187: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_95, 3, 0, 9223372036854775807, 2);  select_95 = None
        add_329: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_186, slice_scatter_187);  slice_scatter_186 = slice_scatter_187 = None
        mul_553: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_96, view_668);  slice_96 = view_668 = None
        add_330: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_329, mul_553);  add_329 = mul_553 = None
        slice_scatter_188: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_95, 3, 64, 9223372036854775807);  slice_95 = None
        slice_scatter_189: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_328, 3, 0, 64);  add_328 = None
        add_331: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_188, slice_scatter_189);  slice_scatter_188 = slice_scatter_189 = None
        slice_scatter_190: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_97, 3, 64, 9223372036854775807);  slice_97 = None
        slice_scatter_191: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_330, 3, 0, 64);  add_330 = None
        add_332: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_190, slice_scatter_191);  slice_scatter_190 = slice_scatter_191 = None
        permute_237: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_661, [0, 2, 1, 3]);  view_661 = None
        clone_71: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_237, memory_format = torch.contiguous_format);  permute_237 = None
        view_670: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_71, _shape_param_720);  clone_71 = _shape_param_720 = None
        view_671: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_332, _shape_param_721);  add_332 = _shape_param_721 = None
        view_672: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_331, _shape_param_722);  add_331 = _shape_param_722 = None
        view_673: "bf16[128, 4096]" = torch.ops.aten.view.default(view_670, _shape_param_723);  view_670 = _shape_param_723 = None
        permute_238: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_673, [1, 0])
        mm_284: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_238, arg476_1);  permute_238 = None
        mm_285: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_673, arg486_1);  view_673 = arg486_1 = None
        view_674: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_285, _shape_param_724);  mm_285 = _shape_param_724 = None
        convert_element_type_600: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_674, torch.float32);  view_674 = None
        add_333: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_591, convert_element_type_600);  convert_element_type_591 = convert_element_type_600 = None
        convert_element_type_601: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_284, torch.float32);  mm_284 = None
        view_675: "bf16[128, 4096]" = torch.ops.aten.view.default(view_671, _shape_param_725);  view_671 = _shape_param_725 = None
        permute_239: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_675, [1, 0])
        mm_286: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_239, arg476_1);  permute_239 = None
        mm_287: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_675, arg487_1);  view_675 = arg487_1 = None
        view_676: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_287, _shape_param_726);  mm_287 = _shape_param_726 = None
        convert_element_type_602: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_676, torch.float32);  view_676 = None
        add_334: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_333, convert_element_type_602);  add_333 = convert_element_type_602 = None
        convert_element_type_603: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_286, torch.float32);  mm_286 = None
        view_677: "bf16[128, 4096]" = torch.ops.aten.view.default(view_672, _shape_param_727);  view_672 = _shape_param_727 = None
        permute_240: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_677, [1, 0])
        mm_288: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_240, arg476_1);  permute_240 = arg476_1 = None
        mm_289: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_677, arg488_1);  view_677 = arg488_1 = None
        view_678: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_289, _shape_param_728);  mm_289 = _shape_param_728 = None
        convert_element_type_604: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_678, torch.float32);  view_678 = None
        add_335: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_334, convert_element_type_604);  add_334 = convert_element_type_604 = None
        convert_element_type_605: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_288, torch.float32);  mm_288 = None
        mul_554: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_335, arg489_1);  arg489_1 = None
        mul_555: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_554, 4096)
        sum_171: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True)
        mul_556: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_554, arg490_1);  mul_554 = None
        sum_172: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_556, [2], True);  mul_556 = None
        mul_557: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg490_1, sum_172);  sum_172 = None
        sub_75: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_555, sum_171);  mul_555 = sum_171 = None
        sub_76: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_75, mul_557);  sub_75 = mul_557 = None
        mul_558: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg491_1, sub_76);  arg491_1 = sub_76 = None
        mul_559: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_335, arg490_1);  arg490_1 = None
        sum_173: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 1]);  mul_559 = None
        sum_174: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_335, [0, 1]);  add_335 = None
        add_336: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_322, mul_558);  add_322 = mul_558 = None
        convert_element_type_606: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_336, torch.bfloat16)
        view_679: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_606, _shape_param_729);  convert_element_type_606 = _shape_param_729 = None
        mm_290: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_679, arg492_1);  arg492_1 = None
        permute_241: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_679, [1, 0])
        mm_291: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_241, arg493_1);  arg493_1 = None
        sum_175: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_679, [0], True, dtype = torch.float32)
        view_680: "f32[4096]" = torch.ops.aten.view.default(sum_175, _shape_param_730);  sum_175 = _shape_param_730 = None
        convert_element_type_607: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_680, torch.bfloat16);  view_680 = None
        view_681: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_290, _shape_param_731);  mm_290 = _shape_param_731 = None
        convert_element_type_608: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_681, torch.float32);  view_681 = None
        convert_element_type_609: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_291, torch.float32);  mm_291 = None
        convert_element_type_610: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_607, torch.float32);  convert_element_type_607 = None
        view_682: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg494_1, _shape_param_732);  arg494_1 = _shape_param_732 = None
        mul_560: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_682, 0.5)
        mul_561: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_608, mul_560);  mul_560 = None
        convert_element_type_611: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_682, torch.float32)
        pow_49: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_611, 3.0)
        mul_562: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_337: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_682, mul_562);  view_682 = mul_562 = None
        mul_563: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_337, 0.7978845608028654);  add_337 = None
        tanh_24: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_563);  mul_563 = None
        add_338: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_24, 1.0)
        mul_564: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_608, add_338);  convert_element_type_608 = add_338 = None
        convert_element_type_612: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_564, torch.bfloat16);  mul_564 = None
        mul_565: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_24, tanh_24);  tanh_24 = None
        sub_77: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_565);  mul_565 = None
        mul_566: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_561, sub_77);  mul_561 = sub_77 = None
        mul_567: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_566, 0.7978845608028654);  mul_566 = None
        convert_element_type_613: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_567, torch.bfloat16)
        mul_568: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_567, 0.044715);  mul_567 = None
        pow_50: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_611, 2.0);  convert_element_type_611 = None
        mul_569: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_50, 3.0);  pow_50 = None
        mul_570: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
        convert_element_type_614: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_570, torch.bfloat16);  mul_570 = None
        add_339: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_613, convert_element_type_614);  convert_element_type_613 = convert_element_type_614 = None
        mul_571: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_612, 0.5);  convert_element_type_612 = None
        add_340: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_339, mul_571);  add_339 = mul_571 = None
        view_683: "bf16[128, 16384]" = torch.ops.aten.view.default(add_340, _shape_param_733);  add_340 = _shape_param_733 = None
        mm_292: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_683, arg495_1);  arg495_1 = None
        permute_242: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_683, [1, 0])
        mm_293: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_242, arg496_1);  permute_242 = None
        sum_176: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_683, [0], True, dtype = torch.float32);  view_683 = None
        view_684: "f32[16384]" = torch.ops.aten.view.default(sum_176, _shape_param_734);  sum_176 = _shape_param_734 = None
        convert_element_type_615: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_684, torch.bfloat16);  view_684 = None
        view_685: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_292, _shape_param_735);  mm_292 = _shape_param_735 = None
        convert_element_type_616: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_685, torch.float32);  view_685 = None
        convert_element_type_617: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_293, torch.float32);  mm_293 = None
        convert_element_type_618: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_615, torch.float32);  convert_element_type_615 = None
        mm_294: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_241, arg497_1);  permute_241 = arg497_1 = None
        mm_295: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_679, arg498_1);  view_679 = arg498_1 = None
        view_686: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_295, _shape_param_736);  mm_295 = _shape_param_736 = None
        convert_element_type_619: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_294, torch.float32);  mm_294 = None
        view_687: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_686, _shape_param_737);  view_686 = _shape_param_737 = None
        permute_243: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_687, [0, 2, 1, 3]);  view_687 = None
        view_688: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_243, _shape_param_738);  permute_243 = _shape_param_738 = None
        bmm_96: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg499_1, view_688);  arg499_1 = None
        bmm_97: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_688, arg500_1);  view_688 = arg500_1 = None
        view_689: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_96, _shape_param_739);  bmm_96 = _shape_param_739 = None
        view_690: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_97, _shape_param_740);  bmm_97 = _shape_param_740 = None
        convert_element_type_620: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_690, torch.float32);  view_690 = None
        mul_572: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_620, arg501_1);  convert_element_type_620 = None
        sum_177: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [-1], True)
        neg_72: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg501_1);  arg501_1 = None
        fma_24: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_72, sum_177, mul_572);  neg_72 = sum_177 = mul_572 = None
        convert_element_type_621: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_24, torch.bfloat16);  fma_24 = None
        div_25: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_621, 16.0);  convert_element_type_621 = None
        view_691: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_25, _shape_param_741);  div_25 = _shape_param_741 = None
        bmm_98: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg502_1, view_691);  arg502_1 = None
        bmm_99: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_691, arg503_1);  view_691 = arg503_1 = None
        view_692: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_98, _shape_param_742);  bmm_98 = _shape_param_742 = None
        view_693: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_99, _shape_param_743);  bmm_99 = _shape_param_743 = None
        convert_element_type_622: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_692, torch.float32);  view_692 = None
        permute_244: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_622, [0, 1, 3, 2]);  convert_element_type_622 = None
        convert_element_type_623: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_244, torch.bfloat16);  permute_244 = None
        convert_element_type_624: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_693, torch.bfloat16);  view_693 = None
        permute_245: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_624, [0, 2, 1, 3]);  convert_element_type_624 = None
        permute_246: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_623, [0, 2, 1, 3]);  convert_element_type_623 = None
        slice_98: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_245, 3, 0, 64)
        slice_99: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_245, 3, 64, 256);  permute_245 = None
        slice_100: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_246, 3, 0, 64)
        slice_101: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_246, 3, 64, 256);  permute_246 = None
        expand_49: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg504_1, _shape_param_744);  arg504_1 = _shape_param_744 = None
        clone_72: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_694: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_72, _shape_param_745);  clone_72 = _shape_param_745 = None
        mul_573: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_98, view_694)
        view_695: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_573, _shape_param_746);  mul_573 = _shape_param_746 = None
        select_96: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_695, -1, 0)
        select_97: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_695, -1, 1);  view_695 = None
        neg_73: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_96);  select_96 = None
        slice_scatter_192: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_73, 3, 1, 9223372036854775807, 2);  neg_73 = None
        slice_scatter_193: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_97, 3, 0, 9223372036854775807, 2);  select_97 = None
        add_341: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_192, slice_scatter_193);  slice_scatter_192 = slice_scatter_193 = None
        expand_50: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg505_1, _shape_param_747);  arg505_1 = _shape_param_747 = None
        clone_73: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_696: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_73, _shape_param_748);  clone_73 = _shape_param_748 = None
        mul_574: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_98, view_696);  slice_98 = None
        add_342: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_341, mul_574);  add_341 = mul_574 = None
        mul_575: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_100, view_694);  view_694 = None
        view_697: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_575, _shape_param_749);  mul_575 = _shape_param_749 = None
        select_98: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_697, -1, 0)
        select_99: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_697, -1, 1);  view_697 = None
        neg_74: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_98);  select_98 = None
        slice_scatter_194: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_74, 3, 1, 9223372036854775807, 2);  neg_74 = None
        slice_scatter_195: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_99, 3, 0, 9223372036854775807, 2);  select_99 = None
        add_343: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_194, slice_scatter_195);  slice_scatter_194 = slice_scatter_195 = None
        mul_576: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_100, view_696);  slice_100 = view_696 = None
        add_344: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_343, mul_576);  add_343 = mul_576 = None
        slice_scatter_196: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_99, 3, 64, 9223372036854775807);  slice_99 = None
        slice_scatter_197: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_342, 3, 0, 64);  add_342 = None
        add_345: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_196, slice_scatter_197);  slice_scatter_196 = slice_scatter_197 = None
        slice_scatter_198: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_101, 3, 64, 9223372036854775807);  slice_101 = None
        slice_scatter_199: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_344, 3, 0, 64);  add_344 = None
        add_346: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_198, slice_scatter_199);  slice_scatter_198 = slice_scatter_199 = None
        permute_247: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_689, [0, 2, 1, 3]);  view_689 = None
        clone_74: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
        view_698: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_74, _shape_param_750);  clone_74 = _shape_param_750 = None
        view_699: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_346, _shape_param_751);  add_346 = _shape_param_751 = None
        view_700: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_345, _shape_param_752);  add_345 = _shape_param_752 = None
        view_701: "bf16[128, 4096]" = torch.ops.aten.view.default(view_698, _shape_param_753);  view_698 = _shape_param_753 = None
        permute_248: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_701, [1, 0])
        mm_296: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_248, arg496_1);  permute_248 = None
        mm_297: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_701, arg506_1);  view_701 = arg506_1 = None
        view_702: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_297, _shape_param_754);  mm_297 = _shape_param_754 = None
        convert_element_type_625: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_702, torch.float32);  view_702 = None
        add_347: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_616, convert_element_type_625);  convert_element_type_616 = convert_element_type_625 = None
        convert_element_type_626: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_296, torch.float32);  mm_296 = None
        view_703: "bf16[128, 4096]" = torch.ops.aten.view.default(view_699, _shape_param_755);  view_699 = _shape_param_755 = None
        permute_249: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_703, [1, 0])
        mm_298: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_249, arg496_1);  permute_249 = None
        mm_299: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_703, arg507_1);  view_703 = arg507_1 = None
        view_704: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_299, _shape_param_756);  mm_299 = _shape_param_756 = None
        convert_element_type_627: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_704, torch.float32);  view_704 = None
        add_348: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_347, convert_element_type_627);  add_347 = convert_element_type_627 = None
        convert_element_type_628: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_298, torch.float32);  mm_298 = None
        view_705: "bf16[128, 4096]" = torch.ops.aten.view.default(view_700, _shape_param_757);  view_700 = _shape_param_757 = None
        permute_250: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_705, [1, 0])
        mm_300: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_250, arg496_1);  permute_250 = arg496_1 = None
        mm_301: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_705, arg508_1);  view_705 = arg508_1 = None
        view_706: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_301, _shape_param_758);  mm_301 = _shape_param_758 = None
        convert_element_type_629: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_706, torch.float32);  view_706 = None
        add_349: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_348, convert_element_type_629);  add_348 = convert_element_type_629 = None
        convert_element_type_630: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_300, torch.float32);  mm_300 = None
        mul_577: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_349, arg509_1);  arg509_1 = None
        mul_578: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_577, 4096)
        sum_178: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_577, [2], True)
        mul_579: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_577, arg510_1);  mul_577 = None
        sum_179: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True);  mul_579 = None
        mul_580: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg510_1, sum_179);  sum_179 = None
        sub_78: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_578, sum_178);  mul_578 = sum_178 = None
        sub_79: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_78, mul_580);  sub_78 = mul_580 = None
        mul_581: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg511_1, sub_79);  arg511_1 = sub_79 = None
        mul_582: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_349, arg510_1);  arg510_1 = None
        sum_180: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_582, [0, 1]);  mul_582 = None
        sum_181: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_349, [0, 1]);  add_349 = None
        add_350: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_336, mul_581);  add_336 = mul_581 = None
        convert_element_type_631: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_350, torch.bfloat16)
        view_707: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_631, _shape_param_759);  convert_element_type_631 = _shape_param_759 = None
        mm_302: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_707, arg512_1);  arg512_1 = None
        permute_251: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_707, [1, 0])
        mm_303: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_251, arg513_1);  arg513_1 = None
        sum_182: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_707, [0], True, dtype = torch.float32)
        view_708: "f32[4096]" = torch.ops.aten.view.default(sum_182, _shape_param_760);  sum_182 = _shape_param_760 = None
        convert_element_type_632: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_708, torch.bfloat16);  view_708 = None
        view_709: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_302, _shape_param_761);  mm_302 = _shape_param_761 = None
        convert_element_type_633: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_709, torch.float32);  view_709 = None
        convert_element_type_634: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_303, torch.float32);  mm_303 = None
        convert_element_type_635: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_632, torch.float32);  convert_element_type_632 = None
        view_710: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg514_1, _shape_param_762);  arg514_1 = _shape_param_762 = None
        mul_583: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_710, 0.5)
        mul_584: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_633, mul_583);  mul_583 = None
        convert_element_type_636: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_710, torch.float32)
        pow_51: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_636, 3.0)
        mul_585: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_51, 0.044715);  pow_51 = None
        add_351: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_710, mul_585);  view_710 = mul_585 = None
        mul_586: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_351, 0.7978845608028654);  add_351 = None
        tanh_25: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_586);  mul_586 = None
        add_352: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_25, 1.0)
        mul_587: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_633, add_352);  convert_element_type_633 = add_352 = None
        convert_element_type_637: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_587, torch.bfloat16);  mul_587 = None
        mul_588: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_25, tanh_25);  tanh_25 = None
        sub_80: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_588);  mul_588 = None
        mul_589: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_584, sub_80);  mul_584 = sub_80 = None
        mul_590: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_589, 0.7978845608028654);  mul_589 = None
        convert_element_type_638: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_590, torch.bfloat16)
        mul_591: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_590, 0.044715);  mul_590 = None
        pow_52: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_636, 2.0);  convert_element_type_636 = None
        mul_592: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_52, 3.0);  pow_52 = None
        mul_593: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_591, mul_592);  mul_591 = mul_592 = None
        convert_element_type_639: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_593, torch.bfloat16);  mul_593 = None
        add_353: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_638, convert_element_type_639);  convert_element_type_638 = convert_element_type_639 = None
        mul_594: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_637, 0.5);  convert_element_type_637 = None
        add_354: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_353, mul_594);  add_353 = mul_594 = None
        view_711: "bf16[128, 16384]" = torch.ops.aten.view.default(add_354, _shape_param_763);  add_354 = _shape_param_763 = None
        mm_304: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_711, arg515_1);  arg515_1 = None
        permute_252: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_711, [1, 0])
        mm_305: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_252, arg516_1);  permute_252 = None
        sum_183: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_711, [0], True, dtype = torch.float32);  view_711 = None
        view_712: "f32[16384]" = torch.ops.aten.view.default(sum_183, _shape_param_764);  sum_183 = _shape_param_764 = None
        convert_element_type_640: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_712, torch.bfloat16);  view_712 = None
        view_713: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_304, _shape_param_765);  mm_304 = _shape_param_765 = None
        convert_element_type_641: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_713, torch.float32);  view_713 = None
        convert_element_type_642: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_305, torch.float32);  mm_305 = None
        convert_element_type_643: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_640, torch.float32);  convert_element_type_640 = None
        mm_306: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_251, arg517_1);  permute_251 = arg517_1 = None
        mm_307: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_707, arg518_1);  view_707 = arg518_1 = None
        view_714: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_307, _shape_param_766);  mm_307 = _shape_param_766 = None
        convert_element_type_644: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_306, torch.float32);  mm_306 = None
        view_715: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_714, _shape_param_767);  view_714 = _shape_param_767 = None
        permute_253: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_715, [0, 2, 1, 3]);  view_715 = None
        view_716: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_253, _shape_param_768);  permute_253 = _shape_param_768 = None
        bmm_100: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg519_1, view_716);  arg519_1 = None
        bmm_101: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_716, arg520_1);  view_716 = arg520_1 = None
        view_717: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_100, _shape_param_769);  bmm_100 = _shape_param_769 = None
        view_718: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_101, _shape_param_770);  bmm_101 = _shape_param_770 = None
        convert_element_type_645: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_718, torch.float32);  view_718 = None
        mul_595: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_645, arg521_1);  convert_element_type_645 = None
        sum_184: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_595, [-1], True)
        neg_75: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg521_1);  arg521_1 = None
        fma_25: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_75, sum_184, mul_595);  neg_75 = sum_184 = mul_595 = None
        convert_element_type_646: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_25, torch.bfloat16);  fma_25 = None
        div_26: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_646, 16.0);  convert_element_type_646 = None
        view_719: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_26, _shape_param_771);  div_26 = _shape_param_771 = None
        bmm_102: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg522_1, view_719);  arg522_1 = None
        bmm_103: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_719, arg523_1);  view_719 = arg523_1 = None
        view_720: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_102, _shape_param_772);  bmm_102 = _shape_param_772 = None
        view_721: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_103, _shape_param_773);  bmm_103 = _shape_param_773 = None
        convert_element_type_647: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_720, torch.float32);  view_720 = None
        permute_254: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_647, [0, 1, 3, 2]);  convert_element_type_647 = None
        convert_element_type_648: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_254, torch.bfloat16);  permute_254 = None
        convert_element_type_649: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_721, torch.bfloat16);  view_721 = None
        permute_255: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_649, [0, 2, 1, 3]);  convert_element_type_649 = None
        permute_256: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_648, [0, 2, 1, 3]);  convert_element_type_648 = None
        slice_102: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_255, 3, 0, 64)
        slice_103: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_255, 3, 64, 256);  permute_255 = None
        slice_104: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_256, 3, 0, 64)
        slice_105: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_256, 3, 64, 256);  permute_256 = None
        expand_51: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg524_1, _shape_param_774);  arg524_1 = _shape_param_774 = None
        clone_75: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_722: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_75, _shape_param_775);  clone_75 = _shape_param_775 = None
        mul_596: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_102, view_722)
        view_723: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_596, _shape_param_776);  mul_596 = _shape_param_776 = None
        select_100: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_723, -1, 0)
        select_101: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_723, -1, 1);  view_723 = None
        neg_76: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_100);  select_100 = None
        slice_scatter_200: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_76, 3, 1, 9223372036854775807, 2);  neg_76 = None
        slice_scatter_201: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_101, 3, 0, 9223372036854775807, 2);  select_101 = None
        add_355: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_200, slice_scatter_201);  slice_scatter_200 = slice_scatter_201 = None
        expand_52: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg525_1, _shape_param_777);  arg525_1 = _shape_param_777 = None
        clone_76: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_724: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_76, _shape_param_778);  clone_76 = _shape_param_778 = None
        mul_597: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_102, view_724);  slice_102 = None
        add_356: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_355, mul_597);  add_355 = mul_597 = None
        mul_598: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_104, view_722);  view_722 = None
        view_725: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_598, _shape_param_779);  mul_598 = _shape_param_779 = None
        select_102: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_725, -1, 0)
        select_103: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_725, -1, 1);  view_725 = None
        neg_77: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_102);  select_102 = None
        slice_scatter_202: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_77, 3, 1, 9223372036854775807, 2);  neg_77 = None
        slice_scatter_203: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_103, 3, 0, 9223372036854775807, 2);  select_103 = None
        add_357: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_202, slice_scatter_203);  slice_scatter_202 = slice_scatter_203 = None
        mul_599: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_104, view_724);  slice_104 = view_724 = None
        add_358: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_357, mul_599);  add_357 = mul_599 = None
        slice_scatter_204: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_103, 3, 64, 9223372036854775807);  slice_103 = None
        slice_scatter_205: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_356, 3, 0, 64);  add_356 = None
        add_359: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_204, slice_scatter_205);  slice_scatter_204 = slice_scatter_205 = None
        slice_scatter_206: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_105, 3, 64, 9223372036854775807);  slice_105 = None
        slice_scatter_207: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_358, 3, 0, 64);  add_358 = None
        add_360: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_206, slice_scatter_207);  slice_scatter_206 = slice_scatter_207 = None
        permute_257: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_717, [0, 2, 1, 3]);  view_717 = None
        clone_77: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
        view_726: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_77, _shape_param_780);  clone_77 = _shape_param_780 = None
        view_727: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_360, _shape_param_781);  add_360 = _shape_param_781 = None
        view_728: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_359, _shape_param_782);  add_359 = _shape_param_782 = None
        view_729: "bf16[128, 4096]" = torch.ops.aten.view.default(view_726, _shape_param_783);  view_726 = _shape_param_783 = None
        permute_258: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_729, [1, 0])
        mm_308: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_258, arg516_1);  permute_258 = None
        mm_309: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_729, arg526_1);  view_729 = arg526_1 = None
        view_730: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_309, _shape_param_784);  mm_309 = _shape_param_784 = None
        convert_element_type_650: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_730, torch.float32);  view_730 = None
        add_361: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_641, convert_element_type_650);  convert_element_type_641 = convert_element_type_650 = None
        convert_element_type_651: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_308, torch.float32);  mm_308 = None
        view_731: "bf16[128, 4096]" = torch.ops.aten.view.default(view_727, _shape_param_785);  view_727 = _shape_param_785 = None
        permute_259: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_731, [1, 0])
        mm_310: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_259, arg516_1);  permute_259 = None
        mm_311: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_731, arg527_1);  view_731 = arg527_1 = None
        view_732: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_311, _shape_param_786);  mm_311 = _shape_param_786 = None
        convert_element_type_652: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_732, torch.float32);  view_732 = None
        add_362: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_361, convert_element_type_652);  add_361 = convert_element_type_652 = None
        convert_element_type_653: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_310, torch.float32);  mm_310 = None
        view_733: "bf16[128, 4096]" = torch.ops.aten.view.default(view_728, _shape_param_787);  view_728 = _shape_param_787 = None
        permute_260: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_733, [1, 0])
        mm_312: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_260, arg516_1);  permute_260 = arg516_1 = None
        mm_313: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_733, arg528_1);  view_733 = arg528_1 = None
        view_734: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_313, _shape_param_788);  mm_313 = _shape_param_788 = None
        convert_element_type_654: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_734, torch.float32);  view_734 = None
        add_363: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_362, convert_element_type_654);  add_362 = convert_element_type_654 = None
        convert_element_type_655: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_312, torch.float32);  mm_312 = None
        mul_600: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_363, arg529_1);  arg529_1 = None
        mul_601: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_600, 4096)
        sum_185: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_600, [2], True)
        mul_602: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_600, arg530_1);  mul_600 = None
        sum_186: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True);  mul_602 = None
        mul_603: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg530_1, sum_186);  sum_186 = None
        sub_81: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_601, sum_185);  mul_601 = sum_185 = None
        sub_82: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_81, mul_603);  sub_81 = mul_603 = None
        mul_604: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(arg531_1, sub_82);  arg531_1 = sub_82 = None
        mul_605: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_363, arg530_1);  arg530_1 = None
        sum_187: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 1]);  mul_605 = None
        sum_188: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_363, [0, 1]);  add_363 = None
        add_364: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_350, mul_604);  add_350 = mul_604 = None
        convert_element_type_656: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_364, torch.bfloat16)
        view_735: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_656, _shape_param_789);  convert_element_type_656 = _shape_param_789 = None
        mm_314: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_735, arg532_1);  arg532_1 = None
        permute_261: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_735, [1, 0])
        mm_315: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_261, arg533_1);  arg533_1 = None
        sum_189: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_735, [0], True, dtype = torch.float32)
        view_736: "f32[4096]" = torch.ops.aten.view.default(sum_189, _shape_param_790);  sum_189 = _shape_param_790 = None
        convert_element_type_657: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_736, torch.bfloat16);  view_736 = None
        view_737: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_314, _shape_param_791);  mm_314 = _shape_param_791 = None
        convert_element_type_658: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_737, torch.float32);  view_737 = None
        convert_element_type_659: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_315, torch.float32);  mm_315 = None
        convert_element_type_660: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_657, torch.float32);  convert_element_type_657 = None
        view_738: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg534_1, _shape_param_792);  arg534_1 = _shape_param_792 = None
        mul_606: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_738, 0.5)
        mul_607: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_658, mul_606);  mul_606 = None
        convert_element_type_661: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_738, torch.float32)
        pow_53: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_661, 3.0)
        mul_608: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_365: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_738, mul_608);  view_738 = mul_608 = None
        mul_609: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_365, 0.7978845608028654);  add_365 = None
        tanh_26: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_609);  mul_609 = None
        add_366: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_26, 1.0)
        mul_610: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_658, add_366);  convert_element_type_658 = add_366 = None
        convert_element_type_662: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_610, torch.bfloat16);  mul_610 = None
        mul_611: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_26, tanh_26);  tanh_26 = None
        sub_83: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_611);  mul_611 = None
        mul_612: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_607, sub_83);  mul_607 = sub_83 = None
        mul_613: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_612, 0.7978845608028654);  mul_612 = None
        convert_element_type_663: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_613, torch.bfloat16)
        mul_614: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_613, 0.044715);  mul_613 = None
        pow_54: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_661, 2.0);  convert_element_type_661 = None
        mul_615: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_54, 3.0);  pow_54 = None
        mul_616: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_614, mul_615);  mul_614 = mul_615 = None
        convert_element_type_664: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_616, torch.bfloat16);  mul_616 = None
        add_367: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_663, convert_element_type_664);  convert_element_type_663 = convert_element_type_664 = None
        mul_617: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_662, 0.5);  convert_element_type_662 = None
        add_368: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_367, mul_617);  add_367 = mul_617 = None
        view_739: "bf16[128, 16384]" = torch.ops.aten.view.default(add_368, _shape_param_793);  add_368 = _shape_param_793 = None
        mm_316: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_739, arg535_1);  arg535_1 = None
        permute_262: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_739, [1, 0])
        mm_317: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_262, arg536_1);  permute_262 = None
        sum_190: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_739, [0], True, dtype = torch.float32);  view_739 = None
        view_740: "f32[16384]" = torch.ops.aten.view.default(sum_190, _shape_param_794);  sum_190 = _shape_param_794 = None
        convert_element_type_665: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_740, torch.bfloat16);  view_740 = None
        view_741: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_316, _shape_param_795);  mm_316 = _shape_param_795 = None
        convert_element_type_666: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_741, torch.float32);  view_741 = None
        convert_element_type_667: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_317, torch.float32);  mm_317 = None
        convert_element_type_668: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_665, torch.float32);  convert_element_type_665 = None
        mm_318: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_261, arg537_1);  permute_261 = arg537_1 = None
        mm_319: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_735, arg538_1);  view_735 = arg538_1 = None
        view_742: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_319, _shape_param_796);  mm_319 = _shape_param_796 = None
        convert_element_type_669: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_318, torch.float32);  mm_318 = None
        view_743: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_742, _shape_param_797);  view_742 = _shape_param_797 = None
        permute_263: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_743, [0, 2, 1, 3]);  view_743 = None
        view_744: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_263, _shape_param_798);  permute_263 = _shape_param_798 = None
        bmm_104: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg539_1, view_744);  arg539_1 = None
        bmm_105: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_744, arg540_1);  view_744 = arg540_1 = None
        view_745: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_104, _shape_param_799);  bmm_104 = _shape_param_799 = None
        view_746: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_105, _shape_param_800);  bmm_105 = _shape_param_800 = None
        convert_element_type_670: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_746, torch.float32);  view_746 = None
        mul_618: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_670, arg541_1);  convert_element_type_670 = None
        sum_191: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [-1], True)
        neg_78: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg541_1);  arg541_1 = None
        fma_26: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_78, sum_191, mul_618);  neg_78 = sum_191 = mul_618 = None
        convert_element_type_671: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_26, torch.bfloat16);  fma_26 = None
        div_27: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_671, 16.0);  convert_element_type_671 = None
        view_747: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_27, _shape_param_801);  div_27 = _shape_param_801 = None
        bmm_106: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg542_1, view_747);  arg542_1 = None
        bmm_107: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_747, arg543_1);  view_747 = arg543_1 = None
        view_748: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_106, _shape_param_802);  bmm_106 = _shape_param_802 = None
        view_749: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_107, _shape_param_803);  bmm_107 = _shape_param_803 = None
        convert_element_type_672: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_748, torch.float32);  view_748 = None
        permute_264: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_672, [0, 1, 3, 2]);  convert_element_type_672 = None
        convert_element_type_673: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_264, torch.bfloat16);  permute_264 = None
        convert_element_type_674: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_749, torch.bfloat16);  view_749 = None
        permute_265: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_674, [0, 2, 1, 3]);  convert_element_type_674 = None
        permute_266: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_673, [0, 2, 1, 3]);  convert_element_type_673 = None
        slice_106: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_265, 3, 0, 64)
        slice_107: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_265, 3, 64, 256);  permute_265 = None
        slice_108: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_266, 3, 0, 64)
        slice_109: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_266, 3, 64, 256);  permute_266 = None
        expand_53: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg544_1, _shape_param_804);  arg544_1 = _shape_param_804 = None
        clone_78: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_750: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_78, _shape_param_805);  clone_78 = _shape_param_805 = None
        mul_619: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_106, view_750)
        view_751: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_619, _shape_param_806);  mul_619 = _shape_param_806 = None
        select_104: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_751, -1, 0)
        select_105: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_751, -1, 1);  view_751 = None
        neg_79: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_104);  select_104 = None
        slice_scatter_208: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_79, 3, 1, 9223372036854775807, 2);  neg_79 = None
        slice_scatter_209: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_105, 3, 0, 9223372036854775807, 2);  select_105 = None
        add_369: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_208, slice_scatter_209);  slice_scatter_208 = slice_scatter_209 = None
        expand_54: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg545_1, _shape_param_807);  arg545_1 = _shape_param_807 = None
        clone_79: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
        view_752: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_79, _shape_param_808);  clone_79 = _shape_param_808 = None
        mul_620: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_106, view_752);  slice_106 = None
        add_370: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_369, mul_620);  add_369 = mul_620 = None
        mul_621: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_108, view_750);  view_750 = None
        view_753: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_621, _shape_param_809);  mul_621 = _shape_param_809 = None
        select_106: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_753, -1, 0)
        select_107: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_753, -1, 1);  view_753 = None
        neg_80: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_106);  select_106 = None
        slice_scatter_210: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_80, 3, 1, 9223372036854775807, 2);  neg_80 = None
        slice_scatter_211: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_107, 3, 0, 9223372036854775807, 2);  select_107 = None
        add_371: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_210, slice_scatter_211);  slice_scatter_210 = slice_scatter_211 = None
        mul_622: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_108, view_752);  slice_108 = view_752 = None
        add_372: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_371, mul_622);  add_371 = mul_622 = None
        slice_scatter_212: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_107, 3, 64, 9223372036854775807);  slice_107 = None
        slice_scatter_213: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_370, 3, 0, 64);  add_370 = None
        add_373: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_212, slice_scatter_213);  slice_scatter_212 = slice_scatter_213 = None
        slice_scatter_214: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_109, 3, 64, 9223372036854775807);  slice_109 = None
        slice_scatter_215: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_372, 3, 0, 64);  add_372 = None
        add_374: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_214, slice_scatter_215);  slice_scatter_214 = slice_scatter_215 = None
        permute_267: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_745, [0, 2, 1, 3]);  view_745 = None
        clone_80: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_267, memory_format = torch.contiguous_format);  permute_267 = None
        view_754: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_80, _shape_param_810);  clone_80 = _shape_param_810 = None
        view_755: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_374, _shape_param_811);  add_374 = _shape_param_811 = None
        view_756: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_373, _shape_param_812);  add_373 = _shape_param_812 = None
        view_757: "bf16[128, 4096]" = torch.ops.aten.view.default(view_754, _shape_param_813);  view_754 = _shape_param_813 = None
        permute_268: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_757, [1, 0])
        mm_320: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_268, arg536_1);  permute_268 = None
        mm_321: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_757, arg546_1);  view_757 = arg546_1 = None
        view_758: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_321, _shape_param_814);  mm_321 = _shape_param_814 = None
        convert_element_type_675: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_758, torch.float32);  view_758 = None
        add_375: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_666, convert_element_type_675);  convert_element_type_666 = convert_element_type_675 = None
        convert_element_type_676: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_320, torch.float32);  mm_320 = None
        view_759: "bf16[128, 4096]" = torch.ops.aten.view.default(view_755, _shape_param_815);  view_755 = _shape_param_815 = None
        permute_269: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_759, [1, 0])
        mm_322: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_269, arg536_1);  permute_269 = None
        mm_323: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_759, arg547_1);  view_759 = arg547_1 = None
        view_760: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_323, _shape_param_816);  mm_323 = _shape_param_816 = None
        convert_element_type_677: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_760, torch.float32);  view_760 = None
        add_376: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_375, convert_element_type_677);  add_375 = convert_element_type_677 = None
        convert_element_type_678: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_322, torch.float32);  mm_322 = None
        view_761: "bf16[128, 4096]" = torch.ops.aten.view.default(view_756, _shape_param_817);  view_756 = _shape_param_817 = None
        permute_270: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_761, [1, 0])
        mm_324: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_270, arg536_1);  permute_270 = arg536_1 = None
        mm_325: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_761, arg548_1);  view_761 = arg548_1 = None
        view_762: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_325, _shape_param_818);  mm_325 = _shape_param_818 = None
        convert_element_type_679: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_762, torch.float32);  view_762 = None
        add_377: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_376, convert_element_type_679);  add_376 = convert_element_type_679 = None
        convert_element_type_680: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_324, torch.float32);  mm_324 = None
        mul_623: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_377, arg549_1);  arg549_1 = None
        mul_624: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_623, 4096)
        sum_192: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_623, [2], True)
        add_378: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(arg550_1, arg551_1);  arg550_1 = None
        sub_84: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_378, arg552_1);  add_378 = arg552_1 = None
        mul_625: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_84, arg553_1);  sub_84 = None
        mul_626: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_623, mul_625);  mul_623 = None
        sum_193: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_626, [2], True);  mul_626 = None
        mul_627: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_625, sum_193);  sum_193 = None
        sub_85: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_624, sum_192);  mul_624 = sum_192 = None
        sub_86: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_85, mul_627);  sub_85 = mul_627 = None
        div_28: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(arg553_1, 4096);  arg553_1 = None
        mul_628: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_28, sub_86);  div_28 = sub_86 = None
        mul_629: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_377, mul_625);  mul_625 = None
        sum_194: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 1]);  mul_629 = None
        sum_195: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_377, [0, 1]);  add_377 = None
        add_379: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_364, mul_628);  add_364 = mul_628 = None
        convert_element_type_681: "bf16[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(add_379, torch.bfloat16)
        view_763: "bf16[128, 4096]" = torch.ops.aten.view.default(convert_element_type_681, _shape_param_819);  convert_element_type_681 = _shape_param_819 = None
        mm_326: "bf16[128, 16384]" = torch.ops.aten.mm.default(view_763, arg554_1);  arg554_1 = None
        permute_271: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_763, [1, 0])
        mm_327: "bf16[4096, 16384]" = torch.ops.aten.mm.default(permute_271, arg555_1);  arg555_1 = None
        sum_196: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_763, [0], True, dtype = torch.float32)
        view_764: "f32[4096]" = torch.ops.aten.view.default(sum_196, _shape_param_820);  sum_196 = _shape_param_820 = None
        convert_element_type_682: "bf16[4096]" = torch.ops.prims.convert_element_type.default(view_764, torch.bfloat16);  view_764 = None
        view_765: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(mm_326, _shape_param_821);  mm_326 = _shape_param_821 = None
        convert_element_type_683: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_765, torch.float32);  view_765 = None
        convert_element_type_684: "f32[4096, 16384]" = torch.ops.prims.convert_element_type.default(mm_327, torch.float32);  mm_327 = None
        convert_element_type_685: "f32[4096]" = torch.ops.prims.convert_element_type.default(convert_element_type_682, torch.float32);  convert_element_type_682 = None
        view_766: "bf16[1, 128, 16384]" = torch.ops.aten.view.default(arg556_1, _shape_param_822);  arg556_1 = _shape_param_822 = None
        mul_630: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_766, 0.5)
        mul_631: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_683, mul_630);  mul_630 = None
        convert_element_type_686: "f32[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(view_766, torch.float32)
        pow_55: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_686, 3.0)
        mul_632: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_55, 0.044715);  pow_55 = None
        add_380: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_766, mul_632);  view_766 = mul_632 = None
        mul_633: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_380, 0.7978845608028654);  add_380 = None
        tanh_27: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_633);  mul_633 = None
        add_381: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_27, 1.0)
        mul_634: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_683, add_381);  convert_element_type_683 = add_381 = None
        convert_element_type_687: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_634, torch.bfloat16);  mul_634 = None
        mul_635: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(tanh_27, tanh_27);  tanh_27 = None
        sub_87: "f32[1, 128, 16384]" = torch.ops.aten.sub.Tensor(1, mul_635);  mul_635 = None
        mul_636: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_631, sub_87);  mul_631 = sub_87 = None
        mul_637: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_636, 0.7978845608028654);  mul_636 = None
        convert_element_type_688: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_637, torch.bfloat16)
        mul_638: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_637, 0.044715);  mul_637 = None
        pow_56: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_686, 2.0);  convert_element_type_686 = None
        mul_639: "f32[1, 128, 16384]" = torch.ops.aten.mul.Scalar(pow_56, 3.0);  pow_56 = None
        mul_640: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_638, mul_639);  mul_638 = mul_639 = None
        convert_element_type_689: "bf16[1, 128, 16384]" = torch.ops.prims.convert_element_type.default(mul_640, torch.bfloat16);  mul_640 = None
        add_382: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(convert_element_type_688, convert_element_type_689);  convert_element_type_688 = convert_element_type_689 = None
        mul_641: "bf16[1, 128, 16384]" = torch.ops.aten.mul.Tensor(convert_element_type_687, 0.5);  convert_element_type_687 = None
        add_383: "bf16[1, 128, 16384]" = torch.ops.aten.add.Tensor(add_382, mul_641);  add_382 = mul_641 = None
        view_767: "bf16[128, 16384]" = torch.ops.aten.view.default(add_383, _shape_param_823);  add_383 = _shape_param_823 = None
        mm_328: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_767, arg557_1);  arg557_1 = None
        permute_272: "bf16[16384, 128]" = torch.ops.aten.permute.default(view_767, [1, 0])
        mm_329: "bf16[16384, 4096]" = torch.ops.aten.mm.default(permute_272, arg558_1);  permute_272 = None
        sum_197: "f32[1, 16384]" = torch.ops.aten.sum.dim_IntList(view_767, [0], True, dtype = torch.float32);  view_767 = None
        view_768: "f32[16384]" = torch.ops.aten.view.default(sum_197, _shape_param_824);  sum_197 = _shape_param_824 = None
        convert_element_type_690: "bf16[16384]" = torch.ops.prims.convert_element_type.default(view_768, torch.bfloat16);  view_768 = None
        view_769: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_328, _shape_param_825);  mm_328 = _shape_param_825 = None
        convert_element_type_691: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_769, torch.float32);  view_769 = None
        convert_element_type_692: "f32[16384, 4096]" = torch.ops.prims.convert_element_type.default(mm_329, torch.float32);  mm_329 = None
        convert_element_type_693: "f32[16384]" = torch.ops.prims.convert_element_type.default(convert_element_type_690, torch.float32);  convert_element_type_690 = None
        mm_330: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_271, arg559_1);  permute_271 = arg559_1 = None
        mm_331: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_763, arg560_1);  view_763 = arg560_1 = None
        view_770: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_331, _shape_param_826);  mm_331 = _shape_param_826 = None
        convert_element_type_694: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_330, torch.float32);  mm_330 = None
        view_771: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(view_770, _shape_param_827);  view_770 = _shape_param_827 = None
        permute_273: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_771, [0, 2, 1, 3]);  view_771 = None
        view_772: "bf16[16, 128, 256]" = torch.ops.aten.view.default(permute_273, _shape_param_828);  permute_273 = _shape_param_828 = None
        bmm_108: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(arg561_1, view_772);  arg561_1 = None
        bmm_109: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_772, arg562_1);  view_772 = arg562_1 = None
        view_773: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_108, _shape_param_829);  bmm_108 = _shape_param_829 = None
        view_774: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_109, _shape_param_830);  bmm_109 = _shape_param_830 = None
        convert_element_type_695: "f32[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(view_774, torch.float32);  view_774 = None
        mul_642: "f32[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_695, arg563_1);  convert_element_type_695 = None
        sum_198: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_642, [-1], True)
        neg_81: "f32[1, 16, 128, 128]" = torch.ops.aten.neg.default(arg563_1);  arg563_1 = None
        fma_27: "f32[1, 16, 128, 128]" = torch.ops.prims.fma.default(neg_81, sum_198, mul_642);  neg_81 = sum_198 = mul_642 = None
        convert_element_type_696: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(fma_27, torch.bfloat16);  fma_27 = None
        div_29: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_696, 16.0);  convert_element_type_696 = None
        view_775: "bf16[16, 128, 128]" = torch.ops.aten.view.default(div_29, _shape_param_831);  div_29 = _shape_param_831 = None
        bmm_110: "bf16[16, 256, 128]" = torch.ops.aten.bmm.default(arg564_1, view_775);  arg564_1 = None
        bmm_111: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_775, arg565_1);  view_775 = arg565_1 = None
        view_776: "bf16[1, 16, 256, 128]" = torch.ops.aten.view.default(bmm_110, _shape_param_832);  bmm_110 = _shape_param_832 = None
        view_777: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_111, _shape_param_833);  bmm_111 = _shape_param_833 = None
        convert_element_type_697: "f32[1, 16, 256, 128]" = torch.ops.prims.convert_element_type.default(view_776, torch.float32);  view_776 = None
        permute_274: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(convert_element_type_697, [0, 1, 3, 2]);  convert_element_type_697 = None
        convert_element_type_698: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(permute_274, torch.bfloat16);  permute_274 = None
        convert_element_type_699: "bf16[1, 16, 128, 256]" = torch.ops.prims.convert_element_type.default(view_777, torch.bfloat16);  view_777 = None
        permute_275: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_699, [0, 2, 1, 3]);  convert_element_type_699 = None
        permute_276: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(convert_element_type_698, [0, 2, 1, 3]);  convert_element_type_698 = None
        slice_110: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_275, 3, 0, 64)
        slice_111: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_275, 3, 64, 256);  permute_275 = None
        slice_112: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(permute_276, 3, 0, 64)
        slice_113: "bf16[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(permute_276, 3, 64, 256);  permute_276 = None
        expand_55: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg566_1, _shape_param_834);  arg566_1 = _shape_param_834 = None
        clone_81: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_778: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_81, _shape_param_835);  clone_81 = _shape_param_835 = None
        mul_643: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_110, view_778)
        view_779: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_643, _shape_param_836);  mul_643 = _shape_param_836 = None
        select_108: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_779, -1, 0)
        select_109: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_779, -1, 1);  view_779 = None
        neg_82: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_108);  select_108 = None
        slice_scatter_216: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_82, 3, 1, 9223372036854775807, 2);  neg_82 = None
        slice_scatter_217: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_109, 3, 0, 9223372036854775807, 2);  select_109 = None
        add_384: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_216, slice_scatter_217);  slice_scatter_216 = slice_scatter_217 = None
        expand_56: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(arg567_1, _shape_param_837);  arg567_1 = _shape_param_837 = None
        clone_82: "bf16[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_780: "bf16[1, 128, 1, 64]" = torch.ops.aten.view.default(clone_82, _shape_param_838);  clone_82 = _shape_param_838 = None
        mul_644: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_110, view_780);  slice_110 = None
        add_385: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_384, mul_644);  add_384 = mul_644 = None
        mul_645: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_112, view_778);  view_778 = None
        view_781: "bf16[1, 128, 16, 32, 2]" = torch.ops.aten.view.default(mul_645, _shape_param_839);  mul_645 = _shape_param_839 = None
        select_110: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_781, -1, 0)
        select_111: "bf16[1, 128, 16, 32]" = torch.ops.aten.select.int(view_781, -1, 1);  view_781 = None
        neg_83: "bf16[1, 128, 16, 32]" = torch.ops.aten.neg.default(select_110);  select_110 = None
        slice_scatter_218: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, neg_83, 3, 1, 9223372036854775807, 2);  neg_83 = None
        slice_scatter_219: "bf16[1, 128, 16, 64]" = torch.ops.aten.slice_scatter.default(full_2, select_111, 3, 0, 9223372036854775807, 2);  full_2 = select_111 = None
        add_386: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(slice_scatter_218, slice_scatter_219);  slice_scatter_218 = slice_scatter_219 = None
        mul_646: "bf16[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_112, view_780);  slice_112 = view_780 = None
        add_387: "bf16[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(add_386, mul_646);  add_386 = mul_646 = None
        slice_scatter_220: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_111, 3, 64, 9223372036854775807);  slice_111 = None
        slice_scatter_221: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_385, 3, 0, 64);  add_385 = None
        add_388: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_220, slice_scatter_221);  slice_scatter_220 = slice_scatter_221 = None
        slice_scatter_222: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, slice_113, 3, 64, 9223372036854775807);  slice_113 = None
        slice_scatter_223: "bf16[1, 128, 16, 256]" = torch.ops.aten.slice_scatter.default(full_3, add_387, 3, 0, 64);  full_3 = add_387 = None
        add_389: "bf16[1, 128, 16, 256]" = torch.ops.aten.add.Tensor(slice_scatter_222, slice_scatter_223);  slice_scatter_222 = slice_scatter_223 = None
        permute_277: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_773, [0, 2, 1, 3]);  view_773 = None
        clone_83: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_277, memory_format = torch.contiguous_format);  permute_277 = None
        view_782: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_83, _shape_param_840);  clone_83 = _shape_param_840 = None
        view_783: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_389, _shape_param_841);  add_389 = _shape_param_841 = None
        view_784: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(add_388, _shape_param_842);  add_388 = _shape_param_842 = None
        view_785: "bf16[128, 4096]" = torch.ops.aten.view.default(view_782, _shape_param_843);  view_782 = _shape_param_843 = None
        permute_278: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_785, [1, 0])
        mm_332: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_278, arg558_1);  permute_278 = None
        mm_333: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_785, arg568_1);  view_785 = arg568_1 = None
        view_786: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_333, _shape_param_844);  mm_333 = _shape_param_844 = None
        convert_element_type_700: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_786, torch.float32);  view_786 = None
        add_390: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(convert_element_type_691, convert_element_type_700);  convert_element_type_691 = convert_element_type_700 = None
        convert_element_type_701: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_332, torch.float32);  mm_332 = None
        view_787: "bf16[128, 4096]" = torch.ops.aten.view.default(view_783, _shape_param_845);  view_783 = _shape_param_845 = None
        permute_279: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_787, [1, 0])
        mm_334: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_279, arg558_1);  permute_279 = None
        mm_335: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_787, arg569_1);  view_787 = arg569_1 = None
        view_788: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_335, _shape_param_846);  mm_335 = _shape_param_846 = None
        convert_element_type_702: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_788, torch.float32);  view_788 = None
        add_391: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_390, convert_element_type_702);  add_390 = convert_element_type_702 = None
        convert_element_type_703: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_334, torch.float32);  mm_334 = None
        view_789: "bf16[128, 4096]" = torch.ops.aten.view.default(view_784, _shape_param_847);  view_784 = _shape_param_847 = None
        permute_280: "bf16[4096, 128]" = torch.ops.aten.permute.default(view_789, [1, 0])
        mm_336: "bf16[4096, 4096]" = torch.ops.aten.mm.default(permute_280, arg558_1);  permute_280 = arg558_1 = None
        mm_337: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_789, arg570_1);  view_789 = arg570_1 = None
        view_790: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_337, _shape_param_848);  mm_337 = _shape_param_848 = None
        convert_element_type_704: "f32[1, 128, 4096]" = torch.ops.prims.convert_element_type.default(view_790, torch.float32);  view_790 = None
        add_392: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_391, convert_element_type_704);  add_391 = convert_element_type_704 = None
        convert_element_type_705: "f32[4096, 4096]" = torch.ops.prims.convert_element_type.default(mm_336, torch.float32);  mm_336 = None
        mul_647: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_392, arg571_1);  arg571_1 = None
        mul_648: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_647, 4096)
        sum_199: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_647, [2], True)
        sub_88: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(arg551_1, arg572_1);  arg551_1 = arg572_1 = None
        mul_649: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_88, arg573_1);  sub_88 = None
        mul_650: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_647, mul_649);  mul_647 = None
        sum_200: "f32[1, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_650, [2], True);  mul_650 = None
        mul_651: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_649, sum_200);  sum_200 = None
        sub_89: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(mul_648, sum_199);  mul_648 = sum_199 = None
        sub_90: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(sub_89, mul_651);  sub_89 = mul_651 = None
        div_30: "f32[1, 128, 1]" = torch.ops.aten.div.Tensor(arg573_1, 4096);  arg573_1 = None
        mul_652: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(div_30, sub_90);  div_30 = sub_90 = None
        mul_653: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(add_392, mul_649);  mul_649 = None
        sum_201: "f32[4096]" = torch.ops.aten.sum.dim_IntList(mul_653, [0, 1]);  mul_653 = None
        sum_202: "f32[4096]" = torch.ops.aten.sum.dim_IntList(add_392, [0, 1]);  add_392 = None
        add_393: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_379, mul_652);  add_379 = mul_652 = None
        ge: "b8[1, 128]" = torch.ops.aten.ge.Scalar(arg574_1, 0)
        lt: "b8[1, 128]" = torch.ops.aten.lt.Scalar(arg574_1, 50400)
        bitwise_and: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_1: "b8[1, 128]" = torch.ops.aten.ne.Scalar(arg574_1, -1)
        bitwise_and_1: "b8[1, 128]" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_1);  bitwise_and = ne_1 = None
        unsqueeze_1: "b8[1, 128, 1]" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_4: "f32[50400, 4096]" = torch.ops.aten.full.default(_shape_param_849, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_849 = None
        _unsafe_masked_index_put_accumulate: "f32[50400, 4096]" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_4, unsqueeze_1, [arg574_1], add_393);  full_4 = unsqueeze_1 = arg574_1 = add_393 = None
        return (convert_element_type_4, convert_element_type_5, sum_5, sum_6, convert_element_type_9, convert_element_type_10, convert_element_type_17, convert_element_type_18, convert_element_type_19, convert_element_type_26, convert_element_type_28, convert_element_type_30, sum_12, sum_13, convert_element_type_34, convert_element_type_35, convert_element_type_42, convert_element_type_43, convert_element_type_44, convert_element_type_51, convert_element_type_53, convert_element_type_55, sum_19, sum_20, convert_element_type_59, convert_element_type_60, convert_element_type_67, convert_element_type_68, convert_element_type_69, convert_element_type_76, convert_element_type_78, convert_element_type_80, sum_26, sum_27, convert_element_type_84, convert_element_type_85, convert_element_type_92, convert_element_type_93, convert_element_type_94, convert_element_type_101, convert_element_type_103, convert_element_type_105, sum_33, sum_34, convert_element_type_109, convert_element_type_110, convert_element_type_117, convert_element_type_118, convert_element_type_119, convert_element_type_126, convert_element_type_128, convert_element_type_130, sum_40, sum_41, convert_element_type_134, convert_element_type_135, convert_element_type_142, convert_element_type_143, convert_element_type_144, convert_element_type_151, convert_element_type_153, convert_element_type_155, sum_47, sum_48, convert_element_type_159, convert_element_type_160, convert_element_type_167, convert_element_type_168, convert_element_type_169, convert_element_type_176, convert_element_type_178, convert_element_type_180, sum_54, sum_55, convert_element_type_184, convert_element_type_185, convert_element_type_192, convert_element_type_193, convert_element_type_194, convert_element_type_201, convert_element_type_203, convert_element_type_205, sum_61, sum_62, convert_element_type_209, convert_element_type_210, convert_element_type_217, convert_element_type_218, convert_element_type_219, convert_element_type_226, convert_element_type_228, convert_element_type_230, sum_68, sum_69, convert_element_type_234, convert_element_type_235, convert_element_type_242, convert_element_type_243, convert_element_type_244, convert_element_type_251, convert_element_type_253, convert_element_type_255, sum_75, sum_76, convert_element_type_259, convert_element_type_260, convert_element_type_267, convert_element_type_268, convert_element_type_269, convert_element_type_276, convert_element_type_278, convert_element_type_280, sum_82, sum_83, convert_element_type_284, convert_element_type_285, convert_element_type_292, convert_element_type_293, convert_element_type_294, convert_element_type_301, convert_element_type_303, convert_element_type_305, sum_89, sum_90, convert_element_type_309, convert_element_type_310, convert_element_type_317, convert_element_type_318, convert_element_type_319, convert_element_type_326, convert_element_type_328, convert_element_type_330, sum_96, sum_97, convert_element_type_334, convert_element_type_335, convert_element_type_342, convert_element_type_343, convert_element_type_344, convert_element_type_351, convert_element_type_353, convert_element_type_355, sum_103, sum_104, convert_element_type_359, convert_element_type_360, convert_element_type_367, convert_element_type_368, convert_element_type_369, convert_element_type_376, convert_element_type_378, convert_element_type_380, sum_110, sum_111, convert_element_type_384, convert_element_type_385, convert_element_type_392, convert_element_type_393, convert_element_type_394, convert_element_type_401, convert_element_type_403, convert_element_type_405, sum_117, sum_118, convert_element_type_409, convert_element_type_410, convert_element_type_417, convert_element_type_418, convert_element_type_419, convert_element_type_426, convert_element_type_428, convert_element_type_430, sum_124, sum_125, convert_element_type_434, convert_element_type_435, convert_element_type_442, convert_element_type_443, convert_element_type_444, convert_element_type_451, convert_element_type_453, convert_element_type_455, sum_131, sum_132, convert_element_type_459, convert_element_type_460, convert_element_type_467, convert_element_type_468, convert_element_type_469, convert_element_type_476, convert_element_type_478, convert_element_type_480, sum_138, sum_139, convert_element_type_484, convert_element_type_485, convert_element_type_492, convert_element_type_493, convert_element_type_494, convert_element_type_501, convert_element_type_503, convert_element_type_505, sum_145, sum_146, convert_element_type_509, convert_element_type_510, convert_element_type_517, convert_element_type_518, convert_element_type_519, convert_element_type_526, convert_element_type_528, convert_element_type_530, sum_152, sum_153, convert_element_type_534, convert_element_type_535, convert_element_type_542, convert_element_type_543, convert_element_type_544, convert_element_type_551, convert_element_type_553, convert_element_type_555, sum_159, sum_160, convert_element_type_559, convert_element_type_560, convert_element_type_567, convert_element_type_568, convert_element_type_569, convert_element_type_576, convert_element_type_578, convert_element_type_580, sum_166, sum_167, convert_element_type_584, convert_element_type_585, convert_element_type_592, convert_element_type_593, convert_element_type_594, convert_element_type_601, convert_element_type_603, convert_element_type_605, sum_173, sum_174, convert_element_type_609, convert_element_type_610, convert_element_type_617, convert_element_type_618, convert_element_type_619, convert_element_type_626, convert_element_type_628, convert_element_type_630, sum_180, sum_181, convert_element_type_634, convert_element_type_635, convert_element_type_642, convert_element_type_643, convert_element_type_644, convert_element_type_651, convert_element_type_653, convert_element_type_655, sum_187, sum_188, convert_element_type_659, convert_element_type_660, convert_element_type_667, convert_element_type_668, convert_element_type_669, convert_element_type_676, convert_element_type_678, convert_element_type_680, sum_194, sum_195, convert_element_type_684, convert_element_type_685, convert_element_type_692, convert_element_type_693, convert_element_type_694, convert_element_type_701, convert_element_type_703, convert_element_type_705, sum_201, sum_202, _unsafe_masked_index_put_accumulate)



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
